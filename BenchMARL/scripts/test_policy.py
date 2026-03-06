import torch
import numpy as np
import os
import pickle
import networkx as nx
import itertools
from pathlib import Path
from PIL import Image
from tensordict import TensorDict

from benchmarl.experiment import Experiment
from benchmarl.environments.mactp.common import MactpTask
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

# ==========================================================
# 核心算法：Held-Karp (动态规划求解 TSP/Steiner 路径)
# ==========================================================
def calculate_oracle_tsp(G, start_node, goal_nodes, fallback_penalty=5.0):
    """
    使用 Held-Karp 算法计算从 start_node 出发访问所有 goal_nodes 的最短路径成本。
    复杂度: O(2^G * G^2)，其中 G 是目标数量。
    """
    nodes = [start_node] + list(goal_nodes)
    num_nodes = len(nodes)
    num_goals = len(goal_nodes)
    
    if num_goals == 0:
        return 0.0

    # 1. 预计算距离矩阵 (仅针对起点和目标点)
    dist_matrix = np.full((num_nodes, num_nodes), float('inf'))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                dist_matrix[i, j] = 0
                continue
            try:
                # 使用 NetworkX 计算两点间最短路
                d = nx.shortest_path_length(G, nodes[i], nodes[j], weight='weight')
                dist_matrix[i, j] = d
            except nx.NetworkXNoPath:
                dist_matrix[i, j] = float('inf')

    # 2. 动态规划初始化
    # dp[(mask, last_node_idx)] = min_cost
    # mask 的第 k 位表示 goal_nodes[k] 是否已访问
    dp = {}

    # 初始状态：从起点到第一个目标点的距离
    for i in range(num_goals):
        d = dist_matrix[0, i + 1]
        dp[(1 << i, i)] = d if d != float('inf') else fallback_penalty

    # 3. 状态转移
    for size in range(2, num_goals + 1):
        for subset in itertools.combinations(range(num_goals), size):
            mask = 0
            for bit in subset:
                mask |= (1 << bit)
            
            for next_goal in subset:
                prev_mask = mask ^ (1 << next_goal)
                best_dist = float('inf')
                for last_goal in subset:
                    if last_goal == next_goal:
                        continue
                    
                    d_edge = dist_matrix[last_goal + 1, next_goal + 1]
                    cost_to_next = d_edge if d_edge != float('inf') else fallback_penalty
                    current_path_cost = dp[(prev_mask, last_goal)] + cost_to_next
                    if current_path_cost < best_dist:
                        best_dist = current_path_cost
                
                dp[(mask, next_goal)] = best_dist

    # 4. 最终结果：完成 full_mask 的所有状态中的最小值
    full_mask = (1 << num_goals) - 1
    res = [dp[(full_mask, i)] for i in range(num_goals)]
    return float(min(res)) if res else 0.0

# ==========================================================
# 评估主函数
# ==========================================================
def run_evaluation(checkpoint_path: str, num_episodes: int = 20, use_actual_oracle: bool = True):
    checkpoint_ptr = Path(checkpoint_path)
    eval_dir = checkpoint_ptr.parent.parent / "new_test_evaluate_results_graph_data_64_16_16"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cpu")
    print(f"--- Evaluation Mode: Force CPU | Oracle: {'Exact' if use_actual_oracle else 'Approx'} ---")

    # 1. Load Config
    config_file = checkpoint_ptr.parent.parent / "config.pkl"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "rb") as f:
        _ = pickle.load(f) 
        task = MactpTask.TEST.get_from_yaml()
        _ = pickle.load(f) 
        algorithm_config = pickle.load(f)
        model_config = pickle.load(f)
        seed = pickle.load(f)
        experiment_config = pickle.load(f)
        critic_model_config = pickle.load(f)
        callbacks = pickle.load(f)

    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cpu"
    experiment_config.evaluation_episodes = 1 
    experiment_config.loggers = []  

    exp = Experiment(
        task=task, algorithm_config=algorithm_config, model_config=model_config,
        seed=seed, config=experiment_config, callbacks=callbacks,
        critic_model_config=critic_model_config
    )
    exp.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    policy = exp.policy.to(device)
    env = exp.test_env.to(device)
    num_agents = env.num_agents

    gif_dir = eval_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    # Metric tracking
    all_rewards, all_costs, all_penalties = [], [], []
    all_lengths, all_oracles, all_cr = [], [], []
    success_cr = [] 
    episode_details = [] 
    successes = 0
    total_episodes = 0

    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        for ep_batch in range(num_episodes):
            td = env.reset()
            E = env.num_envs
            
            # --- Oracle 计算 ---
            batch_oracle_costs = []
            for i in range(E):
                gd = env.graph_data[i]
                start_node = gd.s[0].item()
                goal_nodes = gd.g.tolist()
                
                # [防御代码] 防止计算爆炸
                if use_actual_oracle and len(goal_nodes) > 15:
                    raise RuntimeError(
                        f"Detected {len(goal_nodes)} goals in episode {total_episodes + i}. "
                        f"Held-Karp computation will explode. Please set use_actual_oracle=False "
                        f"to use approximation or reduce number of goals."
                    )

                # 构建 NetworkX 真实图 (排除阻断边)
                true_mask = (gd.edge_realisation != 3) 
                G = nx.Graph()
                G.add_nodes_from(range(env.max_nodes))
                edge_index = gd.edge_index[:, true_mask].cpu().numpy()
                weights = gd.A[true_mask].cpu().numpy()
                for u, v, w in zip(edge_index[0], edge_index[1], weights):
                    G.add_edge(u, v, weight=w)
                
                if use_actual_oracle:
                    oracle_val = calculate_oracle_tsp(G, start_node, goal_nodes)
                else:
                    # 近似逻辑：起点到每个目标的距离之和
                    oracle_val = 0.0
                    for g in goal_nodes:
                        try:
                            oracle_val += nx.shortest_path_length(G, start_node, g, weight='weight')
                        except nx.NetworkXNoPath:
                            oracle_val += 5.0
                
                batch_oracle_costs.append(max(float(oracle_val), 1e-6))

            # --- 环境运行循环 ---
            env_finished = torch.zeros(E, dtype=torch.bool, device=device)
            env_rewards = torch.zeros(E, device=device)
            env_penalties = torch.zeros(E, device=device)
            env_lengths = torch.zeros(E, device=device)
            env_success = torch.zeros(E, dtype=torch.bool, device=device)
            env_coverage = torch.zeros(E, device=device)
            frames = [[] for _ in range(E)]

            for t in range(env.num_phases):
                for i in range(E):
                    if not env_finished[i]:
                        rgb = env.render(env_idx=i)
                        frames[i].append(Image.fromarray(rgb))

                td = policy(td)
                td = env.step(td)
                next_td = td["next"]
                done = next_td["done"].squeeze(-1)
                current_rewards = next_td["agents", "reward"].sum(dim=1).squeeze(-1)
                just_finished = done & (~env_finished)

                for i in range(E):
                    if not env_finished[i]:
                        env_rewards[i] += current_rewards[i]
                        env_lengths[i] += 1
                        if just_finished[i]:
                            env_penalties[i] = next_td["penalty"][i].item()
                            env_success[i] = torch.equal(env.destination_mask[i], env.goals_visited_mask[i])
                            env_coverage[i] = float(env.goals_visited_mask[i].sum().item() / max(len(gd.g), 1))

                env_finished |= done
                if env_finished.all(): break
                td = step_mdp(td, keep_other=True)
                
            # --- 结果记录 ---
            for i in range(E):
                ep_idx = total_episodes
                total_episodes += 1
                
                reward_val = env_rewards[i].item()
                penalty_val = env_penalties[i].item()
                ep_cost = -(reward_val + penalty_val)
                oracle = batch_oracle_costs[i]
                cr = ep_cost / oracle
                is_success = bool(env_success[i].item())

                all_rewards.append(reward_val)
                all_costs.append(ep_cost)
                all_penalties.append(penalty_val)
                all_lengths.append(env_lengths[i].item())
                all_oracles.append(oracle)
                all_cr.append(cr)

                if is_success:
                    successes += 1
                    success_cr.append(cr)
                
                episode_details.append({
                    "idx": ep_idx, "success": is_success, "cost": ep_cost,
                    "penalty": penalty_val, "len": env_lengths[i].item(),
                    "oracle": oracle, "coverage": env_coverage[i].item(), "cr": cr
                })

                if len(frames[i]) > 0:
                    status = "success" if is_success else "fail"
                    frames[i][0].save(gif_dir / f"ep_{ep_idx:03d}_{status}.gif", save_all=True, append_images=frames[i][1:], duration=200, loop=0)

    # --- 统计与输出 ---
    avg_reward = np.mean(all_rewards)
    avg_cost = np.mean(all_costs)
    avg_penalty = np.mean(all_penalties)
    avg_success_cr = np.mean(success_cr) if success_cr else 0.0
    avg_cr = np.mean(all_cr)

    with open(eval_dir / "results.md", "w") as f:
        f.write(f"## Summary Metrics\n")
        f.write(f"- Success Rate: {successes}/{total_episodes} ({(successes/total_episodes)*100:.2f}%)\n")
        f.write(f"- Average Reward: {avg_reward:.4f}\n")
        f.write(f"- Average Cost: {avg_cost:.4f}\n")
        f.write(f"- Average Penalty: {avg_penalty:.4f}\n")
        f.write(f"- Average Oracle Cost: {np.mean(all_oracles):.4f}\n")
        f.write(f"- Average Competitive Ratio (Total): {avg_cr:.4f}\n")
        f.write(f"- **Average Successful Competitive Ratio**: {avg_success_cr:.4f}\n\n")

        f.write(f"## Detailed Episode Log\n")
        f.write(f"| Index | Success | Cost | Penalty | Length | Oracle | Coverage | CR |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for d in episode_details:
            f.write(f"| {d['idx']} | {'✅' if d['success'] else '❌'} | {d['cost']:.2f} | {d['penalty']:.2f} | {d['len']:.0f} | {d['oracle']:.2f} | {d['coverage']:.2f} | {d['cr']:.4f} |\n")

    print(f"Evaluation finished. Results saved to {eval_dir}")

if __name__ == "__main__":
    # 请替换为您实际的 checkpoint 路径
    # IQL
    IQL_CHECKPOINT = "/home/pemb7543/DeC_MACTP/BenchMARL/MACTP_test/iql_default_graphqnet__26_02_26-02_30_35_2d3ae245/checkpoints/checkpoint_500000.pt"
    # VDN
    VDN_CHECKPOINT = "/home/pemb7543/DeC_MACTP/BenchMARL/MACTP_test/vdn_default_graphqnet__26_02_25-22_49_07_437d4bff/checkpoints/checkpoint_500000.pt"
    # MAPPO
    MAPPO_CHECKPOINT = "/home/pemb7543/DeC_MACTP/BenchMARL/MACTP_test/mappo_default_graphactorgnn__26_02_25-21_28_43_24478972/checkpoints/checkpoint_500000.pt"
    # run_evaluation(IQL_CHECKPOINT, num_episodes=100)
    # run_evaluation(VDN_CHECKPOINT, num_episodes=100)
    run_evaluation(MAPPO_CHECKPOINT, num_episodes=100, use_actual_oracle=False)