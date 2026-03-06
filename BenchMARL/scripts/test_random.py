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
from torchrl.envs.utils import step_mdp

# ==========================================================
# 随机策略：基于 Action Mask 的随机采样
# ==========================================================
def masked_random_actions(action_mask: torch.Tensor) -> torch.Tensor:
    """
    根据动作掩码进行随机采样。
    action_mask 维度: [E, A, K] (Env, Agent, Action_Space)
    """
    probs = action_mask.float()
    # 检查是否存在所有动作都被掩蔽的极端情况
    if (probs.sum(dim=-1) == 0).any():
        bad = (probs.sum(dim=-1) == 0).nonzero(as_tuple=False)
        raise RuntimeError(f"CRITICAL: some (env,agent) have all actions masked out: {bad.tolist()}")

    E, A, K = probs.shape
    flat = probs.reshape(E * A, K)
    # torch.multinomial 在 flat 上进行采样
    acts = torch.multinomial(flat, 1).squeeze(-1).reshape(E, A).to(torch.int64)
    return acts

# ==========================================================
# 核心算法：Held-Karp (用于计算理论最优基准)
# ==========================================================
def calculate_oracle_tsp(G, start_node, goal_nodes, fallback_penalty=5.0):
    nodes = [start_node] + list(goal_nodes)
    num_nodes = len(nodes)
    num_goals = len(goal_nodes)
    if num_goals == 0: return 0.0

    dist_matrix = np.full((num_nodes, num_nodes), float('inf'))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                dist_matrix[i, j] = 0
                continue
            try:
                d = nx.shortest_path_length(G, nodes[i], nodes[j], weight='weight')
                dist_matrix[i, j] = d
            except nx.NetworkXNoPath:
                dist_matrix[i, j] = float('inf')

    dp = {}
    for i in range(num_goals):
        d = dist_matrix[0, i + 1]
        dp[(1 << i, i)] = d if d != float('inf') else fallback_penalty

    for size in range(2, num_goals + 1):
        for subset in itertools.combinations(range(num_goals), size):
            mask = 0
            for bit in subset: mask |= (1 << bit)
            for next_goal in subset:
                prev_mask = mask ^ (1 << next_goal)
                res = [dp[(prev_mask, lg)] + (dist_matrix[lg+1, next_goal+1] if dist_matrix[lg+1, next_goal+1] != float('inf') else fallback_penalty)
                       for lg in subset if lg != next_goal]
                dp[(mask, next_goal)] = min(res) if res else fallback_penalty

    full_mask = (1 << num_goals) - 1
    final_res = [dp[(full_mask, i)] for i in range(num_goals)]
    return float(min(final_res)) if final_res else 0.0

# ==========================================================
# 随机探索评估脚本
# ==========================================================
def run_random_exploration(config_path: str, num_episodes: int = 100, use_actual_oracle: bool = True):
    config_ptr = Path(config_path)
    # 结果保存至随机探索专用目录
    eval_dir = config_ptr.parent / "random_exploration_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cpu")
    print(f"--- Running Random Exploration Baseline | Oracle: {'Exact' if use_actual_oracle else 'Approx'} ---")

    # 1. 加载配置以创建环境
    with open(config_path, "rb") as f:
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

    # 实例化实验对象以获取测试环境
    exp = Experiment(
        task=task, algorithm_config=algorithm_config, model_config=model_config,
        seed=seed, config=experiment_config, callbacks=callbacks,
        critic_model_config=critic_model_config
    )
    env = exp.test_env.to(device)
    num_agents = env.num_agents

    gif_dir = eval_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    # 指标追踪
    all_rewards, all_costs, all_penalties = [], [], []
    all_lengths, all_oracles, all_cr = [], [], []
    success_cr = [] 
    episode_details = [] 
    successes = 0
    total_episodes = 0

    for ep_batch in range(num_episodes):
        td = env.reset()
        E = env.num_envs
        
        # --- Oracle 计算 ---
        batch_oracle_costs = []
        for i in range(E):
            gd = env.graph_data[i]
            goal_nodes = gd.g.tolist()
            
            # [防御代码]
            if use_actual_oracle and len(goal_nodes) > 15:
                raise RuntimeError(f"Goal nodes ({len(goal_nodes)}) too high for Held-Karp. Disable use_actual_oracle.")

            true_mask = (gd.edge_realisation != 3) 
            G = nx.Graph()
            G.add_nodes_from(range(env.max_nodes))
            edge_index = gd.edge_index[:, true_mask].cpu().numpy()
            weights = gd.A[true_mask].cpu().numpy()
            for u, v, w in zip(edge_index[0], edge_index[1], weights):
                G.add_edge(u, v, weight=w)
            
            val = calculate_oracle_tsp(G, gd.s[0].item(), goal_nodes) if use_actual_oracle else \
                  sum(nx.shortest_path_length(G, gd.s[0].item(), g, weight='weight') for g in goal_nodes)
            batch_oracle_costs.append(max(float(val), 1e-6))

        # --- 随机运行循环 ---
        env_finished = torch.zeros(E, dtype=torch.bool, device=device)
        env_rewards = torch.zeros(E, device=device)
        env_penalties = torch.zeros(E, device=device)
        env_lengths = torch.zeros(E, device=device)
        env_success = torch.zeros(E, dtype=torch.bool, device=device)
        env_coverage = torch.zeros(E, device=device)
        frames = [[] for _ in range(E)]

        for t in range(env.num_phases):
            # 渲染
            for i in range(E):
                if not env_finished[i]:
                    frames[i].append(Image.fromarray(env.render(env_idx=i)))

            # 获取动作掩码并随机执行
            action_mask = td["agents", "action_mask"]
            actions = masked_random_actions(action_mask)
            
            # 构造输入 TensorDict
            td.set(("agents", "action"), actions)
            
            td = env.step(td)
            next_td = td["next"]
            done = next_td["done"].squeeze(-1)
            just_finished = done & (~env_finished)

            for i in range(E):
                if not env_finished[i]:
                    env_rewards[i] += next_td["agents", "reward"].sum(dim=1).squeeze(-1)[i]
                    env_lengths[i] += 1
                    if just_finished[i]:
                        env_penalties[i] = next_td["penalty"][i].item()
                        env_success[i] = torch.equal(env.destination_mask[i], env.goals_visited_mask[i])
                        env_coverage[i] = float(env.goals_visited_mask[i].sum().item() / max(len(env.graph_data[i].g), 1))

            env_finished |= done
            if env_finished.all(): break
            td = step_mdp(td, keep_other=True)
            
        # --- 数据持久化 ---
        for i in range(E):
            ep_idx = total_episodes
            total_episodes += 1
            cost = -(env_rewards[i].item() + env_penalties[i].item())
            oracle = batch_oracle_costs[i]
            cr = cost / oracle
            is_success = bool(env_success[i].item())

            all_rewards.append(env_rewards[i].item()); all_costs.append(cost)
            all_penalties.append(env_penalties[i].item()); all_lengths.append(env_lengths[i].item())
            all_oracles.append(oracle); all_cr.append(cr)

            if is_success:
                successes += 1
                success_cr.append(cr)
            
            episode_details.append({
                "idx": ep_idx, "success": is_success, "cost": cost,
                "penalty": env_penalties[i].item(), "len": env_lengths[i].item(),
                "oracle": oracle, "coverage": env_coverage[i].item(), "cr": cr
            })

            if len(frames[i]) > 0:
                    status = "success" if is_success else "fail"
                    frames[i][0].save(gif_dir / f"ep_{ep_idx:03d}_{status}.gif", save_all=True, append_images=frames[i][1:], duration=200, loop=0)

    # --- 写入结果报告 ---
    with open(eval_dir / "random_results.md", "w") as f:
        f.write(f"# Random Exploration Baseline Results\n")
        f.write(f"This baseline shows the difficulty of the task when actions are chosen randomly among valid options.\n\n")
        f.write(f"## Summary Metrics\n")
        f.write(f"- Success Rate: {successes}/{total_episodes} ({(successes/total_episodes)*100:.2f}%)\n")
        f.write(f"- Average Cost: {np.mean(all_costs):.4f}\n")
        f.write(f"- Average Oracle Cost: {np.mean(all_oracles):.4f}\n")
        f.write(f"- **Average Successful Competitive Ratio**: {np.mean(success_cr) if success_cr else 'N/A'}\n")
        f.write(f"- Average Total CR: {np.mean(all_cr):.4f}\n\n")

        f.write(f"## Detailed Log\n")
        f.write(f"| Index | Success | Cost | Oracle | Coverage | CR |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for d in episode_details:
            f.write(f"| {d['idx']} | {'✅' if d['success'] else '❌'} | {d['cost']:.2f} | {d['oracle']:.2f} | {d['coverage']:.2f} | {d['cr']:.4f} |\n")

    print(f"Random exploration baseline complete. Results in {eval_dir}")

if __name__ == "__main__":
    # 使用与测试相同的配置文件路径
    CONFIG_PATH = "/home/pemb7543/DeC_MACTP/BenchMARL/MACTP_test/mappo_default_graphactorgnn__26_02_25-21_28_43_24478972/config.pkl"
    run_random_exploration(CONFIG_PATH, num_episodes=100, use_actual_oracle=False)