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

import argparse
import json
from typing import  Optional

from benchmarl.experiment.ippo_scheduler_callback import IppoSchedulerCallback

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

    # 1. 预计算距离矩阵
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

    # 2. DP 初始化
    dp = {}
    for i in range(num_goals):
        d = dist_matrix[0, i + 1]
        dp[(1 << i, i)] = d if d != float('inf') else fallback_penalty

    # 3. 状态转移
    for size in range(2, num_goals + 1):
        for subset in itertools.combinations(range(num_goals), size):
            mask = sum(1 << bit for bit in subset)
            for next_goal in subset:
                prev_mask = mask ^ (1 << next_goal)
                best_dist = min(
                    dp[(prev_mask, last_goal)] + (
                        dist_matrix[last_goal + 1, next_goal + 1]
                        if dist_matrix[last_goal + 1, next_goal + 1] != float('inf')
                        else fallback_penalty
                    )
                    for last_goal in subset if last_goal != next_goal
                )
                dp[(mask, next_goal)] = best_dist

    # 4. 最终结果
    full_mask = (1 << num_goals) - 1
    results = [dp.get((full_mask, i), float('inf')) for i in range(num_goals)]
    return float(min(r for r in results if r != float('inf'))) if results else 0.0


def build_graph_from_env(gd, max_nodes):
    """从环境图数据构建 NetworkX 图（排除阻断边）。"""
    true_mask = (gd.edge_realisation != 3)
    G = nx.Graph()
    G.add_nodes_from(range(max_nodes))
    edge_index = gd.edge_index[:, true_mask].cpu().numpy()
    weights = gd.A[true_mask].cpu().numpy()
    for u, v, w in zip(edge_index[0], edge_index[1], weights):
        G.add_edge(u, v, weight=w)
    return G


def compute_oracle_for_env(gd, max_nodes, use_actual_oracle, fallback_penalty=5.0):
    """为单个环境计算 oracle 成本。"""
    start_node = gd.s[0].item()
    goal_nodes = gd.g.tolist()
    G = build_graph_from_env(gd, max_nodes)

    if use_actual_oracle:
        if len(goal_nodes) > 15:
            raise RuntimeError(
                f"检测到 {len(goal_nodes)} 个目标点,Held-Karp 计算量将爆炸。"
                f"请设置 use_actual_oracle=False 或减少目标数量。"
            )
        oracle_val = calculate_oracle_tsp(G, start_node, goal_nodes, fallback_penalty)
    else:
        oracle_val = 0.0
        for g in goal_nodes:
            try:
                oracle_val += nx.shortest_path_length(G, start_node, g, weight='weight')
            except nx.NetworkXNoPath:
                oracle_val += fallback_penalty

    return max(float(oracle_val), 1e-6)

# ==========================================================
# 评估主函数
# ==========================================================
def run_evaluation(
    checkpoint_path: str,
    graph_path: str,
    num_episodes: int = 20,
    use_actual_oracle: bool = True,
    save_gifs: bool = True,
    yaml_path: Optional[str] = None,
):
    checkpoint_ptr = Path(checkpoint_path)
    eval_dir = checkpoint_ptr.parent.parent / "new_test_evaluate_results" / graph_path
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"{'='*60}")
    print(f"评估模式: CPU | Oracle: {'精确(Held-Karp)' if use_actual_oracle else '近似(贪心)'}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"输出目录:   {eval_dir}")
    print(f"{'='*60}")

    # ----------------------------------------------------------
    # 1. 加载配置
    # ----------------------------------------------------------
    config_file = checkpoint_ptr.parent.parent / "config.pkl"
    if not config_file.exists():
        raise FileNotFoundError(f"Config 文件未找到: {config_file}")

    with open(config_file, "rb") as f:
        _               = pickle.load(f)
        task = MactpTask.TEST.get_from_yaml(path=yaml_path)
        _               = pickle.load(f)
        algorithm_config = pickle.load(f)
        model_config    = pickle.load(f)
        seed            = pickle.load(f)
        experiment_config = pickle.load(f)
        critic_model_config = pickle.load(f)
        callbacks       = pickle.load(f)

    experiment_config.sampling_device = "cpu"
    experiment_config.train_device    = "cpu"
    experiment_config.evaluation_episodes = 1
    experiment_config.loggers = []

    exp = Experiment(
        task=task, algorithm_config=algorithm_config, model_config=model_config,
        seed=seed, config=experiment_config, callbacks=callbacks,
        critic_model_config=critic_model_config,
    )
    exp.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    policy = exp.policy.to(device)
    env    = exp.test_env.to(device)

    if save_gifs:
        gif_dir = eval_dir / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 2. 指标追踪
    # ----------------------------------------------------------
    all_rewards, all_costs, all_penalties = [], [], []
    all_lengths, all_oracles, all_cr      = [], [], []
    success_cr     = []
    episode_details = []
    successes      = 0
    total_episodes = 0

    # ----------------------------------------------------------
    # 3. 评估循环
    # ----------------------------------------------------------
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        for batch_idx in range(num_episodes):
            td = env.reset()
            E  = env.num_envs

            # --- 缓存每个环境的图数据（修复 gd 变量污染问题）---
            graph_data_cache = [env.graph_data[i] for i in range(E)]

            # --- Oracle 计算 ---
            batch_oracle_costs = []
            for i in range(E):
                oracle_val = compute_oracle_for_env(
                    graph_data_cache[i], env.max_nodes, use_actual_oracle
                )
                batch_oracle_costs.append(oracle_val)

            # --- 环境运行循环 ---
            env_finished = torch.zeros(E, dtype=torch.bool, device=device)
            env_rewards  = torch.zeros(E, device=device)
            env_penalties = torch.zeros(E, device=device)
            env_lengths  = torch.zeros(E, device=device)
            env_success  = torch.zeros(E, dtype=torch.bool, device=device)
            env_coverage = torch.zeros(E, device=device)
            frames = [[] for _ in range(E)]

            for t in range(env.num_phases):
                # 渲染未完成的环境
                if save_gifs:
                    for i in range(E):
                        if not env_finished[i]:
                            rgb = env.render(env_idx=i)
                            frames[i].append(Image.fromarray(rgb))

                td = policy(td)
                td = env.step(td)
                next_td       = td["next"]
                done          = next_td["done"].squeeze(-1)
                # reward shape: [E, num_agents, 1] -> [E]
                current_rewards = next_td["agents", "reward"].sum(dim=1).squeeze(-1)
                just_finished = done & (~env_finished)

                for i in range(E):
                    if not env_finished[i]:
                        env_rewards[i] += current_rewards[i]
                        env_lengths[i] += 1
                        if just_finished[i]:
                            env_penalties[i] = next_td["penalty"][i].item()
                            gd_i = graph_data_cache[i]   # ✅ 使用缓存，避免变量污染
                            env_success[i]  = torch.equal(
                                env.destination_mask[i], env.goals_visited_mask[i]
                            )
                            env_coverage[i] = float(
                                env.goals_visited_mask[i].sum().item() / max(len(gd_i.g), 1)
                            )
                            # ✅ 在此处渲染终止帧
                            if save_gifs:
                                rgb = env.render(env_idx=i)
                                frames[i].append(Image.fromarray(rgb))

                env_finished |= done
                if env_finished.all():
                    break
                td = step_mdp(td, keep_other=True)

            # --- 结果记录 ---
            for i in range(E):
                ep_idx = total_episodes
                total_episodes += 1

                reward_val  = env_rewards[i].item()
                penalty_val = env_penalties[i].item()
                ep_cost     = -(reward_val + penalty_val)
                oracle      = batch_oracle_costs[i]
                cr          = ep_cost / oracle
                is_success  = bool(env_success[i].item())

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
                    "oracle": oracle, "coverage": env_coverage[i].item(), "cr": cr,
                })

                if save_gifs and frames[i]:
                    status = "success" if is_success else "fail"
                    frames[i][0].save(
                        gif_dir / f"ep_{ep_idx:03d}_{status}.gif",
                        save_all=True, append_images=frames[i][1:],
                        duration=200, loop=0,
                    )

            # 进度日志
            current_sr = successes / total_episodes * 100
            print(
                f"[Batch {batch_idx+1:>4d}/{num_episodes}] "
                f"Episodes: {total_episodes} | "
                f"SR: {current_sr:.1f}% | "
                f"CR(last): {cr:.4f}"
            )

    # ----------------------------------------------------------
    # 4. 统计与输出
    # ----------------------------------------------------------
    avg_reward     = np.mean(all_rewards)
    avg_cost       = np.mean(all_costs)
    avg_penalty    = np.mean(all_penalties)
    avg_oracle     = np.mean(all_oracles)
    avg_cr         = np.mean(all_cr)
    avg_success_cr = np.mean(success_cr) if success_cr else float('nan')
    sr             = successes / total_episodes

    print(f"\n{'='*60}")
    print(f"评估完成 | 共 {total_episodes} episodes")
    print(f"  成功率:          {successes}/{total_episodes} ({sr*100:.2f}%)")
    print(f"  平均 Reward:     {avg_reward:.4f}")
    print(f"  平均 Cost:       {avg_cost:.4f}")
    print(f"  平均 Penalty:    {avg_penalty:.4f}")
    print(f"  平均 Oracle:     {avg_oracle:.4f}")
    print(f"  平均 CR (全部):  {avg_cr:.4f}")
    print(f"  平均 CR (成功):  {avg_success_cr:.4f}")
    print(f"{'='*60}\n")

    # Markdown 报告
    with open(eval_dir / "results.md", "w") as f:
        f.write(f"# Evaluation Results\n\n")
        f.write(f"- **Checkpoint**: `{checkpoint_path}`\n")
        f.write(f"- **Graph**: `{graph_path}`\n")
        f.write(f"- **Oracle Mode**: {'Exact (Held-Karp)' if use_actual_oracle else 'Approx (Greedy)'}\n\n")
        f.write(f"## Summary Metrics\n\n")
        f.write(f"| Metric | Value |\n| :--- | :--- |\n")
        f.write(f"| Success Rate | {successes}/{total_episodes} ({sr*100:.2f}%) |\n")
        f.write(f"| Avg Reward | {avg_reward:.4f} |\n")
        f.write(f"| Avg Cost | {avg_cost:.4f} |\n")
        f.write(f"| Avg Penalty | {avg_penalty:.4f} |\n")
        f.write(f"| Avg Oracle Cost | {avg_oracle:.4f} |\n")
        f.write(f"| Avg CR (Total) | {avg_cr:.4f} |\n")
        f.write(f"| **Avg CR (Success)** | **{avg_success_cr:.4f}** |\n\n")
        f.write(f"## Episode Log\n\n")
        f.write(f"| # | ✅ | Cost | Penalty | Len | Oracle | Coverage | CR |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for d in episode_details:
            f.write(
                f"| {d['idx']} | {'✅' if d['success'] else '❌'} "
                f"| {d['cost']:.2f} | {d['penalty']:.2f} | {d['len']:.0f} "
                f"| {d['oracle']:.2f} | {d['coverage']:.2f} | {d['cr']:.4f} |\n"
            )

    # JSON 原始数据（便于后续分析）
    with open(eval_dir / "results.json", "w") as f:
        json.dump({
            "summary": {
                "success_rate": sr, "avg_reward": avg_reward,
                "avg_cost": avg_cost, "avg_penalty": avg_penalty,
                "avg_oracle": avg_oracle, "avg_cr": avg_cr,
                "avg_success_cr": avg_success_cr,
            },
            "episodes": episode_details,
        }, f, indent=2)

    print(f"结果已保存至: {eval_dir}")
    return sr, avg_cr, avg_success_cr


# ==========================================================
# CLI 入口
# ==========================================================
CHECKPOINTS = {
    "magnarl": "/home/pemb7543/DeC_MACTP/Train/magnarl_train_magnarlactorgnn__26_03_17-21_46_45_b48119e9/checkpoints/checkpoint_500000.pt",
    "ignarl":  "/home/pemb7543/DeC_MACTP/Train/ignarl_train_ignarlactorgnn__26_03_07-15_48_05_b2070763/checkpoints/checkpoint_500000.pt",
    "ippo":    "/home/pemb7543/DeC_MACTP/Train/ippo_train_graphactorgnn__26_03_07-16_13_57_400b8d5d/checkpoints/checkpoint_500000.pt",
    "mappo":   "/home/pemb7543/DeC_MACTP/Train/mappo_train_graphactorgnn__26_03_06-20_22_00_3920cbbe/checkpoints/checkpoint_500000.pt",
    "iql":     "/home/pemb7543/DeC_MACTP/Train/iql_train_graphqnet__26_03_06-20_46_33_6e77aa5f/checkpoints/checkpoint_500000.pt",
    "vdn":     "/home/pemb7543/DeC_MACTP/Train/vdn_train_graphqnet__26_03_06-22_20_37_172c509d/checkpoints/checkpoint_500000.pt",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test policy.")
    parser.add_argument("--graph",       type=str, default="graph_data_128_4_8")
    parser.add_argument("--episodes",    type=int, default=100)
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Checkpoint 路径，或算法名称: magnarl/ignarl/ippo/mappo/iql/vdn")
    parser.add_argument("--all",         action="store_true",
                        help="评估所有已注册的 checkpoint")
    parser.add_argument("--no-gif",      action="store_true",
                        help="禁用 GIF 保存（加速评估）")
    parser.add_argument("--exact-oracle", action="store_true",
                        help="使用精确 Held-Karp oracle（目标数量较少时使用）")
    parser.add_argument("--yaml-path", type=str, default=None,
    help="自定义 task yaml 路径，None 则使用 BenchMARL 默认配置")
    args = parser.parse_args()

    save_gifs    = not args.no_gif
    use_oracle   = args.exact_oracle

    if args.all:
         # 遍历 CHECKPOINTS 字典中所有算法，逐一评估
        for name, ckpt in CHECKPOINTS.items():
            print(f"\n>>> 评估算法: {name.upper()}")
            run_evaluation(
                checkpoint_path=ckpt, graph_path=args.graph,
                num_episodes=args.episodes,
                use_actual_oracle=use_oracle,
                save_gifs=save_gifs,
                yaml_path=args.yaml_path,
            )
    elif args.checkpoints is not None:
        # 接收一个字符串：
        # 情况A：算法名称（如 "magnarl"）→ 从字典查找路径
        # 情况B：直接传入完整 checkpoint 路径
        ckpt = CHECKPOINTS.get(args.checkpoints, args.checkpoints)
        run_evaluation(
            checkpoint_path=ckpt, graph_path=args.graph,
            num_episodes=args.episodes,
            use_actual_oracle=use_oracle,
            save_gifs=save_gifs,
            yaml_path=args.yaml_path,
        )
    else:
        # 默认评估 magnarl
        run_evaluation(
            checkpoint_path=CHECKPOINTS["magnarl"], graph_path=args.graph,
            num_episodes=args.episodes,
            use_actual_oracle=use_oracle,
            save_gifs=save_gifs,
        )