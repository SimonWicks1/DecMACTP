import torch
import numpy as np
import pickle
import networkx as nx
import itertools
import json
from pathlib import Path
from PIL import Image

from benchmarl.experiment import Experiment
from benchmarl.environments.mactp.common import MactpTask
from torchrl.envs.utils import step_mdp

import argparse
from typing import  Optional

# ==========================================================
# 随机策略：基于 Action Mask 的随机采样
# ==========================================================
def masked_random_actions(action_mask: torch.Tensor) -> torch.Tensor:
    """
    根据动作掩码进行随机采样。
    action_mask shape: [E, A, K]
    """
    probs = action_mask.float()
    all_masked = probs.sum(dim=-1) == 0
    if all_masked.any():
        bad = all_masked.nonzero(as_tuple=False)
        raise RuntimeError(
            f"存在 (env, agent) 的所有动作均被掩蔽: {bad.tolist()}"
        )
    E, A, K = probs.shape
    flat = probs.reshape(E * A, K)
    acts = torch.multinomial(flat, 1).squeeze(-1).reshape(E, A).to(torch.int64)
    return acts


# ==========================================================
# Held-Karp Oracle
# ==========================================================
def calculate_oracle_tsp(G, start_node, goal_nodes, fallback_penalty=5.0):
    nodes = [start_node] + list(goal_nodes)
    num_goals = len(goal_nodes)
    if num_goals == 0:
        return 0.0

    num_nodes = len(nodes)
    dist_matrix = np.full((num_nodes, num_nodes), float('inf'))
    for i in range(num_nodes):
        dist_matrix[i, i] = 0
        for j in range(i + 1, num_nodes):
            try:
                d = nx.shortest_path_length(G, nodes[i], nodes[j], weight='weight')
                dist_matrix[i, j] = dist_matrix[j, i] = d
            except nx.NetworkXNoPath:
                pass

    dp = {}
    for i in range(num_goals):
        d = dist_matrix[0, i + 1]
        dp[(1 << i, i)] = d if d != float('inf') else fallback_penalty

    for size in range(2, num_goals + 1):
        for subset in itertools.combinations(range(num_goals), size):
            mask = sum(1 << b for b in subset)
            for next_g in subset:
                prev_mask = mask ^ (1 << next_g)
                dp[(mask, next_g)] = min(
                    dp[(prev_mask, lg)] + (
                        dist_matrix[lg + 1, next_g + 1]
                        if dist_matrix[lg + 1, next_g + 1] != float('inf')
                        else fallback_penalty
                    )
                    for lg in subset if lg != next_g
                )

    full_mask = (1 << num_goals) - 1
    return float(min(dp.get((full_mask, i), float('inf')) for i in range(num_goals)))


def build_graph(gd, max_nodes):
    """从环境图数据构建 NetworkX 图（排除阻断边）。"""
    true_mask = gd.edge_realisation != 3
    G = nx.Graph()
    G.add_nodes_from(range(max_nodes))
    edge_index = gd.edge_index[:, true_mask].cpu().numpy()
    weights = gd.A[true_mask].cpu().numpy()
    for u, v, w in zip(edge_index[0], edge_index[1], weights):
        G.add_edge(u, v, weight=w)
    return G


def compute_oracle(gd, max_nodes, use_exact, fallback_penalty=5.0):
    """计算单个环境的 oracle 成本。"""
    start = gd.s[0].item()
    goals = gd.g.tolist()
    G = build_graph(gd, max_nodes)

    if use_exact:
        if len(goals) > 15:
            raise RuntimeError(
                f"目标数 {len(goals)} 超过 Held-Karp 安全上限(15)，"
                f"请使用 --approx-oracle。"
            )
        val = calculate_oracle_tsp(G, start, goals, fallback_penalty)
    else:
        val = 0.0
        for g in goals:
            try:
                val += nx.shortest_path_length(G, start, g, weight='weight')
            except nx.NetworkXNoPath:
                val += fallback_penalty

    return max(float(val), 1e-6)


# ==========================================================
# 难度分析：任务内在特征统计
# ==========================================================
def compute_task_difficulty_stats(gd, max_nodes, fallback_penalty=5.0):
    """
    计算任务内在难度指标，与策略性能无关。
    返回一个字典，包含图结构和任务复杂度信息。
    """
    G = build_graph(gd, max_nodes)
    start = gd.s[0].item()
    goals = gd.g.tolist()
    num_goals = len(goals)

    # 图连通性
    num_components = nx.number_connected_components(G)
    is_connected = num_components == 1

    # 起点到所有目标的最短路径
    dists_to_goals = []
    unreachable = 0
    for g in goals:
        try:
            dists_to_goals.append(
                nx.shortest_path_length(G, start, g, weight='weight')
            )
        except nx.NetworkXNoPath:
            unreachable += 1

    # 目标点之间的平均距离（衡量目标分散程度）
    inter_goal_dists = []
    for i, g1 in enumerate(goals):
        for g2 in goals[i + 1:]:
            try:
                inter_goal_dists.append(
                    nx.shortest_path_length(G, g1, g2, weight='weight')
                )
            except nx.NetworkXNoPath:
                pass

    return {
        "num_goals":            num_goals,
        "num_nodes":            G.number_of_nodes(),
        "num_edges":            G.number_of_edges(),
        "num_components":       num_components,
        "is_connected":         is_connected,
        "unreachable_goals":    unreachable,
        "avg_dist_to_goals":    float(np.mean(dists_to_goals)) if dists_to_goals else float('inf'),
        "max_dist_to_goals":    float(np.max(dists_to_goals))  if dists_to_goals else float('inf'),
        "avg_inter_goal_dist":  float(np.mean(inter_goal_dists)) if inter_goal_dists else float('inf'),
        "graph_density":        nx.density(G),
    }


# ==========================================================
# 主评估函数
# ==========================================================
def run_random_exploration(
    config_path: str,
    graph_path: str,
    num_episodes: int = 100,
    use_exact_oracle: bool = False,
    save_gifs: bool = True,
    analyze_difficulty: bool = True,
    yaml_path: Optional[str] = None,
):
    config_ptr = Path(config_path)
    eval_dir = config_ptr.parent / "random_exploration_results" / graph_path   # ← 修改此处
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"{'='*60}")
    print(f"随机探索 Baseline")
    print(f"Oracle 模式: {'精确 (Held-Karp)' if use_exact_oracle else '近似 (贪心)'}")
    print(f"难度分析: {'开启' if analyze_difficulty else '关闭'}")
    print(f"输出目录: {eval_dir}")
    print(f"{'='*60}")

    # ----------------------------------------------------------
    # 1. 加载配置 & 构建环境
    # ----------------------------------------------------------
    with open(config_path, "rb") as f:
        _                   = pickle.load(f)
        task = MactpTask.TEST.get_from_yaml(path=yaml_path) 
        _                   = pickle.load(f)
        algorithm_config    = pickle.load(f)
        model_config        = pickle.load(f)
        seed                = pickle.load(f)
        experiment_config   = pickle.load(f)
        critic_model_config = pickle.load(f)
        callbacks           = pickle.load(f)

    experiment_config.sampling_device = "cpu"
    experiment_config.train_device    = "cpu"
    experiment_config.evaluation_episodes = 1
    experiment_config.loggers = []

    exp = Experiment(
        task=task, algorithm_config=algorithm_config, model_config=model_config,
        seed=seed, config=experiment_config, callbacks=callbacks,
        critic_model_config=critic_model_config,
    )
    env = exp.test_env.to(device)

    if save_gifs:
        gif_dir = eval_dir / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 2. 指标追踪
    # ----------------------------------------------------------
    all_rewards, all_costs, all_penalties = [], [], []
    all_lengths, all_oracles, all_cr      = [], [], []
    success_cr      = []
    episode_details = []
    difficulty_stats = []
    successes        = 0
    total_episodes   = 0

    # ----------------------------------------------------------
    # 3. 评估主循环
    # ----------------------------------------------------------
    for batch_idx in range(num_episodes):
        td = env.reset()
        E  = env.num_envs

        # 缓存图数据，避免变量污染
        graph_data_cache = [env.graph_data[i] for i in range(E)]

        # --- Oracle & 难度统计 ---
        batch_oracle_costs = []
        for i in range(E):
            gd = graph_data_cache[i]
            batch_oracle_costs.append(
                compute_oracle(gd, env.max_nodes, use_exact_oracle)
            )
            if analyze_difficulty:
                difficulty_stats.append(
                    compute_task_difficulty_stats(gd, env.max_nodes)
                )

        # --- 随机运行 ---
        env_finished  = torch.zeros(E, dtype=torch.bool,  device=device)
        env_rewards   = torch.zeros(E,                     device=device)
        env_penalties = torch.zeros(E,                     device=device)
        env_lengths   = torch.zeros(E,                     device=device)
        env_success   = torch.zeros(E, dtype=torch.bool,  device=device)
        env_coverage  = torch.zeros(E,                     device=device)
        frames        = [[] for _ in range(E)]

        for t in range(env.num_phases):
            # 渲染动作执行前的状态
            if save_gifs:
                for i in range(E):
                    if not env_finished[i]:
                        frames[i].append(Image.fromarray(env.render(env_idx=i)))

            # 随机采样动作
            action_mask = td["agents", "action_mask"]
            actions     = masked_random_actions(action_mask)
            td.set(("agents", "action"), actions)

            td      = env.step(td)
            next_td = td["next"]
            done    = next_td["done"].squeeze(-1)
            rewards = next_td["agents", "reward"].sum(dim=1).squeeze(-1)
            just_finished = done & (~env_finished)

            for i in range(E):
                if not env_finished[i]:
                    env_rewards[i] += rewards[i]
                    env_lengths[i] += 1
                    if just_finished[i]:
                        env_penalties[i] = next_td["penalty"][i].item()
                        gd_i = graph_data_cache[i]
                        env_success[i]  = torch.equal(
                            env.destination_mask[i], env.goals_visited_mask[i]
                        )
                        env_coverage[i] = float(
                            env.goals_visited_mask[i].sum().item()
                            / max(len(gd_i.g), 1)
                        )
                        # ✅ 修复：渲染终止帧
                        if save_gifs:
                            frames[i].append(Image.fromarray(env.render(env_idx=i)))

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
            cost        = -(reward_val + penalty_val)
            oracle      = batch_oracle_costs[i]
            cr          = cost / oracle
            is_success  = bool(env_success[i].item())

            all_rewards.append(reward_val)
            all_costs.append(cost)
            all_penalties.append(penalty_val)
            all_lengths.append(env_lengths[i].item())
            all_oracles.append(oracle)
            all_cr.append(cr)

            if is_success:
                successes += 1
                success_cr.append(cr)

            episode_details.append({
                "idx":      ep_idx,
                "success":  is_success,
                "cost":     cost,
                "penalty":  penalty_val,
                "len":      env_lengths[i].item(),
                "oracle":   oracle,
                "coverage": env_coverage[i].item(),
                "cr":       cr,
            })

            if save_gifs and frames[i]:
                status = "success" if is_success else "fail"
                frames[i][0].save(
                    gif_dir / f"ep_{ep_idx:03d}_{status}.gif",
                    save_all=True, append_images=frames[i][1:],
                    duration=200, loop=0,
                )

        # 进度日志
        print(
            f"[Batch {batch_idx+1:>4d}/{num_episodes}] "
            f"SR: {successes}/{total_episodes} ({successes/total_episodes*100:.1f}%) | "
            f"CR(last): {cr:.4f}"
        )

    # ----------------------------------------------------------
    # 4. 汇总统计
    # ----------------------------------------------------------
    sr             = successes / total_episodes
    avg_cost       = np.mean(all_costs)
    avg_oracle     = np.mean(all_oracles)
    avg_cr         = np.mean(all_cr)
    avg_success_cr = np.mean(success_cr) if success_cr else float('nan')
    avg_reward     = np.mean(all_rewards)
    avg_penalty    = np.mean(all_penalties)
    avg_length     = np.mean(all_lengths)

    print(f"\n{'='*60}")
    print(f"随机 Baseline 完成 | 共 {total_episodes} episodes")
    print(f"  成功率:         {successes}/{total_episodes} ({sr*100:.2f}%)")
    print(f"  平均 Cost:      {avg_cost:.4f}")
    print(f"  平均 Oracle:    {avg_oracle:.4f}")
    print(f"  平均 CR (全部): {avg_cr:.4f}")
    print(f"  平均 CR (成功): {avg_success_cr:.4f}")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # 5. 难度分析报告
    # ----------------------------------------------------------
    difficulty_summary = {}
    if analyze_difficulty and difficulty_stats:
        keys = difficulty_stats[0].keys()
        difficulty_summary = {
            k: {
                "mean": float(np.mean([d[k] for d in difficulty_stats])),
                "std":  float(np.std( [d[k] for d in difficulty_stats])),
                "min":  float(np.min( [d[k] for d in difficulty_stats])),
                "max":  float(np.max( [d[k] for d in difficulty_stats])),
            }
            for k in keys if isinstance(difficulty_stats[0][k], (int, float))
        }
        # 不可达目标点的比例
        unreachable_ratio = np.mean(
            [d["unreachable_goals"] / max(d["num_goals"], 1)
             for d in difficulty_stats]
        )
        print("--- 任务难度分析 ---")
        print(f"  平均目标数:           {difficulty_summary['num_goals']['mean']:.2f}")
        print(f"  平均图密度:           {difficulty_summary['graph_density']['mean']:.4f}")
        print(f"  平均连通分量数:       {difficulty_summary['num_components']['mean']:.2f}")
        print(f"  不可达目标比例:       {unreachable_ratio*100:.2f}%")
        print(f"  平均起点到目标距离:   {difficulty_summary['avg_dist_to_goals']['mean']:.2f}")
        print(f"  平均目标间距离:       {difficulty_summary['avg_inter_goal_dist']['mean']:.2f}")

    # ----------------------------------------------------------
    # 6. 写入报告
    # ----------------------------------------------------------
    with open(eval_dir / "random_results.md", "w") as f:
        f.write("# Random Exploration Baseline\n\n")
        f.write(
            "> 注意：随机策略性能同时受**任务难度**和**动作空间大小**影响，"
            "不能单独作为任务难度的度量。请结合任务难度分析部分综合判断。\n\n"
        )
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | Value |\n| :--- | :--- |\n")
        f.write(f"| Success Rate | {successes}/{total_episodes} ({sr*100:.2f}%) |\n")
        f.write(f"| Avg Reward | {avg_reward:.4f} |\n")
        f.write(f"| Avg Cost | {avg_cost:.4f} |\n")
        f.write(f"| Avg Penalty | {avg_penalty:.4f} |\n")
        f.write(f"| Avg Episode Length | {avg_length:.2f} |\n")
        f.write(f"| Avg Oracle Cost | {avg_oracle:.4f} |\n")
        f.write(f"| Avg CR (Total) | {avg_cr:.4f} |\n")
        f.write(f"| **Avg CR (Success)** | **{avg_success_cr:.4f}** |\n\n")

        if difficulty_summary:
            f.write("## Task Difficulty Analysis\n\n")
            f.write("| Metric | Mean | Std | Min | Max |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            labels = {
                "num_goals":           "Num Goals",
                "num_edges":           "Num Edges",
                "num_components":      "Connected Components",
                "graph_density":       "Graph Density",
                "unreachable_goals":   "Unreachable Goals",
                "avg_dist_to_goals":   "Avg Dist to Goals",
                "max_dist_to_goals":   "Max Dist to Goals",
                "avg_inter_goal_dist": "Avg Inter-Goal Dist",
            }
            for k, label in labels.items():
                if k in difficulty_summary:
                    s = difficulty_summary[k]
                    f.write(
                        f"| {label} | {s['mean']:.3f} | {s['std']:.3f} "
                        f"| {s['min']:.3f} | {s['max']:.3f} |\n"
                    )

        f.write("\n## Episode Log\n\n")
        f.write("| # | ✅ | Cost | Penalty | Len | Oracle | Coverage | CR |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for d in episode_details:
            f.write(
                f"| {d['idx']} | {'✅' if d['success'] else '❌'} "
                f"| {d['cost']:.2f} | {d['penalty']:.2f} | {d['len']:.0f} "
                f"| {d['oracle']:.2f} | {d['coverage']:.2f} | {d['cr']:.4f} |\n"
            )

    # JSON 原始数据
    with open(eval_dir / "random_results.json", "w") as f:
        json.dump({
            "summary": {
                "success_rate": sr, "avg_cost": avg_cost,
                "avg_oracle": avg_oracle, "avg_cr": avg_cr,
                "avg_success_cr": avg_success_cr,
            },
            "difficulty": difficulty_summary,
            "episodes":   episode_details,
        }, f, indent=2)

    print(f"结果已保存至: {eval_dir}")
    return sr, avg_cr, avg_success_cr


# ==========================================================
# CLI 入口
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Exploration Baseline")
    parser.add_argument("--config",   type=str,
        default="/home/pemb7543/DeC_MACTP/Train/"
                "magnarl_train_magnarlactorgnn__26_03_17-21_46_45_b48119e9/config.pkl")
    parser.add_argument("--graph",    type=str, default="graph_data_128_4_8")  # ← 新增
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--exact-oracle",   action="store_true")
    parser.add_argument("--no-gif",         action="store_true")
    parser.add_argument("--no-difficulty",  action="store_true")
    parser.add_argument("--yaml-path", type=str, default=None)
    args = parser.parse_args()

    run_random_exploration(
        config_path        = args.config,
        graph_path         = args.graph,           # ← 新增
        num_episodes       = args.episodes,
        use_exact_oracle   = args.exact_oracle,
        save_gifs          = not args.no_gif,
        analyze_difficulty = not args.no_difficulty,
        yaml_path          = args.yaml_path,
    )