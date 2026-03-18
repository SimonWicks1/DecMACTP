#!/usr/bin/env python3
import torch
from torchrl.envs.utils import check_env_specs

from tensordict import TensorDict

# 按你的实际路径修改
from benchmarl.environments.mactp.common import MactpClass, MactpTask
# 如果 MactpClass 不在这个位置，就改为你实际的模块路径

# For render
import cv2  # 用于实时显示
from PIL import Image # 用于保存 GIF
import numpy as np


def masked_random_actions(action_mask: torch.Tensor) -> torch.Tensor:
    """
    action_mask: bool, shape [E, A, N+1]
    返回 actions: int64, shape [E, A]
    """
    probs = action_mask.float()
    # 每个 (E,A) 至少有一个合法动作
    if (probs.sum(dim=-1) == 0).any():
        bad = (probs.sum(dim=-1) == 0).nonzero(as_tuple=False)
        raise RuntimeError(f"CRITICAL: some (env,agent) have all actions masked out: {bad.tolist()}")

    # 展平后 multinomial，再 reshape 回来
    E, A, K = probs.shape
    flat = probs.reshape(E * A, K)
    acts = torch.multinomial(flat, 1).squeeze(-1).reshape(E, A).to(torch.int64)
    return acts


def assert_actions_valid(action_mask: torch.Tensor, actions: torch.Tensor) -> None:
    """
    action_mask: [E,A,K], actions: [E,A]
    """
    chosen = torch.gather(action_mask, dim=-1, index=actions.unsqueeze(-1))  # [E,A,1]
    if not bool(chosen.all().item()):
        bad = (~chosen.squeeze(-1)).nonzero(as_tuple=False)
        raise AssertionError(f"Invalid actions at indices (env,agent): {bad.tolist()}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4

    # 1) 构造 TaskClass，并给它一个 config
    # 这里模拟 BenchMARL：TaskClass.config 来自 default.yaml 的读取结果
    # 你需要把 graph_generator dict 填成你实际 default.yaml 的结构
    task = MactpTask.TRAIN.get_task()

    # 2) 通过 TaskClass.get_env_fun 构造环境
    env_fn = task.get_env_fun(
        num_envs=num_envs,
        continuous_actions=False,
        device=device,
    )
    env = env_fn()

    print("\n=== Basic env info ===")
    print("env.batch_size:", env.batch_size)
    print("env.num_envs:", env.num_envs)
    print("env.num_agents:", env.num_agents)
    print("env.max_nodes:", env.max_nodes)

    # 3) 检查 specs（强烈建议）
    print("\n=== check_env_specs ===")
    check_env_specs(env)
    print("Specs check passed.")

    # 4) 检查 TaskClass spec 分拣是否正确
    print("\n=== TaskClass outputs ===")
    print("group_map:", task.group_map(env))
    print("action_spec keys:", env.full_action_spec.keys(True, True))
    print("observation_spec (task) keys:", task.observation_spec(env).keys(True, True))
    mask_spec = task.action_mask_spec(env)
    print("action_mask_spec is None?", mask_spec is None)
    if mask_spec is not None:
        print("action_mask_spec keys:", mask_spec.keys(True, True))

    # 5) Rollout: reset
    td = env.reset()
    print("\n=== After reset ===")
    print("td.batch_size:", td.batch_size)
    print("agents batch_size:", td["agents"].batch_size)
    print("action_mask shape:", tuple(td["agents", "action_mask"].shape))
    # 假设 env 是你已经创建好的环境实例 (num_envs=4)
    print(f"Checking diversity across {env.num_envs} environments...\n")

    # 1. 检查静态权重 (Static Weights) 是否全等
    weights = env.static_weights  # Shape: [4, N, N]
    diff_0_1 = (weights[0] - weights[1]).abs().sum().item()
    diff_1_2 = (weights[1] - weights[2]).abs().sum().item()

    print(f"Weight diff (Env 0 vs 1): {diff_0_1:.4f}")
    print(f"Weight diff (Env 1 vs 2): {diff_1_2:.4f}")

    if diff_0_1 == 0 and diff_1_2 == 0:
        print("⚠️ 警告：所有环境的边权重完全相同！可能是 Seed 失效或数据集只有一个样本。")
    else:
        print("✅ 正常：环境间的权重存在差异，生成了不同的图。")

    # 2. 检查起始位置 (Start Locations)
    starts = env.current_locations # Shape: [4, A]
    print(f"\nStart locations:\n{starts.cpu().numpy()}")

    if (starts[0] == starts[1]).all():
        print("ℹ️ 提示：起始位置相同（这在 CTP 中很常见，如果是固定起点任务）")
    else:
        print("✅ 正常：起始位置不同")


    print("\n=== Graph data inspection ===")

    for i in range(env.num_envs):
        graph_data = env.graph_data[i]
        print(f"Environment {i}:")
        print(f"  starts: {graph_data.s}")
        print(f"  goals: {graph_data.g}")
        print(f"  Static weights shape: {env.static_weights[i].shape}")

    
    frames = [[] for _ in range(env.num_envs)] # 用于保存每个环境的帧
    total_reward = 0.0
    for t in range(50):
        action_mask = td["agents", "action_mask"]  # [E,A,K]
        actions = masked_random_actions(action_mask)
        assert_actions_valid(action_mask, actions)

        td.set(("agents", "action"), actions)
        td = env.step(td)
        next_td = td["next"]

        rewards = next_td["agents", "reward"]  # [E,A,1]
        done = next_td["done"]                 # [E,1]（按你的 env 实现）
        total_reward += float(rewards.sum().item())

        if t == 0:
            print("\n--- Debug at step 0 ---")
            print("actions[0]:", actions[0].tolist())
            print("rewards shape:", tuple(rewards.shape))
            print("done shape:", tuple(done.shape))
            print("------------------------\n")

        td = next_td
        # 捕获渲染帧
        for i in range(env.num_envs):
            rgb_array = env.render(env_idx=i)
            if rgb_array is not None:
                frames[i].append(Image.fromarray(rgb_array))

        
        if done.all():
            print(f"All environments done at step {t}. Resetting all.")
            break  # 这里直接 break，后面会 partial reset 测试

    for i in range(env.num_envs):
        if len(frames[i]) > 0:
            gif_path = f"mactp_simulation_env_{i}.gif"
            frames[i][0].save(
                gif_path, 
                save_all=True, 
                append_images=frames[i][1:], 
                duration=100, 
                loop=0
            )
            print(f"✅ GIF saved successfully to: {gif_path}")
        else:
            print(f"⚠️ No frames captured for env {i}.")
    
    print(f"\nTotal reward (sum over E,A,t): {total_reward:.4f}")

    # 6) 显式测试：只 reset env 0 和 env 2
    print("\n=== Partial reset test ===")
    partial = torch.tensor([1, 0, 1, 0], dtype=torch.bool, device=device)
    reset_td = TensorDict({"_reset": partial}, batch_size=[env.num_envs])
    td2 = env.reset(reset_td)
    print("Partial reset ok. td2.batch_size:", td2.batch_size)


if __name__ == "__main__":
    main()
