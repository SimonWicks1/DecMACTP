#!/usr/bin/env python3

import gnarl
import numpy as np
import torch as th
import os
import yaml
from gnarl.envs.generate.graph_generator import RandomSetGraphGenerator
from gnarl.envs.generate.data import GraphProblemDataset

from gnarl.util.bc import complete_config

import gymnasium as gym
import importlib

def get_env_cls(env_name: str):
    env_spec = gym.registry[env_name]
    if callable(env_spec.entry_point):
        # Edit by Xiao
        # This is to handle the case where the entry_point is already a class we write in gnarl.envs
        env_cls = env_spec.entry_point
    else:
        module_path, class_name = env_spec.entry_point.rsplit(":", 1)
        module = importlib.import_module(module_path)
        env_cls = getattr(module, class_name)
    return env_cls

def main():

    # 读取/合并配置：complete_config 会为缺省项补默认值，或派生字段
    # 请确保路径正确
    config = yaml.safe_load(open("/home/pemb7543/MAGNARL/GNARL-MACTP/configs/mactp2.yaml", "r"))
    config = complete_config(config)

    print("Constructing test env")
    env_cls = get_env_cls(config["env"])
    
    ds = [
        GraphProblemDataset(
            root=config["train_data"]["graph_dir"],
            split="train",
            algorithm=config["algorithm"],
            num_nodes=n,
            num_samples=num_samples,
            seed=config["train_data"]["seed"],
            graph_generator=config["train_data"]["graph_generator"],
            graph_generator_kwargs=config["train_data"].get("graph_generator_kwargs"),
            # Edit by Xiao
            # 新增多智能体参数
            num_starts=config["train_data"].get("num_starts", 1),
            num_goals=config["train_data"].get("num_goals", 1),
            pre_filter=(getattr(env_cls, "pre_filter", None)),
            pre_transform=(getattr(env_cls, "pre_transform", None)),
        )
        for n, num_samples in config["train_data"]["node_samples"].items()
    ]
    env_seed = config["train_data"]["seed"]
    
    # 初始化环境
    env = gym.make(
        config["env"],
        max_nodes=max(d.num_nodes for d in ds),
        render_mode="human",
        num_agents=config["train_data"].get("num_starts", 1),
        num_goals=config["train_data"].get("num_goals", 1),
        graph_generator=RandomSetGraphGenerator(datasets=ds, seed=env_seed),
        **config.get("env_kwargs", {}),
    )
    print("Max nodes in env:", max(d.num_nodes for d in ds))

    # 设置保存目录
    env.unwrapped.folder = "./renders"  # 明确指定保存目录到 unwrapped 环境
    os.makedirs(env.unwrapped.folder, exist_ok=True)
    
    # Reset 环境
    obs_dict, info_dict = env.reset()
    env.unwrapped.render()
    total_reward = 0.0

    print("============== Starting test ==============")
    
    # 获取智能体列表
    agents = env.unwrapped.agents
    print(f"Agents: {agents}")

    for step in range(500):
        # 1. 获取动作掩码 (Dict {agent_id: mask})
        # 注意：这里直接调用 unwrapped 的方法，因为标准 Gym 接口可能没有暴露这个自定义方法
        action_masks = env.unwrapped.action_masks()
        
        actions = {}
        
        # 2. 为每个智能体选择动作
        for agent_id in agents:
            mask = action_masks[agent_id]
            valid_actions = np.where(mask)[0]
            
            if len(valid_actions) > 0:
                # 随机选择一个合法动作
                action = np.random.choice(valid_actions)
                actions[agent_id] = int(action)
            else:
                # 理论上不应该发生，因为 TERMINATE 总是合法的（对于已终止智能体）或者至少可以停留在原地
                print(f"Warning: No valid actions for {agent_id}, forcing terminate.")
                actions[agent_id] = env.unwrapped.TERMINATE_ACTION

        print(f"Step {step} - Taking actions: {actions}")

        # 3. 执行动作 (传入字典)
        # 返回值现在都是字典形式: {agent_id: value}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # 调试信息打印 (只在第一步打印详细信息)
        if step == 0:
            print("\n--- Debug Info at Step 0 ---")
            first_agent = agents[0]
            print(f"Observation keys for {first_agent}:", obs[first_agent].keys())
            
            curr_nodes = obs[first_agent].get("current_nodes", None)
            if curr_nodes is not None:
                print(f"current_nodes shape: {curr_nodes.shape}")
                # 打印当前位置矩阵中值为1的坐标
                indices = np.where(curr_nodes == 1)
                print(f"Active indices in current_nodes: {list(zip(*indices))}")
            
            print(f"Action masks shape for {first_agent}: {action_masks[first_agent].shape}")
            print("----------------------------\n")

        # 4. 累加奖励 (取平均或总和用于显示)
        # 假设是合作任务，通常看团队总奖励，这里简单把所有人的加起来
        step_total_reward = sum(rewards.values())
        total_reward += step_total_reward
        
        env.unwrapped.render()

        # 5. 检查终止条件
        # 如果所有智能体都 terminated，或者环境 truncated (超时)
        all_done = all(terminateds.values())
        any_truncated = any(truncateds.values())
        
        if all_done or any_truncated:
            reason = "Terminated" if all_done else "Truncated"
            print(f"The episode ended at step {step}! Reason: {reason}")
            break
    
    print(f"Total reward obtained: {total_reward}")
    print(f"============== Ending test ==============")

if __name__ == "__main__":
    main()