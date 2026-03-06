#!/usr/bin/env python3

import gnarl
import numpy as np
import torch as th
import os
import yaml
from gnarl.envs.generate.graph_generator import RandomSetGraphGenerator
from gnarl.envs.generate.data import GraphProblemDataset

from gnarl.util.bc import complete_config

from sb3_contrib.common.maskable.utils import get_action_masks


import gymnasium as gym
import numpy as np

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
    config = yaml.safe_load(open("GNARL-CTP/configs/ctp.yaml", "r"))
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
    env = gym.make(
        config["env"],
        max_nodes=max(d.num_nodes for d in ds),
        render_mode="human",
        graph_generator=RandomSetGraphGenerator(datasets=ds, seed=env_seed),
        **config.get("env_kwargs", {}),
    )

    # 设置保存目录
    env.folder = "./renders"  # 明确指定保存目录
    os.makedirs(env.folder, exist_ok=True)
    
    env.reset()
    env.render()
    Total_reward = 0.0
    print(f"============== Starting test==============")
    for step in range(500):
        # 获取动作掩码并选择动作
        action_masks = get_action_masks(env)
        if step == 0:
            print("Action masks at step", step, ":", action_masks)
            print("Shape of action masks at step", step, ":", action_masks.shape if action_masks is not None else None)
        actions = []
        if isinstance(action_masks, np.ndarray):
            # 单智能体
            valid_actions = np.where(action_masks)[0]
            print(f"Valid actions: {valid_actions}")
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = 0
            actions = action
        else:
            # 多智能体
            for mask in action_masks:
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    actions.append(np.random.choice(valid_actions))
                else:
                    actions.append(0)
        # 执行动作
        observation, reward, terminated, _, info = env.step(actions)
        if step == 0:
            # print("Initial observation:", observation)
            for k,v in observation.items():
                print("Observation key: {k}, value: {v.shape if isinstance(v, np.ndarray) else v}")
            print("Starts:", env.unwrapped.graph_data.s)
            print("current_node:", observation.get("current_node", None))

        Total_reward += reward
        # env.render()
        env.unwrapped.render()
        # print(env.render_mode)
        if terminated:
            print(f"All destinations visited at step {step}!")
            break
    
    # 保存GIF - 关键步骤！
    if hasattr(env.unwrapped, '_save_render'):
        env.unwrapped._save_render()
        print(f"GIF saved to: {getattr(env, '_gif_path', 'unknown path')}")
    else:
        print("Environment does not have _save_render method")
    print(f"Total reward obtained: {Total_reward}")
    print(f"============== Ending test==============")




if __name__ == "__main__":
    main()
