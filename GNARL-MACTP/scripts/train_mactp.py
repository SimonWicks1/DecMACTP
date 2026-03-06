#!/usr/bin/env python3

import gnarl
import numpy as np
import torch as th
import os
import yaml
import argparse

from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

import wandb
from wandb.integration.sb3 import WandbCallback

from gnarl.agent.mapolicy import MaMaskableNodeActorCriticPolicy
from gnarl.agent.imitation.imitation import (
    behavioural_cloning,
    make_data_loader,
    split_dataset,
)
from gnarl.util.evaluation import (
    ExpertPolicy,
    calculate_env_split,
)
from gnarl.util.envs import make_matrain_env, make_maeval_env
from gnarl.util.classes import get_clean_kwargs
from gnarl.util.bc import get_bc_experience, complete_config
from gnarl.util.callbacks import MaskableEvalCallback

# For random seed reproducibility
import random
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run GNARL MACTP environment training.")
    parser.add_argument("-c", "--config", type=str, default="/home/pemb7543/MAGNARL/GNARL-MACTP/configs/mactp2.yaml")
    parser.add_argument("-w", "--wandb", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None, help="Specific seed for this run.")
    parser.add_argument("--group", type=str, default="mactp-multi-seed", help="WandB group name.")
    return parser.parse_args()

def train_ppo(
    train_env: VecEnv, val_envs: list[VecEnv], run, config: dict, policy_path=None
):
    """
    用 MaskablePPO 训练集中式多智能体策略。
    """
    ppo_kwargs = get_clean_kwargs(
        MaskablePPO.__init__,
        warn=False,
        kwargs=config["PPO"],
    )

    # 创建PPO模型
    model = MaskablePPO(
        MaMaskableNodeActorCriticPolicy,
        train_env,
        **ppo_kwargs,
        policy_kwargs=config["policy_kwargs"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )

    # 加载预训练模型
    if policy_path is not None and os.path.exists(policy_path):
        print(f"Loading pre-trained model from {policy_path}.")
        if "best_model.pt" in policy_path:
            model.policy.load_state_dict(
                th.load(policy_path, weights_only=False)["state_dict"]
            )
        else:
            model.policy.load_state_dict(th.load(policy_path, weights_only=True))
        print("Pre-trained model loaded successfully.")
    else:
        print("No pre-trained model found, starting training from scratch.")

    # 配置评估回调
    per_env_samples = calculate_env_split(
        node_samples=config["val_data"]["node_samples"],
        max_envs=config["val_data"]["num_envs"],
    )
    eval_episode_list = [c * e for n, c, e in per_env_samples]
    
    eval_callback = MaskableEvalCallback(
        eval_env=val_envs,
        best_model_save_path=f"models/{run.id}",
        log_path=f"eval_logs/{run.id}",
        eval_freq=max(config["PPO"]["eval_freq"] // train_env.num_envs, 1),
        n_eval_episodes=eval_episode_list,
        deterministic=config["val_data"]["deterministic"],
        render=False,
        log_prefix="ppo",
    )

    # 回调函数列表
    callbacks = [eval_callback]
    callbacks.append(
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    )

    # 训练模型
    model.learn(
        total_timesteps=config["PPO"]["timesteps"],
        progress_bar=True,
        callback=callbacks,
    )

    # 保存模型
    ppo_model_path = f"models/{run.id}/ppo.pt"
    os.makedirs(os.path.dirname(ppo_model_path), exist_ok=True)
    th.save(model.policy.state_dict(), ppo_model_path)
    
    artifact = wandb.Artifact(f"{run.id}_ppo_final_model", type="model")
    artifact.add_file(ppo_model_path)
    wandb.log_artifact(artifact)

    print(f"PPO training completed. Model saved to {ppo_model_path}")

def train_imitation(
    train_env: VecEnv,
    val_envs: list[VecEnv],
    config: dict,
    run,
    experience_path=None,
):
    """训练行为克隆模型"""
    dataset = get_bc_experience(experience_path, train_env, config)

    print("Creating data loader for Behavioural Cloning.")
    torch_generator = th.Generator().manual_seed(config["BC"]["seed"])
    train_data, val_data = split_dataset(
        dataset,
        split=0.8,
        generator=torch_generator,
    )
    data_loader = make_data_loader(
        train_data,
        shuffle=True,
        batch_size=config["BC"].get("batch_size", 64),
    )
    val_loader = make_data_loader(
        val_data,
        shuffle=False,
        batch_size=config["BC"].get("batch_size", 64),
    )

    print("Training a policy using Behavioural Cloning.")
    policy = MaMaskableNodeActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=get_linear_fn(
            config["BC"]["learning_rate"], config["BC"]["learning_rate"], 1.0
        ),
        **config["policy_kwargs"],
    )
    
    wandb.watch(policy, log="all", log_freq=100)

    per_env_samples = calculate_env_split(
        node_samples=config["val_data"]["node_samples"],
        max_envs=config["val_data"]["num_envs"],
    )
    eval_episode_list = [c * e for n, c, e in per_env_samples]

    policy = behavioural_cloning(
        policy=policy,
        data_loader=data_loader,
        val_loader=val_loader,
        val_envs=val_envs,
        **config["BC"],
        progress_bar=True,
        best_model_save_path=f"models/{run.id}",
        deterministic_eval=config["val_data"]["deterministic"],
        n_eval_episodes=eval_episode_list,
    )

    # 保存BC策略
    bc_model_path = f"models/{run.id}/bc.pt"
    os.makedirs(os.path.dirname(bc_model_path), exist_ok=True)
    th.save(policy.state_dict(), bc_model_path)
    print(f"BC policy saved as {bc_model_path}")
    
    artifact = wandb.Artifact(f"{run.id}_bc_final_model", type="model")
    artifact.add_file(bc_model_path)
    wandb.log_artifact(artifact)

def run_demonstration(vec_env, config): 
    """
    Fixed Multi-Agent Demonstration Loop.
    Correctly handles MultiDiscrete masking by reshaping the flattened mask.
    """
    rng = np.random.default_rng(config["seed"])
    obs = vec_env.reset()
    
    # Check if we need to extract obs from a tuple (Gym 0.26+)
    if isinstance(obs, tuple):
        obs = obs[0]
        
    print("Starting demonstration rollout...")
    step = 0
    
    # Get environment dimensions from the Vector Env action space
    # action_space is MultiDiscrete([max_nodes, max_nodes, ...])
    if hasattr(vec_env.action_space, 'nvec'):
        n_agents = len(vec_env.action_space.nvec)
        max_nodes = vec_env.action_space.nvec[0]
    else:
        # Fallback if not easily accessible (assumes all envs same)
        n_agents = config.get("num_agents", config.get("train_data", {}).get("num_starts", 1))
        max_nodes = config["train_data"]["max_nodes"]

    while step < 50:  # Run for 50 steps max
        # 1. Get Flattened Masks: Shape (num_envs, num_agents * max_nodes)
        action_masks = get_action_masks(vec_env)
        
        actions_list = []
        
        # 2. Loop through each environment in the VecEnv
        for i in range(vec_env.num_envs):
            env_mask = action_masks[i]
            
            # --- CRITICAL FIX: Reshape the mask for Multi-Agent ---
            # We must split the 1D mask back into (Agents, Nodes)
            try:
                reshaped_mask = np.array(env_mask).reshape(n_agents, max_nodes)
            except ValueError:
                # If reshaping fails, it might be a single agent setup
                reshaped_mask = np.array(env_mask).reshape(1, -1)
            
            env_actions = []
            
            # 3. Sample an action for EACH agent
            for agent_idx in range(n_agents):
                agent_specific_mask = reshaped_mask[agent_idx]
                valid_nodes = np.where(agent_specific_mask)[0]
                
                if len(valid_nodes) > 0:
                    choice = rng.choice(valid_nodes)
                    env_actions.append(choice)
                else:
                    env_actions.append(0) # Fallback
            
            actions_list.append(env_actions)

        # 4. Convert to Numpy Array
        # Shape: (num_envs, num_agents) -> e.g. [[12, 5, 8], [2, 9, 11]]
        actions = np.array(actions_list)
        
        # 5. Step
        obs, rewards, done, infos = vec_env.step(actions)
        
        if step % 10 == 0:
            print(f"Step {step}: Actions {actions[0]}, Done: {done[0]}")
        
        step += 1
        if done[0]:
            print("Episode finished early.")
            break

    print(f"Demonstration completed in {step} steps")

def main():
    args = parse_args()

    # Load Config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return
    config = yaml.safe_load(open(args.config, "r"))
    config = complete_config(config)


    # th.manual_seed(config["seed"])
    # np.random.seed(config["seed"])

    # Eidt by Xiao: Set random seeds
    # NEW: Generate a random seed if you want it to change every time
    # 命令行 --seed 覆盖配置中的 seed（同时也给 PPO 子配置同步）
    if args.seed is not None:
        config["seed"] = args.seed
        if "PPO" in config:
            config["PPO"]["seed"] = args.seed
    train_seed = config["seed"]
    th.manual_seed(train_seed)
    np.random.seed(train_seed)
    random.seed(train_seed)
    print(f"--- RUNNING WITH SEED: {train_seed} ---")
    
    # 初始化wandb
    run = wandb.init(
        project="GNARL-MACTP",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        mode="offline",
        group=args.group,  # Groups the 10 runs together for mean/std plots
        name=f"seed_{train_seed}_{config['env']}_{list(config['train_data']['node_samples'].keys())}train",
        tags=[
            config["algorithm"],
            f"train-{list(config['train_data']['node_samples'].keys())}",
            f"network-{config['policy_kwargs']['network_kwargs']['network']}",
            f"aggr-{config['policy_kwargs']['network_kwargs']['aggr']}",
            f"pooling-{config['policy_kwargs']['pooling_type']}",
            f"embed-{config['policy_kwargs']['embed_dim']}",
            f"seed-{train_seed}",
            "multi-agent",  # 添加多智能体标签
        ],
        notes="Multi-agent CTP training",
    )

    # 检查文件是否存在，然后保存
    if os.path.exists(args.config):
        wandb.save(args.config)
        print(f"配置文件 {args.config} 已保存到 WandB 运行目录中。")
    else:
        print(f"警告：找不到配置文件 {args.config}。")

    print(f"============== Beginning run: {run.id} ==============")

    # 创建环境
    print("Constructing train env")
    train_env = make_matrain_env(config, run.id)
    print("Constructing val env")
    val_envs = make_maeval_env(config, run.id, "val")

    # 获取环境规格并注入策略参数
    env_spec = train_env.get_attr("graph_spec")[0]
    config["policy_kwargs"]["graph_spec"] = env_spec

    # 运行演示
    print("Running demonstration of the environment...")
    run_demonstration(train_env, config)

    # 训练流程
    if "BC" in config:
        print("Starting Behavioural Cloning training...")
        train_imitation(
            train_env,
            val_envs,
            config,
            run,
            experience_path=config["BC"].get("data_path"),
        )
    
    if "PPO" in config:
        print("Starting PPO training...")
        policy_path = f"models/{run.id}/bc_best_model.pt" if "BC" in config else None
        train_ppo(
            train_env,
            val_envs,
            run,
            config,
            policy_path=policy_path,
        )

    print(f"============== Ending run: {run.id} ==============")
    
    run.finish()


if __name__ == "__main__":
    main()