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

from gnarl.agent.policy import MaskableNodeActorCriticPolicy
from gnarl.agent.imitation.imitation import (
    behavioural_cloning,
    make_data_loader,
    split_dataset,
)
from gnarl.util.evaluation import (
    ExpertPolicy,
    calculate_env_split,
)
from gnarl.util.envs import make_train_env, make_eval_env
from gnarl.util.classes import get_clean_kwargs
from gnarl.util.bc import get_bc_experience, complete_config
from gnarl.util.callbacks import MaskableEvalCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Run GNARL environment training.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed in the config.",
    )
    return parser.parse_args()


def train_ppo(
    train_env: VecEnv, val_envs: list[VecEnv], run, config: dict, policy_path=None
):
    """
    用 MaskablePPO (from sb3_contrib) 训练策略。
    - 支持从 BC 预训练的权重加载（policy_path）。
    - 使用自定义评估回调 MaskableEvalCallback 做周期性评估与最佳模型保存。
    """
    ppo_kwargs = get_clean_kwargs(
        MaskablePPO.__init__,
        warn=False,
        kwargs=config["PPO"],
    )

    # Create the PPO model
    model = MaskablePPO(
        MaskableNodeActorCriticPolicy,
        train_env,
        **ppo_kwargs,
        policy_kwargs=config["policy_kwargs"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )

    # Load the model if a path is provided
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

    # Create list of callbacks
    callbacks = [
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        eval_callback,
    ]

    # Train the model
    model.learn(
        total_timesteps=config["PPO"]["timesteps"],
        progress_bar=True,
        callback=callbacks,
    )

    # Save the model
    ppo_model_path = f"models/{run.id}/ppo.pt"
    os.makedirs(os.path.dirname(ppo_model_path), exist_ok=True)
    th.save(model.policy.state_dict(), ppo_model_path)
    artifact = wandb.Artifact(f"{run.id}_ppo_final_model", type="model")
    artifact.add_file(ppo_model_path)
    wandb.log_artifact(artifact)


def train_imitation(
    train_env: VecEnv,
    val_envs: list[VecEnv],
    config: dict,
    run,
    experience_path=None,
):

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
    policy = MaskableNodeActorCriticPolicy(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        lr_schedule=get_linear_fn(
            config["BC"]["learning_rate"], config["BC"]["learning_rate"], 1.0
        ),  # start, end, duration
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

    # Save the trained BC policy
    bc_model_path = f"models/{run.id}/bc.pt"
    os.makedirs(os.path.dirname(bc_model_path), exist_ok=True)
    th.save(policy.state_dict(), bc_model_path)
    print(f"BC policy saved as {bc_model_path}")
    artifact = wandb.Artifact(f"{run.id}_bc_final_model", type="model")
    artifact.add_file(bc_model_path)
    wandb.log_artifact(artifact)


def run_demonstration(vec_env, config):
    """
    演示（仅 1 个 rollout）：
    - 如果配置中包含 BC，则使用 ExpertPolicy 在当前 vec_env 上进行一次策略 rollout；
    - 否则随机从“当前时刻的有效动作掩码”中采样一个可执行动作；
    - 打印每个时间步的动作与对应的“有效动作索引列表”，用于 sanity check。
    """
    # Perform one rollout of the expert policy on the vec_env and print the actions
    rng = np.random.default_rng(config["seed"]) #创建具有固定种子的随机数生成器，确保实验可重复
    obs = vec_env.reset()
    done = np.zeros(vec_env.num_envs, dtype=bool)
    actions_list = [] #记录每个时间步的动作
    valid_action_list = [] #录每个时间步的有效动作掩码
    while not done[0]:
        # Why we have to mask the actions? Because some actions may be invalid in certain states. Xiao
        # 获取当前状态下所有环境的有效动作掩码, 掩码是布尔数组，True表示该动作在当前状态下可执行
        valid_action_list.append( (vec_env))
        if "BC" in config:
            # 如果要做 BC，就调用专家策略来产生示范动作（可确定性/带参数）
            policy = ExpertPolicy(vec_env.get_attr("unwrapped")[0].__class__, rng)
            actions, _ = policy.predict(
                obs,
                deterministic=config["val_data"]["deterministic"],
                **config.get("expert_policy_kwargs", {}),
            )
        else:
            # 否则，从每个并行环境的“可行动作集合”里随机采样一个动作
            # Because the valid_action_list is 每个时间步的有效动作掩码, 
            # so we use valid_action_list[-1] to get the latest one.
            # np.where(v): 返回掩码 v 中为 True 的索引数组 eg:(array([0, 2]),)
            # np.where(v)[0]: 获取第一个元素，即索引数组本身 eg: array([0, 2])
            actions = np.array(
                [int(rng.choice(np.where(v)[0])) for v in valid_action_list[-1]]
            )
        actions_list.append(actions)
        obs, _, done, _ = vec_env.step(actions)
    print("Actions taken during the rollout:")
    # enumerate() 是 Python 内置函数，用于在循环同时获取元素索引和元素值
    # 这里用于打印每个时间步的动作及此时的有效动作索引列表
    # for step, actions in enumerate(actions_list):
    #     # # Edit by Xiao 
    #     # #why valid_action_list[step][0]? Because valid_action_list is a list of arrays,
    #     # #each array corresponds to a time step, and each array contains the action masks for all environments. 
    #     # #valid_action_list[step][0] gets the action mask for the first environment at that time step. Xiao
    #     # # But why only the first environment? Because we just want to demonstrate the actions taken in one environment. Xiao
    #     # # actions[0]: the actions taken by the first environment at this time step
    #     # valid_idxs = [i for i, v in enumerate(valid_action_list[step][0]) if v]
    #     # print(f"Step {step}: {actions[0]}, valid actions: {valid_idxs}")
    #     # # Ended Edit by Xiao
    #     mask_for_env0 = valid_action_list[step][0]  # shape: (n_travelers, max_nodes)
    #     # 打印每个 traveler 的有效动作索引
    #     for t_idx, mask in enumerate(mask_for_env0):
    #         valid_idxs = [i for i, v in enumerate(mask) if bool(v)]
    #         print(f"Step {step}: Traveler {t_idx} action {actions[t_idx]}, valid actions: {valid_idxs}")


def main():
    args = parse_args()
    # 读取/合并配置：complete_config 会为缺省项补默认值，或派生字段
    config = yaml.safe_load(open(args.config, "r"))
    config = complete_config(config)

    # 命令行 --seed 覆盖配置中的 seed（同时也给 PPO 子配置同步）
    if args.seed is not None:
        config["seed"] = args.seed
        if "PPO" in config:
            config["PPO"]["seed"] = args.seed

    # Set random seeds But why both? Xiao
    # 设随机种子（torch 与 numpy 都设）
    # 这回答了代码里的注释“为什么两个都要设？”：因为训练中既用到 torch（模型/优化器/数据加载）
    # 也用到 numpy（例如专家策略、环境初始化或 dataset 抽样），两者需各自固定
    th.manual_seed(config["seed"])
    np.random.seed(config["seed"])  # type: ignore

    run = wandb.init(
        project="GNARL-CTP",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,
        mode="disabled" if not args.wandb else "online", #默认禁用；只有传 -w 才 online
        # 运行名：环境名_训练节点规模；同时打上若干模型结构/算法tag以便检索
        name=f"{config['env']}_{list(config['train_data']['node_samples'].keys())}train",  # type: ignore
        tags=[
            config["algorithm"],  # type: ignore
            f"train-{list(config['train_data']['node_samples'].keys())}",  # type: ignore
            f"network-{config['policy_kwargs']['network_kwargs']['network']}",  # type: ignore
            f"aggr-{config['policy_kwargs']['network_kwargs']['aggr']}",  # type: ignore
            f"pooling-{config['policy_kwargs']['pooling_type']}",  # type: ignore
            f"embed-{config['policy_kwargs']['embed_dim']}",  # type: ignore
        ],
        notes="",
    )

    print(f"============== Beginning run: {run.id} ==============")

    print("Constructing train env")
    train_env = make_train_env(config, run.id)
    print("Constructing val env")
    val_envs = make_eval_env(config, run.id, "val")

    # Get the spec attribute from the first environment in vec_env
    # 从训练环境读取图规格（graph_spec），注入到策略构造参数中（GNN 需要此元信息）
    env_spec = train_env.get_attr("graph_spec")[0]
    config["policy_kwargs"]["graph_spec"] = env_spec  # type: ignore

    print("Running demonstration of the expert policy...")
    run_demonstration(train_env, config)

    if "BC" in config:
        print("Starting Behavioural Cloning training...")
        # Train a policy using Behavioural Cloning
        train_imitation(
            train_env,
            val_envs,
            config,
            run,
            experience_path=config["BC"]["data_path"],  # type: ignore
        )
    if "PPO" in config:
        print("Starting PPO training...")
        # Train the policy using PPO
        train_ppo(
            train_env,
            val_envs,
            run,
            config,
            policy_path=(
                f"models/{run.id}/bc_best_model.pt" if "BC" in config else None
            ),
        )

    print(f"============== Ending run: {run.id} ==============")


if __name__ == "__main__":
    main()
