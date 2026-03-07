import torch

from benchmarl.algorithms import IgnarlConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.mactp.common import MactpTask

from benchmarl.models import IgnarlActorGNNConfig, IgnarlCriticGNNConfig

from benchmarl.experiment.callback import Callback
import torch
import numpy as np

import argparse

class IgnarlSchedulerCallback(Callback):
    def __init__(self, 
                 lr_end=1e-5, 
                 critic_coef_start=0.1, 
                 critic_coef_end=0.5, 
                 anneal_frames=None):
        super().__init__()
        self.lr_end = lr_end
        self.critic_coef_start = critic_coef_start
        self.critic_coef_end = critic_coef_end
        self.anneal_frames = anneal_frames

    def on_setup(self):
        exp = self.experiment
        if self.anneal_frames is None:
            # 使用配置中的最大帧数
            self.anneal_frames = exp.config.get_max_n_frames(exp.on_policy) // 2
        self.lr_init = exp.config.lr

    def on_batch_collected(self, batch):
        exp = self.experiment
        current_frame = exp.total_frames
        progress = min(1.0, current_frame / self.anneal_frames)

        # 1. 更新 critic_coef
        new_critic_coef_val = self.critic_coef_start + progress * (self.critic_coef_end - self.critic_coef_start)
        exp.algorithm.critic_coef = new_critic_coef_val
        
        # 将 float 转换为 Tensor 并移动到训练设备上
        new_critic_coef_tensor = torch.as_tensor(
            new_critic_coef_val, 
            device=exp.config.train_device, 
            dtype=torch.float32
        )

        for group in exp.group_map.keys():
            if group in exp.losses:
                # 修正：赋值 Tensor 而不是 float
                exp.losses[group].critic_coeff = new_critic_coef_tensor

        # 2. 更新学习率 (lr)
        new_lr = self.lr_init - progress * (self.lr_init - self.lr_end)
        for group in exp.optimizers.keys():
            for loss_name in exp.optimizers[group].keys():
                optimizer = exp.optimizers[group][loss_name]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

    def on_train_end(self, training_td, group):
        exp = self.experiment
        # 记录日志时，如果它是 Tensor，建议调用 .item() 转换为数值
        exp.logger.log(
            {
                "schedulers/critic_coef": exp.algorithm.critic_coef,
                "schedulers/lr": exp.optimizers[group]["loss_objective"].param_groups[0]["lr"]
            },
            step=exp.n_iters_performed
        )


def main():
    # 增加命令行参数解析
    parser = argparse.ArgumentParser(description="Run MARL training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # 将外部传入的 seed 赋值给变量
    seed = args.seed
    
    # 打印日志以便追踪
    print(f"========== Running Experiment with SEED: {seed} ==========")
    # ------------------------------------------------------------
    # 1) Experiment config
    # ------------------------------------------------------------
    experiment_config = ExperimentConfig.get_from_yaml()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.train_device = device
    experiment_config.sampling_device = device

    experiment_config.gamma = 0.99

    # on-policy collection
    experiment_config.on_policy_collected_frames_per_batch = 2000
    experiment_config.on_policy_minibatch_size = 512
    experiment_config.on_policy_n_minibatch_iters = 4
    experiment_config.on_policy_n_envs_per_worker = 4

    # logging / eval
    experiment_config.loggers = ["csv", "tensorboard"]
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 10_000
    experiment_config.evaluation_episodes = 100
    experiment_config.render = False
    experiment_config.checkpoint_interval = 10_000

    # 修正：你原来写了 5_00_000（可运行但不建议），这里显式写 500_000
    experiment_config.max_n_frames = 500_000
    
    # ------------------------------------------------------------
    # 2) Task (MACTP)
    # ------------------------------------------------------------
    # 关键：这里要求环境已经移除 observation 里的 "state"
    task = MactpTask.TRAIN.get_from_yaml()

    # ------------------------------------------------------------
    # 3) Algorithm: MAPPO (CTDE)
    # ------------------------------------------------------------
    algorithm_config = IgnarlConfig(
        clip_epsilon=0.2,
        entropy_coef=0.01,
        critic_coef=1.0,
        loss_critic_type="l2",
        lmbda=0.95,
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=False,
        share_param_critic=True,   
        minibatch_advantage=False,
    )

    # ------------------------------------------------------------
    # 4) Actor model: Shared Graph Encoder + Agent-conditioned Pointer Head
    # ------------------------------------------------------------
    # 说明：
    # - node_features / edge_features 这两个字段是 BenchMARL ModelConfig 的必填项（即使你的模型内部不用它们）
    # - 你的 env 里 ("agents","node_features") 是 7 通道：[dest, visited, opt, pess, self_pos, density, terminated]
    # - ("agents","edge_weights") 和 ("agents","edge_status") -> edge_features 我们内部构成：weight + status_emb
    actor_model_config = IgnarlActorGNNConfig(
        node_features=7,
        edge_features=2,
        num_gnn_layers=2,
        status_emb_dim=8,
        gnn_hidden_dim=64,
        pooling_type="mean",
        temperature=2.0,
    )


    # ------------------------------------------------------------
    # 5) Critic model: Shared Graph Encoder + Centralized Pooling
    # ------------------------------------------------------------
    critic_model_config = IgnarlCriticGNNConfig(
        node_features=7,
        edge_features=2,
        num_gnn_layers=2,
        status_emb_dim=8,
        gnn_hidden_dim=64,
        pooling_type="mean",
        use_loc=True,
        detach_trunk=True,   # ✅
        # detach_trunk=False, 
    )
    # ------------------------------------------------------------
    # 6) Run experiment
    # ------------------------------------------------------------

    ignarl_callback = IgnarlSchedulerCallback(
        lr_end=5e-6,
        critic_coef_start=0.1,
        critic_coef_end=0.5,
        anneal_frames=500_000  # 根据你的总帧数设定
    )
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
        # callbacks=[ignarl_callback],
    )

    experiment.run()


if __name__ == "__main__":
    main()