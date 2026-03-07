import torch

from benchmarl.algorithms import IppoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.mactp.common import MactpTask

# 导入我们重构后的 Actor 和 IPPO Critic
from benchmarl.models import GraphActorGNNConfig, IndependentGraphCriticGNNConfig

from benchmarl.experiment.callback import Callback
import numpy as np
import argparse

class IppoSchedulerCallback(Callback):
    """
    专门为共享主干 (Shared Trunk) 设计的调度器。
    通过动态调整 Critic 损失系数和学习率，防止价值网络的巨大梯度冲刷掉策略网络的特征。
    """
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
            # 使用配置中的最大帧数进行退火计算
            self.anneal_frames = exp.config.get_max_n_frames(exp.on_policy) // 2
        self.lr_init = exp.config.lr

    def on_batch_collected(self, batch):
        exp = self.experiment
        current_frame = exp.total_frames
        progress = min(1.0, current_frame / self.anneal_frames)

        # 1. 动态更新 critic_coef (缓解共享主干的梯度冲突)
        new_critic_coef_val = self.critic_coef_start + progress * (self.critic_coef_end - self.critic_coef_start)
        exp.algorithm.critic_coef = new_critic_coef_val
        
        # 将 float 转换为 Tensor 并移动到训练设备上，符合 TorchRL/BenchMARL 底层要求
        new_critic_coef_tensor = torch.as_tensor(
            new_critic_coef_val, 
            device=exp.config.train_device, 
            dtype=torch.float32
        )

        for group in exp.group_map.keys():
            if group in exp.losses:
                exp.losses[group].critic_coeff = new_critic_coef_tensor

        # 2. 线性学习率衰减 (Linear LR Annealing)，帮助共享网络在后期稳定收敛
        new_lr = self.lr_init - progress * (self.lr_init - self.lr_end)
        for group in exp.optimizers.keys():
            for loss_name in exp.optimizers[group].keys():
                optimizer = exp.optimizers[group][loss_name]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

    def on_train_end(self, training_td, group):
        exp = self.experiment
        # 记录关键调度参数，方便在 Tensorboard/Wandb 中对齐分析
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

    experiment_config.max_n_frames = 500_000


    # ------------------------------------------------------------
    # 2) Task (MACTP)
    # ------------------------------------------------------------
    task = MactpTask.TRAIN.get_from_yaml()

    # ------------------------------------------------------------
    # 3) Algorithm: IPPO (去中心化执行与去中心化训练)
    # ------------------------------------------------------------
    algorithm_config = IppoConfig(
        clip_epsilon=0.2,
        entropy_coef=0.01,
        critic_coef=1.0,  # 初始值会被 IppoSchedulerCallback 动态覆盖
        loss_critic_type="l2",
        lmbda=0.95,
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=False,
        # BenchMARL 中的 share_param_critic=True 指的是智能体(Agents)之间共享同一种 Critic 结构
        # 并不是 Actor 和 Critic 共享主干（主干共享由我们的 GNN 代码内部通过单例注册表实现）
        share_param_critic=True,     
        minibatch_advantage=False,
    )

    # ------------------------------------------------------------
    # 4) Actor model: 使用通过 get_or_create_graph_trunk 链接共享主干的模型
    # ------------------------------------------------------------
    actor_model_config = GraphActorGNNConfig(
        node_features=7,
        edge_features=2,
        num_gnn_layers=2,
        status_emb_dim=8,
        gnn_hidden_dim=64,
    )

    # ------------------------------------------------------------
    # 5) Critic model: IPPO 独立评估 Critic，同样链接共享主干
    # ------------------------------------------------------------
    critic_model_config = IndependentGraphCriticGNNConfig(
        node_features=7,
        edge_features=2,
        num_gnn_layers=2,
        status_emb_dim=8,
        gnn_hidden_dim=64,
        use_phase=True,
        detach_trunk=True,  # 确保开启
    )

    # ------------------------------------------------------------
    # 6) Run experiment
    # ------------------------------------------------------------
    # 实例化我们的动态调度器
    ippo_callback = IppoSchedulerCallback(
        lr_end=5e-6,
        critic_coef_start=0.1,  # 强制降低 Critic 初始梯度权重，保护策略特征提取
        critic_coef_end=0.5,
        anneal_frames=500_000
    )

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
        callbacks=[ippo_callback],  # 挂载回调
    )

    experiment.run()


if __name__ == "__main__":
    main()