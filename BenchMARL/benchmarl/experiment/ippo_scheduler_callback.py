import torch
from tensordict import TensorDict
from benchmarl.experiment.callback import Callback


class IppoSchedulerCallback(Callback):
    """
    专门为共享主干 (Shared Trunk) 设计的调度器。
    动态调整 Critic 损失系数和学习率，防止价值网络梯度冲刷策略特征。
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
            self.anneal_frames = exp.config.get_max_n_frames(exp.on_policy) // 2
        self.lr_init = exp.config.lr

    def on_batch_collected(self, batch):
        exp = self.experiment
        current_frame = exp.total_frames
        progress = min(1.0, current_frame / self.anneal_frames)

        new_critic_coef_val = self.critic_coef_start + progress * (
            self.critic_coef_end - self.critic_coef_start
        )
        exp.algorithm.critic_coef = new_critic_coef_val

        new_critic_coef_tensor = torch.as_tensor(
            new_critic_coef_val,
            device=exp.config.train_device,
            dtype=torch.float32
        )
        for group in exp.group_map.keys():
            if group in exp.losses:
                exp.losses[group].critic_coeff = new_critic_coef_tensor

        new_lr = self.lr_init - progress * (self.lr_init - self.lr_end)
        for group in exp.optimizers.keys():
            for loss_name in exp.optimizers[group].keys():
                optimizer = exp.optimizers[group][loss_name]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

    def on_train_end(self, training_td, group):
        exp = self.experiment
        exp.logger.log(
            {
                "schedulers/critic_coef": exp.algorithm.critic_coef,
                "schedulers/lr": exp.optimizers[group][
                    "loss_objective"
                ].param_groups[0]["lr"],
            },
            step=exp.n_iters_performed,
        )