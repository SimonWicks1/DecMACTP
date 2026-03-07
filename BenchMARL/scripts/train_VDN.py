import torch

from benchmarl.algorithms import VdnConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.mactp.common import MactpTask

from benchmarl.models import GraphQNetConfig


import argparse

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

    print(f"--- Training Device: {experiment_config.train_device} ---")
    print(f"--- Sampling Device: {experiment_config.sampling_device} ---")

   
    experiment_config.gamma = 0.99
    
    experiment_config.off_policy_n_envs_per_worker = 2
    experiment_config.off_policy_collected_frames_per_batch = 2000
    experiment_config.off_policy_n_optimizer_steps = 1000
    experiment_config.off_policy_train_batch_size = 256
    experiment_config.off_policy_memory_size = 10_000

    experiment_config.loggers = ["csv", "tensorboard"]
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 10_000
    experiment_config.evaluation_episodes = 100
    experiment_config.render = False
    experiment_config.checkpoint_interval = 10_000

    experiment_config.max_n_frames = 500_000



    # ------------------------------------------------------------
    # 2) Task
    # ------------------------------------------------------------
    task = MactpTask.TRAIN.get_from_yaml()

    # ------------------------------------------------------------
    # 3) Algorithm: VDN
    # ------------------------------------------------------------
    algorithm_config = VdnConfig(
        delay_value=True,
        loss_function= "smooth_l1",
    )

    # ------------------------------------------------------------
    # 4) Model: GraphQNet (per-agent action_value)
    # ------------------------------------------------------------
    model_config = GraphQNetConfig(
        node_features=7,
        edge_features=2,
        num_gnn_layers=2,
        status_emb_dim=8,
        hidden_dim=64,
        use_action_mask=True,
    )

    # ------------------------------------------------------------
    # 5) Run experiment
    # ------------------------------------------------------------
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=seed,
        config=experiment_config,
    )
    experiment.run()


if __name__ == "__main__":
    main()