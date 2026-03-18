import torch
import argparse

from benchmarl.algorithms import MagnarlConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.mactp.common import MactpTask
from benchmarl.models import MagnarlActorGNNConfig, MagnarlCriticGNNConfig

from benchmarl.experiment.magnarl_diagnostics_callback import MagnarlDiagnosticsCallback


def main():
    parser = argparse.ArgumentParser(description="Run MAGNARL training.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    print(f"========== Running MAGNARL with SEED: {seed} ==========")

    experiment_config = ExperimentConfig.get_from_yaml()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.train_device = device
    experiment_config.sampling_device = device

    experiment_config.gamma = 0.99
    experiment_config.on_policy_collected_frames_per_batch = 2000
    experiment_config.on_policy_minibatch_size = 512
    experiment_config.on_policy_n_minibatch_iters = 4
    experiment_config.on_policy_n_envs_per_worker = 4

    experiment_config.loggers = ["csv", "tensorboard"]
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 10_000
    experiment_config.evaluation_episodes = 100
    experiment_config.render = False
    experiment_config.checkpoint_interval = 10_000
    experiment_config.max_n_frames = 500_000

    task = MactpTask.TRAIN.get_from_yaml()

    algorithm_config = MagnarlConfig(
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

    actor_model_config = MagnarlActorGNNConfig(
        node_features=7,
        edge_features=2,
        num_gnn_layers=2,
        status_emb_dim=8,
        gnn_hidden_dim=64,
    )

    critic_model_config = MagnarlCriticGNNConfig(
        node_features=7,
        edge_features=2,
        num_gnn_layers=2,
        status_emb_dim=8,
        gnn_hidden_dim=64,
    )

    callbacks = [
        MagnarlDiagnosticsCallback(group="agents", log_every_collection=True),
    ]

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
        callbacks=callbacks,
    )

    torch.autograd.set_detect_anomaly(True)
    experiment.run()


if __name__ == "__main__":
    main()