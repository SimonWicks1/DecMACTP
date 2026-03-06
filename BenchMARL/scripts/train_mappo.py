import torch

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.mactp.common import MactpTask

from benchmarl.models import GraphActorGNNConfig, GraphCriticGNNConfig


def main():
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

    seed = 42

    # ------------------------------------------------------------
    # 2) Task (MACTP)
    # ------------------------------------------------------------
    # 关键：这里要求环境已经移除 observation 里的 "state"
    task = MactpTask.TRAIN.get_from_yaml()

    # ------------------------------------------------------------
    # 3) Algorithm: MAPPO (CTDE)
    # ------------------------------------------------------------
    algorithm_config = MappoConfig(
        clip_epsilon=0.2,
        entropy_coef=0.01,
        critic_coef=1.0,
        loss_critic_type="l2",
        lmbda=0.95,
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=False,
        share_param_critic=True,     # 团队任务更稳
        minibatch_advantage=False,
    )

    # ------------------------------------------------------------
    # 4) Actor model: Shared Graph Encoder + Agent-conditioned Pointer Head
    # ------------------------------------------------------------
    # 说明：
    # - node_features / edge_features 这两个字段是 BenchMARL ModelConfig 的必填项（即使你的模型内部不用它们）
    # - 你的 env 里 ("agents","node_features") 是 7 通道：[dest, visited, opt, pess, self_pos, density, terminated]
    # - ("agents","edge_weights") 和 ("agents","edge_status") -> edge_features 我们内部构成：weight + status_emb
    actor_model_config = GraphActorGNNConfig(
        node_features=7,
        edge_features=2,
        num_gnn_layers=2,
        status_emb_dim=8,
        gnn_hidden_dim=64,
    )

    # ------------------------------------------------------------
    # 5) Critic model: Shared Graph Encoder + Centralized Pooling
    # ------------------------------------------------------------
    critic_model_config = GraphCriticGNNConfig(
        node_features=7,
        edge_features=2,
        num_gnn_layers=2,
        status_emb_dim=8,
        gnn_hidden_dim=64,
        use_phase=True,
    )

    # ------------------------------------------------------------
    # 6) Run experiment
    # ------------------------------------------------------------
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
    )

    experiment.run()


if __name__ == "__main__":
    main()