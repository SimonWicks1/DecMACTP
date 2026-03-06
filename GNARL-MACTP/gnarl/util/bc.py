from gnarl.agent.imitation.imitation import (
    collect_experience,
    load_experience_dataset,
)
from gnarl.util.evaluation import ExpertPolicy
import os
import torch as th
from functools import partial
from gnarl.util.weights_init import multipurpose_init
from gnarl.envs import ENV_MAPPING
import numpy as np
from gnarl.util.classes import dict2string


def get_bc_experience(experience_path, train_env, config):
    """
    Get the experience dataset for behavioral cloning.
    If the dataset does not exist at the specified path, it collects new experience
    using an expert policy and saves it to that path.

    Args:
        experience_path (str): Path to the experience dataset.
        train_env (VecEnv): The training environment.
        config (dict): Configuration dictionary containing BC settings.
    Returns:
        dataset: The loaded or newly collected experience dataset.
    """
    rng = np.random.default_rng(config["BC"]["seed"])  # type: ignore

    if experience_path is not None and os.path.exists(experience_path):
        try:
            print(f"Loading expert transitions from {experience_path}")
            dataset = load_experience_dataset(experience_path)
            return dataset
        except Exception as e:
            print(f"Failed to load expert transitions: {e}")

    print("Collecting expert transitions for Behavioural Cloning.")
    os.makedirs(os.path.dirname(experience_path), exist_ok=True)
    expert_policy = ExpertPolicy(train_env.get_attr("unwrapped")[0].__class__, rng)
    collect_experience(
        lambda obs: expert_policy.predict(
            obs, deterministic=False, **config.get("expert_policy_kwargs", {})
        ),
        vec_env=train_env,
        n_episodes=config["BC"]["min_episodes"],
        save_frequency=1000,
        save_dir=experience_path,
    )

    dataset = load_experience_dataset(experience_path)
    return dataset


def get_bc_data_path(config) -> str:
    p = (
        f"experience/BC_{config['env']}_"
        + f"{list(config['train_data']['node_samples'].keys())}train-"
        + f"{config['BC']['min_episodes']}eps-"
        + f"{config['BC']['seed']}seed-"
        + f"{config['train_data']['graph_generator']}"
    )
    if "expert_policy_kwargs" in config:
        p += f"-{dict2string(config['expert_policy_kwargs'])}"
    return p


def complete_config(config: dict):
    config["env"] = ENV_MAPPING[config["algorithm"]]
    config["policy_kwargs"]["weights_init"] = partial(
        multipurpose_init,
        gnn_init=config["weights_init"].get("gnn_init"),
        encoder_init=config["weights_init"].get("encoder_init"),
    )

    if "BC" in config:
        config["BC"]["data_path"] = get_bc_data_path(config)

    if max(config["test_data"]["node_samples"].keys()) > 500:
        # Disable validation of distribution parameters for large environments
        # This is a workaround for an issue with large categorical distributions
        th.distributions.Distribution.set_default_validate_args(False)

    return config
