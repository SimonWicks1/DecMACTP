#!/usr/bin/env python3

import gnarl
import numpy as np
import torch as th
import yaml
import argparse
import wandb
import os
import zipfile
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from gnarl.agent.policy import MaskableNodeActorCriticPolicy
from gnarl.util.envs import make_eval_env
from gnarl.util.evaluation import (
    evaluate_policy_function,
    ExpertPolicy,
)

from gnarl.util.bc import complete_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GNARL model.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-r",
        "--run_id",
        type=str,
        help="WandB run ID for fetching the model.",
    )
    group.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Path to a local model file.",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "-e",
        "--expert",
        action="store_true",
        default=False,
        help="Evaluate the expert policy before the trained model.",
    )

    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=None,
        help="Maximum step to select model artifact from (requires -w). If specified, uses the model with the largest artifact._history_step <= this step.",
    )
    args = parser.parse_args()
    if args.wandb and not args.run_id:
        parser.error("-w/--wandb requires -r/--run_id to be specified.")
    if args.step is not None and not args.wandb:
        parser.error("-s/--step requires -w/--wandb to be specified.")
    return args


def evaluate_expert_policy(
    test_envs: list[VecEnv],
    config: dict,
):
    rng = np.random.default_rng(config["seed"])
    expert_policy = ExpertPolicy(test_envs[0].get_attr("unwrapped")[0].__class__, rng)
    evaluate_policy_function(
        test_envs,
        config,
        "expert",
        expert_policy,
        deterministic=config["test_data"]["deterministic"],  # type: ignore
        expert_kwargs=config.get("expert_policy_kwargs", {}),
    )


def maybe_load_from_zip(model_path: str) -> dict:
    try:
        d = th.load(model_path, weights_only=False)["state_dict"]
        return d
    except RuntimeError as e:
        with zipfile.ZipFile(model_path, "r") as z:
            with z.open("policy.pth") as f:
                d = th.load(f)
                return d


def load_policy_from_run(
    run,
    run_id: str,
    method: str,
    test_envs: list[VecEnv],
    config: dict,
    max_step: int = None,
    model_path: str = None,
) -> tuple[MaskableNodeActorCriticPolicy, dict]:

    # Helper to find the best artifact by step
    def get_best_artifact_by_step(run, run_id, method, max_step):
        api = wandb.Api()
        prefix = f"{run_id}_{method}_best_model"
        # List all versions of the artifact
        artifact_versions = api.artifacts("model", f"{run.project}/{prefix}")
        best = None
        best_step = -1
        for artifact in artifact_versions:
            step = getattr(artifact, "_history_step", None)
            if step is not None and step <= max_step and step > best_step:
                best = artifact
                best_step = step
        return best

    if model_path is not None:
        print(f"Loading model from local path: {model_path}")
    elif run is None or run._settings.mode == "disabled":
        print("Warning: WandB is disabled, trying to load local model.")
        artifact_dir = f"models/{run_id}"
        model_path = os.path.join(artifact_dir, f"{method}_best_model.pt")
    else:
        # Check if step is specified in args (monkeypatch: pass via run object)
        if max_step is not None:
            artifact = get_best_artifact_by_step(run, run_id, method, max_step)
            if artifact is None:
                raise RuntimeError(f"No artifact found for step <= {max_step}")
            print(f"Using artifact at step {getattr(artifact, '_history_step', '?')}")
        else:
            artifact = run.use_artifact(
                f"{run_id}_{method}_best_model:latest",
                type="model",
            )
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, f"{method}_best_model.pt")

    print(f"Loading model from {model_path}")

    policy = MaskableNodeActorCriticPolicy(
        observation_space=test_envs[0].observation_space,
        action_space=test_envs[0].action_space,
        lr_schedule=get_linear_fn(1e-5, 1e-5, 1.0),
        graph_spec=test_envs[0].get_attr("graph_spec")[0],
        **config["policy_kwargs"],  # type: ignore
    )
    policy.load_state_dict(maybe_load_from_zip(model_path))

    return policy, config


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    config = complete_config(config)

    RENDER = False

    th.manual_seed(config["seed"])
    np.random.seed(config["seed"])  # type: ignore

    run = wandb.init(
        project="GNARL-CTP",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,
        mode="disabled" if not args.wandb else "online",
        name=f"{config['env']}_{args.run_id}-{list(config['test_data']['node_samples'].keys())}eval",  # type: ignore
        tags=[
            config["algorithm"],  # type: ignore
            f"eval-{list(config['test_data']['node_samples'].keys())}",  # type: ignore
            f"network-{config['policy_kwargs']['network_kwargs']['network']}",  # type: ignore
            f"aggr-{config['policy_kwargs']['network_kwargs']['aggr']}",  # type: ignore
            f"pooling-{config['policy_kwargs']['pooling_type']}",  # type: ignore
            f"embed-{config['policy_kwargs']['embed_dim']}",  # type: ignore
        ],
        notes="",
    )

    print("Constructing test envs")
    test_envs = make_eval_env(config, run.id, "test", render=RENDER)

    if args.expert:
        print("Evaluating expert policy")
        evaluate_expert_policy(test_envs, config)

    method = "ppo" if "PPO" in config else "bc"

    print("Loading policy from run")
    policy, config = load_policy_from_run(
        run,
        args.run_id,
        method,
        test_envs,
        config,
        max_step=args.step,
        model_path=args.path,
    )

    print("Beginning evaluation")
    evaluate_policy_function(
        test_envs,
        config,
        method,
        policy,
        deterministic=config["test_data"]["deterministic"],  # type: ignore
        render=RENDER,
    )

    for env in test_envs:
        env.close()


if __name__ == "__main__":
    main()
