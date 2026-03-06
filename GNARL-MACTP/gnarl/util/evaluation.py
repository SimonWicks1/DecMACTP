import os

import numpy as np
import wandb
from tqdm import tqdm

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from gnarl.envs.alg_env import PhasedNodeSelectEnv
from gnarl.util.envs import calculate_env_split

from typing import Any, Callable, Optional, Union
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
import warnings


def is_wandb() -> bool:
    """Check if wandb is not disabled."""
    return wandb.run is not None and wandb.run._settings.mode != "disabled"


class ExpertPolicy:
    """Wrapper for expert policies used for collecting experience for behavioral cloning."""

    def __init__(self, env_class: PhasedNodeSelectEnv, rng: np.random.Generator):
        self.env_class = env_class
        self.rng = rng

    def predict(
        self,
        obs: dict,
        deterministic: bool,
        *args,
        **kwargs,
    ):
        action_probs = self.env_class.expert_policy(obs, **kwargs)
        if action_probs.shape[-1] == 1:
            # Policy returns a single action instead of probs
            actions = action_probs.squeeze(-1)
            action_probs = None
        elif deterministic:
            actions = np.argmax(action_probs, axis=-1)
        else:
            actions = np.array(
                [self.rng.choice(len(probs), p=probs) for probs in action_probs]
            )
        return actions, action_probs


def change_obs_action_space(
    policy: ActorCriticPolicy | MaskableActorCriticPolicy | ExpertPolicy,
    eval_env: VecEnv,
) -> ActorCriticPolicy | MaskableActorCriticPolicy | ExpertPolicy:
    """Change the observation and action space of a policy to match that of a new environment."""
    if not isinstance(policy, MaskableActorCriticPolicy):
        return policy
    constructor_args = policy._get_constructor_parameters()
    constructor_args["observation_space"] = eval_env.observation_space
    constructor_args["action_space"] = eval_env.action_space
    eval_policy = policy.__class__(**constructor_args)
    eval_policy.load_state_dict(policy.state_dict())
    return eval_policy


def evaluate_one_env(
    model: MaskableActorCriticPolicy | ActorCriticPolicy | ExpertPolicy,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    warn: bool = True,
    use_masking: bool = True,
    **kwargs,
) -> tuple[
    list[float],
    list[int],
    list[bool],
    list[int],
    list[float | None],
    list[float | None],
    list[int],
]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :param use_masking: Whether or not to use invalid action masks during evaluation
    :return: Returns list of rewards, lengths, num_nodes,
        objective_evaluations, expert_objectives (if they exist), env_index
    """

    if use_masking and not is_masking_supported(env):
        raise ValueError(
            "Environment does not support action masking. Consider using ActionMasker wrapper"
        )

    is_monitor_wrapped = False

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    objective_evaluations = []
    num_nodes = []
    expert_objectives = []
    success_buffer = []
    env_index = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    if render:
        env.render()
    current_expert_objectives = [None] * n_envs
    for i in range(n_envs):
        num_nodes.append(env.get_attr("num_nodes", [i])[0])
        if hasattr(env.envs[i].unwrapped, "graph_data"):
            graph_data = env.envs[i].unwrapped.graph_data
            if hasattr(graph_data, "expert_objective"):
                current_expert_objectives[i] = graph_data.expert_objective.item()

    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        if use_masking:
            action_masks = get_action_masks(env)
            actions, state = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
                action_masks=action_masks,
                **kwargs,
            )
        else:
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
                **kwargs,
            )
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

                    is_success = info.get("is_success")
                    success_buffer.append(is_success)
                    if hasattr(env.envs[i].unwrapped, "objective_function"):
                        J = env.envs[i].unwrapped.objective_function(
                            **info["terminal_observation"]
                        )
                        objective_evaluations.append(J)
                    else:
                        objective_evaluations.append(None)
                    expert_objectives.append(current_expert_objectives[i])
                    env_index.append(i)

                    # Capture expert objective for the NEXT episode
                    if hasattr(env.envs[i].unwrapped, "graph_data"):
                        graph_data = env.envs[i].unwrapped.graph_data
                        if hasattr(graph_data, "expert_objective"):
                            current_expert_objectives[i] = (
                                graph_data.expert_objective.item()
                            )
                        else:
                            current_expert_objectives[i] = None
                    else:
                        current_expert_objectives[i] = None

            if episode_counts[i] < episode_count_targets[i]:
                num_nodes.append(env.get_attr("num_nodes", [i])[0])
        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    return (
        episode_rewards,
        episode_lengths,
        success_buffer,
        num_nodes,
        objective_evaluations,
        expert_objectives,
        env_index,
    )


def evaluate_policy(
    model: MaskableActorCriticPolicy | ActorCriticPolicy | ExpertPolicy,
    envs: list[VecEnv],
    n_eval_episodes: list[int],
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    warn: bool = True,
    use_masking: bool = True,
    **kwargs,
) -> tuple[
    list[list[float]],
    list[list[int]],
    list[list[bool]],
    list[list[int]],
    list[list[float | None]],
    list[list[float | None]],
    list[list[int]],
]:
    """Run evaluation on multiple environments and aggregate the results."""
    outputs = []
    for venv, episodes in zip(envs, n_eval_episodes):
        policy = change_obs_action_space(model, venv)
        try:
            outputs.append(
                evaluate_one_env(
                    model=policy,
                    env=venv,
                    n_eval_episodes=episodes,
                    deterministic=deterministic,
                    render=render,
                    callback=callback,
                    reward_threshold=reward_threshold,
                    warn=warn,
                    use_masking=use_masking,
                    **kwargs,
                )
            )
        except Exception as e:
            print(f"Error evaluating on environment {venv}: {e}. Skipping.")

    if not outputs:
        print("No environments were evaluated successfully. Returning None values.")
        return [None for _ in range(7)]  # type: ignore[list]

    return [
        [outputs[i][j] for i in range(len(outputs))] for j in range(len(outputs[0]))
    ]  # type: ignore[index]


def get_values_by_n(
    val_list: list[list[float | int]], n_list: list[list[int]]
) -> dict[int, list[float | int]]:
    values_by_n = {}
    for vals, ns in zip(val_list, n_list):
        for n, val in zip(ns, vals):
            if n not in values_by_n:
                values_by_n[n] = []
            values_by_n[n].append(val)
    return values_by_n


def process_evaluation_output(
    pfx: str,
    rewards: list[list[float]],
    lengths: list[list[int]],
    success: list[list[bool]],
    num_nodes: list[list[int]],
    objectives: list[list[float | None]],
    expert_objs: list[list[float | None]],
) -> dict[str, Any]:

    if not rewards:
        return {}

    def flt(lst: list[list[Any]]) -> list[Any]:
        return [item for sublist in lst for item in sublist]

    def sts(
        data: list[float | int],
    ) -> dict[str, Any]:
        d = {
            "mean": float(np.mean(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
        }
        return d

    use_success = not any(any(s is None for s in suc) for suc in success)
    use_objective = not any(any(o is None for o in obj) for obj in objectives)
    use_expert = (
        use_objective
        and all(len(e) > 0 for e in expert_objs)
        and not any(any(e is None for e in exp) for exp in expert_objs)
    )
    # Get the expert objectives if available
    if use_expert:
        diffs = [np.array(o) - e for o, e in zip(objectives, expert_objs)]
        rels = [
            np.array(o) / e if e != 0 else 0 for o, e in zip(objectives, expert_objs)
        ]

    # Calculate overall stats
    l = {f"{pfx}/n_episodes": len(flt(lengths))}
    l.update({f"{pfx}/length/{k}": v for k, v in sts(flt(lengths)).items()})
    l.update({f"{pfx}/reward/{k}": v for k, v in sts(flt(rewards)).items()})

    if use_success:
        l.update({f"{pfx}/success/{k}": v for k, v in sts(flt(success)).items()})
    if use_objective:
        l.update({f"{pfx}/objective/{k}": v for k, v in sts(flt(objectives)).items()})
    if use_expert:
        l.update({f"{pfx}/obj_diff/{k}": v for k, v in sts(flt(diffs)).items()})
        l.update({f"{pfx}/obj_ratio/{k}": v for k, v in sts(flt(rels)).items()})

    # Calculate tabular stats by number of nodes
    table_headings = ["n"]
    rw_n = get_values_by_n(rewards, num_nodes)
    len_n = get_values_by_n(lengths, num_nodes)
    rw_stats = {n: sts(rw_n[n]) for n in rw_n}
    len_stats = {n: sts(len_n[n]) for n in len_n}
    table_stats = {n: [n] for n in rw_n.keys()}
    table_headings += [f"reward/{k}" for k in list(rw_stats.values())[0].keys()]
    table_headings += [f"length/{k}" for k in list(len_stats.values())[0].keys()]
    for n in rw_n:
        table_stats[n] += rw_stats[n].values()
        table_stats[n] += len_stats[n].values()
    if use_success:
        suc_n = get_values_by_n(success, num_nodes)
        suc_stats = {n: sts(suc_n[n]) for n in suc_n}
        table_headings += [f"success/{k}" for k in list(suc_stats.values())[0].keys()]
        for n in rw_n:
            table_stats[n] += suc_stats[n].values()
    if use_objective:
        obj_n = get_values_by_n(objectives, num_nodes)
        obj_stats = {n: sts(obj_n[n]) for n in obj_n}
        table_headings += [f"objective/{k}" for k in list(obj_stats.values())[0].keys()]
        for n in rw_n:
            table_stats[n] += obj_stats[n].values()
    if use_expert:
        diff_n = get_values_by_n(diffs, num_nodes)
        rel_n = get_values_by_n(rels, num_nodes)
        diff_stats = {n: sts(diff_n[n]) for n in diff_n}
        rel_stats = {n: sts(rel_n[n]) for n in rel_n}
        table_headings += [f"obj_diff/{k}" for k in list(diff_stats.values())[0].keys()]
        table_headings += [f"obj_ratio/{k}" for k in list(rel_stats.values())[0].keys()]
        for n in rw_n:
            table_stats[n] += diff_stats[n].values()
            table_stats[n] += rel_stats[n].values()
    # Create a wandb table
    if is_wandb():
        table = wandb.Table(columns=table_headings)
        for data in table_stats.values():
            table.add_data(*data)
    else:
        table = {"headings": table_headings, "data": list(table_stats.values())}

    l[f"{pfx}/table"] = table

    return l


def heirarchical_comparison(
    a: dict[str, float | int], b: dict[str, float | int], order: list[str]
) -> bool:
    """
    Compare two dictionaries based on a predefined order of keys.
    The first key in the order is the most important, and so on.
    Returns True if 'a' is better than 'b', False otherwise.
    """
    for key in order:
        if key not in a or key not in b:
            continue
        if a[key] > b[key]:
            return True
        elif a[key] < b[key]:
            return False
    return False


def save_best_model(
    best: dict[str, float | int],
    achieved: dict[str, float | int],
    model,
    best_model_save_path: str | None,
    method: str,
    verbose: int = 0,
) -> bool:

    if heirarchical_comparison(
        achieved, best, ["mean_success", "mean_reward", "mean_length"]
    ):
        best.update(achieved)
        if verbose > 0:
            print(
                f"New best mean success/reward/length: {best['mean_success']:.2f}/{best['mean_reward']:.2f}/{best['mean_length']:.2f}!"
            )
        if best_model_save_path is not None:
            model_path = os.path.join(best_model_save_path, f"{method}_best_model.pt")
            model.save(model_path)
            # Log as a new artifact version
            artifact = wandb.Artifact(
                f"{wandb.run.id}_{method}_best_model", type="model"
            )
            artifact.add_file(model_path)
            artifact.metadata = {
                "mean_success": best["mean_success"],
                "mean_reward": best["mean_reward"],
                "mean_length": best["mean_length"],
            }
            wandb.log_artifact(artifact)

        wandb.log(best)
        return True
    wandb.log(best)
    return False


def evaluate_policy_function(
    eval_envs: list[VecEnv],
    config: dict[str, Any],
    policy_name: str,
    policy_fn: ExpertPolicy | MaskableActorCriticPolicy,
    deterministic: bool = True,
    expert_kwargs: dict[str, Any] = None,
    render: bool = False,
):
    print(f"Evaluating {policy_name} policy.")

    if expert_kwargs is None:
        expert_kwargs = {}

    per_env_samples = calculate_env_split(
        node_samples=config["test_data"]["node_samples"],
        max_envs=config["test_data"]["num_envs"],
    )
    eval_episode_list = [c * e for n, c, e in per_env_samples]

    # NOTE: Won't be accurate if edges are used in calculation of max steps
    max_steps_fn = eval_envs[0].get_attr("unwrapped")[0].__class__.get_max_episode_steps
    total_steps = sum(
        [
            max_steps_fn(n, n * n) * c
            for n, c in config["test_data"]["node_samples"].items()
        ]
    )
    pbar = tqdm(
        total=total_steps,
        desc=f"Evaluating {policy_name} policy",
    )

    completed_episodes = [0]  # List for mutable integer to track completed episodes

    def progress_callback(_locals, _globals):
        completed_episodes[0] += int(_locals.get("done", False))
        pbar.update(1)
        pbar.set_postfix({"completed": completed_episodes[0]})
        return True

    (
        episode_rewards,
        episode_lengths,
        episode_success,
        num_nodes,
        objective_values,
        expert_objectives,
        env_index,
    ) = evaluate_policy(
        policy_fn,
        eval_envs,
        n_eval_episodes=eval_episode_list,
        deterministic=deterministic,
        callback=progress_callback,
        render=render,
        **expert_kwargs,
    )
    pbar.close()

    log_info = process_evaluation_output(
        f"evaluation/{policy_name}",
        episode_rewards,
        episode_lengths,
        episode_success,
        num_nodes,
        objective_values,
        expert_objectives,
    )
    if is_wandb():
        wandb.log(data=log_info)
    else:
        print(log_info)
    print(
        f"Mean success: {log_info.get(f'evaluation/{policy_name}/success/mean')}, "
        + f"Mean reward: {log_info.get(f'evaluation/{policy_name}/reward/mean')}, "
        + f"Mean episode length: {log_info.get(f'evaluation/{policy_name}/length/mean')}"
    )
    return objective_values, env_index
