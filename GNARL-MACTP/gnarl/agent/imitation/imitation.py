import torch as th
from torch.utils.data import DataLoader
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.optim as optim
import tqdm
from typing import Optional, Callable, Iterator
import torch.nn.functional as F
import numpy as np
import dataclasses
from torch.utils.data.dataset import Dataset, random_split
import itertools
import wandb
from stable_baselines3.common.vec_env import VecEnv
from gnarl.util.evaluation import (
    evaluate_policy,
    process_evaluation_output,
    save_best_model,
)
import os
import pickle
import tempfile


def is_wandb() -> bool:
    return wandb.run is not None and wandb.run._settings.mode != "disabled"


class ExperienceBuffer:
    """Incremental experience buffer that saves data to disk as it's collected."""

    def __init__(self, save_dir: str, save_frequency: int = 1000):
        """
        Args:
            save_dir: Directory to save experience files
            save_frequency: Save to disk every N steps (default: 50k for larger chunks)
        """
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self.buffer = []
        self.file_counter = 0
        self.total_steps = 0

        os.makedirs(save_dir, exist_ok=True)

    def add_step(self, obs, action, action_prob, reward, done, next_obs):
        """Add a single step to the buffer."""
        step_data = {
            "obs": obs,
            "action": action,
            "action_prob": action_prob,
            "reward": reward,
            "done": done,
            "next_obs": next_obs,
        }
        self.buffer.append(step_data)
        self.total_steps += 1

        # save to disk if buffer is full
        if len(self.buffer) >= self.save_frequency:
            self._save_buffer()

    def _save_buffer(self):
        """Save current buffer to disk and clear it."""
        if not self.buffer:
            return

        filename = os.path.join(
            self.save_dir, f"experience_chunk_{self.file_counter:04d}.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)

        self.buffer = []
        self.file_counter += 1

    def finalise(self):
        """Save any remaining data in buffer."""
        if self.buffer:
            self._save_buffer()


class Experience(Dataset):
    def __init__(self, experience_dir: str, cache_size: int = 3, prefetch: bool = True):
        """
        Load experience from directory of saved chunks.

        Args:
            experience_dir: Directory containing experience chunk files
            cache_size: Number of chunks to keep in memory cache
            prefetch: Whether to prefetch adjacent chunks
        """
        self.experience_dir = experience_dir
        self.cache_size = cache_size
        self.prefetch = prefetch
        self.chunk_cache = {}  # {chunk_idx: chunk_data}
        self.cache_order = []  # LRU order

        self.chunk_files = sorted(
            [
                f
                for f in os.listdir(experience_dir)
                if f.startswith("experience_chunk_") and f.endswith(".pkl")
            ]
        )

        if not self.chunk_files:
            raise ValueError(f"No experience files found in {experience_dir}")

        self._count_samples()

    def _count_samples(self):
        """Count total number of samples across all chunks."""
        self.total_samples = 0
        self.chunk_sizes = []

        for chunk_file in self.chunk_files:
            chunk_path = os.path.join(self.experience_dir, chunk_file)
            with open(chunk_path, "rb") as f:
                chunk_data = pickle.load(f)
            chunk_size = len(chunk_data)
            self.chunk_sizes.append(chunk_size)
            self.total_samples += chunk_size

    def _find_chunk_and_index(self, global_index):
        """Find which chunk contains the global index and the local index within that chunk."""
        if global_index >= self.total_samples:
            raise IndexError(
                f"Index {global_index} out of range (total: {self.total_samples})"
            )

        current_offset = 0
        for chunk_idx, chunk_size in enumerate(self.chunk_sizes):
            if global_index < current_offset + chunk_size:
                local_index = global_index - current_offset
                return chunk_idx, local_index
            current_offset += chunk_size

        raise IndexError(f"Could not find chunk for index {global_index}")

    def _load_chunk(self, chunk_idx: int):
        """Load a chunk into cache with LRU eviction."""
        if chunk_idx in self.chunk_cache:
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.chunk_cache[chunk_idx]

        chunk_file = self.chunk_files[chunk_idx]
        chunk_path = os.path.join(self.experience_dir, chunk_file)
        with open(chunk_path, "rb") as f:
            chunk_data = pickle.load(f)

        # add to cache
        self.chunk_cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)

        # evict oldest if cache is full
        if len(self.chunk_cache) > self.cache_size:
            oldest_chunk = self.cache_order.pop(0)
            del self.chunk_cache[oldest_chunk]

        if self.prefetch and chunk_idx + 1 < len(self.chunk_files):
            next_idx = chunk_idx + 1
            if (
                next_idx not in self.chunk_cache
                and len(self.chunk_cache) < self.cache_size
            ):
                try:
                    next_chunk_file = self.chunk_files[next_idx]
                    next_chunk_path = os.path.join(self.experience_dir, next_chunk_file)
                    with open(next_chunk_path, "rb") as f:
                        next_chunk_data = pickle.load(f)
                    self.chunk_cache[next_idx] = next_chunk_data
                    self.cache_order.append(next_idx)
                except Exception:
                    pass

        return chunk_data

    def __getitem__(self, index):
        chunk_idx, local_index = self._find_chunk_and_index(index)

        chunk_data = self._load_chunk(chunk_idx)
        step_data = chunk_data[local_index]

        def convert_obs_to_tensor(obs):
            if isinstance(obs, dict):
                return {
                    k: th.tensor(v) if isinstance(v, np.ndarray) else v
                    for k, v in obs.items()
                }
            elif isinstance(obs, np.ndarray):
                return th.tensor(obs)
            else:
                return obs

        # convert to expected format
        item = {
            "obs": convert_obs_to_tensor(step_data["obs"]),
            "acts": (
                th.tensor(step_data["action"]).unsqueeze(-1)
                if isinstance(step_data["action"], (int, float))
                else (
                    th.tensor(step_data["action"])
                    if isinstance(step_data["action"], np.ndarray)
                    else step_data["action"]
                )
            ),
            "rewards": (
                th.tensor(step_data["reward"]).unsqueeze(-1)
                if isinstance(step_data["reward"], (int, float))
                else (
                    th.tensor(step_data["reward"])
                    if isinstance(step_data["reward"], np.ndarray)
                    else step_data["reward"]
                )
            ),
            "done": (
                th.tensor(step_data["done"]).unsqueeze(-1)
                if isinstance(step_data["done"], (int, float, bool))
                else (
                    th.tensor(step_data["done"])
                    if isinstance(step_data["done"], np.ndarray)
                    else step_data["done"]
                )
            ),
            "next_obs": convert_obs_to_tensor(step_data["next_obs"]),
        }

        if step_data["action_prob"] is not None:
            item["act_probs"] = (
                th.tensor(step_data["action_prob"])
                if isinstance(step_data["action_prob"], np.ndarray)
                else step_data["action_prob"]
            )

        return item

    def __len__(self):
        return self.total_samples


def collect_experience(
    policy: Callable,
    vec_env: VecEnv,
    n_episodes: int = 10,
    save_dir: Optional[str] = None,
    save_frequency: int = 1000,
) -> str:
    """
    Collects experience from the environment using the given policy.
    Saves data incrementally to avoid memory issues.

    Args:
        policy: Policy to collect experience with
        vec_env: Vectorised environment
        n_episodes: Number of episodes to collect
        save_dir: Directory to save experience. If None, uses temp directory
        save_frequency: Save to disk every N steps

    Returns:
        Path to directory containing saved experience
    """
    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="experience_")

    buffer = ExperienceBuffer(save_dir, save_frequency)
    episode_counter = 0

    obs = vec_env.reset()
    with tqdm.tqdm(total=n_episodes, desc="Collecting episodes") as pbar:
        while episode_counter < n_episodes:
            act, act_probs = policy(obs)
            next_obs, rew, done, info = vec_env.step(act)

            for env_idx in range(vec_env.num_envs):
                if isinstance(obs, dict):
                    env_obs = {k: v[env_idx] for k, v in obs.items()}
                elif isinstance(obs, (list, tuple)):
                    env_obs = obs[env_idx]
                else:
                    env_obs = obs[env_idx]

                if isinstance(next_obs, dict):
                    env_next_obs = {k: v[env_idx] for k, v in next_obs.items()}
                elif isinstance(next_obs, (list, tuple)):
                    env_next_obs = next_obs[env_idx]
                else:
                    env_next_obs = next_obs[env_idx]

                # handle terminal observations
                if done[env_idx] and "episode" in info[env_idx]:
                    env_next_obs = info[env_idx]["terminal_observation"]

                buffer.add_step(
                    obs=env_obs,
                    action=act[env_idx],
                    action_prob=act_probs[env_idx] if act_probs is not None else None,
                    reward=rew[env_idx],
                    done=done[env_idx],
                    next_obs=env_next_obs,
                )

            episode_done = done.sum().item()
            episode_counter += episode_done
            pbar.update(episode_done)

            obs = next_obs

    buffer.finalise()

    print(f"Collected {buffer.total_steps} steps saved to {save_dir}")
    return save_dir


def load_experience_dataset(
    experience_dir: str, cache_size: int = 3, prefetch: bool = True
) -> Experience:
    """
    Load experience dataset from directory.

    Args:
        experience_dir: Directory containing experience chunks
        cache_size: Number of chunks to keep in memory (default: 3)
        prefetch: Whether to prefetch adjacent chunks (default: True)

    Returns:
        Experience dataset
    """
    return Experience(experience_dir, cache_size=cache_size, prefetch=prefetch)


def split_dataset(
    experience: Experience,
    split: float,
    generator: Optional[th.Generator] = None,
) -> tuple[Dataset, Dataset]:

    train_size = int(len(experience) * split)
    val_size = len(experience) - train_size
    train_dataset, val_dataset = random_split(
        experience, [train_size, val_size], generator=generator
    )
    return train_dataset, val_dataset


def make_data_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset: The dataset to load.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at every epoch.

    Returns:
        A DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )


@dataclasses.dataclass(frozen=True)
class BatchIteratorWithEpochEndCallback:
    """Loops through batches from a batch loader and calls a callback after every epoch.

    Will throw an exception when an epoch contains no batches.
    """

    batch_loader: DataLoader
    n_epochs: int
    on_epoch_end: Optional[Callable[[int], None]]

    def __iter__(self) -> Iterator:
        def batch_iterator() -> Iterator:
            # Note: the islice here ensures we do not exceed self.n_epochs
            for epoch_num in itertools.islice(itertools.count(), self.n_epochs):
                some_batch_was_yielded = False
                for batch in self.batch_loader:
                    yield batch
                    some_batch_was_yielded = True

                if not some_batch_was_yielded:
                    raise AssertionError(
                        f"Data loader returned no data during epoch "
                        f"{epoch_num} -- did it reset correctly?",
                    )
                if self.on_epoch_end is not None:
                    self.on_epoch_end(epoch_num)

        return batch_iterator()


def log_bc_epoch_stats(policy, data_loader, epoch):
    """
    Computes and logs statistics for behavioural cloning at the end of an epoch.

    Args:
        policy: The trained policy.
        data_loader: DataLoader for the (validation) set.
        epoch: Current epoch number.
    """
    if data_loader is not None:
        policy.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_entropy = 0.0
        all_preds = []
        all_targets = []

        with th.no_grad():
            for data in data_loader:
                obs = data["obs"]
                acts = data["acts"]
                dist = policy.get_distribution(obs)
                logits = dist.distribution.logits
                log_probs = logits.log_softmax(dim=-1)
                if "act_probs" in data:
                    act_probs = data["act_probs"]
                    loss = F.kl_div(log_probs, act_probs, reduction="batchmean")
                else:
                    loss = F.cross_entropy(
                        log_probs, acts.squeeze(-1).squeeze(-1), reduction="mean"
                    )
                total_loss += loss.item() * acts.size(0)

                preds = logits.argmax(dim=-1)
                total_correct += (preds == acts.squeeze(-1)).sum().item()
                total_samples += acts.size(0)
                total_entropy += dist.distribution.entropy().sum().item()
                all_preds.append(preds.cpu())
                all_targets.append(acts.cpu())

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        avg_entropy = total_entropy / total_samples
        all_preds = th.cat(all_preds)
        all_targets = th.cat(all_targets)

        log_info = {
            "bc/epoch_loss": avg_loss,
            "bc/epoch_acc": accuracy,
            "bc/epoch_entropy": avg_entropy,
            "bc/epoch": epoch,
        }
        if is_wandb():
            wandb.log(log_info)
        else:
            print(log_info)
    print(
        f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.4f}, entropy={avg_entropy:.4f}"
    )


def compute_critic_loss(
    policy: ActorCriticPolicy | MaskableActorCriticPolicy,
    gamma: float,
    rewards: th.Tensor,
    dones: th.Tensor,
    next_obs: dict[str, th.Tensor],
    values: th.Tensor,
):
    """Compute the critic loss using Mean Squared Error."""
    k = policy.predict_values(next_obs)
    targets = rewards.unsqueeze(-1) + gamma * k * ~(dones.bool()).unsqueeze(-1)
    return F.mse_loss(values, targets)


def compute_actor_loss(
    log_probs: th.Tensor,
    target: th.Tensor,
    loss_fn: Callable,
):
    """Compute the actor loss using the specified loss function."""
    if loss_fn.__name__ == "kl_div":
        return loss_fn(log_probs, target, reduction="batchmean")
    else:
        return loss_fn(log_probs, target, reduction="mean")


def get_log_probs_and_values(
    policy: ActorCriticPolicy | MaskableActorCriticPolicy,
    obs: dict[str, th.Tensor],
):
    """Get log probabilities and state values from the policy in one call."""
    features = policy.extract_features(obs, policy.pi_features_extractor)
    latent_pi, latent_vf = policy.mlp_extractor(features)
    distribution = policy._get_action_dist_from_latent(latent_pi)
    log_probs = distribution.distribution.logits.log_softmax(dim=-1)
    values = policy.value_net(latent_vf)
    return log_probs, values


def behavioural_cloning(
    policy: ActorCriticPolicy | MaskableActorCriticPolicy,
    data_loader: DataLoader,
    val_loader: Optional[DataLoader],
    val_envs: list[VecEnv],
    n_eval_episodes: list[int],
    gamma: float = 1,
    vf_coef: float = 0.5,
    eval_freq: int = 1000,
    n_epochs: int = 10,
    progress_bar: bool = True,
    learning_rate: float = 1e-3,
    best_model_save_path: Optional[str] = None,
    deterministic_eval: bool = True,
    **unused_kwargs,
):
    """
    Args:
        policy: The policy to be trained.
        data_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set (can be None).
        val_envs: List of vectorised environments for evaluation.
        n_eval_episodes: Number of episodes to evaluate per environment.
        gamma: Discount factor for future rewards.
        vf_coef: Coefficient for the value function loss.
        eval_freq: Frequency (in batches) to run evaluation.
        n_epochs: Number of epochs to train.
        progress_bar: Whether to display a progress bar during training.
        learning_rate: Initial learning rate for the optimizer.
        best_model_save_path: Directory to save the best model (if None, won't save).
        deterministic_eval: Whether to use deterministic actions during evaluation.
        **unused_kwargs: Additional unused keyword arguments.

    Returns:
        The trained policy.
    """
    if best_model_save_path is not None:
        os.makedirs(best_model_save_path, exist_ok=True)

    batch_iterator = BatchIteratorWithEpochEndCallback(
        batch_loader=data_loader,
        n_epochs=n_epochs,
        on_epoch_end=lambda epoch: log_bc_epoch_stats(policy, val_loader, epoch),
    )

    optimiser = optim.Adam(policy.parameters(), lr=learning_rate)
    policy.train()

    best = {
        "mean_success": 0.0,
        "mean_reward": -np.inf,
        "mean_length": np.inf,
    }

    if progress_bar:
        batch_iterator = tqdm.tqdm(
            batch_iterator,
            total=len(data_loader) * n_epochs,
            desc="Training BC policy",
            unit="batch",
        )

    for batch_idx, data in enumerate(batch_iterator):
        optimiser.zero_grad()

        obs = data["obs"]
        if "act_probs" in data:
            target = data["act_probs"]
            act_loss_fn = F.kl_div
        else:
            target = data["acts"].squeeze(-1).squeeze(-1)
            act_loss_fn = F.cross_entropy

        log_probs, values = get_log_probs_and_values(policy, obs)

        actor_loss = compute_actor_loss(log_probs, target, act_loss_fn)
        if vf_coef != 0:
            critic_loss = compute_critic_loss(
                policy, gamma, data["rewards"], data["done"], data["next_obs"], values
            )
        else:
            critic_loss = th.tensor(0.0)

        loss = actor_loss + vf_coef * critic_loss
        loss.backward()
        optimiser.step()

        # Logging training stats
        epoch = batch_idx // len(data_loader)
        log_info = {
            "bc/actor_loss": actor_loss.item(),
            "bc/critic_loss": critic_loss.item(),
            "bc/batch_idx": batch_idx,
            "bc/epoch": epoch,
        }
        if is_wandb():
            wandb.log(log_info)

        if progress_bar:
            batch_iterator.set_postfix(
                {
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "epoch": epoch + 1,
                }
            )

        if (batch_idx + 1) % eval_freq == 0:
            print(f"\n--- Running evaluation at batch {batch_idx + 1} ---")

            (
                episode_rewards,
                episode_lengths,
                episode_success,
                num_nodes,
                objective_values,
                expert_objectives,
                _,
            ) = evaluate_policy(
                policy,
                val_envs,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic_eval,
            )

            log_info = process_evaluation_output(
                "bc/val",
                episode_rewards,
                episode_lengths,
                episode_success,
                num_nodes,
                objective_values,
                expert_objectives,
            )
            if is_wandb():
                wandb.log(log_info)
            else:
                print(log_info)

            mean_success = log_info.get("bc/val/success/mean", 0.0)
            mean_reward = log_info.get("bc/val/reward/mean", -np.inf)
            mean_length = log_info.get("bc/val/length/mean", np.inf)
            print(
                f"Evaluation results: mean_success={mean_success:.2f}, mean_reward={mean_reward:.2f}, mean_length={mean_length:.2f}"
            )

            save_best_model(
                best,
                {
                    "mean_success": mean_success,
                    "mean_reward": mean_reward,
                    "mean_length": mean_length,
                },
                policy,
                best_model_save_path,
                "bc",
                verbose=1,
            )

    if progress_bar:
        batch_iterator.close()

    return policy
