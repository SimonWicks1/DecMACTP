from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from typing import Any
import numpy as np
from gnarl.util.evaluation import (
    process_evaluation_output,
    save_best_model,
    evaluate_policy,
)


class MaskableEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent. Supports invalid action masking.

    Args:
        eval_env (list[VecEnv]): The environments used for evaluation.
        n_eval_episodes (list[int]): The number of episodes to test the agent in each eval env.
        eval_freq (int): The frequency (in steps) at which to evaluate the agent.
        callback (BaseCallback): Callback to trigger after each evaluation.
        best_model_save_path (str): Path to a folder where the best model
            according to performance on the eval env will be saved.
        log_path (str): Path to a folder where the evaluations (``evaluations.npz``)
            will be saved. It will be updated at each evaluation.
        log_prefix (str): Prefix to add to the logger keys.
        callback_on_new_best (BaseCallback) : Callback to trigger
            when there is a new best model according to the ``mean_success``, ``mean_reward``
            or ``mean_length`` (if any of these is not None).
        deterministic (bool): Whether the evaluation should use deterministic or stochastic actions.
        render (bool): Whether to render or not the environment during evaluation.
        verbose (bool): Whether to print or not the evaluation results.
        warn (bool): Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
            wrapped with a Monitor wrapper).
        use_masking (bool): Whether to use invalid action masks during evaluation.
    """

    def __init__(self, *args, use_masking: bool = True, log_prefix="", **kwargs):
        self.eval_envs = kwargs.pop("eval_env")
        kwargs["eval_env"] = self.eval_envs[0]
        super().__init__(*args, **kwargs)
        self.use_masking = use_masking
        self.pfx = log_prefix + "/val" if log_prefix else "val"
        self.best = {
            "mean_success": 0.0,
            "mean_reward": -np.inf,
            "mean_length": np.inf,
        }

    def _on_step(self) -> bool:
        continue_training = True

        def flat(lst: list[list[Any]]) -> list[Any]:
            """Flatten a list of lists."""
            return [item for sublist in lst for item in sublist]

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            (
                episode_rewards,
                episode_lengths,
                episode_success,
                num_nodes,
                objective_values,
                expert_objectives,
                _,
            ) = evaluate_policy(
                self.model.policy,
                self.eval_envs,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                warn=self.warn,
                callback=self._log_success_callback,
                use_masking=self.use_masking,
            )

            # Flatten the lists
            flt_episode_rewards = flat(episode_rewards)
            flt_episode_lengths = flat(episode_lengths)

            if self.log_path is not None:
                assert isinstance(flt_episode_rewards, list)
                assert isinstance(flt_episode_lengths, list)
                
                # 1. Update internal lists
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(flt_episode_rewards)
                self.evaluations_length.append(flt_episode_lengths)

                # 2. Prepare dictionary for saving
                save_dict = {}

                # Handle successes
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    save_dict['successes'] = self.evaluations_successes

                # Add main metrics
                save_dict['timesteps'] = self.evaluations_timesteps
                save_dict['results'] = self.evaluations_results
                save_dict['ep_lengths'] = self.evaluations_length

                # 3. Explicit Conversion to Object Arrays with Debugging
                # We convert BEFORE np.savez to pinpoint if the error is in data structure
                converted_dict = {}
                
                try:
                    for key, val in save_dict.items():
                        # Explicitly use dtype=object to handle variable length sub-lists
                        converted_dict[key] = np.array(val, dtype=object)
                    
                    # 4. Save
                    np.savez(
                        self.log_path,
                        **converted_dict
                    )

                except Exception as e:
                    # === DEBUGGING BLOCK ===
                    print("\n" + "!"*40)
                    print("CRITICAL ERROR DURING EVALUATION SAVE")
                    print(f"Error: {e}")
                    print("-" * 30)
                    
                    # Analyze the 'results' list which causes the common error
                    results = self.evaluations_results
                    print(f"evaluations_results len: {len(results)}")
                    print("Detailed inspection of 'results' elements:")
                    
                    for i, res in enumerate(results):
                        length = len(res) if hasattr(res, '__len__') else 'Scalar'
                        type_name = type(res).__name__
                        # Print details for the first few and the last one (where mismatch usually happens)
                        if i < 3 or i == len(results) - 1:
                            print(f"  Idx {i}: Type={type_name}, Len={length}, Sample={res[:3] if hasattr(res, '__getitem__') else res}")
                    
                    if len(results) > 1:
                        len_prev = len(results[-2])
                        len_curr = len(results[-1])
                        if len_prev != len_curr:
                            print(f"\n[MISMATCH DETECTED] Idx {-2} len={len_prev} vs Idx {-1} len={len_curr}")
                            
                    print("!"*40 + "\n")
                    # Re-raise the error to stop the process as requested
                    raise e
                
            mean_reward, std_reward = np.mean(flt_episode_rewards), np.std(
                flt_episode_rewards
            )
            mean_ep_length, std_ep_length = np.mean(flt_episode_lengths), np.std(
                flt_episode_lengths
            )
            self.last_mean_reward = float(mean_reward)

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            log_info = process_evaluation_output(
                self.pfx,
                episode_rewards,
                episode_lengths,
                episode_success,
                num_nodes,
                objective_values,
                expert_objectives,
            )
            for key, value in log_info.items():
                self.logger.record(key, value)

            if len(self._is_success_buffer) > 0:
                mean_success = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * mean_success:.2f}%")
            else:
                mean_success = 0

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if save_best_model(
                self.best,
                {
                    "mean_success": mean_success,
                    "mean_reward": mean_reward,
                    "mean_length": mean_ep_length,
                },  # type: ignore[arg-type]
                self.model,
                self.best_model_save_path,
                "ppo",
                self.verbose,
            ):
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
