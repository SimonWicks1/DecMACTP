#!/usr/bin/env python3
import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tensordict import TensorDict

# Environment imports
from benchmarl.environments.mactp.common import MactpTask

def masked_random_actions(action_mask: torch.Tensor) -> torch.Tensor:
    probs = action_mask.float()
    if (probs.sum(dim=-1) == 0).any():
        bad = (probs.sum(dim=-1) == 0).nonzero(as_tuple=False)
        raise RuntimeError(f"CRITICAL: some (env,agent) have all actions masked out: {bad.tolist()}")

    E, A, K = probs.shape
    flat = probs.reshape(E * A, K)
    acts = torch.multinomial(flat, 1).squeeze(-1).reshape(E, A).to(torch.int64)
    return acts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4
    num_episodes_to_run = 5 
    output_dir = Path("test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    task = MactpTask.DEFAULT.get_task()
    env_fn = task.get_env_fun(num_envs=num_envs, continuous_actions=False, device=device)
    env = env_fn()

    # Metric tracking
    all_rewards, all_costs, all_penalties = [], [], []
    all_lengths, all_oracles, all_cr = [], [], []
    successes = 0
    failed_indices = []
    failure_coverage = []
    total_episodes = 0

    num_agents = env.num_agents

    for ep_batch in range(num_episodes_to_run):
        td = env.reset()
        E = env.num_envs
        
        # Calculate num_goals after reset
        num_goals = int(env.destination_mask[0].sum().item())
        if num_goals == 0: num_goals = 1 

        batch_oracle_costs = []
        for i in range(E):
            gd = env.graph_data[i]
            start_node = gd.s[0].item()
            goal_nodes = gd.g.tolist()
            gt_dist = env.gt_distances[i]
            oracle_val = sum([gt_dist[start_node, g].item() for g in goal_nodes])
            batch_oracle_costs.append(max(float(oracle_val), 1e-6))

        env_finished = torch.zeros(E, dtype=torch.bool, device=device)
        env_rewards = torch.zeros(E, device=device)
        frames = [[] for _ in range(E)]

        for t in range(env.num_phases):
            action_mask = td["agents", "action_mask"]
            actions = masked_random_actions(action_mask)
            
            td.set(("agents", "action"), actions)
            td = env.step(td)
            next_td = td["next"]

            active_mask = ~env_finished
            if active_mask.any():
                current_rewards = next_td["agents", "reward"].sum(dim=1).squeeze(-1)
                env_rewards[active_mask] += current_rewards[active_mask]

            if ep_batch == 0:
                for i in range(E):
                    if not env_finished[i]:
                        rgb = env.render(env_idx=i)
                        frames[i].append(Image.fromarray(rgb))

            done = next_td["done"].squeeze(-1)
            env_finished |= done
            
            if env_finished.all():
                for i in range(E):
                    total_episodes += 1
                    is_success = torch.equal(env.destination_mask[i], env.goals_visited_mask[i])
                    final_penalty = float(next_td["penalty"][i].item())
                    
                    ep_reward = float(env_rewards[i].item())
                    ep_cost = -(ep_reward + final_penalty)
                    ep_oracle = batch_oracle_costs[i]
                    
                    all_rewards.append(ep_reward)
                    all_costs.append(ep_cost)
                    all_penalties.append(final_penalty)
                    all_lengths.append(float(t + 1))
                    all_oracles.append(ep_oracle)
                    all_cr.append(ep_cost / ep_oracle)

                    if is_success:
                        successes += 1
                    else:
                        failed_indices.append(total_episodes - 1)
                        cov = float(env.goals_visited_mask[i].sum().item() / num_goals)
                        failure_coverage.append(cov)

                    if ep_batch == 0 and len(frames[i]) > 0:
                        gif_path = output_dir / f"simulation_ep_{total_episodes-1}.gif"
                        frames[i][0].save(gif_path, save_all=True, append_images=frames[i][1:], duration=100, loop=0)
                break
            td = next_td

    # Summary Statistics using NumPy
    def get_stats(data_list):
        if not data_list: return 0.0, 0.0
        arr = np.array(data_list)
        return float(np.mean(arr)), float(np.std(arr))

    avg_reward, std_reward = get_stats(all_rewards)
    avg_cost, std_cost = get_stats(all_costs)
    avg_penalty, std_penalty = get_stats(all_penalties)
    avg_len, std_len = get_stats(all_lengths)
    avg_oracle, std_oracle = get_stats(all_oracles)
    avg_cr, std_cr = get_stats(all_cr)
    avg_fail_cov = np.mean(failure_coverage) if failure_coverage else 0.0

    # Write results.md
    with open(output_dir / "results.md", "w") as f:
        f.write(f"## Environment Configuration\n")
        f.write(f"- Number of Agents: {num_agents} | Number of Goals: {num_goals}\n\n")
        f.write(f"## Summary Metrics\n")
        f.write(f"- Success Rate: {successes}/{total_episodes} = {(successes/total_episodes)*100:.2f}%\n")
        f.write(f"- Average Reward over {total_episodes} episodes: {avg_reward:.4f} ± {std_reward:.4f}\n")
        f.write(f"- Average Cost over {total_episodes} episodes: {avg_cost:.4f} ± {std_cost:.4f}\n")
        f.write(f"- Average Penalty over {total_episodes} episodes: {avg_penalty:.4f} ± {std_penalty:.4f}\n")
        f.write(f"- Average Episode Length: {avg_len:.2f} ± {std_len:.2f}\n")
        f.write(f"- Average Oracle Cost: {avg_oracle:.4f} ± {std_oracle:.4f}\n")
        f.write(f"- Average Competitive Ratio: {avg_cr:.4f} ± {std_cr:.4f}\n")
        f.write(f"- Failed Episodes: {failed_indices}\n")
        f.write(f"- Average Goals Coverage in Failures: {avg_fail_cov:.4f}\n")

    print(f"Testing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()