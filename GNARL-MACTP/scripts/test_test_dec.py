# test_mactp_gppo_render.py
"""
Evaluate a trained MaskablePPO (GPPO) model on MultiTravelerCTPEnv2, WITH rendering.

Example:
    python test_mactp_gppo_render.py --model-path final_mactp_gppo_model.zip --episodes 10 --render

Notes:
- Rendering support depends on MultiTravelerCTPEnv2's render implementation.
- If the environment requires render_mode at construction (Gymnasium style), this script tries to pass it.
"""

import os
import argparse
import time
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Optional

# Stable Baselines3 & Contrib Imports
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# PyTorch Geometric Imports
from torch_geometric.nn import GINEConv, global_mean_pool

# GNARL Imports
from gnarl.envs.mactp_env2 import MultiTravelerCTPEnv2
from gnarl.envs.generate.graph_generator import RandomSetGraphGenerator
from gnarl.envs.generate.data import GraphProblemDataset


# =============================================================================
# 1. GPPO Feature Extractor (GNN for SB3) - must match training
# =============================================================================
class GPPOFeatureExtractor(BaseFeaturesExtractor):
    """
    Processes graph observations from MultiTravelerCTPEnv2 using GINEConv.
    Constructs PyG graphs on-the-fly from the dictionary observations.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.hidden_dim = features_dim
        self.node_in_dim = observation_space["node_features"].shape[1]
        self.edge_in_dim = 2  # [Status, Weight]

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.gnn1_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gnn1 = GINEConv(self.gnn1_mlp, edge_dim=self.edge_in_dim)

        self.gnn2_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gnn2 = GINEConv(self.gnn2_mlp, edge_dim=self.edge_in_dim)

        self.projector = nn.Linear(self.hidden_dim, features_dim)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        node_feats = observations["node_features"]
        edge_status = observations["edge_status"]
        edge_weights = observations["edge_weights"]

        batch_size, num_nodes, _ = node_feats.shape
        device = node_feats.device

        x = node_feats.reshape(-1, self.node_in_dim)

        valid_edges_mask = edge_status > 0
        indices = valid_edges_mask.nonzero(as_tuple=False)

        if indices.shape[0] == 0:
            edge_index = th.empty((2, 0), dtype=th.long, device=device)
            edge_attr = th.empty((0, self.edge_in_dim), dtype=th.float, device=device)
        else:
            b_idx, row, col = indices[:, 0], indices[:, 1], indices[:, 2]
            row_global = b_idx * num_nodes + row
            col_global = b_idx * num_nodes + col
            edge_index = th.stack([row_global, col_global], dim=0)

            status_vals = edge_status[b_idx, row, col].float().unsqueeze(-1)
            weight_vals = edge_weights[b_idx, row, col].float().unsqueeze(-1)
            edge_attr = th.cat([status_vals, weight_vals], dim=-1)

        h = self.node_encoder(x)
        h = self.gnn1(h, edge_index, edge_attr=edge_attr)
        h = th.relu(h)
        h = self.gnn2(h, edge_index, edge_attr=edge_attr)
        h = th.relu(h)

        batch_vec = th.arange(batch_size, device=device).repeat_interleave(num_nodes)
        graph_embedding = global_mean_pool(h, batch_vec)

        return self.projector(graph_embedding)


# =============================================================================
# 2. Multi-Agent Wrapper (Parameter Sharing & Masking) - plus render support
# =============================================================================
class MultiAgentVecEnv(VecEnv):
    """
    Wraps the Multi-Agent environment to behave like a Vectorized Environment.
    Enables Parameter Sharing and exposes 'action_masks' for MaskablePPO.
    """

    def __init__(self, env_fn):
        self.env = env_fn()
        self.num_agents = self.env.num_agents
        self.agents = self.env.agents

        observation_space = self.env.observation_space[self.agents[0]]
        action_space = self.env.action_space[self.agents[0]]

        super().__init__(self.num_agents, observation_space, action_space)
        self.actions = None

    def reset(self):
        obs_dict, _ = self.env.reset()
        return self._stack_obs(obs_dict)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        action_dict = {agent: self.actions[i] for i, agent in enumerate(self.agents)}
        obs_dict, rew_dict, done_dict, trunc_dict, info_dict = self.env.step(action_dict)

        obs = self._stack_obs(obs_dict)
        rews = np.array([rew_dict[a] for a in self.agents], dtype=np.float32)
        dones = np.array([done_dict[a] or trunc_dict[a] for a in self.agents], dtype=bool)
        infos = [info_dict[a] for a in self.agents]

        if np.all(dones):
            new_obs_dict, _ = self.env.reset()
            for i, agent in enumerate(self.agents):
                infos[i]["terminal_observation"] = obs_dict[agent]
            obs = self._stack_obs(new_obs_dict)

        return obs, rews, dones, infos

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            masks = self.env.action_masks()  # Dict {agent: mask}
            return [masks[a] for a in self.agents]
        return getattr(self.env, method_name)(*method_args, **method_kwargs)

    def render(self, mode: str = "human"):
        """
        Forward render to the underlying env.
        Gymnasium often uses env.render() and expects 'render_mode' set at construction;
        this still works if the underlying env implements render directly.
        """
        try:
            return self.env.render()
        except TypeError:
            # Some envs accept a mode argument
            return self.env.render(mode=mode)

    def _stack_obs(self, obs_dict):
        keys = obs_dict[self.agents[0]].keys()
        stacked = {}
        for k in keys:
            stacked[k] = np.stack([obs_dict[a][k] for a in self.agents])
        return stacked

    def close(self):
        self.env.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def seed(self, seed=None):
        return [self.env.reset(seed=seed)]


# =============================================================================
# 3. Test Env Factory (tries to enable rendering via render_mode if supported)
# =============================================================================
def make_test_env(
    graph_dir: str,
    seed: int,
    num_nodes: int,
    num_samples: int,
    num_agents: int,
    num_goals: int,
    split: str,
    render: bool,
    render_mode: str,
):
    os.makedirs(graph_dir, exist_ok=True)

    ds = [
        GraphProblemDataset(
            root=graph_dir,
            split=split,
            algorithm="ctp",
            num_nodes=num_nodes,
            num_samples=num_samples,
            seed=seed,
            graph_generator="er",
            graph_generator_kwargs={"p_range": [0.1, 0.2]},
            num_starts=num_agents,
            num_goals=num_goals,
        )
    ]

    common_kwargs = dict(
        max_nodes=num_nodes,
        graph_generator=RandomSetGraphGenerator(datasets=ds, seed=seed),
        num_agents=num_agents,
        num_goals=num_goals,
    )

    # Some Gymnasium envs require render_mode at init; others ignore it or don't accept it.
    if render:
        try:
            env = MultiTravelerCTPEnv2(**common_kwargs, render_mode=render_mode)
        except TypeError:
            env = MultiTravelerCTPEnv2(**common_kwargs)
    else:
        env = MultiTravelerCTPEnv2(**common_kwargs)

    return env


# =============================================================================
# 4. Evaluation Loop (renders each step if requested)
# =============================================================================
def evaluate(
    model: MaskablePPO,
    vec_env: MultiAgentVecEnv,
    episodes: int,
    deterministic: bool,
    render: bool,
    render_sleep: float,
):
    episode_returns_total = []
    episode_returns_per_agent = []
    episode_lengths = []

    for ep in range(episodes):
        obs = vec_env.reset()

        ep_ret_agents = np.zeros(vec_env.num_envs, dtype=np.float64)
        ep_len = 0

        while True:
            if render:
                vec_env.render()
                if render_sleep > 0:
                    time.sleep(render_sleep)

            masks = vec_env.env_method("action_masks")
            actions, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
            obs, rews, dones, infos = vec_env.step(actions)

            ep_ret_agents += rews
            ep_len += 1

            if np.all(dones):
                # One final render at terminal, if desired
                if render:
                    vec_env.render()
                    if render_sleep > 0:
                        time.sleep(render_sleep)
                break

        episode_returns_per_agent.append(ep_ret_agents.copy())
        episode_returns_total.append(float(ep_ret_agents.sum()))
        episode_lengths.append(ep_len)

        print(
            f"[Episode {ep+1:>4}/{episodes}] "
            f"Return(sum agents)={episode_returns_total[-1]:.3f} | "
            f"Return(per-agent)={np.round(ep_ret_agents, 3)} | "
            f"Length={ep_len}"
        )

    episode_returns_total = np.array(episode_returns_total, dtype=np.float64)
    episode_lengths = np.array(episode_lengths, dtype=np.int64)
    per_agent = np.vstack(episode_returns_per_agent)

    results = {
        "episodes": episodes,
        "return_total_mean": float(episode_returns_total.mean()),
        "return_total_std": float(episode_returns_total.std(ddof=1) if episodes > 1 else 0.0),
        "length_mean": float(episode_lengths.mean()),
        "length_std": float(episode_lengths.std(ddof=1) if episodes > 1 else 0.0),
        "return_per_agent_mean": per_agent.mean(axis=0),
        "return_per_agent_std": per_agent.std(axis=0, ddof=1) if episodes > 1 else np.zeros(vec_env.num_envs),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="final_mactp_gppo_model.zip",
                        help="Path to the saved SB3 model (.zip).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda (if compatible).")

    # Rendering options
    parser.add_argument("--render", action="store_true", default=True, help="Render the environment during evaluation.")
    parser.add_argument("--render-mode", type=str, default="human",
                        help='Render mode passed at env init if supported (e.g., "human", "rgb_array").')
    parser.add_argument("--render-sleep", type=float, default=0.03,
                        help="Sleep (seconds) after each render call to control playback speed.")

    # Env params (match training defaults unless you explicitly want different test settings)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-goals", type=int, default=3)
    parser.add_argument("--graph-dir", type=str, default="./data/test")
    parser.add_argument("--split", type=str, default="test", help='Dataset split string (e.g., "test").')

    args = parser.parse_args()

    set_random_seed(args.seed)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found at: {args.model_path}\n"
            f"Provide --model-path pointing to your saved .zip (e.g., final_mactp_gppo_model.zip)."
        )

    print(f"Creating test environment (split={args.split}, render={args.render})...")

    def _env_fn():
        return make_test_env(
            graph_dir=args.graph_dir,
            seed=args.seed,
            num_nodes=args.num_nodes,
            num_samples=args.num_samples,
            num_agents=args.num_agents,
            num_goals=args.num_goals,
            split=args.split,
            render=args.render,
            render_mode=args.render_mode,
        )

    vec_env = MultiAgentVecEnv(_env_fn)

    print(f"Loading model from: {args.model_path} on device={args.device}")
    model = MaskablePPO.load(args.model_path, env=vec_env, device=args.device)

    print("Running evaluation...")
    results = evaluate(
        model=model,
        vec_env=vec_env,
        episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render,
        render_sleep=args.render_sleep,
    )

    print("\n==================== EVALUATION SUMMARY ====================")
    print(f"Episodes: {results['episodes']}")
    print(f"Return (sum agents) mean ± std: {results['return_total_mean']:.4f} ± {results['return_total_std']:.4f}")
    print(f"Episode length mean ± std:      {results['length_mean']:.2f} ± {results['length_std']:.2f}")

    per_agent_mean = results["return_per_agent_mean"]
    per_agent_std = results["return_per_agent_std"]
    for i in range(len(per_agent_mean)):
        print(f"Agent {i} return mean ± std:     {per_agent_mean[i]:.4f} ± {per_agent_std[i]:.4f}")
    print("============================================================\n")

    vec_env.close()


if __name__ == "__main__":
    main()
