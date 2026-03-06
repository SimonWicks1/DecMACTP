import os
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List

# Stable Baselines3 & Contrib Imports
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# PyTorch Geometric Imports
from torch_geometric.nn import GINEConv, global_mean_pool

# GNARL Imports
from gnarl.envs.mactp_env2 import MultiTravelerCTPEnv2
from gnarl.envs.generate.graph_generator import RandomSetGraphGenerator
from gnarl.envs.generate.data import GraphProblemDataset

# =============================================================================
# 1. GPPO Feature Extractor (GNN for SB3)
# =============================================================================
class GPPOFeatureExtractor(BaseFeaturesExtractor):
    """
    Processes graph observations from MultiTravelerCTPEnv2 using GINEConv.
    Constructs PyG graphs on-the-fly from the dictionary observations.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        self.hidden_dim = features_dim
        # Node features: [IsGoal, Visited, OptCost, PessCost, SelfPos, TeamDensity, Terminated]
        self.node_in_dim = observation_space["node_features"].shape[1] 
        self.edge_in_dim = 2  # [Status, Weight]
        
        # 1. Node Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # 2. GNN Layers (GINEConv)
        self.gnn1_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.gnn1 = GINEConv(self.gnn1_mlp, edge_dim=self.edge_in_dim)
        
        self.gnn2_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.gnn2 = GINEConv(self.gnn2_mlp, edge_dim=self.edge_in_dim)
        
        # 3. Projector
        self.projector = nn.Linear(self.hidden_dim, features_dim)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        node_feats = observations["node_features"] 
        edge_status = observations["edge_status"]    
        edge_weights = observations["edge_weights"] 
        
        batch_size, num_nodes, _ = node_feats.shape
        device = node_feats.device  # Ensure we stay on the same device as inputs
        
        # --- Efficient Batching Strategy ---
        # Flatten node features
        x = node_feats.reshape(-1, self.node_in_dim)
        
        # Find valid edges (Status > 0)
        valid_edges_mask = edge_status > 0
        indices = valid_edges_mask.nonzero(as_tuple=False) 
        
        if indices.shape[0] == 0:
            # Handle edge case with no valid edges (rare but possible)
            # Create dummy self-loops or empty graph
            edge_index = th.empty((2, 0), dtype=th.long, device=device)
            edge_attr = th.empty((0, self.edge_in_dim), dtype=th.float, device=device)
        else:
            b_idx, row, col = indices[:, 0], indices[:, 1], indices[:, 2]
            
            # Calculate global indices for disjoint batch graph
            row_global = b_idx * num_nodes + row
            col_global = b_idx * num_nodes + col
            edge_index = th.stack([row_global, col_global], dim=0)
            
            # Extract Attributes
            status_vals = edge_status[b_idx, row, col].float().unsqueeze(-1)
            weight_vals = edge_weights[b_idx, row, col].float().unsqueeze(-1)
            edge_attr = th.cat([status_vals, weight_vals], dim=-1)
        
        # --- GNN Pass ---
        h = self.node_encoder(x)
        h = self.gnn1(h, edge_index, edge_attr=edge_attr)
        h = th.relu(h)
        h = self.gnn2(h, edge_index, edge_attr=edge_attr)
        h = th.relu(h)
        
        # --- Pooling ---
        batch_vec = th.arange(batch_size, device=device).repeat_interleave(num_nodes)
        graph_embedding = global_mean_pool(h, batch_vec)
        
        return self.projector(graph_embedding)

# =============================================================================
# 2. Multi-Agent Wrapper (Parameter Sharing & Masking)
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
        """
        Intercepts calls from MaskablePPO.
        Specifically handles 'action_masks' to return a list of masks [Mask_Agent1, Mask_Agent2, ...].
        """
        if method_name == "action_masks":
            masks = self.env.action_masks() # Returns Dict {agent: mask}
            return [masks[a] for a in self.agents]
        
        return getattr(self.env, method_name)(*method_args, **method_kwargs)

    def _stack_obs(self, obs_dict):
        keys = obs_dict[self.agents[0]].keys()
        stacked = {}
        for k in keys:
            stacked[k] = np.stack([obs_dict[a][k] for a in self.agents])
        return stacked

    def close(self):
        self.env.close()
    
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def get_attr(self, attr_name, indices=None): return [getattr(self.env, attr_name)] * self.num_envs
    def set_attr(self, attr_name, value, indices=None): setattr(self.env, attr_name, value)
    def seed(self, seed=None): return [self.env.reset(seed=seed)]

# =============================================================================
# 3. Main Training Execution
# =============================================================================
def make_env():
    GRAPH_DIR = "./data/train"
    os.makedirs(GRAPH_DIR, exist_ok=True)
    SEED = 42
    NUM_NODES = 16
    NUM_SAMPLES = 100
    NUM_AGENTS = 3
    
    ds = [
        GraphProblemDataset(
            root=GRAPH_DIR,
            split="train",
            algorithm="ctp",
            num_nodes=NUM_NODES,
            num_samples=NUM_SAMPLES,
            seed=SEED,
            graph_generator="er", 
            graph_generator_kwargs={"p_range": [0.1, 0.2]},
            num_starts=NUM_AGENTS,
            num_goals=3 
        )
    ]
    
    env = MultiTravelerCTPEnv2(
        max_nodes=NUM_NODES,
        graph_generator=RandomSetGraphGenerator(datasets=ds, seed=SEED),
        num_agents=NUM_AGENTS,
        num_goals=3
    )
    return env

def main():
    log_dir = "./mactp_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Force CPU to avoid CUDA kernel mismatches
    device = "cpu"
    print(f"Forcing training on device: {device}")

    print("Initializing Environment...")
    vec_env = MultiAgentVecEnv(make_env)
    
    print(f"Agents: {vec_env.num_envs}")

    policy_kwargs = dict(
        features_extractor_class=GPPOFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]) 
    )

    print("Initializing GPPO (MaskablePPO) Model...")
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device=device  # Explicitly set device to CPU
    )

    print(f"Starting Training... Logging to {log_dir}")
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./logs/', 
        name_prefix='mactp_gppo'
    )

    try:
        model.learn(
            total_timesteps=200_000, 
            callback=checkpoint_callback, 
            tb_log_name="GPPO_Run"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted manually.")

    print("Saving final model...")
    model.save("final_mactp_gppo_model")
    vec_env.close()
    print("Done!")

if __name__ == "__main__":
    set_random_seed(42)
    main()