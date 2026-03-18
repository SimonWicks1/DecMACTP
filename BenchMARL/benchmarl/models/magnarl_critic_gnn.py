from __future__ import annotations

from dataclasses import dataclass, MISSING

import torch
from torch import nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.magnarl_shared_trunk import get_or_create_shared_trunk


class MagnarlCriticGNN(Model):
    """Centralized team-value critic using the v2 idealized coordination trunk.
    Trunk is evaluated under no_grad because actor already trains the shared trunk.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        num_gnn_layers: int = 2,
        status_emb_dim: int = 8,
        gnn_hidden_dim: int = 64,
        pooling_type: str = "mean",
        comm_num_heads: int = 4,
        comm_top_k: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._agent_group = kwargs.get("agent_group", "agents")
        self._model_index = int(kwargs.get("model_index", 0))

        self.trunk = get_or_create_shared_trunk(
            agent_group=self._agent_group,
            model_index=self._model_index,
            device=self.device,
            embed_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            status_emb_dim=status_emb_dim,
            pooling_type=pooling_type,
            comm_num_heads=comm_num_heads,
            comm_top_k=comm_top_k,
        )
        self.value_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, 1),
        ).to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        group = self._agent_group
        node_features = tensordict.get((group, "node_features")).to(self.device)
        edge_status = tensordict.get((group, "edge_status")).to(self.device)
        edge_weights = tensordict.get((group, "edge_weights")).to(self.device)
        phase = tensordict.get((group, "phase")).to(self.device)

        batch_shape = tensordict.batch_size
        flat_batch = 1
        for s in batch_shape:
            flat_batch *= int(s)

        A = node_features.shape[-3]
        N = node_features.shape[-2]

        node_features = node_features.reshape(flat_batch, A, N, -1)
        edge_status = edge_status.reshape(flat_batch, A, N, N)
        edge_weights = edge_weights.reshape(flat_batch, A, N, N)
        phase = phase.reshape(flat_batch, A, -1)

        with torch.no_grad():
            _, hG, hComm, _, _ = self.trunk(node_features, edge_status, edge_weights, phase)
        hG = hG.detach()
        hComm = hComm.detach()

        team_g = hG.mean(dim=1)
        team_c = hComm.mean(dim=1)
        value = self.value_head(torch.cat([team_g, team_c], dim=-1))
        tensordict.set(self.out_key, value.reshape(*batch_shape, 1))
        return tensordict


@dataclass
class MagnarlCriticGNNConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    num_gnn_layers: int = 2
    status_emb_dim: int = 8
    gnn_hidden_dim: int = 64
    pooling_type: str = "mean"
    comm_num_heads: int = 4
    comm_top_k: int = 4

    @staticmethod
    def associated_class():
        return MagnarlCriticGNN