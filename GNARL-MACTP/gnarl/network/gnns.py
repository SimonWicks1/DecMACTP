# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# # Adapted from https://github.com/HekpoMaH/DEAR/blob/a3187b86c1ef9eb2a482ae020ed390db9a822cf2/layers/gnns.py
# Modifications: Removed GIN and GAT, removed hidden embeddings, added class dict - June 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng
from torch_geometric.utils import to_dense_batch, to_dense_adj, unbatch_edge_index
from typing import Callable, Optional


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, epsilon=1e-5):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.offset = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        inv = self.scale * torch.rsqrt(var + self.epsilon)
        x = x - mean
        x = inv * x + self.offset
        return x


class MPNN(nng.MessagePassing):

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        aggr="max",
        bias=False,
        flow="source_to_target",
        use_ln_MLP=True,
        use_ln_GNN=True,
        num_layers=3,
        weights_init: Optional[Callable] = None,
        **unused_kwargs,
    ):
        super(MPNN, self).__init__(aggr=aggr, flow=flow)
        self.use_ln = use_ln_MLP

        msg_modules = []
        for _ in range(num_layers - 1):
            msg_modules.extend(
                [
                    LayerNorm(out_channels) if use_ln_MLP else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels, bias=bias),
                ]
            )
        self.M = nn.Sequential(*msg_modules)
        self.i_map = nn.Linear(in_channels, out_channels, bias=bias)
        self.j_map = nn.Linear(in_channels, out_channels, bias=bias)

        self.edge_map = nn.Linear(edge_dim, out_channels, bias=bias)
        self.graph_map = nn.Linear(edge_dim, out_channels, bias=bias)

        edge_modules = [nn.Linear(2 * in_channels + edge_dim, out_channels, bias=bias)]
        for _ in range(num_layers - 1):
            edge_modules.extend(
                [
                    LayerNorm(out_channels) if use_ln_MLP else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels, bias=bias),
                ]  # type: ignore
            )
        self.M_e = nn.Sequential(*edge_modules)

        self.U1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.U2 = nn.Linear(out_channels, out_channels, bias=bias)

        self.out_channels = out_channels
        self.ln = LayerNorm(out_channels) if use_ln_GNN else nn.Identity()

        if weights_init is not None:
            self.apply(weights_init)

    def forward(self, node_fts, edge_attr, graph_fts, edge_index, batch, **kwargs):

        graph_fts_padded = torch.zeros(
            node_fts.shape[0], graph_fts.shape[1], device=edge_index.device
        )
        graph_fts_padded[: batch.shape[0]] = graph_fts[batch]

        node_embeddings = self.propagate(
            edge_index,
            x=node_fts,
            edge_attr=edge_attr,
            graph_fts=graph_fts_padded,
        )
        edge_embeddings = self.edge_updater(
            edge_index,
            x=node_fts,
            edge_attr=edge_attr,
        )
        if not self.training:
            node_embeddings = torch.clamp(node_embeddings, -1e9, 1e9)
        return node_embeddings, edge_embeddings

    def message(self, x_i, x_j, edge_attr, graph_fts_i):
        mapped = self.i_map(x_i)
        mapped += self.j_map(x_j)
        mapped += self.edge_map(edge_attr)
        mapped += self.graph_map(graph_fts_i)
        return self.M(mapped)

    def edge_update(self, x_i, x_j, edge_attr):
        m_e = self.M_e(torch.cat((x_i, x_j, edge_attr), dim=1))
        return m_e

    def update(self, aggr_out, x):
        h_1 = self.U1(x)
        h_2 = self.U2(aggr_out)

        ret = self.ln(h_1 + h_2)

        return ret


class TripletMPNN(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        aggr="max",
        bias=False,
        num_layers=2,
        use_ln_MLP=True,
        use_ln_GNN=True,
        triplet_fts=8,
        weights_init: Optional[Callable] = None,
        **unused_kwargs,
    ):
        super(TripletMPNN, self).__init__()
        assert aggr == "max", "Max only mode, soz!"

        self.out_channels = out_channels
        graph_dim = edge_dim
        self._init_triplet_networks(in_channels, edge_dim, graph_dim, triplet_fts, bias)

        mpnn_lst = []
        for in_dim in [in_channels, in_channels, edge_dim, graph_dim]:
            modules = [nn.Linear(in_dim, out_channels, bias=bias)]
            mpnn_lst.append(nn.Sequential(*modules))

        modules = []
        for _ in range(num_layers):
            modules.extend(
                [
                    LayerNorm(out_channels) if use_ln_MLP else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels, bias=bias),
                ]
            )
        mpnn_lst.append(nn.Sequential(*modules))
        self.M = nn.ModuleList(mpnn_lst)

        self.U1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.U2 = nn.Linear(out_channels, out_channels, bias=bias)
        self.U3 = nn.Linear(triplet_fts, out_channels, bias=bias)

        self.ln = LayerNorm(out_channels) if use_ln_GNN else nn.Identity()

        if weights_init is not None:
            self.apply(weights_init)

    def _init_triplet_networks(self, node_dim, edge_dim, graph_dim, triplet_fts, bias):
        """Initialize the triplet networks."""
        tri_lst = []
        for in_dim in [
            node_dim,
            node_dim,
            node_dim,
            edge_dim,
            edge_dim,
            edge_dim,
            graph_dim,
        ]:
            modules = [nn.Linear(in_dim, triplet_fts, bias=bias)]
            tri_lst.append(nn.Sequential(*modules))
        self.M_tri = nn.ModuleList(tri_lst)

    def MPNN_forward_dense(self, z_dense, e_dense, graph_fts, mask, msgs_mask):
        msg_1 = self.M[0](z_dense)  # B x N x H
        msg_2 = self.M[1](z_dense)  # B x N x H
        msg_e = self.M[2](e_dense)  # B x N x N x H
        msg_g = self.M[3](graph_fts)  # B x H
        msg_1[~mask] = 0
        msg_2[~mask] = 0
        msg_e[~msgs_mask] = 0
        msgs = (
            msg_1[:, None, :, :]
            + msg_2[:, :, None, :]
            + msg_e
            + msg_g[:, None, None, :]
        )  # B x N x N x H
        assert not torch.isnan(msgs).any()
        msgs = self.M[-1](msgs)
        assert not torch.isnan(msgs).any(), breakpoint()
        msgs[~msgs_mask] = -1e9
        msgs = msgs.max(1).values
        assert not torch.isnan(msgs).any()
        h_1 = self.U1(z_dense)
        assert not torch.isnan(h_1).any()
        h_2 = self.U2(msgs)
        assert not torch.isnan(h_2).any()
        ret = h_1 + h_2
        assert not torch.isnan(ret).any()
        return ret, msgs

    def triplet_forward_dense(self, z_dense, e_dense, graph_fts, mask, msgs_mask):
        assert not torch.isnan(z_dense).any()
        tri_1 = self.M_tri[0](z_dense)
        tri_2 = self.M_tri[1](z_dense)
        tri_3 = self.M_tri[2](z_dense)
        tri_e_1 = self.M_tri[3](e_dense)
        tri_e_2 = self.M_tri[4](e_dense)
        tri_e_3 = self.M_tri[5](e_dense)
        tri_g = self.M_tri[6](graph_fts)
        tri_1[~mask] = 0
        tri_2[~mask] = 0
        tri_3[~mask] = 0

        tri_msgs = (
            tri_1[:, :, None, None, :]  #   (B, N, 1, 1, H)
            + tri_2[:, None, :, None, :]  # + (B, 1, N, 1, H)
            + tri_3[:, None, None, :, :]  # + (B, 1, 1, N, H)
            + tri_e_1[:, :, :, None, :]  # + (B, N, N, 1, H)
            + tri_e_2[:, :, None, :, :]  # + (B, N, 1, N, H)
            + tri_e_3[:, None, :, :, :]  # + (B, 1, N, N, H)
            + tri_g[:, None, None, None, :]  # + (B, 1, 1, 1, H)
        )  # = (B, N, N, N, H)
        assert not torch.isnan(tri_msgs).any()
        msk_tri = (
            mask[:, None, None, :] | mask[:, None, :, None] | mask[:, :, None, None]
        )
        tri_msgs[~msk_tri] = -1e9
        tri_msgs_pooled = tri_msgs.max(1).values
        tri_msgs = F.relu(self.U3(tri_msgs_pooled))  # B x N x N x H

        ret, msgs = self.MPNN_forward_dense(
            z_dense, e_dense, graph_fts, mask, msgs_mask
        )
        return ret, msgs, tri_msgs

    def forward(
        self, node_fts, edge_attr, graph_fts, edge_index, batch, *args, **kwargs
    ):
        z = node_fts
        z_dense, mask = to_dense_batch(z, batch=batch)  # BxNxH
        e_dense = to_dense_adj(edge_index, batch=batch, edge_attr=edge_attr)  # BxNxNxH
        adj_mat = to_dense_adj(edge_index, batch=batch).bool()
        ret, msgs, tri_msgs = self.triplet_forward_dense(
            z_dense,
            e_dense,
            graph_fts,
            mask,
            adj_mat,
        )
        ret = self.ln(ret)
        ret = ret[mask]

        ebatch = batch[edge_index[0]]
        local_edge_index = unbatch_edge_index(edge_index, batch)
        e12 = torch.cat(local_edge_index, dim=-1)

        assert (ret != -1e9).all(), breakpoint()
        return ret, tri_msgs[ebatch, e12[0], e12[1]]


_NETWORKS = {"MPNN": MPNN, "TripletMPNN": TripletMPNN}


def get_network_class(network_name: str):
    if network_name not in _NETWORKS:
        raise ValueError(
            f"Unknown network {network_name}, available networks: {_NETWORKS.keys()}"
        )
    return _NETWORKS[network_name]
