from gymnasium import spaces

import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from typing import Callable, Any, Optional

from gnarl.network.gnns import get_network_class
from gnarl.util.graph_format import (
    matrix_features_to_batch,
)


class MaGraphFeatureTransformer(BaseFeaturesExtractor):
    """
    [Modified] Supports 'magent' type features and syncs agent masks with VALID graph nodes.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        graph_spec: dict,
        node_max: int,
        embed_dim: int = 64,
        weights_init: Optional[Callable] = None,
    ) -> None:
        self.node_max = node_max
        self.embed_dim = embed_dim
        self.graph_spec = graph_spec

        features_dim = 1  # unused
        super().__init__(observation_space, features_dim=features_dim)

        self.encoders = self._construct_encoders(embed_dim)

        if weights_init is not None:
            for key, encoder in self.encoders.items():
                weights_init(encoder, key=key, graph_spec=self.graph_spec)

    def forward(self, observations) -> Batch:
        """Convert the observations to a graph Batch object."""
        node_features, edge_features, graph_features, adj_matrix = (
            self._encode_features(observations)
        )
        # This converts dense to sparse and DROPS isolated nodes
        batch = matrix_features_to_batch(
            node_features, edge_features, graph_features, adj_matrix
        )
        
        # Edit by Xiao: For the new multi-channel agent node feature
        # # === [CRITICAL FIX] Attach raw agent locations ===
        # # We must apply the EXACT same filtering logic as matrix_features_to_batch
        # # to ensure the agent masks match the sparse batch.x dimensions.
        # if "current_nodes" in observations:
        #     # 1. Get Raw Positions: (Batch, Num_Agents, Max_Nodes)
        #     raw_pos = observations["current_nodes"]
            
        #     # 2. Transpose to (Batch, Max_Nodes, Num_Agents)
        #     # This aligns the nodes dim with the mask we are about to create
        #     raw_pos_transposed = raw_pos.transpose(-1, -2)
            
        #     # 3. Create Valid Node Mask based on Adjacency Matrix
        #     # Logic copied from matrix_features_to_batch:
        #     # has_edge = (adj.sum(dim=0) > 0) | (adj.sum(dim=1) > 0)
        #     if "adj" in observations:
        #         adj = observations["adj"] # (Batch, Max_Nodes, Max_Nodes)
                
        #         # Compute degree (in + out) > 0
        #         row_sum = adj.sum(dim=-1) # (Batch, Max_Nodes)
        #         col_sum = adj.sum(dim=-2) # (Batch, Max_Nodes)
        #         valid_node_mask = (row_sum > 0) | (col_sum > 0) # Boolean (Batch, Max_Nodes)
                
        #         # 4. Apply Mask using Boolean Indexing
        #         # This selects only the valid nodes and flattens the Batch/Node dims,
        #         # automatically preserving the sequence order (Graph 0 nodes, then Graph 1...)
        #         # Result shape: (Total_Valid_Nodes, Num_Agents) e.g., (52, 3)
        #         flat_pos = raw_pos_transposed[valid_node_mask]
                
        #         batch.agent_loc_masks = flat_pos.float()
        #     else:
        #         # Fallback: If no adjacency provided, assume all nodes valid (unlikely in this repo)
        #         batch.agent_loc_masks = raw_pos_transposed.reshape(-1, raw_pos.shape[1]).float()
        # === [关键：提取 Agent Location Masks] ===
        if "current_nodes" in observations:
            # 形状: (Batch, Agents, Nodes, Channels)
            # Channel 0 是位置信息
            raw_pos = observations["current_nodes"][:, :, :, 0]
            
            # 变换为 (Batch, Nodes, Agents) 以匹配 PyG 的节点维度逻辑
            raw_pos_transposed = raw_pos.transpose(-1, -2)
            
            # 使用与 matrix_features_to_batch 相同的过滤逻辑
            if "adj" in observations:
                adj = observations["adj"]
                valid_node_mask = (adj.sum(dim=-1) > 0) | (adj.sum(dim=-2) > 0)
                # 结果形状: (Total_Valid_Nodes, Num_Agents)
                batch.agent_loc_masks = raw_pos_transposed[valid_node_mask].float()
            else:
                batch.agent_loc_masks = raw_pos_transposed.reshape(-1, raw_pos.shape[1]).float()
        # End Edit by Xiao
        return batch

    def _construct_encoders(self, embed_dim: int) -> nn.ModuleDict:
        encoders = {}
        for key, s in self.graph_spec.items():
            typ = s[2]
            if typ == "categorical":
                num_categories = s[3]
                encoders[key] = nn.Linear(num_categories, embed_dim)
            elif typ == "magent":
                # # [Modified] Input dim is num_agents
                # input_dim = s[3]
                # encoders[key] = nn.Linear(input_dim, embed_dim)
                # Edit by Xiao: Input dim is the num of channels 
                num_channels = s[4] 
                encoders[key] = nn.Linear(num_channels, embed_dim)
            elif typ == "mask":
                encoders[key] = nn.Linear(1, embed_dim)
            else:
                encoders[key] = nn.Linear(1, embed_dim)

        return nn.ModuleDict(encoders)

    def _encode_features(
        self, observations: dict[str, th.Tensor]
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        
        batch_size = next(iter(observations.values())).shape[0]

        edge_features = {}
        node_features = {}
        graph_features = {}
        adj_matrix = observations["adj"]
        
        for key, feature in observations.items():
            s = self.graph_spec.get(key, (None, None, None))
            stage, location, typ = s[:3]
            
            if stage is None: continue 
            if key in ["adj", "xc", "yc", "tau", "edge_realisation", "total_cost", "phase"]: continue 

            # ===== NODE FEATURES =====
            if location == "node":
                if typ == "pointer":
                    feature = F.one_hot(feature.long(), num_classes=self.node_max)
                    edge_features[key] = self.encoders[key](feature.unsqueeze(-1).to(th.float32))
                
                elif typ == "categorical":
                    feature = F.one_hot(feature.long(), num_classes=s[3])
                    node_features[key] = self.encoders[key](feature.to(th.float32))
                
                # Edut by Xiao: For the new multi-channel agent node feature
                # elif typ == "magent":
                #     # [Transposed] (Batch, Agents, Nodes) -> (Batch, Nodes, Agents)
                #     feat_transposed = feature.transpose(-1, -2)
                #     node_features[key] = self.encoders[key](feat_transposed.to(th.float32))
                # 处理多智能体特征 (Shared + Pool)
                if typ == "magent":
                    # 1. 输入 feature 形状: (Batch, Agents, Nodes, Channels)
                    # 2. 我们希望先对每个智能体的 (Node, Channel) 进行编码
                    # 将维度调整为 (Batch, Agents, Nodes, Channels) -> (Batch, Agents, Nodes, embed_dim)
                    agent_node_enc = self.encoders[key](feature.float())
                    
                    # 3. 执行“池化”操作（这里使用 Sum，也可以改为 Mean 以获得更好的泛化性）
                    # 这一步消除了智能体 ID 的顺序依赖，并将特征降维回 (Batch, Nodes, embed_dim)
                    node_features[key] = th.sum(agent_node_enc, dim=1)
                # End Edit by Xiao
                    
                elif typ == "mask":
                    node_features[key] = self.encoders[key](feature.unsqueeze(-1).to(th.float32))
                
                else:
                    node_features[key] = self.encoders[key](feature.unsqueeze(-1).to(th.float32))

            # ===== EDGE FEATURES =====
            elif location == "edge":
                if typ == "categorical":
                    feature = F.one_hot(feature.long(), num_classes=s[3])
                    edge_features[key] = self.encoders[key](feature.to(th.float32))
                else:
                    edge_features[key] = self.encoders[key](feature.unsqueeze(-1).to(th.float32))

            # ===== GRAPH FEATURES =====
            elif location == "graph":
                if typ == "categorical":
                    feature = F.one_hot(feature.squeeze(-1).long(), num_classes=s[3])
                    graph_features[key] = self.encoders[key](feature.to(th.float32))
                else:
                    graph_features[key] = self.encoders[key](feature.to(th.float32))

        if not node_features:
            node_features = {"node": th.zeros(batch_size, self.node_max, self.embed_dim, device=adj_matrix.device)}
        encoded_node_features = th.sum(th.stack(list(node_features.values()), dim=0), dim=0)

        if not edge_features:
            edge_features = {"edge": th.zeros(batch_size, self.node_max, self.node_max, self.embed_dim, device=adj_matrix.device)}
        encoded_edge_features = th.sum(th.stack(list(edge_features.values()), dim=0), dim=0)

        if not graph_features:
            graph_features = {"graph": th.zeros(batch_size, self.embed_dim, device=adj_matrix.device)}
        encoded_graph_features = th.sum(th.stack(list(graph_features.values()), dim=0), dim=0)

        return (
            encoded_node_features,
            encoded_edge_features,
            encoded_graph_features,
            adj_matrix,
        )
    
class MaGraphFeatureEncoderProcessor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        pooling_type: str = "max",
        weights_init: Optional[Callable] = None,
        output_features: list[str] = ["node"], 
        cat_raw_features: bool = False,
        network_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim_vf = embed_dim
        self.latent_dim_pi = 0 
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type
        
        if network_kwargs is None: network_kwargs = {}
        processor_class = get_network_class(network_kwargs["network"])

        self.processor = processor_class(
            in_channels=embed_dim,
            out_channels=embed_dim,
            edge_dim=embed_dim,
            weights_init=weights_init,
            **network_kwargs,
        )

    def forward(self, batch: Batch) -> tuple[th.Tensor, th.Tensor]:
        node_embedding, edge_embedding = self.processor(
            node_fts=batch.x,
            edge_attr=batch.edge_attr,
            graph_fts=batch.graph_attr,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )

        if self.pooling_type == "max":
            graph_embedding = global_max_pool(node_embedding, batch.batch)
        elif self.pooling_type == "mean":
            graph_embedding = global_mean_pool(node_embedding, batch.batch)
        elif self.pooling_type == "sum":
            graph_embedding = global_add_pool(node_embedding, batch.batch)

        processed_batch = Batch(
            x=node_embedding,
            edge_index=batch.edge_index,
            edge_attr=edge_embedding,
            graph_attr=graph_embedding,
            batch=batch.batch,
        )
        
        # Pass through agent_loc_masks if it exists
        if hasattr(batch, 'agent_loc_masks'):
            processed_batch.agent_loc_masks = batch.agent_loc_masks

        return processed_batch, graph_embedding

    def forward_critic(self, x: Batch) -> th.Tensor:
        return self.forward(x)[1]

    def forward_actor(self, x: Batch) -> Batch:
        return self.forward(x)[0]


class MaNodeSimilarityMatchAgg(nn.Module):
    """
    [Modified] Agent-Specific Action Network.
    Constructs a unique Query vector for each agent by combining:
      1. Global Graph Embedding
      2. The embedding of the node where the agent is currently located.
    """

    def __init__(
        self,
        embed_dim: int,
        max_nodes: int,
        num_agents: int,  # [New]
        distance_metric: str = "euclidean",
        temp: float = 1.0,
        weights_init: Optional[Callable] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes
        self.num_agents = num_agents
        self.distance_metric = distance_metric
        
        # [New] Projection layer to combine Global Emb + Location Emb -> Query
        # Input: embed_dim * 2 (Global + Local), Output: embed_dim
        self.query_projector = nn.Linear(self.embed_dim * 2, self.embed_dim)
        
        self.softmax_temp = nn.Parameter(th.tensor(temp), requires_grad=True)

        if weights_init is not None:
            weights_init(self.query_projector)

    def compute_embedding_similarities(self, all_node_embeddings, agent_queries):
        """
        all_node_embeddings: (Batch_Size, Max_Nodes, Embed_Dim)
        agent_queries: (Batch_Size, Num_Agents, Embed_Dim)
        
        Returns: (Batch_Size, Num_Agents, Max_Nodes)
        """
        if self.distance_metric == "euclidean":
            # cdist requires (B, N, D) and (B, M, D)
            # Returns (B, Num_Agents, Max_Nodes)
            similarities = -th.cdist(agent_queries, all_node_embeddings, p=2)

        elif self.distance_metric == "cosine":
            # Manual cosine calc for 3D tensors
            # Normalize both
            a_norm = F.normalize(agent_queries, p=2, dim=-1)
            n_norm = F.normalize(all_node_embeddings, p=2, dim=-1)
            # (B, N_Agents, D) @ (B, D, Max_Nodes) -> (B, N_Agents, Max_Nodes)
            similarities = th.bmm(a_norm, n_norm.transpose(1, 2))
        
        return similarities

    def forward(self, batch: Batch) -> th.Tensor:
        """
        Returns:
            th.Tensor: (Batch_Size, Num_Agents * Max_Nodes) - Flattened logits
        """
        node_embeddings = batch.x  # (Batch_Total_Nodes, D)
        graph_embedding = batch.graph_attr # (Batch_Size, D)
        
        # 1. Retrieve raw agent positions (Batch_Total_Nodes, Num_Agents)
        if not hasattr(batch, 'agent_loc_masks'):
             raise ValueError("Batch is missing 'agent_loc_masks'. Check MaGraphFeatureTransformer.")
        
        agent_masks = batch.agent_loc_masks 
        
        agent_queries = []
        
        # 2. Construct Query for each agent
        for i in range(self.num_agents):
            # Extract the mask for agent i: (Batch_Total_Nodes, 1)
            mask_i = agent_masks[:, i].unsqueeze(-1)
            
            # Mask the node embeddings. Only the node where agent is will be non-zero.
            # (Batch_Total_Nodes, D)
            masked_nodes = node_embeddings * mask_i
            
            # Pool to get the single embedding vector per graph in the batch
            # Since only 1 node is active per graph, 'add' pooling retrieves exactly that vector.
            # (Batch_Size, D)
            agent_loc_embedding = global_add_pool(masked_nodes, batch.batch)
            
            # Combine Global + Local
            # (Batch_Size, 2*D)
            combined_context = th.cat([graph_embedding, agent_loc_embedding], dim=-1)
            
            # Project to Query
            # (Batch_Size, D)
            query_i = self.query_projector(combined_context)
            agent_queries.append(query_i)
            
        # Stack to (Batch_Size, Num_Agents, D)
        stacked_queries = th.stack(agent_queries, dim=1)
        
        # 3. Reshape Node Embeddings for Batch Processing
        # (Batch_Size, Max_Nodes, D)
        dense_node_embeddings, _ = to_dense_batch(
            node_embeddings, 
            batch.batch, 
            max_num_nodes=self.max_nodes
        )
        
        # 4. Compute Similarities
        # (Batch_Size, Num_Agents, Max_Nodes)
        logits = self.compute_embedding_similarities(dense_node_embeddings, stacked_queries)
        
        logits = logits / self.softmax_temp
        
        # SB3 MultiDiscrete expects flattened logits: (Batch_Size, Num_Agents * Max_Nodes)
        return logits.view(logits.shape[0], -1)


class MaMaskableNodeActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete, 
        lr_schedule: Callable[[float], float],
        graph_spec: dict,
        embed_dim: int = 64,
        pooling_type: str = "max",
        distance_metric: str = "euclidean",
        temp: float = 1.0,
        weights_init: Optional[Callable] = None,
        network_kwargs: dict = None,
        *args,
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type
        self.distance_metric = distance_metric
        self.temp = temp
        self.weights_init = weights_init
        self.network_kwargs = network_kwargs
        
        # [New] Detect Multi-Agent Config
        if isinstance(action_space, spaces.MultiDiscrete):
            self.num_agents = len(action_space.nvec)
            self.node_max = action_space.nvec[0]
        else:
            # Fallback for single agent
            self.num_agents = 1
            self.node_max = action_space.n
            
        self.graph_spec = {
            k: v for k, v in graph_spec.items() if v[0] in ["input", "state"]
        }

        self.extractor_args = {
            "output_features": ["node", "graph"],
            "cat_raw_features": False,
        }

        kwargs.setdefault("features_extractor_class", MaGraphFeatureTransformer)
        features_extractor_kwargs = kwargs.setdefault("features_extractor_kwargs", {})
        features_extractor_kwargs.update(
            {
                "embed_dim": embed_dim,
                "weights_init": self.weights_init,
                "graph_spec": self.graph_spec,
                "node_max": self.node_max,
            }
        )

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        # override action net
        self.action_net = MaNodeSimilarityMatchAgg(
            embed_dim=self.embed_dim,
            max_nodes=self.node_max,
            num_agents=self.num_agents, # [Passed Here]
            distance_metric=self.distance_metric,
            temp=self.temp,
            weights_init=self.weights_init,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MaGraphFeatureEncoderProcessor(
            embed_dim=self.embed_dim,
            pooling_type=self.pooling_type,
            weights_init=self.weights_init,
            network_kwargs=self.network_kwargs,
            **self.extractor_args,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                graph_spec=self.graph_spec,
                embed_dim=self.embed_dim,
                pooling_type=self.pooling_type,
                distance_metric=self.distance_metric,
                temp=self.temp,
                weights_init=self.weights_init,
                network_kwargs=self.network_kwargs,
            )
        )
        return data