#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .cnn import Cnn, CnnConfig
from .common import (
    EnsembleModelConfig,
    Model,
    ModelConfig,
    SequenceModel,
    SequenceModelConfig,
)
from .deepsets import Deepsets, DeepsetsConfig
from .gnn import Gnn, GnnConfig
from .gru import Gru, GruConfig
from .lstm import Lstm, LstmConfig
from .mlp import Mlp, MlpConfig
from .graph_critic_gnn import GraphCriticGNN, GraphCriticGNNConfig
from .graph_actor_gnn import GraphActorGNN, GraphActorGNNConfig
from .graphQnet import GraphQNet, GraphQNetConfig
from .indepent_critic_gnn import IndependentGraphCriticGNN, IndependentGraphCriticGNNConfig
from .ignarl_actor_gnn import IgnarlActorGNN, IgnarlActorGNNConfig
# from .ignarl_shared_trunk import SharedEncodeProcessTrunk, get_or_create_shared_trunk
from .ignarl_critic_gnn import IgnarlCriticGNN, IgnarlCriticGNNConfig
from .magnarl_actor_gnn import MagnarlActorGNN, MagnarlActorGNNConfig
from .magnarl_critic_gnn import MagnarlCriticGNN, MagnarlCriticGNNConfig    

classes = [
    "Mlp",
    "MlpConfig",
    "Gnn",
    "GnnConfig",
    "Cnn",
    "CnnConfig",
    "Deepsets",
    "DeepsetsConfig",
    "Gru",
    "GruConfig",
    "Lstm",
    "LstmConfig",
    "GraphCriticGNNConfig",
    "GraphActorGNNConfig",
    "IndependentGraphCriticGNNConfig",
    "IgnarlActorGNN",
    "IgnarlActorGNNConfig",
    # "SharedEncodeProcessTrunk",
    # "get_or_create_shared_trunk",
    "IgnarlCriticGNN",
    "IgnarlCriticGNNConfig",
    "MagnarlActorGNN",
    "MagnarlActorGNNConfig",
    "MagnarlCriticGNN",
    "MagnarlCriticGNNConfig",
]

model_config_registry = {
    "mlp": MlpConfig,
    "gnn": GnnConfig,
    "cnn": CnnConfig,
    "deepsets": DeepsetsConfig,
    "gru": GruConfig,
    "lstm": LstmConfig,
    "critic_gnn":GraphCriticGNNConfig,
    "actor_gnn":GraphActorGNNConfig,
    "independent_critic_gnn": IndependentGraphCriticGNNConfig,
    "ignarl_actor_gnn": IgnarlActorGNNConfig,
    "ignarl_critic_gnn": IgnarlCriticGNNConfig,
    "magnarl_actor_gnn": MagnarlActorGNNConfig,
    "magnarl_critic_gnn": MagnarlCriticGNNConfig,
}
