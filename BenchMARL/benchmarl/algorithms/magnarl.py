from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Type

from torchrl.objectives import ClipPPOLoss

from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.mappo import Mappo, MappoConfig


class Magnarl(Mappo):
    """
    MAGNARL = MAPPO-style CTDE training + communication-aware GNARL trunk.

    Actor and critic may share (parts of) the same local encode/process trunk.
    As in IGNARL, we deduplicate critic parameters that are already optimized by
    the actor optimizer to avoid stepping shared parameters twice.
    """

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        actor_params = list(loss.actor_network_params.flatten_keys().values())
        critic_params = list(loss.critic_network_params.flatten_keys().values())

        actor_ids = {id(p) for p in actor_params}
        critic_unique = [p for p in critic_params if id(p) not in actor_ids]

        return {
            "loss_objective": actor_params,
            "loss_critic": critic_unique,
        }


@dataclass
class MagnarlConfig(MappoConfig):
    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Magnarl