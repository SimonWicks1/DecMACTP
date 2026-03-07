from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type

from torchrl.objectives import ClipPPOLoss

from benchmarl.algorithms.ippo import Ippo, IppoConfig
from benchmarl.algorithms.common import Algorithm
from tensordict import TensorDictBase


class Ignarl(Ippo):
    """
    IGNARL = IPPO training + GNARL-style shared trunk actor/critic.

    Important: actor and critic share the same trunk module instance.
    IPPO by default optimizes actor and critic params in two separate optimizers.
    To prevent shared trunk params from being stepped twice, we deduplicate parameters here.
    """
# benchmarl/algorithms/ignarl.py
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Type

from torchrl.objectives import ClipPPOLoss

from benchmarl.algorithms.ippo import Ippo, IppoConfig
from benchmarl.algorithms.common import Algorithm


class Ignarl(Ippo):
    """
    IGNARL = IPPO training + GNARL-style shared trunk.
    To avoid shared-trunk multi-step/backward issues, we:
      - (recommended) detach trunk in critic forward (see IgnarlCriticGNNConfig.detach_trunk=True)
      - ensure shared parameters are not optimized twice by excluding shared params from critic optimizer.
    """
    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        actor_params = list(loss.actor_network_params.flatten_keys().values())
        critic_params = list(loss.critic_network_params.flatten_keys().values())

        actor_ids = {id(p) for p in actor_params}
        critic_unique = [p for p in critic_params if id(p) not in actor_ids]

        return {"loss_objective": actor_params, "loss_critic": critic_unique}


@dataclass
class IgnarlConfig(IppoConfig):
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
        return Ignarl