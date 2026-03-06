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

    # 在 ignarl.py 中修改
    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        # 将 Actor Loss 和 Critic Loss 加和，由 "loss_objective" 统一负责
        # 注意：ClipPPOLoss 内部已经根据 critic_coef 处理了权重
        loss_vals.set(
            "loss_objective", 
            loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
        )
        # 删除单独的项，防止被 experiment.py 遍历到
        del loss_vals["loss_entropy"]
        del loss_vals["loss_critic"]
        return loss_vals

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        # 将 Actor 和 Critic 的所有参数合并给一个优化器
        all_params = set(loss.actor_network_params.flatten_keys().values())
        all_params.update(loss.critic_network_params.flatten_keys().values())
        
        return {
            "loss_objective": list(all_params),
        }

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