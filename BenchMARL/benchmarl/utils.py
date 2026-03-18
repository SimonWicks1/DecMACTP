#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import contextlib
import importlib
import random
import typing
from typing import Any, Callable, Dict, List, Union

import torch
import yaml
from torchrl.data import Composite
from torchrl.envs import Compose, EnvBase, InitTracker, TensorDictPrimer, TransformedEnv

if typing.TYPE_CHECKING:
    from benchmarl.models import ModelConfig

_has_numpy = importlib.util.find_spec("numpy") is not None


DEVICE_TYPING = Union[torch.device, str, int]


def _read_yaml_config(config_file: str) -> Dict[str, Any]:
    with open(config_file) as config:
        yaml_string = config.read()
    config_dict = yaml.safe_load(yaml_string)
    if config_dict is None:
        config_dict = {}
    if "defaults" in config_dict.keys():
        del config_dict["defaults"]
    return config_dict


def _class_from_name(name: str):
    name_split = name.split(".")
    module_name = ".".join(name_split[:-1])
    class_name = name_split[-1]
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if _has_numpy:
        import numpy

        numpy.random.seed(seed)


@contextlib.contextmanager
def local_seed():
    torch_state = torch.random.get_rng_state()
    if _has_numpy:
        import numpy as np

        np_state = np.random.get_state()
    py_state = random.getstate()

    yield

    torch.random.set_rng_state(torch_state)
    if _has_numpy:
        np.random.set_state(np_state)
    random.setstate(py_state)


def _add_rnn_transforms(
    env_fun: Callable[[], EnvBase],
    group_map: Dict[str, List[str]],
    model_config: "ModelConfig",
) -> Callable[[], EnvBase]:
    """
    This function adds RNN specific transforms to the environment

    Args:
        env_fun (callable): a function that takes no args and creates an environment
        group_map (Dict[str,List[str]]): the group_map of the agents
        model_config (ModelConfig): the model configuration

    Returns: a function that takes no args and creates an environment

    """

    def model_fun():
        env = env_fun()
        spec_actor = Composite(
            {
                group: Composite(
                    model_config._get_model_state_spec_inner(group=group).expand(
                        len(agents),
                        *model_config._get_model_state_spec_inner(group=group).shape
                    ),
                    shape=(len(agents),),
                )
                for group, agents in group_map.items()
            }
        )

        out_env = TransformedEnv(
            env,
            Compose(
                *(
                    [InitTracker(init_key="is_init")]
                    + (
                        [
                            TensorDictPrimer(
                                spec_actor, reset_key="_reset", expand_specs=True
                            )
                        ]
                        if len(spec_actor.keys(True, True)) > 0
                        else []
                    )
                )
            ),
        )
        return out_env

    return model_fun

def compute_joint_behavior_metrics(
    actions: torch.Tensor,
    positions: torch.Tensor,
    action_mask: torch.Tensor = None,
    uncertain_frontier_map: torch.Tensor = None,
    goals_visited_mask: torch.Tensor = None,
):
    """
    Args:
        actions: [B, A] long
        positions: [B, A] long
        action_mask: optional [B, A, N]
        uncertain_frontier_map: optional [B, A, N] or [B, N]
        goals_visited_mask: optional [B, N]

    Returns:
        dict[str, Tensor] with scalar batch metrics
    """
    B, A = actions.shape
    device = actions.device

    # 1) pairwise same-action conflict rate
    a_i = actions.unsqueeze(2)  # [B, A, 1]
    a_j = actions.unsqueeze(1)  # [B, 1, A]
    same_action = (a_i == a_j)
    eye = torch.eye(A, device=device, dtype=torch.bool).unsqueeze(0)
    same_action = same_action & (~eye)
    # conflict_rate = same_action.float().mean()
    same_action = same_action & (~eye)              # [B, A, A]
    conflict_rate = same_action.float().mean(dim=(-1, -2))   # [B]

    # 2) idle ratio
    idle_ratio = (actions == positions).float().mean()

    # 3) mean pairwise dispersion in discrete node index space
    # 这里先用 index 差作为轻量 proxy；后续你可以替换成 graph shortest-path distance
    pos_i = positions.unsqueeze(2).float()
    pos_j = positions.unsqueeze(1).float()
    dispersion = (pos_i - pos_j).abs()
    dispersion = dispersion.masked_fill(eye, 0.0)
    mean_agent_dispersion = dispersion.sum() / ((~eye).float().sum() * B + 1e-8)

    # 4) action overlap mass (proxy)
    action_overlap = same_action.any(dim=-1).float().mean()

    out = {
        "conflict_rate": conflict_rate,
        "idle_ratio": idle_ratio,
        "mean_agent_dispersion": mean_agent_dispersion,
        "pairwise_action_overlap": action_overlap,
    }

    # 5) optional frontier redundancy
    if uncertain_frontier_map is not None:
        if uncertain_frontier_map.dim() == 2:
            uncertain_frontier_map = uncertain_frontier_map.unsqueeze(1).expand(B, A, -1)
        # chosen_frontier = torch.gather(
        #     uncertain_frontier_map.float(), dim=-1, index=actions.unsqueeze(-1)
        # ).squeeze(-1)  # [B, A]
        safe_actions = actions.clamp(0, uncertain_frontier_map.shape[-1] - 1)
        chosen_frontier = torch.gather(
            uncertain_frontier_map.float(), dim=-1, index=safe_actions.unsqueeze(-1)
        ).squeeze(-1)
        frontier_redundancy_rate = (
            (chosen_frontier.sum(dim=-1) > 1).float().mean()
        )
        out["frontier_redundancy_rate"] = frontier_redundancy_rate

    # 6) optional goal redundancy
    if goals_visited_mask is not None:
        unvisited = (1.0 - goals_visited_mask.float())  # [B, N]
        safe_actions = actions.clamp(0, unvisited.shape[-1] - 1)
        chosen_unvisited = torch.gather(
            unvisited, dim=-1, index=safe_actions
        )  # [B, A]
        goal_redundancy_rate = ((chosen_unvisited.sum(dim=-1) > 1).float().mean())
        out["goal_redundancy_rate"] = goal_redundancy_rate

    return out