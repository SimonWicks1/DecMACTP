import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from .mactp_torchrl import MultiTravelerCTPTorchRLEnv


class MactpClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        device: DEVICE_TYPING,
        seed: Optional[int] = None,
    ) -> Callable[[], EnvBase]:
        if continuous_actions:
            raise ValueError("MACTP only supports discrete actions.")

        config = copy.deepcopy(self.config)
        max_nodes = config.get("max_nodes", None)
        num_agents = config.get("num_agents", None)
        graph_generator = config.get("graph_generator", None)
        seed = seed if seed is not None else config.get("seed", 0)
        if graph_generator is None:
            raise ValueError("Config must contain `graph_generator`.")

        return lambda: MultiTravelerCTPTorchRLEnv(
            max_nodes=max_nodes,
            num_agents=num_agents,
            graph_generator=graph_generator,
            device=device,
            num_envs=num_envs,
            seed=seed,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return env.num_phases

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {"agents": [f"agent_{i}" for i in range(env.num_agents)]}
    
    def observation_spec(self, env: EnvBase) -> Composite:
        """
        Level-1 Actor/Critic 输入（CTDE）：
        - 允许：node_features（含 self_pos/density 等团队统计）、edge_status、edge_weights、phase、action_mask
        - 禁止：state、penalty、agent_id（同质 agent 保持置换对称）、以及重复/可能泄露的全局键
        """
        full = env.observation_spec_unbatched.clone()

        # 只保留 Level-1 keys（必须与 env _make_tensordict 输出一致）
        allowed_agent_keys = {
            "node_features",
            "edge_status",
            "edge_weights",
            "phase",
            "action_mask",
        }

        out = Composite()
        for group in self.group_map(env):
            if group not in full.keys():
                continue
            group_full = full[group]
            group_out = Composite(shape=group_full.shape)
            for k in allowed_agent_keys:
                if k in group_full.keys():
                    group_out[k] = group_full[k]
            if not group_out.is_empty():
                out[group] = group_out

        return out
    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        # """
        # 核心修复：将 state 张量重新包装为 BenchMARL 期望的单键 Composite 容器
        # """
        # from torchrl.data import Composite
        
        # spec = env.observation_spec_unbatched.clone()
        # if "state" in spec.keys():
        #     # 必须返回一个 Composite 对象，且内部仅包含一个叶子节点
        #     return Composite({"state": spec["state"]})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        """
        修复：同样使用 unbatched spec
        """
        # --- 修改开始 ---
        # 原代码: observation_spec = env.observation_spec.clone()
        full = env.observation_spec_unbatched.clone()
        out = Composite()
        for group in self.group_map(env):
            if group not in full.keys():
                continue
            group_full = full[group]
            if "action_mask" not in group_full.keys():
                continue
            group_out = Composite({"action_mask": group_full["action_mask"]}, shape=group_full.shape)
            out[group] = group_out
        return None if out.is_empty() else out

    def action_spec(self, env: EnvBase) -> Composite:
        """
        修复：返回 unbatched action spec
        """
        # --- 修改开始 ---
        # 原代码: return env.full_action_spec
        return env.action_spec_unbatched
        # --- 修改结束 ---


    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return "mactp"


class MactpTask(Task):
    DEFAULT = None
    TEST = None
    TRAIN = None

    @staticmethod
    def associated_class():
        return MactpClass