from functools import partial
import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecMonitor,
    VecEnv,
)
from gnarl.envs.generate.graph_generator import (
    FixedSetGraphGenerator,
    RandomSetGraphGenerator,
)
from gnarl.envs.generate.data import GraphProblemDataset
import importlib


class VariableTimeLimit(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_steps = None
        self.elapsed_steps = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.max_steps = self.env.unwrapped.max_episode_steps
        self.elapsed_steps = 0
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1

        if self.elapsed_steps >= self.max_steps:
            truncated = True
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info


class RenderOnDone(Wrapper):
    """
    Ensures the final frame is rendered before a VecEnv auto-resets a done env.
    Wrap the base env so that when (terminated or truncated) occurs, we call
    env.render() immediately (while the terminal state is still present),
    allowing GIF/video buffers to capture the last frame.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            if (terminated or truncated) and getattr(
                self.env, "render_mode", None
            ) == "human":
                # Render once more to capture the terminal state before VecEnv resets
                self.env.render()
                self._save_render()

        except Exception:
            # Rendering issues should not break training/evaluation
            pass
        return obs, reward, terminated, truncated, info


def get_env_cls(env_name: str):
    env_spec = gym.registry[env_name]
    if callable(env_spec.entry_point):
        # Edit by Xiao
        # This is to handle the case where the entry_point is already a class we write in gnarl.envs
        env_cls = env_spec.entry_point
    else:
        module_path, class_name = env_spec.entry_point.rsplit(":", 1)
        module = importlib.import_module(module_path)
        env_cls = getattr(module, class_name)
    return env_cls


def make_train_env(config, run_id) -> VecEnv:
    """
    Creates a vectorised environment for evaluation where each sub-environment
    is initialized with a generator.
    """

    env_cls = get_env_cls(config["env"])

    ds = [
        GraphProblemDataset(
            root=config["train_data"]["graph_dir"],
            split="train",
            algorithm=config["algorithm"],
            num_nodes=n,
            num_samples=num_samples,
            seed=config["train_data"]["seed"],
            graph_generator=config["train_data"]["graph_generator"],
            graph_generator_kwargs=config["train_data"].get("graph_generator_kwargs"),
            # Edit by Xiao
            # 新增多智能体参数
            num_starts=config["train_data"].get("num_starts", 1),
            num_goals=config["train_data"].get("num_goals", 1),
            pre_filter=(getattr(env_cls, "pre_filter", None)),
            pre_transform=(getattr(env_cls, "pre_transform", None)),
        )
        for n, num_samples in config["train_data"]["node_samples"].items()
    ]

    def create_env(seed_offset=0):
        env_seed = config["train_data"]["seed"] + seed_offset

        env = gym.make(
            config["env"],
            max_nodes=max(d.num_nodes for d in ds),
            graph_generator=RandomSetGraphGenerator(datasets=ds, seed=env_seed),
            **config.get("env_kwargs", {}),
        )
        env = VariableTimeLimit(env)
        return env

    env_fns = []
    for i in range(config["train_data"]["num_envs"]):
        thunk = partial(create_env, seed_offset=i)
        env_fns.append(thunk)

    if config["train_data"].get("subproc_env", False):
        vec_env_cls = SubprocVecEnv
    else:
        vec_env_cls = DummyVecEnv
    vec_env = vec_env_cls(env_fns)
    vec_env = VecMonitor(vec_env, f"runs/{run_id}")
    vec_env.seed(config["train_data"]["seed"])

    return vec_env


def make_matrain_env(config, run_id) -> VecEnv:
    """
    Creates a vectorised environment for evaluation where each sub-environment
    is initialized with a generator.
    """

    env_cls = get_env_cls(config["env"])

    ds = [
        GraphProblemDataset(
            root=config["train_data"]["graph_dir"],
            split="train",
            algorithm=config["algorithm"],
            num_nodes=n,
            num_samples=num_samples,
            seed=config["train_data"]["seed"],
            graph_generator=config["train_data"]["graph_generator"],
            graph_generator_kwargs=config["train_data"].get("graph_generator_kwargs"),
            # Edit by Xiao
            # 新增多智能体参数
            num_starts=config["train_data"].get("num_starts", 1),
            num_goals=config["train_data"].get("num_goals", 1),
            pre_filter=(getattr(env_cls, "pre_filter", None)),
            pre_transform=(getattr(env_cls, "pre_transform", None)),
        )
        for n, num_samples in config["train_data"]["node_samples"].items()
    ]
    
    def create_env(seed_offset=0):
        env_seed = config["train_data"]["seed"] + seed_offset

        env = gym.make(
            config["env"],
            max_nodes=max(d.num_nodes for d in ds),
            num_agents=config["train_data"].get("num_starts", 1),
            num_goals=config["train_data"].get("num_goals", 1),
            graph_generator=RandomSetGraphGenerator(datasets=ds, seed=env_seed),
            **config.get("env_kwargs", {}),
        )
        env = VariableTimeLimit(env)
        return env

    env_fns = []
    for i in range(config["train_data"]["num_envs"]):
        thunk = partial(create_env, seed_offset=i)
        env_fns.append(thunk)

    if config["train_data"].get("subproc_env", False):
        vec_env_cls = SubprocVecEnv
    else:
        vec_env_cls = DummyVecEnv
    vec_env = vec_env_cls(env_fns)
    vec_env = VecMonitor(vec_env, f"runs/{run_id}")
    vec_env.seed(config["train_data"]["seed"])

    return vec_env

def calculate_env_split(
    node_samples: dict[int, int], max_envs: int
) -> list[tuple[int, int, int]]:
    """
    Calculate the number of samples per environment based on the node sizes and
    the maximum number of environments.
    """
    env_allocation = []
    for n, count in node_samples.items():
        c = count
        while c > 0:
            if c >= max_envs:
                s = c // max_envs
                c -= s * max_envs
                env_allocation.append((n, s, max_envs))
            else:
                env_allocation.append((n, 1, c))
                c = 0
    return env_allocation


def make_eval_env(config, run_id, split="test", render=False) -> list[VecEnv]:
    """
    Creates a vectorised environment for evaluation where each sub-environment
    is initialized with a different graph.
    """
    if split not in ["test", "val"]:
        raise ValueError("Split must be either 'test' or 'val'.")
    cfg = config["test_data"] if split == "test" else config["val_data"]

    per_env_samples = calculate_env_split(
        node_samples=cfg["node_samples"], max_envs=cfg["num_envs"]
    )
    per_node_samples = {}
    for num_nodes, num_samples, num_envs in per_env_samples:
        if num_nodes not in per_node_samples:
            per_node_samples[num_nodes] = 0
        per_node_samples[num_nodes] += num_samples * num_envs
    print(f"Constructing evaluation environments with parameters: {per_env_samples}")

    env_cls = get_env_cls(config["env"])

    datasets_by_n = {
        num_nodes: GraphProblemDataset(
            root=cfg["graph_dir"],
            split=split,
            algorithm=config["algorithm"],
            num_nodes=num_nodes,
            num_samples=num_samples,
            seed=cfg["seed"],
            graph_generator=cfg["graph_generator"],
            graph_generator_kwargs=cfg.get("graph_generator_kwargs"),
            pre_filter=(getattr(env_cls, "pre_filter", None)),
            pre_transform=(getattr(env_cls, "pre_transform", None)),
        )
        for num_nodes, num_samples in per_node_samples.items()
    }

    subsets = []
    prev_samples = {num_nodes: 0 for num_nodes in datasets_by_n.keys()}
    for num_nodes, num_samples, num_envs in per_env_samples:
        subsets_e = []
        for _ in range(num_envs):
            subsets_e.append(
                np.arange(
                    prev_samples[num_nodes], prev_samples[num_nodes] + num_samples
                )
            )
            prev_samples[num_nodes] += num_samples
        subsets.append(subsets_e)

    def create_env(dataset: GraphProblemDataset, subset: list[int], seed_offset=0):
        env_seed = cfg["seed"] + seed_offset

        env = gym.make(
            config["env"],
            max_nodes=dataset.num_nodes,
            graph_generator=FixedSetGraphGenerator(
                dataset,
                seed=env_seed,
                subset=subset,
            ),
            render_mode="human" if render else None,
            folder=f"eval_runs/{run_id}/render" if render else None,
            env_id=seed_offset if render else None,
            **config.get("env_kwargs", {}),
        )
        env = VariableTimeLimit(env)
        if render:
            env = RenderOnDone(env)
        return env

    venvs = []
    for e, env_set in enumerate(per_env_samples):
        env_fns = []
        num_nodes, num_samples, num_envs = env_set
        for i in range(num_envs):
            thunk = partial(
                create_env,
                dataset=datasets_by_n[num_nodes],
                subset=subsets[e][i],
                seed_offset=i,
            )
            env_fns.append(thunk)

        if cfg.get("subproc_env", False):
            vec_env_cls = SubprocVecEnv
        else:
            vec_env_cls = DummyVecEnv
        vec_env = vec_env_cls(env_fns)
        vec_env = VecMonitor(vec_env, f"eval_runs/{run_id}/")
        vec_env.seed(cfg["seed"])
        venvs.append(vec_env)

    return venvs


def make_maeval_env(config, run_id, split="test", render=False) -> list[VecEnv]:
    """
    Creates a vectorised environment for evaluation where each sub-environment
    is initialized with a different graph.
    """
    if split not in ["test", "val"]:
        raise ValueError("Split must be either 'test' or 'val'.")
    cfg = config["test_data"] if split == "test" else config["val_data"]

    # 1. 计算环境分割
    per_env_samples = calculate_env_split(
        node_samples=cfg["node_samples"], max_envs=cfg["num_envs"]
    )
    per_node_samples = {}
    for num_nodes, num_samples, num_envs in per_env_samples:
        if num_nodes not in per_node_samples:
            per_node_samples[num_nodes] = 0
        per_node_samples[num_nodes] += num_samples * num_envs
    print(f"Constructing evaluation environments with parameters: {per_env_samples}")

    env_cls = get_env_cls(config["env"])

    # 2. 创建数据集 (包含多智能体参数)
    datasets_by_n = {
        num_nodes: GraphProblemDataset(
            root=cfg["graph_dir"],
            split=split,
            algorithm=config["algorithm"],
            num_nodes=num_nodes,
            num_samples=num_samples,
            seed=cfg["seed"],
            graph_generator=cfg["graph_generator"],
            graph_generator_kwargs=cfg.get("graph_generator_kwargs"),
            # [Multi-Agent Args]
            num_starts=cfg.get("num_starts", 1),
            num_goals=cfg.get("num_goals", 1),
            pre_filter=(getattr(env_cls, "pre_filter", None)),
            pre_transform=(getattr(env_cls, "pre_transform", None)),
        )
        for num_nodes, num_samples in per_node_samples.items()
    }

    # 3. 准备子集索引
    subsets = []
    prev_samples = {num_nodes: 0 for num_nodes in datasets_by_n.keys()}
    for num_nodes, num_samples, num_envs in per_env_samples:
        subsets_e = []
        for _ in range(num_envs):
            subsets_e.append(
                np.arange(
                    prev_samples[num_nodes], prev_samples[num_nodes] + num_samples
                )
            )
            prev_samples[num_nodes] += num_samples
        subsets.append(subsets_e)

    # 4. 定义单个环境创建函数
    def create_env(dataset: GraphProblemDataset, subset: list[int], seed_offset=0):
        env_seed = cfg["seed"] + seed_offset
        
        env = gym.make(
            config["env"],
            max_nodes=dataset.num_nodes,
            # [CRITICAL FIX] 必须传递 num_agents，否则 eval env 会默认为 2，导致维度报错
            num_agents=cfg.get("num_starts", 1), 
            num_goals=cfg.get("num_goals", 1), 
            graph_generator=FixedSetGraphGenerator(
                dataset,
                seed=env_seed,
                subset=subset,
            ),
            render_mode="human" if render else None,
            folder=f"eval_runs/{run_id}/render" if render else None,
            env_id=seed_offset if render else None,
            **config.get("env_kwargs", {}),
        )
        env = VariableTimeLimit(env)
        if render:
            env = RenderOnDone(env)
        return env

    # 5. 构建 Vectorized Environments
    venvs = []
    for e, env_set in enumerate(per_env_samples):
        env_fns = []
        num_nodes, num_samples, num_envs = env_set
        for i in range(num_envs):
            thunk = partial(
                create_env,
                dataset=datasets_by_n[num_nodes],
                subset=subsets[e][i],
                seed_offset=i,
            )
            env_fns.append(thunk)

        if cfg.get("subproc_env", False):
            vec_env_cls = SubprocVecEnv
        else:
            vec_env_cls = DummyVecEnv
        
        # 注意：Eval Env 通常不需要 VecMonitor 记录日志到 tensorboard，
        # 除非你想看 Eval 的原始 reward 曲线。EvalCallback 会自己处理 evaluation logging。
        vec_env = vec_env_cls(env_fns)
        # vec_env = VecMonitor(vec_env, f"eval_runs/{run_id}/") # 可选
        vec_env.seed(cfg["seed"])
        venvs.append(vec_env)

    return venvs

    """
    Creates a vectorised environment for training.
    """
    env_cls = get_env_cls(config["env"])

    ds = [
        GraphProblemDataset(
            root=config["train_data"]["graph_dir"],
            split="train",
            algorithm=config["algorithm"],
            num_nodes=n,
            num_samples=num_samples,
            seed=config["train_data"]["seed"],
            graph_generator=config["train_data"]["graph_generator"],
            graph_generator_kwargs=config["train_data"].get("graph_generator_kwargs"),
            # [Multi-Agent Args]
            num_starts=config["train_data"].get("num_starts", 1),
            num_goals=config["train_data"].get("num_goals", 1),
            pre_filter=(getattr(env_cls, "pre_filter", None)),
            pre_transform=(getattr(env_cls, "pre_transform", None)),
        )
        for n, num_samples in config["train_data"]["node_samples"].items()
    ]
    
    def create_env(seed_offset=0):
        env_seed = config["train_data"]["seed"] + seed_offset

        env = gym.make(
            config["env"],
            max_nodes=max(d.num_nodes for d in ds),
            # [Correct] 训练环境这里已经写对了，保持即可
            num_agents=config["train_data"].get("num_starts", 1),
            graph_generator=RandomSetGraphGenerator(datasets=ds, seed=env_seed),
            **config.get("env_kwargs", {}),
        )
        env = VariableTimeLimit(env)
        return env

    env_fns = []
    for i in range(config["train_data"]["num_envs"]):
        thunk = partial(create_env, seed_offset=i)
        env_fns.append(thunk)

    if config["train_data"].get("subproc_env", False):
        vec_env_cls = SubprocVecEnv
    else:
        vec_env_cls = DummyVecEnv
    vec_env = vec_env_cls(env_fns)
    vec_env = VecMonitor(vec_env, f"runs/{run_id}")
    vec_env.seed(config["train_data"]["seed"])

    return vec_env