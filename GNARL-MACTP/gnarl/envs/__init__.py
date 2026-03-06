from gymnasium.envs.registration import register

register(
    id="BellmanFord-v1",
    entry_point="gnarl.envs.clrs_envs:BellmanFordEnv",
    nondeterministic=False,
)

register(
    id="BFS-v1",
    entry_point="gnarl.envs.clrs_envs:BFSEnv",
    nondeterministic=False,
)

register(
    id="DFS-v1",
    entry_point="gnarl.envs.clrs_envs:DFSEnv",
    nondeterministic=False,
)

register(
    id="MSTPrim-v1",
    entry_point="gnarl.envs.clrs_envs:MSTPrimEnv",
    nondeterministic=False,
)

register(
    id="TSP-v1",
    entry_point="gnarl.envs.np_envs:TSPEnv",
    nondeterministic=False,
)

register(
    id="MVC-v1",
    entry_point="gnarl.envs.np_envs:MVCEnv",
    nondeterministic=False,
)

register(
    id="RGC-v1",
    entry_point="gnarl.envs.np_envs:RGCEnv",
    nondeterministic=True,
)

register(
    id="CTP-v1",
    entry_point="gnarl.envs.ctp_env:CTPEnv",
    nondeterministic=False,
)

register(
    id="MACTP-v1",
    entry_point="gnarl.envs.mactp_env:MultiTravelerCTPEnv",
    nondeterministic=False,
)

register(
    id="MACTP-v2",
    entry_point="gnarl.envs.mactp_env2:MultiTravelerCTPEnv2",
    nondeterministic=False,
)

ENV_MAPPING = {
    "bfs": "BFS-v1",
    "bellman_ford": "BellmanFord-v1",
    "dfs": "DFS-v1",
    "mst_prim": "MSTPrim-v1",
    "tsp": "TSP-v1",
    "mvc": "MVC-v1",
    "rgc": "RGC-v1",
    "ctp": "CTP-v1",
    "mactp": "MACTP-v2",
}
