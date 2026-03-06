from clrs._src.specs import Stage, Location, Type, SPECS

SPECS = dict(SPECS)
SPECS.update(
    {
        "tsp": {
            "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
            "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
            "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
            "xc": (Stage.INPUT, Location.NODE, Type.SCALAR),  # not encoded
            "yc": (Stage.INPUT, Location.NODE, Type.SCALAR),  # not encoded
        },
        "mvc": {
            "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
            "nw": (Stage.INPUT, Location.NODE, Type.SCALAR),  # node weights
        },
        "rgc": {
            "tau": (Stage.INPUT, Location.GRAPH, Type.SCALAR),  # not encoded
        },
        "ctp": {
            "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
            "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
            "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
            "g": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
            "stochastic_edges": (Stage.INPUT, Location.EDGE, Type.MASK),
            "edge_probs": (Stage.INPUT, Location.EDGE, Type.SCALAR),
            "edge_realisation": (
                Stage.INPUT,
                Location.EDGE,
                Type.CATEGORICAL,
                4,
            ),  # not encoded
        },
        "mactp": {
            "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
            "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
            "s": (Stage.INPUT, Location.NODE, Type.MASK),      # 多个起始节点
            "g": (Stage.INPUT, Location.NODE, Type.MASK),      # 多个目标节点
            "stochastic_edges": (Stage.INPUT, Location.EDGE, Type.MASK),
            "edge_probs": (Stage.INPUT, Location.EDGE, Type.SCALAR),
            "edge_realisation": (
                Stage.INPUT,
                Location.EDGE,
                Type.CATEGORICAL,
                4,
            ),  # 0: 非随机边, 1: ?, 2: 可通行随机边, 3: 阻塞随机边
        }
        # # Centralized Multi-Agent CTP specification -- for the first version of mactp env
        # "mactp": {
        #     # graph / edge-level (same as ctp)
        #     "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
        #     "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
        #     "stochastic_edges": (Stage.INPUT, Location.EDGE, Type.MASK),
        #     "edge_probs": (Stage.INPUT, Location.EDGE, Type.SCALAR),
        #     "edge_realisation": (
        #         Stage.INPUT,
        #         Location.EDGE,
        #         Type.CATEGORICAL,
        #         4,
        #     ),  # not encoded

        #     # standard node-level fields
        #     "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),  # start(s)
        #     "g": (Stage.INPUT, Location.NODE, Type.MASK_ONE),  # goal(s) / destinations mask

        #     # multi-agent specific (centralized view)
        #     # traveler_positions: one-hot per traveler (shape: n_travelers x num_nodes)
        #     "traveler_positions": (Stage.INPUT, Location.NODE, Type.MASK),
        #     # traveler_costs: per-traveler scalar list (encoded at graph level)
        #     "traveler_costs": (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        #     # destinations and visited masks (node-level)
        #     "destinations": (Stage.INPUT, Location.NODE, Type.MASK),
        #     "visited_mask": (Stage.INPUT, Location.NODE, Type.MASK),
        #     # global/graph-level scalars
        #     "remaining_destinations": (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        #     "total_cost": (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        #     # action masks for centralized joint-action (shape: n_travelers x num_nodes)
        #     "action_masks": (Stage.INPUT, Location.NODE, Type.MASK),
        # },
    }
)
