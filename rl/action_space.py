from enum import IntEnum
from typing import List, Dict, Union

class Strategy(IntEnum):
    ROUTE = 0
    CLARIFY = 1
    SUGGEST = 2
    ESCALATE = 3

# Mapping Strategy to its constituent actions
STRATEGY_MAP = {
    Strategy.ROUTE: [
        "route_bug",
        "route_ml_module",
        "route_build_infra",
        "route_docs",
        "route_billing",
        "route_product"
    ],
    Strategy.CLARIFY: [
        "ask_uncertainty",
        "ask_error_type",
        "ask_version",
        "ask_platform",
        "ask_hardware",
        "ask_vague_request"
    ],
    Strategy.SUGGEST: [
        "suggest_top1",
        "suggest_top2",
        "suggest_top3",
        "suggest_high_conf_only"
    ],
    Strategy.ESCALATE: [
        "escalate_human"
    ]
}

# Flattened list of all 17 actions
ACTIONS = []
for strategy in Strategy:
    ACTIONS.extend(STRATEGY_MAP[strategy])

# Bi-directional mappings
ACTION_TO_INDEX = {name: i for i, name in enumerate(ACTIONS)}
INDEX_TO_ACTION = {i: name for i, name in enumerate(ACTIONS)}

# Reverse mapping: action_id -> strategy
ACTION_ID_TO_STRATEGY = {}
for strategy, actions in STRATEGY_MAP.items():
    for action_name in actions:
        idx = ACTION_TO_INDEX[action_name]
        ACTION_ID_TO_STRATEGY[idx] = strategy

def get_action_names() -> List[str]:
    """Returns the list of all action names in order."""
    return ACTIONS

def get_action_count() -> int:
    """Returns the total number of actions (17)."""
    return len(ACTIONS)

def get_strategy_actions(strategy: Union[Strategy, str]) -> List[int]:
    """Returns the indices of all actions belonging to a strategy."""
    if isinstance(strategy, str):
        strategy = Strategy[strategy.upper()]
    names = STRATEGY_MAP[strategy]
    return [ACTION_TO_INDEX[name] for name in names]

def get_action_strategy(action_id: int) -> Strategy:
    """Returns the high-level strategy for a given action ID."""
    return ACTION_ID_TO_STRATEGY.get(action_id)

def is_route(action_id: int) -> bool:
    return get_action_strategy(action_id) == Strategy.ROUTE

def is_clarify(action_id: int) -> bool:
    return get_action_strategy(action_id) == Strategy.CLARIFY

def is_suggest(action_id: int) -> bool:
    return get_action_strategy(action_id) == Strategy.SUGGEST

def is_escalate(action_id: int) -> bool:
    return get_action_strategy(action_id) == Strategy.ESCALATE

def get_action_name(action_id: int) -> str:
    return INDEX_TO_ACTION[action_id]

if __name__ == "__main__":
    print(f"Total actions: {get_action_count()}")
    for i, name in INDEX_TO_ACTION.items():
        strategy = get_action_strategy(i)
        print(f"{i:2}: {name:<25} (Strategy: {strategy.name})")
