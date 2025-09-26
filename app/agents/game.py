"""Multi-agent decision game implementation."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Dict, Iterable, List, Mapping, Tuple

from .base import Agent, AgentAction, AgentContext, UtilityMatrix
from .registry import weight_map


ACTIONS: Tuple[AgentAction, ...] = (
    AgentAction.SELL,
    AgentAction.HOLD,
    AgentAction.BUY_S,
    AgentAction.BUY_M,
    AgentAction.BUY_L,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class Decision:
    action: AgentAction
    confidence: float
    target_weight: float
    feasible_actions: List[AgentAction]
    utilities: UtilityMatrix


def compute_utilities(agents: Iterable[Agent], context: AgentContext) -> UtilityMatrix:
    utilities: UtilityMatrix = {}
    for action in ACTIONS:
        utilities[action] = {}
        for agent in agents:
            score = _clamp(agent.score(context, action))
            utilities[action][agent.name] = score
    return utilities


def feasible_actions(agents: Iterable[Agent], context: AgentContext) -> List[AgentAction]:
    feas: List[AgentAction] = []
    for action in ACTIONS:
        if all(agent.feasible(context, action) for agent in agents):
            feas.append(action)
    return feas


def nash_bargain(utilities: UtilityMatrix, weights: Mapping[str, float], disagreement: Mapping[str, float]) -> Tuple[AgentAction, float]:
    best_action = AgentAction.HOLD
    best_score = float("-inf")
    for action, agent_scores in utilities.items():
        if action not in utilities:
            continue
        log_product = 0.0
        valid = True
        for agent_name, score in agent_scores.items():
            w = weights.get(agent_name, 0.0)
            if w == 0:
                continue
            gap = score - disagreement.get(agent_name, 0.0)
            if gap <= 0:
                valid = False
                break
            log_product += w * log(gap)
        if not valid:
            continue
        if log_product > best_score:
            best_score = log_product
            best_action = action
    if best_score == float("-inf"):
        return AgentAction.HOLD, 0.0
    confidence = _aggregate_confidence(utilities[best_action], weights)
    return best_action, confidence


def vote(utilities: UtilityMatrix, weights: Mapping[str, float]) -> Tuple[AgentAction, float]:
    scores: Dict[AgentAction, float] = {}
    for action, agent_scores in utilities.items():
        scores[action] = sum(weights.get(agent, 0.0) * score for agent, score in agent_scores.items())
    best_action = max(scores, key=scores.get)
    confidence = _aggregate_confidence(utilities[best_action], weights)
    return best_action, confidence


def _aggregate_confidence(agent_scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    weighted = sum(weights.get(agent, 0.0) * score for agent, score in agent_scores.items())
    return weighted / total


def target_weight_for_action(action: AgentAction) -> float:
    mapping = {
        AgentAction.SELL: -1.0,
        AgentAction.HOLD: 0.0,
        AgentAction.BUY_S: 0.01,
        AgentAction.BUY_M: 0.02,
        AgentAction.BUY_L: 0.03,
    }
    return mapping[action]


def decide(context: AgentContext, agents: Iterable[Agent], weights: Mapping[str, float], method: str = "nash") -> Decision:
    agent_list = list(agents)
    norm_weights = weight_map(dict(weights))
    utilities = compute_utilities(agent_list, context)
    feas_actions = feasible_actions(agent_list, context)
    if not feas_actions:
        return Decision(AgentAction.HOLD, 0.0, 0.0, [], utilities)

    filtered_utilities = {action: utilities[action] for action in feas_actions}
    hold_scores = utilities.get(AgentAction.HOLD, {})

    if method == "vote":
        action, confidence = vote(filtered_utilities, norm_weights)
    else:
        action, confidence = nash_bargain(filtered_utilities, norm_weights, hold_scores)
        if action not in feas_actions:
            action, confidence = vote(filtered_utilities, norm_weights)

    weight = target_weight_for_action(action)
    return Decision(action, confidence, weight, feas_actions, utilities)
