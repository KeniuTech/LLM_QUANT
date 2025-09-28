"""Multi-agent decision game implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .base import Agent, AgentAction, AgentContext, UtilityMatrix
from .departments import DepartmentContext, DepartmentDecision, DepartmentManager
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
    department_decisions: Dict[str, DepartmentDecision] = field(default_factory=dict)
    department_votes: Dict[str, float] = field(default_factory=dict)
    requires_review: bool = False


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


def decide(
    context: AgentContext,
    agents: Iterable[Agent],
    weights: Mapping[str, float],
    method: str = "nash",
    department_manager: Optional[DepartmentManager] = None,
    department_context: Optional[DepartmentContext] = None,
) -> Decision:
    agent_list = list(agents)
    utilities = compute_utilities(agent_list, context)
    feas_actions = feasible_actions(agent_list, context)
    if not feas_actions:
        return Decision(
            action=AgentAction.HOLD,
            confidence=0.0,
            target_weight=0.0,
            feasible_actions=[],
            utilities=utilities,
        )

    raw_weights = dict(weights)
    department_decisions: Dict[str, DepartmentDecision] = {}
    department_votes: Dict[str, float] = {}

    if department_manager:
        dept_context = department_context
        if dept_context is None:
            dept_context = DepartmentContext(
                ts_code=context.ts_code,
                trade_date=context.trade_date,
                features=dict(context.features),
                market_snapshot=dict(getattr(context, "market_snapshot", {}) or {}),
                raw=dict(getattr(context, "raw", {}) or {}),
            )
        department_decisions = department_manager.evaluate(dept_context)
        for code, decision in department_decisions.items():
            agent_key = f"dept_{code}"
            dept_agent = department_manager.agents.get(code)
            weight = dept_agent.settings.weight if dept_agent else 1.0
            raw_weights[agent_key] = weight
            scores = _department_scores(decision)
            for action in ACTIONS:
                utilities.setdefault(action, {})[agent_key] = scores[action]
            bucket = _department_vote_bucket(decision.action)
            if bucket:
                department_votes[bucket] = department_votes.get(bucket, 0.0) + weight * decision.confidence

    filtered_utilities = {action: utilities[action] for action in feas_actions}
    hold_scores = utilities.get(AgentAction.HOLD, {})
    norm_weights = weight_map(raw_weights)

    if method == "vote":
        action, confidence = vote(filtered_utilities, norm_weights)
    else:
        action, confidence = nash_bargain(filtered_utilities, norm_weights, hold_scores)
        if action not in feas_actions:
            action, confidence = vote(filtered_utilities, norm_weights)

    weight = target_weight_for_action(action)
    requires_review = _department_conflict_flag(department_votes)
    return Decision(
        action=action,
        confidence=confidence,
        target_weight=weight,
        feasible_actions=feas_actions,
        utilities=utilities,
        department_decisions=department_decisions,
        department_votes=department_votes,
        requires_review=requires_review,
    )


def _department_scores(decision: DepartmentDecision) -> Dict[AgentAction, float]:
    conf = _clamp(decision.confidence)
    scores: Dict[AgentAction, float] = {action: 0.2 for action in ACTIONS}
    if decision.action is AgentAction.SELL:
        scores[AgentAction.SELL] = 0.7 + 0.3 * conf
        scores[AgentAction.HOLD] = 0.4 * (1 - conf)
        scores[AgentAction.BUY_S] = 0.2 * (1 - conf)
        scores[AgentAction.BUY_M] = 0.15 * (1 - conf)
        scores[AgentAction.BUY_L] = 0.1 * (1 - conf)
    elif decision.action in {AgentAction.BUY_S, AgentAction.BUY_M, AgentAction.BUY_L}:
        for action in (AgentAction.BUY_S, AgentAction.BUY_M, AgentAction.BUY_L):
            if action is decision.action:
                scores[action] = 0.6 + 0.4 * conf
            else:
                scores[action] = 0.3 + 0.3 * conf
        scores[AgentAction.HOLD] = 0.3 * (1 - conf) + 0.25
        scores[AgentAction.SELL] = 0.15 * (1 - conf)
    else:  # HOLD 或未知
        scores[AgentAction.HOLD] = 0.6 + 0.4 * conf
        scores[AgentAction.SELL] = 0.3 * (1 - conf)
        scores[AgentAction.BUY_S] = 0.3 * (1 - conf)
        scores[AgentAction.BUY_M] = 0.3 * (1 - conf)
        scores[AgentAction.BUY_L] = 0.3 * (1 - conf)
    return {action: _clamp(score) for action, score in scores.items()}


def _department_vote_bucket(action: AgentAction) -> str:
    if action is AgentAction.SELL:
        return "sell"
    if action in {AgentAction.BUY_S, AgentAction.BUY_M, AgentAction.BUY_L}:
        return "buy"
    if action is AgentAction.HOLD:
        return "hold"
    return ""


def _department_conflict_flag(votes: Mapping[str, float]) -> bool:
    if not votes:
        return False
    total = sum(votes.values())
    if total <= 0:
        return True
    top = max(votes.values())
    if top < total * 0.45:
        return True
    if len(votes) > 1:
        sorted_votes = sorted(votes.values(), reverse=True)
        if len(sorted_votes) >= 2 and (sorted_votes[0] - sorted_votes[1]) < total * 0.1:
            return True
    return False
