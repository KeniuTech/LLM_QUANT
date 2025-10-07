"""Multi-agent decision game implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .base import Agent, AgentAction, AgentContext, UtilityMatrix
from .departments import DepartmentContext, DepartmentDecision, DepartmentManager
from .registry import weight_map
from .risk import RiskAgent, RiskRecommendation
from .protocols import (
    DialogueMessage,
    DialogueRole,
    GameStructure,
    MessageType,
    ProtocolHost,
    RoundSummary,
)


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
class BeliefUpdate:
    belief: Dict[str, object]
    rationale: Optional[str] = None


@dataclass
class RiskAssessment:
    status: str
    reason: str
    recommended_action: Optional[AgentAction] = None
    notes: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "status": self.status,
            "reason": self.reason,
            "notes": dict(self.notes),
        }
        if self.recommended_action is not None:
            payload["recommended_action"] = self.recommended_action.value
        return payload


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
    rounds: List[RoundSummary] = field(default_factory=list)
    risk_assessment: Optional[RiskAssessment] = None
    belief_updates: Dict[str, BeliefUpdate] = field(default_factory=dict)


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
    host = ProtocolHost()
    host_trace = host.bootstrap_trace(
        session_id=f"{context.ts_code}:{context.trade_date}",
        ts_code=context.ts_code,
        trade_date=context.trade_date,
    )
    department_round: Optional[RoundSummary] = None
    risk_round: Optional[RoundSummary] = None
    execution_round: Optional[RoundSummary] = None
    belief_updates: Dict[str, BeliefUpdate] = {}

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
        if department_decisions:
            department_round = host.start_round(
                host_trace,
                agenda="department_consensus",
                structure=GameStructure.REPEATED,
            )
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
            if department_round:
                message = _department_message(code, decision)
                host.handle_message(department_round, message)
                belief_updates[code] = BeliefUpdate(
                    belief={
                        "action": decision.action.value,
                        "confidence": decision.confidence,
                        "signals": decision.signals,
                    },
                    rationale=decision.summary,
                )

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
    conflict_flag = _department_conflict_flag(department_votes)

    risk_agent = _find_risk_agent(agent_list)
    risk_assessment = _evaluate_risk(
        context,
        action,
        department_votes,
        conflict_flag,
        risk_agent,
    )
    requires_review = risk_assessment.status != "ok"

    if department_round:
        department_round.notes.setdefault("department_votes", dict(department_votes))
        department_round.outcome = action.value
        host.finalize_round(department_round)

    if requires_review:
        risk_round = host.ensure_round(
            host_trace,
            agenda="risk_review",
            structure=GameStructure.CUSTOM,
        )
        review_message = DialogueMessage(
            sender="risk_guard",
            role=DialogueRole.RISK,
            message_type=MessageType.COUNTER,
            content=_risk_review_message(risk_assessment.reason),
            confidence=1.0,
            references=list(department_votes.keys()),
            annotations={
                "department_votes": dict(department_votes),
                "risk_reason": risk_assessment.reason,
                "recommended_action": (
                    risk_assessment.recommended_action.value
                    if risk_assessment.recommended_action
                    else None
                ),
                "notes": dict(risk_assessment.notes),
            },
        )
        host.handle_message(risk_round, review_message)
        risk_round.notes.setdefault("status", risk_assessment.status)
        risk_round.notes.setdefault("reason", risk_assessment.reason)
        if risk_assessment.recommended_action:
            risk_round.notes.setdefault(
                "recommended_action",
                risk_assessment.recommended_action.value,
            )
        risk_round.outcome = "REVIEW"
        host.finalize_round(risk_round)
        belief_updates["risk_guard"] = BeliefUpdate(
            belief={
                "status": risk_assessment.status,
                "reason": risk_assessment.reason,
                "recommended_action": (
                    risk_assessment.recommended_action.value
                    if risk_assessment.recommended_action
                    else None
                ),
            },
        )
    execution_round = host.ensure_round(
        host_trace,
        agenda="execution_summary",
        structure=GameStructure.REPEATED,
    )
    exec_action = action
    exec_weight = weight
    exec_status = "normal"
    if requires_review and risk_assessment.recommended_action:
        exec_action = risk_assessment.recommended_action
        exec_status = "risk_adjusted"
        exec_weight = target_weight_for_action(exec_action)
    execution_message = DialogueMessage(
        sender="execution_engine",
        role=DialogueRole.EXECUTION,
        message_type=MessageType.DIRECTIVE,
        content=f"执行操作 {exec_action.value}",
        confidence=1.0,
        annotations={
            "target_weight": exec_weight,
            "requires_review": requires_review,
            "execution_status": exec_status,
        },
    )
    host.handle_message(execution_round, execution_message)
    execution_round.outcome = exec_action.value
    execution_round.notes.setdefault("execution_status", exec_status)
    if exec_action is not action:
        execution_round.notes.setdefault("original_action", action.value)
    belief_updates["execution"] = BeliefUpdate(
        belief={
            "execution_status": exec_status,
            "action": exec_action.value,
            "target_weight": exec_weight,
        },
    )
    host.finalize_round(execution_round)
    host.close(host_trace)
    rounds = host_trace.rounds if host_trace.rounds else _build_round_summaries(
        department_decisions,
        action,
        department_votes,
    )
    return Decision(
        action=action,
        confidence=confidence,
        target_weight=weight,
        feasible_actions=feas_actions,
        utilities=utilities,
        department_decisions=department_decisions,
        department_votes=department_votes,
        requires_review=requires_review,
        rounds=rounds,
        risk_assessment=risk_assessment,
        belief_updates=belief_updates,
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


def _department_message(code: str, decision: DepartmentDecision) -> DialogueMessage:
    content = decision.summary or decision.raw_response or decision.action.value
    references = decision.signals or []
    annotations: Dict[str, object] = {
        "risks": decision.risks,
        "supplements": decision.supplements,
    }
    if decision.dialogue:
        annotations["dialogue"] = decision.dialogue
    if decision.telemetry:
        annotations["telemetry"] = decision.telemetry
    return DialogueMessage(
        sender=code,
        role=DialogueRole.PREDICTION,
        message_type=MessageType.DECISION,
        content=content,
        confidence=decision.confidence,
        references=references,
        annotations=annotations,
    )


def _evaluate_risk(
    context: AgentContext,
    action: AgentAction,
    department_votes: Mapping[str, float],
    conflict_flag: bool,
    risk_agent: Optional[RiskAgent],
) -> RiskAssessment:
    external_alerts = []
    if getattr(context, "raw", None):
        alerts = context.raw.get("risk_alerts", [])
        if alerts:
            external_alerts = list(alerts)

    if risk_agent:
        recommendation = risk_agent.assess(context, action, conflict_flag)
        notes = dict(recommendation.notes)
        notes.setdefault("department_votes", dict(department_votes))
        if external_alerts:
            notes.setdefault("external_alerts", external_alerts)
            if recommendation.status == "ok":
                recommendation = RiskRecommendation(
                    status="pending_review",
                    reason="external_alert",
                    recommended_action=recommendation.recommended_action or AgentAction.HOLD,
                    notes=notes,
                )
            else:
                recommendation.notes = notes
        return RiskAssessment(
            status=recommendation.status,
            reason=recommendation.reason,
            recommended_action=recommendation.recommended_action,
            notes=recommendation.notes,
        )

    notes: Dict[str, object] = {
        "conflict": conflict_flag,
        "department_votes": dict(department_votes),
    }
    if external_alerts:
        notes["external_alerts"] = external_alerts
        return RiskAssessment(
            status="pending_review",
            reason="external_alert",
            recommended_action=AgentAction.HOLD,
            notes=notes,
        )
    if conflict_flag:
        return RiskAssessment(
            status="pending_review",
            reason="conflict_threshold",
            notes=notes,
        )
    return RiskAssessment(status="ok", reason="clear", notes=notes)


def _find_risk_agent(agents: Iterable[Agent]) -> Optional[RiskAgent]:
    for agent in agents:
        if isinstance(agent, RiskAgent):
            return agent
    return None


def _risk_review_message(reason: str) -> str:
    mapping = {
        "conflict_threshold": "部门意见分歧，触发风险复核",
        "suspended": "标的停牌，需冻结执行",
        "limit_up": "标的涨停，执行需调整",
        "position_limit": "仓位限制已触发，需调整目标",
        "risk_penalty_extreme": "风险评分极高，建议暂停加仓",
        "risk_penalty_high": "风险评分偏高，建议复核",
        "external_alert": "外部风险告警触发复核",
    }
    return mapping.get(reason, "触发风险复核，需人工确认")


def _build_round_summaries(
    department_decisions: Mapping[str, DepartmentDecision],
    final_action: AgentAction,
    department_votes: Mapping[str, float],
) -> List[RoundSummary]:
    if not department_decisions:
        return []
    messages: List[DialogueMessage] = []
    for code, decision in department_decisions.items():
        content = decision.summary or decision.raw_response or decision.action.value
        references = decision.signals or []
        annotations: Dict[str, object] = {
            "risks": decision.risks,
            "supplements": decision.supplements,
        }
        if decision.dialogue:
            annotations["dialogue"] = decision.dialogue
        if decision.telemetry:
            annotations["telemetry"] = decision.telemetry
        message = DialogueMessage(
            sender=code,
            role=DialogueRole.PREDICTION,
            message_type=MessageType.DECISION,
            content=content,
            confidence=decision.confidence,
            references=references,
            annotations=annotations,
        )
        messages.append(message)
    notes: Dict[str, object] = {
        "department_votes": dict(department_votes),
    }
    summary = RoundSummary(
        index=0,
        agenda="department_consensus",
        structure=GameStructure.REPEATED,
        resolved=True,
        outcome=final_action.value,
        messages=messages,
        notes=notes,
    )
    return [summary]
