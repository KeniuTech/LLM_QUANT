"""Multi-agent decision game implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .base import Agent, AgentAction, AgentContext, UtilityMatrix
from .departments import DepartmentContext, DepartmentDecision, DepartmentManager
from .registry import weight_map
from .beliefs import BeliefRevisionResult, revise_beliefs
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
    belief_revision: Optional[BeliefRevisionResult] = None


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


class DecisionWorkflow:
    def __init__(
        self,
        context: AgentContext,
        agents: Iterable[Agent],
        weights: Mapping[str, float],
        method: str,
        department_manager: Optional[DepartmentManager],
        department_context: Optional[DepartmentContext],
    ) -> None:
        self.context = context
        self.agent_list = list(agents)
        self.method = method
        self.department_manager = department_manager
        self.department_context = department_context
        self.utilities = compute_utilities(self.agent_list, context)
        self.feasible_actions = feasible_actions(self.agent_list, context)
        self.raw_weights = dict(weights)
        self.department_decisions: Dict[str, DepartmentDecision] = {}
        self.department_votes: Dict[str, float] = {}
        self.host = ProtocolHost()
        self.host_trace = self.host.bootstrap_trace(
            session_id=f"{context.ts_code}:{context.trade_date}",
            ts_code=context.ts_code,
            trade_date=context.trade_date,
        )
        self.briefing_round = self.host.start_round(
            self.host_trace,
            agenda="situation_briefing",
            structure=GameStructure.SIGNALING,
        )
        self.host.handle_message(self.briefing_round, _host_briefing_message(context))
        self.host.finalize_round(self.briefing_round)
        self.department_round: Optional[RoundSummary] = None
        self.risk_round: Optional[RoundSummary] = None
        self.execution_round: Optional[RoundSummary] = None
        self.belief_updates: Dict[str, BeliefUpdate] = {}
        self.prediction_round: Optional[RoundSummary] = None
        self.norm_weights: Dict[str, float] = {}
        self.filtered_utilities: Dict[AgentAction, Dict[str, float]] = {}
        self.belief_revision: Optional[BeliefRevisionResult] = None

    def run(self) -> Decision:
        if not self.feasible_actions:
            return Decision(
                action=AgentAction.HOLD,
                confidence=0.0,
                target_weight=0.0,
                feasible_actions=[],
                utilities=self.utilities,
            )

        self._evaluate_departments()
        action, confidence = self._select_action()
        risk_assessment = self._apply_risk(action)
        exec_action = self._finalize_execution(action, risk_assessment)
        self._finalize_conflicts(exec_action)
        rounds = self.host_trace.rounds or _build_round_summaries(
            self.department_decisions,
            action,
            self.department_votes,
        )

        return Decision(
            action=action,
            confidence=confidence,
            target_weight=target_weight_for_action(action),
            feasible_actions=self.feasible_actions,
            utilities=self.utilities,
            department_decisions=self.department_decisions,
            department_votes=self.department_votes,
            requires_review=risk_assessment.status != "ok",
            rounds=rounds,
            risk_assessment=risk_assessment,
            belief_updates=self.belief_updates,
            belief_revision=self.belief_revision,
        )

    def _evaluate_departments(self) -> None:
        if not self.department_manager:
            return

        dept_context = self.department_context or DepartmentContext(
            ts_code=self.context.ts_code,
            trade_date=self.context.trade_date,
            features=dict(self.context.features),
            market_snapshot=dict(getattr(self.context, "market_snapshot", {}) or {}),
            raw=dict(getattr(self.context, "raw", {}) or {}),
        )
        self.department_decisions = self.department_manager.evaluate(dept_context)
        if self.department_decisions:
            self.department_round = self.host.start_round(
                self.host_trace,
                agenda="department_consensus",
                structure=GameStructure.REPEATED,
            )
        for code, decision in self.department_decisions.items():
            agent_key = f"dept_{code}"
            dept_agent = self.department_manager.agents.get(code)
            weight = dept_agent.settings.weight if dept_agent else 1.0
            self.raw_weights[agent_key] = weight
            scores = _department_scores(decision)
            for action in ACTIONS:
                self.utilities.setdefault(action, {})[agent_key] = scores[action]
            bucket = _department_vote_bucket(decision.action)
            if bucket:
                self.department_votes[bucket] = self.department_votes.get(bucket, 0.0) + weight * decision.confidence
            if self.department_round:
                message = _department_message(code, decision)
                self.host.handle_message(self.department_round, message)
                self.belief_updates[code] = BeliefUpdate(
                    belief={
                        "action": decision.action.value,
                        "confidence": decision.confidence,
                        "signals": decision.signals,
                    },
                    rationale=decision.summary,
                )

    def _select_action(self) -> Tuple[AgentAction, float]:
        self.filtered_utilities = {action: self.utilities[action] for action in self.feasible_actions}
        hold_scores = self.utilities.get(AgentAction.HOLD, {})
        self.norm_weights = weight_map(self.raw_weights)
        self.prediction_round = self.host.start_round(
            self.host_trace,
            agenda="prediction_alignment",
            structure=GameStructure.REPEATED,
        )
        prediction_message, prediction_summary = _prediction_summary_message(self.filtered_utilities, self.norm_weights)
        self.host.handle_message(self.prediction_round, prediction_message)
        self.host.finalize_round(self.prediction_round)
        if prediction_summary:
            self.belief_updates["prediction_summary"] = BeliefUpdate(
                belief=prediction_summary,
                rationale="Aggregated utilities shared during alignment round.",
            )

        if self.method == "vote":
            return vote(self.filtered_utilities, self.norm_weights)

        action, confidence = nash_bargain(self.filtered_utilities, self.norm_weights, hold_scores)
        if action not in self.feasible_actions:
            return vote(self.filtered_utilities, self.norm_weights)
        return action, confidence

    def _apply_risk(self, action: AgentAction) -> RiskAssessment:
        conflict_flag = _department_conflict_flag(self.department_votes)
        risk_agent = _find_risk_agent(self.agent_list)
        assessment = _evaluate_risk(
            self.context,
            action,
            self.department_votes,
            conflict_flag,
            risk_agent,
        )
        if self.department_round:
            self.department_round.notes.setdefault("department_votes", dict(self.department_votes))
            self.department_round.outcome = action.value
            self.host.finalize_round(self.department_round)

        if assessment.status != "ok":
            self.risk_round = self.host.ensure_round(
                self.host_trace,
                agenda="risk_review",
                structure=GameStructure.CUSTOM,
            )
            review_message = DialogueMessage(
                sender="risk_guard",
                role=DialogueRole.RISK,
                message_type=MessageType.COUNTER,
                content=_risk_review_message(assessment.reason),
                confidence=1.0,
                references=list(self.department_votes.keys()),
                annotations={
                    "department_votes": dict(self.department_votes),
                    "risk_reason": assessment.reason,
                    "recommended_action": (
                        assessment.recommended_action.value
                        if assessment.recommended_action
                        else None
                    ),
                    "notes": dict(assessment.notes),
                },
            )
            self.host.handle_message(self.risk_round, review_message)
            self.risk_round.notes.setdefault("status", assessment.status)
            self.risk_round.notes.setdefault("reason", assessment.reason)
            if assessment.recommended_action:
                self.risk_round.notes.setdefault(
                    "recommended_action",
                    assessment.recommended_action.value,
                )
            self.risk_round.outcome = "REVIEW"
            self.host.finalize_round(self.risk_round)
            self.belief_updates["risk_guard"] = BeliefUpdate(
                belief={
                    "status": assessment.status,
                    "reason": assessment.reason,
                    "recommended_action": (
                        assessment.recommended_action.value
                        if assessment.recommended_action
                        else None
                    ),
                },
            )
        return assessment

    def _finalize_execution(
        self,
        action: AgentAction,
        assessment: RiskAssessment,
    ) -> AgentAction:
        self.execution_round = self.host.ensure_round(
            self.host_trace,
            agenda="execution_summary",
            structure=GameStructure.REPEATED,
        )
        exec_action = action
        exec_weight = target_weight_for_action(action)
        exec_status = "normal"
        requires_review = assessment.status != "ok"
        if requires_review and assessment.recommended_action:
            exec_action = assessment.recommended_action
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
        self.host.handle_message(self.execution_round, execution_message)
        self.execution_round.outcome = exec_action.value
        self.execution_round.notes.setdefault("execution_status", exec_status)
        if exec_action is not action:
            self.execution_round.notes.setdefault("original_action", action.value)
        self.belief_updates["execution"] = BeliefUpdate(
            belief={
                "execution_status": exec_status,
                "action": exec_action.value,
                "target_weight": exec_weight,
            },
        )
        self.host.finalize_round(self.execution_round)
        self.execution_round.notes.setdefault("target_weight", exec_weight)
        return exec_action

    def _finalize_conflicts(self, exec_action: AgentAction) -> None:
        self.host.close(self.host_trace)
        self.belief_revision = revise_beliefs(self.belief_updates, exec_action)
        if self.belief_revision.conflicts:
            conflict_round = self.host.ensure_round(
                self.host_trace,
                agenda="conflict_resolution",
                structure=GameStructure.CUSTOM,
            )
            conflict_message = DialogueMessage(
                sender="protocol_host",
                role=DialogueRole.HOST,
                message_type=MessageType.COUNTER,
                content="检测到关键冲突，需要后续回合复核。",
                annotations={"conflicts": self.belief_revision.conflicts},
            )
            self.host.handle_message(conflict_round, conflict_message)
            conflict_round.notes.setdefault("conflicts", self.belief_revision.conflicts)
            self.host.finalize_round(conflict_round)
        if self.execution_round:
            self.execution_round.notes.setdefault("consensus_action", self.belief_revision.consensus_action.value)
            self.execution_round.notes.setdefault("consensus_confidence", self.belief_revision.consensus_confidence)
            if self.belief_revision.conflicts:
                self.execution_round.notes.setdefault("conflicts", self.belief_revision.conflicts)
            if self.belief_revision.notes:
                self.execution_round.notes.setdefault("belief_notes", self.belief_revision.notes)


def decide(
    context: AgentContext,
    agents: Iterable[Agent],
    weights: Mapping[str, float],
    method: str = "nash",
    department_manager: Optional[DepartmentManager] = None,
    department_context: Optional[DepartmentContext] = None,
) -> Decision:
    workflow = DecisionWorkflow(
        context,
        agents,
        weights,
        method,
        department_manager,
        department_context,
    )
    return workflow.run()


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


def _host_briefing_message(context: AgentContext) -> DialogueMessage:
    features = getattr(context, "features", {}) or {}
    close = features.get("close") or features.get("daily.close")
    pct_chg = features.get("pct_chg") or features.get("daily.pct_chg")
    snapshot = getattr(context, "market_snapshot", {}) or {}
    index_brief = snapshot.get("index_change")
    lines = [
        f"标的 {context.ts_code}",
        f"交易日 {context.trade_date}",
    ]
    if close is not None:
        lines.append(f"最新收盘价：{close}")
    if pct_chg is not None:
        lines.append(f"涨跌幅：{pct_chg}")
    if index_brief:
        lines.append(f"市场概览：{index_brief}")
    content = "；".join(str(line) for line in lines)
    return DialogueMessage(
        sender="protocol_host",
        role=DialogueRole.HOST,
        message_type=MessageType.META,
        content=content,
    )


def _prediction_summary_message(
    utilities: Mapping[AgentAction, Mapping[str, float]],
    weights: Mapping[str, float],
) -> Tuple[DialogueMessage, Dict[str, object]]:
    if not utilities:
        message = DialogueMessage(
            sender="protocol_host",
            role=DialogueRole.PREDICTION,
            message_type=MessageType.META,
            content="暂无可用的部门或代理评分，默认进入 HOLD 讨论。",
        )
        return message, {}
    aggregates: Dict[AgentAction, float] = {}
    for action, agent_scores in utilities.items():
        aggregates[action] = sum(weights.get(agent, 0.0) * score for agent, score in agent_scores.items())
    ranked = sorted(aggregates.items(), key=lambda item: item[1], reverse=True)
    summary_lines = []
    for action, score in ranked[:3]:
        summary_lines.append(f"{action.value}: {score:.3f}")
    content = "预测合意度：" + " ｜ ".join(summary_lines)
    total_score = sum(max(score, 0.0) for _, score in ranked)
    confidence = 0.0
    if total_score > 0 and ranked:
        confidence = max(ranked[0][1], 0.0) / total_score
    annotations = {
        "aggregates": {action.value: score for action, score in aggregates.items()},
    }
    message = DialogueMessage(
        sender="protocol_host",
        role=DialogueRole.PREDICTION,
        message_type=MessageType.META,
        content=content,
        confidence=confidence,
        annotations=annotations,
    )
    summary = {
        "aggregates": {action.value: aggregates[action] for action in ACTIONS if action in aggregates},
        "confidence": confidence,
    }
    return message, summary


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
