"""Belief revision helpers for multi-round negotiation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .base import AgentAction


@dataclass
class BeliefRevisionResult:
    consensus_action: Optional[AgentAction]
    consensus_confidence: float
    conflicts: List[str]
    notes: Dict[str, object]


def revise_beliefs(belief_updates: Dict[str, "BeliefUpdate"], default_action: AgentAction) -> BeliefRevisionResult:
    action_votes: Dict[AgentAction, int] = {}
    reasons: Dict[str, object] = {}
    for agent, update in belief_updates.items():
        belief = getattr(update, "belief", {}) or {}
        action_value = belief.get("action") if isinstance(belief, dict) else None
        try:
            action = AgentAction(action_value) if action_value else None
        except ValueError:
            action = None
        if action:
            action_votes[action] = action_votes.get(action, 0) + 1
        reasons[agent] = belief

    consensus_action = None
    consensus_confidence = 0.0
    conflicts: List[str] = []
    if action_votes:
        total_votes = sum(action_votes.values())
        consensus_action = max(action_votes.items(), key=lambda kv: kv[1])[0]
        consensus_confidence = action_votes[consensus_action] / total_votes if total_votes else 0.0
        if len(action_votes) > 1:
            conflicts = [action.name for action in action_votes.keys() if action is not consensus_action]

    if consensus_action is None:
        consensus_action = default_action

    notes = {
        "votes": {action.value: count for action, count in action_votes.items()},
        "reasons": reasons,
    }
    return BeliefRevisionResult(
        consensus_action=consensus_action,
        consensus_confidence=consensus_confidence,
        conflicts=conflicts,
        notes=notes,
    )


# avoid circular import typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    from .game import BeliefUpdate
