"""Protocols and data structures for multi-round multi-agent games."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class GameStructure(str, Enum):
    """Supported multi-agent game topologies."""

    REPEATED = "repeated"
    SIGNALING = "signaling"
    BAYESIAN = "bayesian"
    CUSTOM = "custom"


class DialogueRole(str, Enum):
    """Roles participating in the negotiation agenda."""

    HOST = "host"
    PREDICTION = "prediction"
    RISK = "risk"
    EXECUTION = "execution"
    OBSERVER = "observer"


class MessageType(str, Enum):
    """High-level classification of dialogue intents."""

    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    COUNTER = "counter"
    DECISION = "decision"
    DIRECTIVE = "directive"
    META = "meta"


@dataclass
class DialogueMessage:
    """Single utterance in the multi-round dialogue."""

    sender: str
    role: DialogueRole
    message_type: MessageType
    content: str
    confidence: float = 0.0
    references: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "role": self.role.value,
            "message_type": self.message_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "references": list(self.references),
            "timestamp": self.timestamp,
            "annotations": dict(self.annotations),
        }


@dataclass
class BeliefSnapshot:
    """Belief state emitted by an agent before or after revision."""

    agent: str
    role: DialogueRole
    belief: Dict[str, Any]
    confidence: float
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "role": self.role.value,
            "belief": dict(self.belief),
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass
class BeliefRevision:
    """Tracks belief updates triggered during a round."""

    before: BeliefSnapshot
    after: BeliefSnapshot
    justification: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "justification": self.justification,
        }


@dataclass
class RoundSummary:
    """Aggregated view of a single negotiation round."""

    index: int
    agenda: str
    structure: GameStructure
    resolved: bool
    outcome: Optional[str] = None
    messages: List[DialogueMessage] = field(default_factory=list)
    revisions: List[BeliefRevision] = field(default_factory=list)
    constraints_triggered: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "agenda": self.agenda,
            "structure": self.structure.value,
            "resolved": self.resolved,
            "outcome": self.outcome,
            "messages": [message.to_dict() for message in self.messages],
            "revisions": [revision.to_dict() for revision in self.revisions],
            "constraints_triggered": list(self.constraints_triggered),
            "notes": dict(self.notes),
        }


@dataclass
class DialogueTrace:
    """Ordered collection of round summaries for auditing."""

    session_id: str
    ts_code: str
    trade_date: str
    rounds: List[RoundSummary] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "ts_code": self.ts_code,
            "trade_date": self.trade_date,
            "rounds": [summary.to_dict() for summary in self.rounds],
            "metadata": dict(self.metadata),
        }


class GameProtocol(Protocol):
    """Extension point for hosting multi-round agent negotiations."""

    def bootstrap(self, trace: DialogueTrace) -> None:
        """Prepare the protocol-specific state before rounds begin."""

    def start_round(self, trace: DialogueTrace, agenda: str, structure: GameStructure) -> RoundSummary:
        """Open a new round with the given agenda and structure descriptor."""

    def handle_message(self, summary: RoundSummary, message: DialogueMessage) -> None:
        """Process a single dialogue message emitted by an agent."""

    def apply_revision(self, summary: RoundSummary, revision: BeliefRevision) -> None:
        """Register a belief revision triggered by debate or new evidence."""

    def finalize_round(self, summary: RoundSummary) -> None:
        """Mark the round as resolved and perform protocol-specific bookkeeping."""

    def close(self, trace: DialogueTrace) -> None:
        """Finish the negotiation session and emit protocol artifacts."""


class ProtocolHost(GameProtocol):
    """Base implementation for agenda-driven negotiation protocols."""

    def __init__(self) -> None:
        self._current_round: Optional[RoundSummary] = None
        self._trace: Optional[DialogueTrace] = None
        self._round_index: int = 0

    def bootstrap(self, trace: DialogueTrace) -> None:
        trace.metadata.setdefault("host_started", True)
        self._trace = trace
        self._round_index = len(trace.rounds)

    def bootstrap_trace(
        self,
        *,
        session_id: str,
        ts_code: str,
        trade_date: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DialogueTrace:
        trace = DialogueTrace(
            session_id=session_id,
            ts_code=ts_code,
            trade_date=trade_date,
            metadata=dict(metadata or {}),
        )
        self.bootstrap(trace)
        return trace

    def start_round(
        self,
        trace: DialogueTrace,
        agenda: str,
        structure: GameStructure,
    ) -> RoundSummary:
        index = self._round_index
        summary = RoundSummary(
            index=index,
            agenda=agenda,
            structure=structure,
            resolved=False,
        )
        trace.rounds.append(summary)
        self._current_round = summary
        self._round_index += 1
        return summary

    def handle_message(self, summary: RoundSummary, message: DialogueMessage) -> None:
        summary.messages.append(message)

    def apply_revision(self, summary: RoundSummary, revision: BeliefRevision) -> None:
        summary.revisions.append(revision)

    def finalize_round(self, summary: RoundSummary) -> None:
        summary.resolved = True
        summary.notes.setdefault("message_count", len(summary.messages))
        self._current_round = None

    def close(self, trace: DialogueTrace) -> None:
        trace.metadata.setdefault("host_finished", True)
        self._trace = trace

    def current_round(self) -> Optional[RoundSummary]:
        return self._current_round

    @property
    def trace(self) -> Optional[DialogueTrace]:
        return self._trace

    def ensure_round(
        self,
        trace: DialogueTrace,
        agenda: str,
        structure: GameStructure,
    ) -> RoundSummary:
        if self._current_round and not self._current_round.resolved:
            return self._current_round
        return self.start_round(trace, agenda, structure)


def round_to_dict(summary: RoundSummary) -> Dict[str, Any]:
    """Serialize a round summary for persistence layers."""

    return summary.to_dict()


__all__ = [
    "GameStructure",
    "DialogueRole",
    "MessageType",
    "DialogueMessage",
    "BeliefSnapshot",
    "BeliefRevision",
    "RoundSummary",
    "DialogueTrace",
    "GameProtocol",
    "ProtocolHost",
    "round_to_dict",
]
