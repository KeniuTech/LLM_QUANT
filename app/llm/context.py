"""LLM context management and access control."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class DataAccessConfig:
    """Configuration for data access control."""

    allowed_tables: Set[str]
    max_history_days: int
    max_batch_size: int

    def validate_request(
        self, table: str, start_date: str, end_date: Optional[str] = None
    ) -> List[str]:
        """Validate a data access request."""
        errors = []

        if table not in self.allowed_tables:
            errors.append(f"Table {table} not allowed")

        try:
            start_ts = time.strptime(start_date, "%Y%m%d")
            if end_date:
                end_ts = time.strptime(end_date, "%Y%m%d")
                days = (time.mktime(end_ts) - time.mktime(start_ts)) / (24 * 3600)
                if days > self.max_history_days:
                    errors.append(
                        f"Date range exceeds max {self.max_history_days} days"
                    )
                if days < 0:
                    errors.append("End date before start date")
        except ValueError:
            errors.append("Invalid date format (expected YYYYMMDD)")

        return errors


@dataclass
class ContextConfig:
    """Configuration for context management."""

    max_total_tokens: int = 4000
    max_messages: int = 10
    include_system: bool = True
    include_functions: bool = True


@dataclass
class Message:
    """A message in the conversation context."""

    role: str  # system, user, assistant, function
    content: str
    name: Optional[str] = None  # For function calls/results
    function_call: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format for API calls."""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.function_call:
            msg["function_call"] = self.function_call
        return msg

    @property
    def estimated_tokens(self) -> int:
        """Rough estimate of tokens in message."""
        # Very rough estimate: 1 token ≈ 4 chars
        base = len(self.content) // 4
        if self.function_call:
            base += len(json.dumps(self.function_call)) // 4
        return base


@dataclass
class Context:
    """Manages conversation context with token tracking."""

    messages: List[Message] = field(default_factory=list)
    config: ContextConfig = field(default_factory=ContextConfig)
    _token_count: int = 0

    def add_message(self, message: Message) -> None:
        """Add a message to context, maintaining token limit."""
        # Update token count
        new_tokens = message.estimated_tokens
        while (
            self._token_count + new_tokens > self.config.max_total_tokens
            and self.messages
        ):
            # Remove oldest non-system message if needed
            for i, msg in enumerate(self.messages):
                if msg.role != "system" or len(self.messages) <= 1:
                    removed = self.messages.pop(i)
                    self._token_count -= removed.estimated_tokens
                    break

        # Add new message
        self.messages.append(message)
        self._token_count += new_tokens

        # Trim to max messages if needed
        while len(self.messages) > self.config.max_messages:
            for i, msg in enumerate(self.messages):
                if msg.role != "system" or len(self.messages) <= 1:
                    removed = self.messages.pop(i)
                    self._token_count -= removed.estimated_tokens
                    break

    def get_messages(
        self, include_system: bool = None, include_functions: bool = None
    ) -> List[Dict[str, Any]]:
        """Get messages for API call."""
        if include_system is None:
            include_system = self.config.include_system
        if include_functions is None:
            include_functions = self.config.include_functions

        return [
            msg.to_dict()
            for msg in self.messages
            if (include_system or msg.role != "system")
            and (include_functions or msg.role != "function")
        ]

    def clear(self, keep_system: bool = True) -> None:
        """Clear context, optionally keeping system messages."""
        if keep_system:
            system_msgs = [m for m in self.messages if m.role == "system"]
            self.messages = system_msgs
            self._token_count = sum(m.estimated_tokens for m in system_msgs)
        else:
            self.messages.clear()
            self._token_count = 0


class ContextManager:
    """Global manager for conversation contexts."""

    _contexts: Dict[str, Context] = {}
    _configs: Dict[str, ContextConfig] = {}

    @classmethod
    def create_context(
        cls, context_id: str, config: Optional[ContextConfig] = None
    ) -> Context:
        """Create a new context."""
        if context_id in cls._contexts:
            raise ValueError(f"Context {context_id} already exists")
        context = Context(config=config or ContextConfig())
        cls._contexts[context_id] = context
        return context

    @classmethod
    def get_context(cls, context_id: str) -> Optional[Context]:
        """Get existing context."""
        return cls._contexts.get(context_id)

    @classmethod
    def remove_context(cls, context_id: str) -> None:
        """Remove a context."""
        if context_id in cls._contexts:
            del cls._contexts[context_id]

    @classmethod
    def clear_all(cls) -> None:
        """Clear all contexts."""
        cls._contexts.clear()
