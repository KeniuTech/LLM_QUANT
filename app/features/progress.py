"""Optional progress reporting for factor computation."""
from __future__ import annotations

from typing import Optional, Protocol


class FactorProgressProtocol(Protocol):
    """Protocol describing the optional UI progress handler."""

    def start_calculation(self, total_securities: int, total_batches: int) -> None: ...

    def update_progress(
        self,
        current_securities: int,
        current_batch: int,
        message: str = "",
    ) -> None: ...

    def complete_calculation(self, message: str = "") -> None: ...

    def error_occurred(self, error_message: str) -> None: ...


_progress_handler: Optional[FactorProgressProtocol] = None


def register_progress_handler(progress: FactorProgressProtocol | None) -> None:
    """Register a progress handler (typically provided by the UI layer)."""

    global _progress_handler
    _progress_handler = progress


def get_progress_handler() -> Optional[FactorProgressProtocol]:
    """Return the currently registered progress handler if any."""

    return _progress_handler

