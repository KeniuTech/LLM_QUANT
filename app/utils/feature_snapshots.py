"""Shared feature snapshot helpers built on top of DataBroker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from .data_access import DataBroker


@dataclass
class FeatureSnapshotService:
    """Provide batch-oriented access to latest features for multiple symbols."""

    broker: DataBroker

    def __init__(self, broker: Optional[DataBroker] = None) -> None:
        self.broker = broker or DataBroker()

    def load_latest(
        self,
        trade_date: str,
        fields: Sequence[str],
        ts_codes: Sequence[str],
        *,
        auto_refresh: bool = False,
    ) -> Dict[str, Dict[str, object]]:
        """Fetch a snapshot of feature values for the given universe."""

        if not ts_codes:
            return {}
        return self.broker.fetch_batch_latest(
            list(ts_codes),
            trade_date,
            fields,
            auto_refresh=auto_refresh,
        )

    def load_single(
        self,
        trade_date: str,
        ts_code: str,
        fields: Iterable[str],
        *,
        auto_refresh: bool = False,
    ) -> Mapping[str, object]:
        """Convenience wrapper to reuse the snapshot logic for a single symbol."""

        snapshot = self.load_latest(
            trade_date,
            list(fields),
            [ts_code],
            auto_refresh=auto_refresh,
        )
        return snapshot.get(ts_code, {})


__all__ = ["FeatureSnapshotService"]

