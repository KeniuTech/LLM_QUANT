"""Shared feature snapshot helpers built on top of DataBroker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from .data_access import DataBroker
from .logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "feature_snapshot"}


@dataclass
class FeatureSnapshotService:
    """Provide batch-oriented access to latest features for multiple symbols."""

    broker: DataBroker

    def __init__(self, broker: Optional[DataBroker] = None) -> None:
        self.broker = broker or DataBroker()
        LOGGER.debug(
            "初始化特征快照服务 broker=%s",
            type(self.broker).__name__,
            extra=LOG_EXTRA,
        )

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
            LOGGER.debug(
                "跳过快照加载（标的为空） trade_date=%s",
                trade_date,
                extra=LOG_EXTRA,
            )
            return {}
        field_count = len(fields)
        LOGGER.debug(
            "加载特征快照 trade_date=%s universe=%s fields=%s auto_refresh=%s",
            trade_date,
            len(ts_codes),
            field_count,
            auto_refresh,
            extra=LOG_EXTRA,
        )
        snapshot = self.broker.fetch_batch_latest(
            list(ts_codes),
            trade_date,
            fields,
            auto_refresh=auto_refresh,
        )
        LOGGER.debug(
            "特征快照加载完成 trade_date=%s universe=%s",
            trade_date,
            len(snapshot),
            extra=LOG_EXTRA,
        )
        return snapshot

    def load_single(
        self,
        trade_date: str,
        ts_code: str,
        fields: Iterable[str],
        *,
        auto_refresh: bool = False,
    ) -> Mapping[str, object]:
        """Convenience wrapper to reuse the snapshot logic for a single symbol."""

        field_list = list(fields)
        LOGGER.debug(
            "加载单标的快照 trade_date=%s ts_code=%s fields=%s auto_refresh=%s",
            trade_date,
            ts_code,
            len(field_list),
            auto_refresh,
            extra=LOG_EXTRA,
        )
        snapshot = self.load_latest(
            trade_date,
            field_list,
            [ts_code],
            auto_refresh=auto_refresh,
        )
        result = snapshot.get(ts_code, {})
        if not result:
            LOGGER.debug(
                "单标的快照为空 trade_date=%s ts_code=%s",
                trade_date,
                ts_code,
                extra=LOG_EXTRA,
            )
        return result


__all__ = ["FeatureSnapshotService"]
