"""Apply or display the best tuning result for an experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.config import get_config, save_config
from app.utils.tuning import select_best_tuning_result
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply best tuning weights")
    parser.add_argument("experiment_id", help="Experiment identifier")
    parser.add_argument(
        "--metric",
        default="reward",
        help="Metric name for ranking (default: reward)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort metric ascending instead of descending",
    )
    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="Ignore records without weight payload",
    )
    parser.add_argument(
        "--apply-config",
        action="store_true",
        help="Update agent_weights in config with best result weights (fallback to action)",
    )
    return parser


def run_cli(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    best = select_best_tuning_result(
        args.experiment_id,
        metric=args.metric,
        descending=not args.ascending,
        require_weights=args.require_weights,
    )
    if not best:
        LOGGER.error("未找到实验结果 experiment_id=%s", args.experiment_id)
        return 1

    print(json.dumps(best, ensure_ascii=False, indent=2))

    if args.apply_config:
        weights = best.get("weights") or best.get("action")
        if not weights:
            LOGGER.error("最佳结果缺少权重信息，无法更新配置")
            return 2
        cfg = get_config()
        if not cfg.agent_weights:
            LOGGER.warning("配置缺少 agent_weights，初始化默认值")
        cfg.agent_weights.update_from_dict(weights)
        save_config(cfg)
        LOGGER.info("已写入新的 agent_weights 至配置")

    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
