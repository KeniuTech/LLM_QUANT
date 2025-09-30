"""Run epsilon-greedy bandit tuning on DecisionEnv."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.registry import default_agents
from app.backtest.decision_env import DecisionEnv, ParameterSpec
from app.backtest.engine import BtConfig
from app.backtest.optimizer import BanditConfig, EpsilonGreedyBandit
from app.utils.config import get_config


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _parse_param(text: str) -> ParameterSpec:
    parts = text.split(":")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "parameter format must be name:target:min[:max]"
        )
    name, target, minimum = parts[:3]
    maximum = parts[3] if len(parts) == 4 else "1.0"
    return ParameterSpec(
        name=name,
        target=target,
        minimum=float(minimum),
        maximum=float(maximum),
    )


def _resolve_baseline_weights() -> dict:
    cfg = get_config()
    if cfg.agent_weights:
        return cfg.agent_weights.as_dict()
    return {agent.name: 1.0 for agent in default_agents()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DecisionEnv bandit optimizer")
    parser.add_argument("experiment_id", help="Experiment identifier to log results")
    parser.add_argument("name", help="Backtest config name")
    parser.add_argument("start", type=_parse_date, help="Start date YYYYMMDD")
    parser.add_argument("end", type=_parse_date, help="End date YYYYMMDD")
    parser.add_argument(
        "--universe",
        required=True,
        help="Comma separated ts_codes, e.g. 000001.SZ,000002.SZ",
    )
    parser.add_argument(
        "--param",
        action="append",
        required=True,
        help="Parameter spec name:target:min[:max] (target like agent_weights.A_mom)",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def run_cli(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.end < args.start:
        parser.error("end date must not precede start date")

    specs: List[ParameterSpec] = [_parse_param(item) for item in args.param]
    universe = [token.strip() for token in args.universe.split(",") if token.strip()]
    bt_cfg = BtConfig(
        id=args.experiment_id,
        name=args.name,
        start_date=args.start,
        end_date=args.end,
        universe=universe,
        params={},
    )

    env = DecisionEnv(
        bt_config=bt_cfg,
        parameter_specs=specs,
        baseline_weights=_resolve_baseline_weights(),
    )
    optimizer = EpsilonGreedyBandit(
        env,
        BanditConfig(
            experiment_id=args.experiment_id,
            episodes=args.episodes,
            epsilon=args.epsilon,
            seed=args.seed,
        ),
    )
    summary = optimizer.run()
    best = summary.best_episode
    output = {
        "episodes": len(summary.episodes),
        "average_reward": summary.average_reward,
        "best": {
            "reward": best.reward if best else None,
            "action": best.action if best else None,
            "metrics": (best.metrics and json.dumps(best.metrics.risk_breakdown)) if best else None,
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
