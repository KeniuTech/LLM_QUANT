"""Command-line entrypoint for PPO training on DecisionEnv."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from app.agents.registry import default_agents
from app.backtest.decision_env import DecisionEnv, ParameterSpec
from app.backtest.engine import BtConfig
from app.rl import DecisionEnvAdapter, PPOConfig, train_ppo
from app.ui.shared import default_backtest_range
from app.utils.config import get_config


def _parse_universe(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_env(args: argparse.Namespace) -> DecisionEnvAdapter:
    app_cfg = get_config()
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    universe = _parse_universe(args.universe)
    if not universe:
        raise ValueError("universe must contain at least one ts_code")

    agents = default_agents()
    baseline_weights = app_cfg.agent_weights.as_dict()
    for agent in agents:
        baseline_weights.setdefault(agent.name, 1.0)

    specs: List[ParameterSpec] = []
    for name in sorted(baseline_weights):
        specs.append(
            ParameterSpec(
                name=f"weight_{name}",
                target=f"agent_weights.{name}",
                minimum=args.weight_min,
                maximum=args.weight_max,
            )
        )

    bt_cfg = BtConfig(
        id=args.experiment_id,
        name=f"PPO-{args.experiment_id}",
        start_date=start,
        end_date=end,
        universe=universe,
        params={
            "target": args.target,
            "stop": args.stop,
            "hold_days": args.hold_days,
        },
        method=app_cfg.decision_method,
    )
    env = DecisionEnv(
        bt_config=bt_cfg,
        parameter_specs=specs,
        baseline_weights=baseline_weights,
        disable_departments=args.disable_departments,
    )
    return DecisionEnvAdapter(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO policy on DecisionEnv")
    default_start, default_end = default_backtest_range(window_days=60)
    parser.add_argument("--start-date", default=str(default_start))
    parser.add_argument("--end-date", default=str(default_end))
    parser.add_argument("--universe", default="000001.SZ")
    parser.add_argument("--experiment-id", default=f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--target", type=float, default=0.035)
    parser.add_argument("--stop", type=float, default=-0.015)
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-min", type=float, default=0.0)
    parser.add_argument("--weight-max", type=float, default=1.5)
    parser.add_argument("--disable-departments", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("ppo_training_summary.json"))

    args = parser.parse_args()
    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(",") if x.strip())
    adapter = build_env(args)

    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_sizes=hidden_sizes,
        seed=args.seed,
    )

    summary = train_ppo(adapter, config)
    payload = {
        "timesteps": summary.timesteps,
        "episode_rewards": summary.episode_rewards,
        "episode_lengths": summary.episode_lengths,
        "diagnostics_tail": summary.diagnostics[-10:],
        "observation_keys": adapter.keys(),
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Training finished. Summary written to {args.output}")


if __name__ == "__main__":
    main()
