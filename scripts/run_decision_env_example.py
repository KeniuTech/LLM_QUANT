"""Quick example of using DecisionEnv for weight tuning experiments."""
from __future__ import annotations

import json
from datetime import date, timedelta

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.decision_env import DecisionEnv, ParameterSpec
from app.backtest.engine import BtConfig
from app.agents.registry import default_agents
from app.utils.config import get_config


def main() -> None:
    cfg = get_config()
    agents = default_agents()
    baseline_weights = {agent.name: cfg.agent_weights.as_dict().get(agent.name, 1.0) for agent in agents}

    today = date.today()
    bt_cfg = BtConfig(
        id="decision_env_example",
        name="Decision Env Demo",
        start_date=today - timedelta(days=60),
        end_date=today,
        universe=["000001.SZ"],
        params={},
        method=cfg.decision_method,
    )

    specs = [
        ParameterSpec(name="momentum_weight", target="agent_weights.A_mom", minimum=0.1, maximum=0.6),
        ParameterSpec(name="value_weight", target="agent_weights.A_val", minimum=0.1, maximum=0.4),
    ]

    env = DecisionEnv(bt_config=bt_cfg, parameter_specs=specs, baseline_weights=baseline_weights)
    env.reset()
    observation, reward, done, info = env.step([0.5, 0.2])

    print("Observation:", json.dumps(observation, ensure_ascii=False, indent=2))
    print("Reward:", reward)
    print("Done:", done)
    print("Weights:", json.dumps(info.get("weights", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
