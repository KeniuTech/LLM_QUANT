"""Render architecture call graph for multi-agent decision flow."""
from __future__ import annotations

from pathlib import Path

DOT_TEMPLATE = """
digraph LLMQuantCallGraph {
    rankdir=LR;
    node [shape=box, style=rounded];

    BacktestEngine -> LoadMarketData;
    LoadMarketData -> DataBrokerFetch;
    DataBrokerFetch -> BrokerQuery [label="db_session"];
    LoadMarketData -> FeatureAssembly;

    BacktestEngine -> Decide;
    Decide -> ProtocolHost;
    ProtocolHost -> DepartmentRound;
    ProtocolHost -> RiskReview;
    ProtocolHost -> ExecutionSummary;
    Decide -> BeliefRevision;

    ExecutionSummary -> ApplyPortfolio;
    ApplyPortfolio -> RiskEvents;
    ApplyPortfolio -> Alerts;
    ApplyPortfolio -> PersistResults;

    PersistResults -> Reports;
    PersistResults -> UI;
}
"""

def render(output: Path) -> None:
    output.write_text(DOT_TEMPLATE.strip() + "\n", encoding="utf-8")


def main() -> None:
    out_file = Path("docs/architecture_call_graph.dot")
    render(out_file)
    print(f"dot file written: {out_file}")


if __name__ == "__main__":
    main()
