# 多轮博弈决策调用示意

本节概述 `llm_quant` 中多轮博弈执行链路，便于定位关键日志与扩展点。

```
BacktestEngine.simulate_day
 └─ load_market_data
     ├─ DataBroker.fetch_latest
     │   ├─ BrokerQueryEngine.fetch_latest
     │   └─ 缺失字段 → derived_fields/missing_fields (写入 raw/missing_fields)
     ├─ DataBroker.fetch_series (同上)
     └─ assemble feature_map (features / market_snapshot / raw)
 └─ for each symbol → decide (agents.game)
     ├─ compute_utilities / feasible_actions
     ├─ DepartmentManager.evaluate (LLM 部门，可带回 risk/rationale)
     ├─ ProtocolHost (game protocols)
     │   ├─ start_round("department_consensus")
     │   ├─ risk_review (当 conflict / risk assessment 触发)
     │   └─ execution_summary (记录 execution_status)
     ├─ revise_beliefs (beliefs.py) → consensus/conflict
     └─ Decision
         ├─ rounds (RoundSummary 日志)
         ├─ risk_assessment (status/reason/recommended_action)
         ├─ belief_updates / belief_revision (供监控/重播)
         └─ department_votes / utilities
 └─ _apply_portfolio_updates
     ├─ 使用 Decision.risk_assessment 调节执行
     ├─ 执行失败/阻断 → risk_events & alerts.backtest_risk
     └─ executed_trades / nav_series / risk_events → bt_* 表
```

## 关键日志
- `LOG_EXTRA = {"stage": "backtest"}`：缺失字段、派生字段、执行阻断。
- `LOG_EXTRA = {"stage": "data_broker"}`：自动补数触发、查询失败回退。

## 拉通数据
- `app/agents/scopes.py` 维护结构 → 字段映射。
- `Decision.raw` 中 `missing_fields/derived_fields` 可用于缺口诊断。

## 后续建议
1. 将 `belief_revision` 与 `risk_events` 接入监控告警。
2. 结合 `missing_fields` 统计生成数据质量简报。
3. 通过自动化脚本渲染上述流程图/时序图。
