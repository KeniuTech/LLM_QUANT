# 变更记录

## 2025-09-30

- **BacktestEngine 风险闭环强化**
  - 调整撮合逻辑，统一考虑仓位上限、换手约束、滑点与手续费。
  - 新增 `bt_risk_events` 表及落库链路，回测报告输出风险事件统计。
  - 效果：回测结果可复盘风险拦截与执行成本，为 LLM 策略调优提供可靠反馈。

- **DecisionEnv 风险感知奖励**
  - Episode 观测新增换手、风险事件等字段，默认奖励将回撤、风险与换手纳入惩罚项。
  - 效果：强化学习/ Bandit 调参能够权衡收益与风险，符合多智能体自治决策目标。

- **Bandit 调参与权重回收工具**
  - 新增 `EpsilonGreedyBandit` 与 `run_bandit_optimization.py`，自动记录调参结果。
  - 提供 `apply_best_weights.py` 和 `select_best_tuning_result()`，支持一键回收最优权重并写入配置。
  - 效果：建立起“调参→记录→回收”的闭环，便于持续优化 LLM 多智能体参数。

- **DataBroker 取数方式优化**
  - `fetch_latest` 改为整行查询后按需取值，避免列缺失导致的异常。
  - 效果：新增因子或字段时无需调整查询逻辑，降低维护成本。
