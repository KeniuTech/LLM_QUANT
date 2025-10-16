# 业务逻辑体检报告

本报告梳理当前端到端业务链路，并标注出影响可维护性与扩展性的关键风险点，供后续重构排期参考。

## 端到端链路速览
- **数据采集与健康巡检**：命令行入口 `scripts/run_ingestion_job.py` 通过编排层 `app/ingest/tushare.py`，调用 `app/ingest/api_client.py` 与 `app/ingest/coverage.py` 完成 TuShare 拉数、数据补齐与指标巡检。
- **数据接入与覆盖治理**：`DataBroker` (`app/utils/data_access.py`) 负责字段解析、缓存、派生指标，自动补数由可注入的 `coverage_runner` 承担；批量快照能力由 `FeatureSnapshotService` (`app/utils/feature_snapshots.py`) 暴露给上层。
- **因子与特征加工**：`compute_factors` 系列 (`app/features/factors.py`) 借助批量快照与分批校验，输出持久化特征供代理与回测消费。
- **多智能体决策**：`DecisionWorkflow` (`app/agents/game.py`) 将议程控制、部门投票、风险审查、执行总结拆分为可维护的阶段，驱动规则代理与部门 LLM 协同。
- **回测与调参与强化学习**：`BacktestEngine.load_market_data` (`app/backtest/engine.py`) 使用批量快照聚合特征，`DecisionEnv` (`app/backtest/decision_env.py`) 暴露 RL 行为接口。
- **可视化与运营面板**：Streamlit 入口 `app/ui/streamlit_app.py:14-120` 触发数据库初始化、自动补数与多页可视化，消费上述链路的产物。

## 主要发现
### 1. 数据采集模块拆分完成但仍需扩展
- 采集 orchestrator 已收敛在 `app/ingest/tushare.py`，API 调用与覆盖校验分别由 `app/ingest/api_client.py`、`app/ingest/coverage.py` 承担，后续可考虑将因子计算改为显式队列任务。
- `run_ingestion` 通过 `post_tasks` 钩子触发默认因子回填，方便引入异步或多阶段处理策略。

### 2. 数据访问层职责下沉但仍偏厚重
- `DataBroker` 引入可注入的 `coverage_runner` 与批量缓存接口，不过派生指标、行业分析仍集中在同一类，可进一步拆分至子组件。
- 自动补数与覆盖统计改为显式依赖 `app/ingest/coverage.py`，消除了懒加载带来的环状依赖风险。

### 3. 因子流水线新增批量快照
- `FeatureSnapshotService` 批量预取最新字段，`compute_factors` 改为按批拼装特征并共用缓存，减少了重复 SQL。
- 校验与进度汇报依旧集中在 `_compute_batch_factors`，后续可继续剥离统计与写库逻辑以优化测试粒度。

### 4. 多智能体决策流程模块化
- `DecisionWorkflow` 将部门投票、风险审查、执行总结拆分为独立方法，便于插桩和单元测试。
- 部门代理仍直接访问 `DataBroker`，后续可对接 `FeatureSnapshotService` 或数据域策略，统一数据获取边界。

### 5. 回测链路复用批量特征
- `BacktestEngine.load_market_data` 与因子流水线共用快照服务，避免重复的最新值查询。
- 强化学习环境仍按日重新构造 `BacktestEngine`，可在后续迭代中缓存快照或拆分环境状态以进一步加速。

## 优先级建议
1. **完善采集流水线的任务编排**（高优先级）  
   现已拆分 API/覆盖/编排层，建议继续将因子计算与其它后置动作放入独立任务队列，便于并发执行与重试。

2. **解耦 DataBroker 的派生职责**（中高优先级）  
   将行业、情绪、派生指标等逻辑抽出为独立服务，保留 DataBroker 专注于字段解析与缓存；同步补充更细粒度的单元测试。

3. **推广特征快照到部门代理**（中优先级）  
   目前因子与回测已复用 `FeatureSnapshotService`，建议在 LLM 部门工具调用中也接入统一快照，降低重复 SQL。

4. **补齐 DecisionWorkflow 测试与监控**（中优先级）  
   为 `DecisionWorkflow` 各阶段编写单元/集成测试，并将风险评审与信念修正暴露在监控面板中，便于审计。

5. **建立性能与回归基线**（低优先级）  
   构造包含快照缓存的基准数据集，度量因子计算和回测的时延，对后续优化提供数据支持。

以上建议可依次推进，亦可按业务优先级穿插执行。
