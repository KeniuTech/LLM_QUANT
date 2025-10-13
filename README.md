# 多智能体量化投资个人助理

面向 A 股日线级别的多智能体投资研究平台，整合数据采集、因子工程、部门化 LLM 决策、强化学习调参与可视化展示，帮助投研团队快速搭建可落地的策略助手。

## 快速了解

- **端到端闭环**：从行情/新闻抓取、特征存储到多智能体协同决策、回测评估与可视化一体化。
- **多角色博弈**：主持、预测、风险、执行等角色按轮次协同，支持 LLM 与规则代理混合。
- **强化学习调参**：`DecisionEnv` 将回测引擎包装为 RL 环境，已接入 PPO/SAC 等连续动作算法。
- **风险优先**：风险回合可调整最终指令并记录风控证据，提供复核与告警通道。
- **模块化扩展**：数据层、因子库、代理、Prompt、实验脚本均以独立模块维护，便于替换或二次开发。

## 系统构成

- **数据管线**（`app/ingest`, `app/utils/data_access.py`）  
  TuShare/RSS 拉取与限频处理，`DataBroker` 统一提供行情、特征与派生字段，支持健康监控与回退。

- **因子与特征**（`app/features`）  
  `compute_factors()` 负责批量计算与持久化，包含动量、估值、情绪、风险等因子，并预留增量模式与公式校验。

- **多智能体协作**（`app/agents`）  
  规则代理与部门 LLM 通过 `DepartmentManager`、`ProtocolHost` 参与博弈，风险回合可否决或调整仓位。

- **强化学习/优化**（`app/backtest/decision_env.py`, `scripts/train_ppo.py` 等）  
  将参数搜索、权重调节、提示版本选择统一抽象为动作，回测指标写入 `tuning_results` 便于对比。

- **可视化 UI**（`app/ui/streamlit_app.py`）  
  今日计划、回测复盘、监控、自检页面提供策略轨迹、风险监控、实验结果与配置管理。

架构调用链示意可参考 `docs/architecture_call_graph.md`。

## 快速开始

### 1. 环境准备

- Python 3.10+（建议使用虚拟环境）
- TuShare Token（环境变量 `TUSHARE_TOKEN`）
- 可选：LLM 供应商 API Key（`LLM_API_KEY` 等，具体参见 `config.json` 中的 provider 定义）

```bash
python -m venv .venv
source .venv/bin/activate           # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt

export TUSHARE_TOKEN="your-token"   # 必填
export LLM_API_KEY="your-api-key"   # 如需调用 LLM
```

### 2. 初始化数据与回测

```bash
# 拉取最新行情、特征或新闻（示例脚本，可按需修改参数）
python scripts/run_ingestion_job.py --mode daily --date 2024-01-05

# 运行示例决策环境，验证 RL 闭环
python scripts/run_decision_env_example.py
```

### 3. 启动 Streamlit 应用

```bash
streamlit run app/ui/streamlit_app.py
```

应用启动后可在侧边栏配置 LLM Provider、提示模板或监控指标；今日计划页会读取 `agent_utils`、`portfolio_*` 等表展示最新决策。

## 典型工作流

1. **数据补全**：定时运行 `scripts/run_ingestion_job.py`，确保行情、基本面、新闻数据齐备。  
2. **因子计算**：按交易日触发 `compute_factors()`（可通过脚本或定时任务），结果写入 SQLite。  
3. **代理决策**：规则代理 + LLM 部门协商输出交易建议，风险回合进行复核。  
4. **回测与调参**：使用 `scripts/train_ppo.py`、`scripts/run_bandit_optimization.py` 等脚本探索参数空间，成果写入 `tuning_results`。  
5. **可视化复盘**：Streamlit 展示多版本策略表现、风险事件与日志明细，支持“一键重评估”等待办项。  
6. **上线前验证**：沿线下回测 → 前向测试 → 影子运行的节奏推进，必要时接入外部告警。

## 自动化与实验工具

- `scripts/train_ppo.py`：使用 PPO 在 `DecisionEnv` 上训练策略。  
- `scripts/run_bandit_optimization.py`：黑箱调参示例。  
- `scripts/apply_best_weights.py`：将实验权重写回配置。  
- `scripts/render_architecture_diagram.py`：根据代码结构渲染架构图。  
- `scripts/migrations/`：数据库迁移脚本与示例。

更多示例请查看 `scripts/README` 或脚本内联说明。

## 文档索引

- 工作项总览：`docs/TODO.md`  
- 多智能体原理：`docs/principles/multi_agent_decision.md`  
- 强化学习与调参原理：`docs/principles/reinforcement_learning_tuning.md`  
- 风险控制原理：`docs/principles/risk_management.md`  
- 架构调用示意：`docs/architecture_call_graph.md`

上述文档保持与代码同步，请在功能迭代后一并更新。

## 路线图与贡献

- 当前路线图统一维护在 `docs/TODO.md`。欢迎在 Issue 或 PR 中反馈新的需求与优先级。  
- 修改代码前建议阅读 `app/utils/config.py`、`app/agents/game.py`、`app/backtest/engine.py` 了解关键路径。  
- 提交代码请附带必要的单元测试：

```bash
pytest
```

如有新的模块或实验脚本，请在对应文档与工作项中补充说明，确保团队成员可以快速上手。
