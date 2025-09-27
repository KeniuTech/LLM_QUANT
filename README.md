# 多智能体投资助理骨架

## 项目简介

本仓库提供一个面向 A 股日线级别的多智能体投资助理原型，覆盖数据采集、特征抽取、策略博弈、回测展示和 LLM 解释链路。代码以模块化骨架形式呈现，方便在单机环境下快速搭建端到端的量化研究和可视化决策流程。

## 核心模块

- `app/data`：数据库初始化与 Schema 定义。
- `app/utils`：配置、数据库连接、日志和交易日历工具。
- `app/ingest`：TuShare 数据抓取、新闻 RSS、数据覆盖检查器。
- `app/features`：指标与信号计算接口。
- `app/agents`：多智能体博弈实现，包括动量、价值、新闻、流动性、宏观与风险代理。
- `app/backtest`：日线回测引擎与指标计算的占位实现。
- `app/llm`：人类可读卡片与摘要生成入口（仅构建提示，不直接交易）。
- `app/ui`：Streamlit 四页界面骨架，含“自检测试”页。

## 核心技术原理

- **多智能体博弈**：通过 `app/agents` 定义六类风格化代理，利用纳什谈判与加权投票在 `app/agents/game.py` 中聚合交易动作与信心水平。
- **数据覆盖自检**：`app/ingest/tushare.py` 封装 TuShare 拉取、增量更新与覆盖统计，`app/ingest/checker.py` 提供强制补数与窗口化覆盖报告。
- **事件驱动回测**：`app/backtest/engine.py` 构建日频回测循环，将代理决策与投资组合状态解耦，便于扩展成交撮合与绩效统计。
- **可视化与解释**：`app/ui/streamlit_app.py` 提供四大页签（今日计划、回测与复盘、数据与设置、自检测试），结合 Plotly 图形展示和 `app/llm` 提示卡片生成器，支撑人机协作分析。
- **统一日志与持久化**：SQLite 统一存储行情、回测与日志，配合 `DatabaseLogHandler` 在 UI/抓数流程中输出结构化运行轨迹，支持快速追踪与复盘。
- **跨市场数据扩展**：`app/ingest/tushare.py` 追加指数、ETF/公募基金、期货、外汇、港股与美股的增量拉取逻辑，确保多资产因子与宏观代理所需的行情基础数据齐备。

## LLM + 多智能体最佳实践

- **强化结构化特征**：除 A 股行情外，引入资金流、因子、宏观等数据，为六类代理提供更丰富上下文。
- **场景化 Prompt**：在 `app/llm` 中注入代理贡献、宏观状态和风险事件，让 LLM 输出的策略解释与信号一致。
- **闭环反馈机制**：将回测或实盘的真实收益、成交等结果写回 SQLite，用于调整代理权重与 Prompt 语料。
- **多层日志监控**：保留代理评分、决策信心、LLM 提示与 UI 操作日志，帮助定位“谁做的决策、为何失败”。
- **人机协同流程**：在 UI 呈现代理分歧与 LLM 风险提示，分析师可调权、重跑或复核，实现人在环路的策略流程。

## 环境依赖与安装

建议使用 Python 3.10+，并在虚拟环境中安装依赖。

```bash
# 1. 创建并激活虚拟环境（示例使用 venv）
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate

# 2. 安装项目依赖
pip install -r requirements.txt

# 3. 设置 TuShare Token（可写入环境变量或在 UI 中配置）
export TUSHARE_TOKEN="<your-token>"
```

`requirements.txt` 当前涵盖运行框架所需的核心三方库：

- Pandas：数据表结构与指标处理
- Streamlit：交互式前端
- Plotly：行情与指标可视化
- TuShare：行情与基础面数据源

## 快速开始

```bash
# 启动交互界面（内含数据库初始化、开机检查、样例回测入口）
streamlit run app/ui/streamlit_app.py
```

Streamlit `自检测试` 页签提供：
- 数据库初始化快捷按钮；
- TuShare 小范围拉取测试；
- 一键开机检查（可自动补数并展示覆盖摘要）；
- 股票行情可视化（自动加载近段时间价格、成交量，并展示核心指标）。
- 开机检查带进度指示与详细日志，便于排查 TuShare 拉取问题。

`回测与复盘` 页签提供快速回测表单，可调整时间区间、股票池与参数并即时查看回测输出。

## 下一步

1. 在 `app/features` 和 `app/backtest` 中完善信号计算、事件驱动撮合与绩效指标输出。
2. 将代理效用写入 SQLite 的 `agent_utils` 和 `alloc_log` 表，驱动 UI 决策解释。
3. 使用轻量情感分析与热度计算填充 `news`、`heat_daily` 与热点指数。
4. 接入本地小模型或 API 完成人类可读的策略建议卡片，形成端到端体验。
