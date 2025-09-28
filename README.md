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
- **部门化多模型协作**：`app/agents/departments.py` 封装部门级 LLM 调度，`app/llm/client.py` 支持 single/majority/leader 策略，部门结论在 `app/agents/game.py` 与六类基础代理共同博弈，并持久化至 `agent_utils` 供 UI 展示。

## LLM + 多智能体最佳实践

- **强化结构化特征**：除 A 股行情外，引入资金流、因子、宏观等数据，为六类代理提供更丰富上下文。
- **场景化 Prompt**：在 `app/llm` 中注入代理贡献、宏观状态和风险事件，让 LLM 输出的策略解释与信号一致。
- **闭环反馈机制**：将回测或实盘的真实收益、成交等结果写回 SQLite，用于调整代理权重与 Prompt 语料。
- **多层日志监控**：保留代理评分、决策信心、LLM 提示与 UI 操作日志，帮助定位“谁做的决策、为何失败”。
- **人机协同流程**：在 UI 呈现代理分歧与 LLM 风险提示，分析师可调权、重跑或复核，实现人在环路的策略流程。

## 环境依赖与安装

建议使用 Python 3.10+，并在虚拟环境中安装依赖。

```bash
# 1. 创建并激活 Conda 环境
conda create -n llm-quant python=3.11 -y
conda activate llm-quant

# 2. 安装项目依赖
pip install -r requirements.txt

# 3. 设置 TuShare Token
export TUSHARE_TOKEN="<your-token>"
```

`requirements.txt` 当前涵盖运行框架所需的核心三方库：

- Pandas：数据表结构与指标处理
- Streamlit：交互式前端
- Plotly：行情与指标可视化
- TuShare：行情与基础面数据源
- Requests：统一访问 Ollama / OpenAI 兼容 API

### LLM 配置与测试

- 支持本地 Ollama 与多家 OpenAI 兼容云端供应商（如 DeepSeek、文心一言、OpenAI 等），可在 “数据与设置” 页签切换 Provider 并自动加载该 Provider 的候选模型、推荐 Base URL、默认温度与超时时间，亦可切换为自定义值。所有修改会持久化到 `app/data/config.json`，下次启动自动加载。
- 修改 Provider/模型/Base URL/API Key 后点击 “保存 LLM 设置”，更新内容仅在当前会话生效。
- 在 “自检测试” 页新增 “LLM 接口测试”，可输入 Prompt 快速验证调用结果，日志会记录限频与错误信息便于排查。
- 未来可对同一功能的智能体并行调用多个 LLM，采用多数投票等策略增强鲁棒性，当前代码结构已为此预留扩展空间。
- 若使用环境变量自动注入配置，可设置：
  - `TUSHARE_TOKEN`
  - `LLM_API_KEY`

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
2. 丰富 `DepartmentContext`（行情快照、风险指标），让部门评估拥有更完整的上下文。
3. 使用轻量情感分析与热度计算填充 `news`、`heat_daily` 与热点指数。
4. 在 Streamlit 今日计划页增加“重新评估”与日志追踪能力，串联实时调度链路。

## License

本项目采用定制的 “LLM Quant Framework License v1.0”。个人使用、修改与分发需保留出处，任何商业用途须事先与版权方协商并签署付费协议。详情参见仓库根目录的 `LICENSE` 文件。

## 多智能体 LLM 投资流程

- **部门化结构**：动量、价值、新闻、流动性、宏观、风险等代理视作独立业务部门，利用项目现有的数据/特征处理流程向每个部门提供上下文。
- **多 LLM 协作**：每个部门内部可配置多家 LLM 提供商（如 DeepSeek、OpenAI、文心等）作为智能体助手，分别生成分析意见和风险提示；可通过多数投票、仲裁等策略确定部门结论。
- **部门输出**：统一返回部门行动（买入/卖出/持有）、信心水平以及核心理由 (context + LLM 摘要)，当前实现会将摘要、风险提示与票权写入 `agent_utils`。
- **跨部门协调**：沿用 `app/agents/game.py` 的纳什谈判/投票结构，将各部门的结论与六类基础代理共同建模，必要时触发冲突检测并标记复核。
- **日志与可视化**：Streamlit 今日计划页读取 `agent_utils` 展示部门意见、投票细节与全局行动，可快速核查部门分歧与置信度。

## 实施步骤

1. **配置扩展** (`app/utils/config.py` + `config.json`) ✅
   - 部门支持 primary/ensemble、策略（single/majority/leader）、权重，并可在 Streamlit 中编辑主要字段。

2. **部门管控器** ✅
   - `app/agents/departments.py` 提供 `DepartmentAgent`/`DepartmentManager`，封装 Prompt 构建、多模型协商及异常回退。

3. **集成决策链** ✅
   - `app/agents/game.py` 将部门评分嵌入纳什谈判/加权投票，并对冲突设置复核标记；`app/backtest/engine.py` 将结果落库。

4. **UI 与日志**（进行中）
   - 今日计划页展示部门意见、票权与全局策略，后续补充一键重评估、日志钻取。

5. **测试与验证**（待补充）
   - 需完善部门上下文构造与多模型调用的单元/集成测试，结合回测指标对比多 LLM 策略收益差异。
