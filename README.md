# 多智能体投资助理骨架

本仓库提供一个基于多智能体博弈的 A 股日线投资助理代码框架，满足单机可运行、SQLite 存储和 Streamlit UI 的需求。核心模块划分如下：

- `app/data`：数据库初始化与 Schema 定义。
- `app/utils`：配置、数据库连接、日志和交易日历工具。
- `app/ingest`：TuShare 与 RSS 数据拉取骨架。
- `app/features`：指标与信号计算接口。
- `app/agents`：多智能体博弈实现，包括动量、价值、新闻、流动性、宏观与风险代理。
- `app/backtest`：日线回测引擎与指标计算的占位实现。
- `app/llm`：人类可读卡片与摘要生成入口（仅构建提示，不直接交易）。
- `app/ui`：Streamlit 三页界面骨架。

## 快速开始

```bash
python -m app.main  # 初始化数据库
streamlit run app/ui/streamlit_app.py
```

## 下一步

1. 在 `app/ingest` 中补充 TuShare 和 RSS 数据抓取逻辑。
2. 完善 `app/features` 和 `app/backtest` 以实现实际的信号计算与事件驱动回测。
3. 将代理效用写入 SQLite 的 `agent_utils` 和 `alloc_log` 表，驱动 UI 展示。
4. 使用轻量情感分析与热度计算，填充 `news` 和 `heat_daily`。
5. 接入本地小模型或 API 完成 LLM 文本解释，并在 UI 中展示。
