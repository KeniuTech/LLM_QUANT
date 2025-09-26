# 多智能体投资助理骨架

本仓库提供一个基于多智能体博弈的 A 股日线投资助理代码框架，满足单机可运行、SQLite 存储和 Streamlit UI 的需求。核心模块划分如下：

- `app/data`：数据库初始化与 Schema 定义。
- `app/utils`：配置、数据库连接、日志和交易日历工具。
- `app/ingest`：TuShare 数据抓取、新闻 RSS、数据覆盖检查器。
- `app/features`：指标与信号计算接口。
- `app/agents`：多智能体博弈实现，包括动量、价值、新闻、流动性、宏观与风险代理。
- `app/backtest`：日线回测引擎与指标计算的占位实现。
- `app/llm`：人类可读卡片与摘要生成入口（仅构建提示，不直接交易）。
- `app/ui`：Streamlit 四页界面骨架，含“自检测试”页。

## 快速开始

```bash
# 初始化数据库结构
python -m app.cli init-db

# 一键开机检查（默认回溯 365 天，缺失数据会自动补齐）
python -m app.cli boot-check --days 365

# 启动界面
streamlit run app/ui/streamlit_app.py
```

Streamlit `自检测试` 页签提供：
- 数据库初始化快捷按钮；
- TuShare 小范围拉取测试；
- 开机检查器（展示当前数据覆盖范围与股票基础信息完整度）。

## 下一步

1. 在 `app/features` 和 `app/backtest` 中完善信号计算、事件驱动撮合与绩效指标输出。
2. 将代理效用写入 SQLite 的 `agent_utils` 和 `alloc_log` 表，驱动 UI 决策解释。
3. 使用轻量情感分析与热度计算填充 `news`、`heat_daily` 与热点指数。
4. 接入本地小模型或 API 完成人类可读的策略建议卡片，形成端到端体验。
