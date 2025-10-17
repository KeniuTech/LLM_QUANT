# 多智能体量化投资助手 🤖📈

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE) [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

> 一个前沿的多智能体投资研究平台，针对A股日频数据，集成数据采集、因子工程、多智能体大语言模型决策、强化学习调优及丰富的可视化，加速可部署策略开发。

---

## 🚀 快速开始

```bash
# 使用 conda 创建并激活环境
conda create -n llm-quant python=3.10
conda activate llm-quant

# 安装依赖
pip install -r requirements.txt

# 启动 Streamlit 可视化界面（所有任务均在前端完成）
streamlit run app/ui/streamlit_app.py
```

> ⚙️ 环境变量（如 TUSHARE_TOKEN、LLM API Key）可在 Streamlit 配置界面统一维护，无需手动 `export`。

---

## 🌟 特色功能

| 功能                   | 描述                                                                                     |
|------------------------|------------------------------------------------------------------------------------------|
| **端到端流水线**        | 从市场/新闻数据采集到多智能体决策、回测及可视化                                         |
| **多智能体协作**        | 主持人、预测者、风险管理者、执行者等角色轮次交互；支持大语言模型与规则混合               |
| **强化学习调优**        | 将回测引擎封装为强化学习环境（`DecisionEnv`），兼容PPO/SAC等算法                         |
| **风险优先设计**        | 风险轮次可调整或否决交易，记录证据，支持复核和告警                                       |
| **模块化与可扩展性**    | 独立的数据、因子、智能体、提示词及实验模块，便于定制                                    |

---

## 🏗 架构概览

```plaintext
+---------------------+       +-----------------------+       +----------------------+
|  数据流水线         | <---> |  因子与特征计算       | <---> |  多智能体系统         |
| (app/ingest, utils)  |       |                       |       | (app/agents)          |
+---------------------+       +-----------------------+       +----------------------+
         |                                                         |
         |                                                         |
         v                                                         v
+---------------------+                                   +----------------------+
| 强化学习与优化      |                                   | 可视化与界面          |
| (app/backtest)      |                                   | (app/ui/streamlit_app)|
+---------------------+                                   +----------------------+
```

> 详细调用图见 [`docs/architecture_call_graph.md`](docs/architecture_call_graph.md)。

---

## 🎯 演示

体验交互式演示，实时策略规划、风险监控与回测复盘：

[🔗 演示链接占位符](#)

---

## 📚 文档

- **多智能体原则：** [`docs/principles/multi_agent_decision.md`](docs/principles/multi_agent_decision.md)  
- **强化学习调优：** [`docs/principles/reinforcement_learning_tuning.md`](docs/principles/reinforcement_learning_tuning.md)  
- **风险管理：** [`docs/principles/risk_management.md`](docs/principles/risk_management.md)  
- **架构调用图：** [`docs/architecture_call_graph.md`](docs/architecture_call_graph.md)  
- **项目TODO与路线图：** [`docs/TODO.md`](docs/TODO.md)

---

## 🛠 典型工作流程

1. **环境配置：** 通过 Streamlit「系统设置」页填入 TuShare、LLM 等凭据，并在同页管理数据根目录、日志级别等参数。  
2. **数据补全：** 在「今日计划 → 数据自检」中一键触发行情、基本面、新闻拉取与健康检查。  
3. **因子与特征：** 使用「回测与复盘 → 因子计算」面板选择交易日及股票池，实时查看进度与校验报告。  
4. **多智能体决策：** 「今日计划」页直接发起多轮博弈，风险部门结论与对话全量保存，可即时复核。  
5. **回测与调优：** 「实验调参」与「回测与复盘」页提供 PPO、贝叶斯优化等实验入口，支持参数可视化比对。  
6. **可视化与监控：** 借助「投资池/仓位」「风险预警」等看板实时掌握仓位、事件、日志；完成线下验证后再推进实盘。

---

## 📜 许可证

本项目基于 [MIT License](./LICENSE) 开源。

---


*怀揣对多智能体协作、强化学习及透明投资策略的热情构建。*
