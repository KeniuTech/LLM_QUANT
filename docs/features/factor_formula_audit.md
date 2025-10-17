# 因子公式复核指引

本文档总结因子公式复核流程，并说明如何使用新引入的工具快速检查数据库中的持久化因子。

## 1. 快速开始

1. **增量/干跑计算**：
   ```bash
   python scripts/run_factor_pipeline.py --mode incremental --max-days 5
   ```
   - 默认会写入数据库，若只想验证公式可加 `--no-persist`。

2. **执行公式审计**：
   ```bash
   python scripts/run_factor_pipeline.py --mode single --trade-date 20250210 --audit
   ```
   - 同等功能也可通过 Python 调用：
     ```python
     from datetime import date
     from app.features.factor_audit import audit_factors

     summary = audit_factors(date(2025, 2, 10))
     print(summary.to_dict())
     ```

3. **查看得分**：`summary.mismatched` 为不一致条目数；若为 0 表示通过。

## 2. 常见使用场景

- **版本升级后复核**：指定 trade_date 运行审计，确认公式变更未引入漂移。
- **数据库回滚/恢复**：使用 `--no-persist --audit` 快速检查备份数据的完整性。
- **问题定位**：`summary.issues` 提供具体股票代码、因子名与差值，便于对账。

## 3. 结果解读

| 字段 | 说明 |
| --- | --- |
| `score` | 数据质量得分（0-100）。 |
| `blocking` | 会导致任务失败的错误；需优先处理。 |
| `warnings` | 风险提示，可安排在巡检时处理。 |

> 注：公式审计依赖 `factors` 表的历史数据；若发现字段缺失，请先运行 `scripts/run_factor_pipeline.py` 补齐。

