"""Standalone view for reinforcement learning and parameter search experiments."""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from app.agents.registry import default_agents
from app.backtest.decision_env import DecisionEnv, ParameterSpec
from app.backtest.engine import BtConfig
from app.backtest.optimizer import (
    BanditConfig,
    BayesianBandit,
    EpsilonGreedyBandit,
    SuccessiveHalvingOptimizer,
)
from app.rl import TORCH_AVAILABLE, DecisionEnvAdapter, PPOConfig, train_ppo
from app.ui.navigation import navigate_top_menu
from app.llm.templates import TemplateRegistry
from app.utils.config import get_config, save_config
from app.utils.portfolio import (
    get_candidate_pool,
    get_portfolio_settings_snapshot,
)
from app.ui.shared import LOGGER, LOG_EXTRA, default_backtest_range
from app.agents.protocols import GameStructure

_DECISION_ENV_BANDIT_RESULTS_KEY = "decision_env_bandit_results"
_DECISION_ENV_PPO_RESULTS_KEY = "decision_env_ppo_results"


def _render_bandit_summary(
    bandit_state: Optional[Dict[str, object]],
    app_cfg,
) -> None:
    """Display a concise summary of the latest bandit search run."""
    if not bandit_state:
        st.info("尚未执行参数搜索实验，可在下方配置参数后启动探索。")
        return

    st.caption(
        f"实验 ID：{bandit_state.get('experiment_id')} | 策略：{bandit_state.get('strategy')}"
    )
    best_payload = bandit_state.get("best") or {}
    reward = best_payload.get("reward")
    best_index = bandit_state.get("best_index")
    metrics_payload = best_payload.get("metrics") or {}

    if reward is None:
        st.info("实验记录暂未产生有效的最佳结果。")
        return

    col_reward, col_return, col_drawdown, col_sharpe, col_calmar = st.columns(5)
    col_reward.metric("最佳奖励", f"{reward:+.4f}")
    total_return = metrics_payload.get("total_return")
    col_return.metric(
        "累计收益",
        f"{total_return:+.4f}" if total_return is not None else "—",
    )
    max_drawdown = metrics_payload.get("max_drawdown")
    col_drawdown.metric(
        "最大回撤",
        f"{max_drawdown:.3f}" if max_drawdown is not None else "—",
    )
    sharpe_like = metrics_payload.get("sharpe_like")
    col_sharpe.metric(
        "Sharpe",
        f"{sharpe_like:.3f}" if sharpe_like is not None else "—",
    )
    calmar_like = metrics_payload.get("calmar_like")
    col_calmar.metric(
        "Calmar",
        f"{calmar_like:.3f}" if calmar_like is not None else "—",
    )
    st.caption(f"最佳轮次：第 {best_index} 轮")

    with st.expander("动作与参数详情", expanded=False):
        st.write("动作 (raw)：")
        st.json(best_payload.get("action") or {})
        st.write("解析后的参数：")
        st.json(best_payload.get("resolved_action") or {})

    weights_payload = best_payload.get("weights") or {}
    if weights_payload:
        st.write("对应代理权重：")
        st.json(weights_payload)
        if st.button("将最佳权重写入默认配置", key="save_decision_env_bandit_weights"):
            try:
                app_cfg.agent_weights.update_from_dict(weights_payload)
                save_config(app_cfg)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(
                    "保存 bandit 权重失败",
                    extra={**LOG_EXTRA, "error": str(exc)},
                )
                st.error(f"写入配置失败：{exc}")
            else:
                st.success("最佳权重已写入 config.json")

    dept_ctrl = best_payload.get("department_controls") or {}
    if dept_ctrl:
        with st.expander("最佳部门控制参数", expanded=False):
            st.json(dept_ctrl)

    st.caption("完整的 RL/BOHB 日志请切换到“RL/BOHB 日志”标签查看。")


def _render_bandit_logs(bandit_state: Optional[Dict[str, object]]) -> None:
    """Render the detailed BOHB/Bandit episode logs."""
    st.subheader("RL/BOHB 执行日志")
    if not bandit_state:
        st.info("暂无日志，请先在“策略实验管理”中运行一次参数搜索。")
        return

    episodes = bandit_state.get("episodes") or []
    if episodes:
        df = pd.DataFrame(episodes)
        st.dataframe(df, hide_index=True, width="stretch")
        csv_name = f"tuning_logs_{bandit_state.get('experiment_id', 'bandit')}.csv"
        json_name = f"tuning_logs_{bandit_state.get('experiment_id', 'bandit')}.json"
        st.download_button(
            "下载日志 CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=csv_name,
            mime="text/csv",
            key="download_decision_env_bandit_csv",
        )
        st.download_button(
            "下载日志 JSON",
            data=json.dumps(episodes, ensure_ascii=False, indent=2),
            file_name=json_name,
            mime="application/json",
            key="download_decision_env_bandit_json",
        )
    else:
        st.info("暂无迭代记录。")

    if st.button("清除自动探索结果", key="clear_decision_env_bandit"):
        st.session_state.pop(_DECISION_ENV_BANDIT_RESULTS_KEY, None)
        st.success("已清除自动探索结果。")


def _render_ppo_training(
    app_cfg,
    context: Dict[str, object],
    ppo_state: Optional[Dict[str, object]],
) -> None:
    """Render PPO training controls and diagnostics within the PPO tab."""
    st.subheader("PPO 训练（逐日强化学习）")

    specs: List[ParameterSpec] = context.get("specs") or []
    agent_objects = context.get("agent_objects") or []
    selected_structures = context.get("selected_structures") or [GameStructure.REPEATED]
    disable_departments = context.get("disable_departments", True)
    universe_text = context.get("universe_text") or ""
    backtest_params = context.get("backtest_params") or {}
    start_date = context.get("start_date") or date.today()
    end_date = context.get("end_date") or date.today()
    range_valid = context.get("range_valid", True)
    controls_valid = context.get("controls_valid", True)

    if not agent_objects:
        st.info("暂无可调整的代理，无法进行 PPO 训练。")
    elif not specs:
        st.info("请先在“策略实验管理”中配置可调节的参数维度。")
    elif not range_valid or not controls_valid:
        st.warning("请先修正代理或部门的参数范围，再启动 PPO 训练。")
    elif not TORCH_AVAILABLE:
        st.warning("当前环境未检测到 PyTorch，无法运行 PPO 训练。")
    else:
        col_ts, col_rollout, col_epochs = st.columns(3)
        ppo_timesteps = int(
            col_ts.number_input(
                "总时间步",
                min_value=256,
                max_value=200_000,
                value=4096,
                step=256,
                key="decision_env_ppo_timesteps",
            )
        )
        ppo_rollout = int(
            col_rollout.number_input(
                "每次收集步数",
                min_value=32,
                max_value=2048,
                value=256,
                step=32,
                key="decision_env_ppo_rollout",
            )
        )
        ppo_epochs = int(
            col_epochs.number_input(
                "每批优化轮数",
                min_value=1,
                max_value=30,
                value=8,
                step=1,
                key="decision_env_ppo_epochs",
            )
        )

        col_mb, col_gamma, col_lambda = st.columns(3)
        ppo_minibatch = int(
            col_mb.number_input(
                "批次大小",
                min_value=16,
                max_value=1024,
                value=128,
                step=16,
                key="decision_env_ppo_minibatch",
            )
        )
        ppo_gamma = float(
            col_gamma.number_input(
                "折现系数 γ",
                min_value=0.8,
                max_value=0.999,
                value=0.99,
                step=0.01,
                format="%.3f",
                key="decision_env_ppo_gamma",
            )
        )
        ppo_lambda = float(
            col_lambda.number_input(
                "GAE λ",
                min_value=0.5,
                max_value=0.999,
                value=0.95,
                step=0.01,
                format="%.3f",
                key="decision_env_ppo_lambda",
            )
        )

        col_clip, col_entropy, col_value = st.columns(3)
        ppo_clip = float(
            col_clip.number_input(
                "裁剪范围 ε",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.01,
                format="%.2f",
                key="decision_env_ppo_clip",
            )
        )
        ppo_entropy = float(
            col_entropy.number_input(
                "熵系数",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                step=0.005,
                format="%.3f",
                key="decision_env_ppo_entropy",
            )
        )
        ppo_value_coef = float(
            col_value.number_input(
                "价值损失系数",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                key="decision_env_ppo_value_coef",
            )
        )

        col_lr_p, col_lr_v, col_grad = st.columns(3)
        ppo_policy_lr = float(
            col_lr_p.number_input(
                "策略学习率",
                min_value=1e-5,
                max_value=1e-2,
                value=3e-4,
                step=1e-5,
                format="%.5f",
                key="decision_env_ppo_policy_lr",
            )
        )
        ppo_value_lr = float(
            col_lr_v.number_input(
                "价值学习率",
                min_value=1e-5,
                max_value=1e-2,
                value=3e-4,
                step=1e-5,
                format="%.5f",
                key="decision_env_ppo_value_lr",
            )
        )
        ppo_max_grad_norm = float(
            col_grad.number_input(
                "梯度裁剪",
                value=0.5,
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                format="%.1f",
                key="decision_env_ppo_grad_norm",
            )
        )

        col_hidden, col_seed, _ = st.columns(3)
        ppo_hidden_text = col_hidden.text_input(
            "隐藏层结构 (逗号分隔)",
            value="128,128",
            key="decision_env_ppo_hidden",
        )
        ppo_seed_text = col_seed.text_input(
            "随机种子 (可选)",
            value="42",
            key="decision_env_ppo_seed",
        )
        try:
            ppo_hidden = tuple(
                int(v.strip()) for v in ppo_hidden_text.split(",") if v.strip()
            )
        except ValueError:
            ppo_hidden = ()
        ppo_seed = None
        if ppo_seed_text.strip():
            try:
                ppo_seed = int(ppo_seed_text.strip())
            except ValueError:
                st.warning("PPO 随机种子需为整数，已忽略该值。")
                ppo_seed = None

        if st.button("启动 PPO 训练", key="run_decision_env_ppo"):
            if not ppo_hidden:
                st.error("请提供合法的隐藏层结构，例如 128,128。")
            else:
                baseline_weights = app_cfg.agent_weights.as_dict()
                for agent in agent_objects:
                    baseline_weights.setdefault(agent.name, 1.0)

                universe_env = [
                    code.strip() for code in universe_text.split(",") if code.strip()
                ]
                if not universe_env:
                    st.error("请先指定至少一个股票代码。")
                else:
                    bt_cfg_env = BtConfig(
                        id="decision_env_ppo",
                        name="DecisionEnv PPO",
                        start_date=start_date,
                        end_date=end_date,
                        universe=universe_env,
                        params=dict(backtest_params),
                        method=app_cfg.decision_method,
                        game_structures=selected_structures,
                    )
                    env = DecisionEnv(
                        bt_config=bt_cfg_env,
                        parameter_specs=specs,
                        baseline_weights=baseline_weights,
                        disable_departments=disable_departments,
                    )
                    adapter = DecisionEnvAdapter(env)
                    config = PPOConfig(
                        total_timesteps=ppo_timesteps,
                        rollout_steps=ppo_rollout,
                        gamma=ppo_gamma,
                        gae_lambda=ppo_lambda,
                        clip_range=ppo_clip,
                        policy_lr=ppo_policy_lr,
                        value_lr=ppo_value_lr,
                        epochs=ppo_epochs,
                        minibatch_size=ppo_minibatch,
                        entropy_coef=ppo_entropy,
                        value_coef=ppo_value_coef,
                        max_grad_norm=ppo_max_grad_norm,
                        hidden_sizes=ppo_hidden,
                        seed=ppo_seed,
                    )
                    with st.spinner("PPO 训练进行中，请耐心等待..."):
                        try:
                            summary = train_ppo(adapter, config)
                        except Exception as exc:  # noqa: BLE001
                            LOGGER.exception(
                                "PPO 训练失败",
                                extra={**LOG_EXTRA, "error": str(exc)},
                            )
                            st.error(f"PPO 训练失败：{exc}")
                        else:
                            payload = {
                                "timesteps": summary.timesteps,
                                "episode_rewards": summary.episode_rewards,
                                "episode_lengths": summary.episode_lengths,
                                "diagnostics": summary.diagnostics[-25:],
                                "observation_keys": adapter.keys(),
                            }
                            st.session_state[_DECISION_ENV_PPO_RESULTS_KEY] = payload
                            st.success("PPO 训练完成。")

    if ppo_state:
        st.caption(f"最近一次 PPO 训练时间步：{ppo_state.get('timesteps')}")
        rewards = ppo_state.get("episode_rewards") or []
        if rewards:
            st.line_chart(rewards, height=200)
        lengths = ppo_state.get("episode_lengths") or []
        if lengths:
            st.bar_chart(lengths, height=200)
        diagnostics = ppo_state.get("diagnostics") or []
        if diagnostics:
            st.dataframe(pd.DataFrame(diagnostics), hide_index=True, width="stretch")
        st.download_button(
            "下载 PPO 结果 (JSON)",
            data=json.dumps(ppo_state, ensure_ascii=False, indent=2),
            file_name="ppo_training_summary.json",
            mime="application/json",
            key="decision_env_ppo_json",
        )

    st.caption("提示：可在“回测与复盘”页面载入保存的权重并进行对比验证。")


def _render_experiment_management(
    app_cfg,
    portfolio_snapshot: Dict[str, object],
    default_start: date,
    default_end: date,
) -> Dict[str, object]:
    """Render strategy experiment management controls and return context for other tabs."""
    st.subheader("实验基础参数")

    col_dates_1, col_dates_2 = st.columns(2)
    start_date = col_dates_1.date_input(
        "开始日期",
        value=default_start,
        key="tuning_start_date",
    )
    end_date = col_dates_2.date_input(
        "结束日期",
        value=default_end,
        key="tuning_end_date",
    )

    candidate_records, candidate_fallback = get_candidate_pool(limit=50)
    candidate_codes = [item.ts_code for item in candidate_records]
    default_universe = ",".join(candidate_codes) if candidate_codes else "000001.SZ"
    universe_text = st.text_input(
        "股票列表（逗号分隔）",
        value=default_universe,
        key="tuning_universe",
        help="默认载入最新候选池，如需自定义可直接编辑。",
    )
    if candidate_codes:
        message = (
            f"候选池载入 {len(candidate_codes)} 个标的："
            f"{'、'.join(candidate_codes[:10])}{'…' if len(candidate_codes) > 10 else ''}"
        )
        if candidate_fallback:
            message += "（使用最新候选池作为回退）"
        st.caption(message)

    col_target, col_stop, col_hold, col_cap = st.columns(4)
    target = col_target.number_input(
        "目标收益（例：0.035 表示 3.5%）",
        value=0.035,
        step=0.005,
        format="%.3f",
        key="tuning_target",
    )
    stop = col_stop.number_input(
        "止损收益（例：-0.015 表示 -1.5%）",
        value=-0.015,
        step=0.005,
        format="%.3f",
        key="tuning_stop",
    )
    hold_days = col_hold.number_input(
        "持有期（交易日）",
        value=10,
        step=1,
        key="tuning_hold_days",
    )
    initial_capital_default = float(portfolio_snapshot["initial_capital"])
    initial_capital = col_cap.number_input(
        "组合初始资金",
        value=initial_capital_default,
        step=100000.0,
        format="%.0f",
        key="tuning_initial_capital",
    )
    initial_capital = max(0.0, float(initial_capital))
    position_limits = portfolio_snapshot.get("position_limits", {})
    backtest_params = {
        "target": float(target),
        "stop": float(stop),
        "hold_days": int(hold_days),
        "initial_capital": initial_capital,
        "max_position_weight": float(position_limits.get("max_position", 0.2)),
        "max_total_positions": int(position_limits.get("max_total_positions", 20)),
    }
    st.caption(
        "组合约束：单仓上限 {max_pos:.0%} ｜ 最大持仓 {max_count} ｜ 行业敞口 {sector:.0%}".format(
            max_pos=backtest_params["max_position_weight"],
            max_count=position_limits.get("max_total_positions", 20),
            sector=position_limits.get("max_sector_exposure", 0.35),
        )
    )

    structure_options = [item.value for item in GameStructure]
    selected_structure_values = st.multiselect(
        "选择博弈框架",
        structure_options,
        default=structure_options,
        key="tuning_game_structures",
    )
    if not selected_structure_values:
        selected_structure_values = [GameStructure.REPEATED.value]
    selected_structures = [GameStructure(value) for value in selected_structure_values]

    allow_disable = st.columns([1, 1])
    disable_departments = allow_disable[0].checkbox(
        "禁用部门 LLM（仅规则代理，适合离线快速评估）",
        value=True,
        help="关闭部门调用后不依赖外部 LLM 网络，仅根据规则代理权重模拟。",
    )
    allow_disable[1].markdown(
        "[查看回测结果对比](javascript:void(0)) — 请通过顶部导航切换到“回测与复盘”。"
    )

    st.divider()
    st.subheader("实验与调参设置")

    default_experiment_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_id = st.text_input(
        "实验 ID",
        value=default_experiment_id,
        help="用于在 tuning_results 表中区分不同实验。",
        key="decision_env_experiment_id",
    )
    strategy_label = st.text_input(
        "策略说明",
        value="DecisionEnv",
        help="可选：为本次调参记录一个策略名称或备注。",
        key="decision_env_strategy_label",
    )

    agent_objects = default_agents()
    agent_names = [agent.name for agent in agent_objects]
    if not agent_names:
        st.info("暂无可调整的代理。")
        return {}

    selected_agents = st.multiselect(
        "选择调参的代理权重",
        agent_names,
        default=agent_names[:2],
        key="decision_env_agents",
    )

    specs: List[ParameterSpec] = []
    spec_labels: List[str] = []
    range_valid = True
    for agent_name in selected_agents:
        col_min, col_max, col_action = st.columns([1, 1, 2])
        min_key = f"decision_env_min_{agent_name}"
        max_key = f"decision_env_max_{agent_name}"
        action_key = f"decision_env_action_{agent_name}"
        default_min = 0.0
        default_max = 1.0
        min_val = col_min.number_input(
            f"{agent_name} 最小权重",
            min_value=0.0,
            max_value=1.0,
            value=default_min,
            step=0.05,
            key=min_key,
        )
        max_val = col_max.number_input(
            f"{agent_name} 最大权重",
            min_value=0.0,
            max_value=1.0,
            value=default_max,
            step=0.05,
            key=max_key,
        )
        if max_val <= min_val:
            range_valid = False
        col_action.slider(
            f"{agent_name} 动作 (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key=action_key,
        )
        specs.append(
            ParameterSpec(
                name=f"weight_{agent_name}",
                target=f"agent_weights.{agent_name}",
                minimum=min_val,
                maximum=max_val,
            )
        )
        spec_labels.append(f"agent:{agent_name}")

    controls_valid = True
    st.divider()
    st.subheader("部门参数调整（可选）")
    dept_codes = sorted(app_cfg.departments.keys())
    if not dept_codes:
        st.caption("当前未配置部门。")
    else:
        selected_departments = st.multiselect(
            "选择需要调整的部门",
            dept_codes,
            default=[],
            key="decision_env_departments",
        )
        tool_policy_values = ["auto", "none", "required"]
        for dept_code in selected_departments:
            settings = app_cfg.departments.get(dept_code)
            if not settings:
                continue
            st.subheader(f"部门：{settings.title or dept_code}")
            base_temp = 0.2
            if (
                settings.llm
                and settings.llm.primary
                and settings.llm.primary.temperature is not None
            ):
                base_temp = float(settings.llm.primary.temperature)
            prefix = f"decision_env_dept_{dept_code}"
            col_tmin, col_tmax, col_tslider = st.columns([1, 1, 2])
            temp_min = col_tmin.number_input(
                "温度最小值",
                min_value=0.0,
                max_value=2.0,
                value=max(0.0, base_temp - 0.3),
                step=0.05,
                key=f"{prefix}_temp_min",
            )
            temp_max = col_tmax.number_input(
                "温度最大值",
                min_value=0.0,
                max_value=2.0,
                value=min(2.0, base_temp + 0.3),
                step=0.05,
                key=f"{prefix}_temp_max",
            )
            if temp_max <= temp_min:
                controls_valid = False
                st.warning("温度最大值必须大于最小值。")
                temp_max = min(2.0, temp_min + 0.01)
            span = temp_max - temp_min
            ratio_default = 0.0
            if span > 0:
                clamped = min(max(base_temp, temp_min), temp_max)
                ratio_default = (clamped - temp_min) / span
            col_tslider.slider(
                "动作值（映射至温度区间）",
                min_value=0.0,
                max_value=1.0,
                value=float(ratio_default),
                step=0.01,
                key=f"{prefix}_temp_action",
            )
            specs.append(
                ParameterSpec(
                    name=f"dept_temperature_{dept_code}",
                    target=f"department.{dept_code}.temperature",
                    minimum=temp_min,
                    maximum=temp_max,
                )
            )
            spec_labels.append(f"department:{dept_code}:temperature")

            col_tool, col_hint = st.columns([1, 2])
            tool_choice = col_tool.selectbox(
                "函数调用策略",
                tool_policy_values,
                index=tool_policy_values.index("auto"),
                key=f"{prefix}_tool_choice",
            )
            col_hint.caption("映射提示：0→auto，0.5→none，1→required。")
            tool_value = 0.0
            if len(tool_policy_values) > 1:
                tool_value = tool_policy_values.index(tool_choice) / (
                    len(tool_policy_values) - 1
                )
            specs.append(
                ParameterSpec(
                    name=f"dept_tool_{dept_code}",
                    target=f"department.{dept_code}.function_policy",
                    values=tool_policy_values,
                )
            )
            spec_labels.append(f"department:{dept_code}:tool_choice")

            template_id = (settings.prompt_template_id or f"{dept_code}_dept").strip()
            versions = [
                ver for ver in TemplateRegistry.list_versions(template_id) if isinstance(ver, str)
            ]
            if versions:
                active_version = TemplateRegistry.get_active_version(template_id)
                default_version = (
                    settings.prompt_template_version
                    or active_version
                    or versions[0]
                )
                try:
                    default_index = versions.index(default_version)
                except ValueError:
                    default_index = 0
                version_choice = st.selectbox(
                    "提示模板版本",
                    versions,
                    index=default_index,
                    key=f"{prefix}_template_version",
                    help="离散动作将按版本列表顺序映射，可用于强化学习优化。",
                )
                selected_index = versions.index(version_choice)
                specs.append(
                    ParameterSpec(
                        name=f"dept_prompt_version_{dept_code}",
                        target=f"department.{dept_code}.prompt_template_version",
                        values=list(versions),
                    )
                )
                spec_labels.append(f"department:{dept_code}:prompt_version")
                st.caption(
                    f"激活版本：{active_version or '默认'} ｜ 当前选择：{version_choice}"
                )
            else:
                st.caption("当前模板未注册可选提示词版本，继续沿用激活版本。")

    if specs:
        st.caption("动作维度顺序：" + "，".join(spec_labels))

    return {
        "start_date": start_date,
        "end_date": end_date,
        "universe_text": universe_text,
        "backtest_params": backtest_params,
        "selected_structures": selected_structures,
        "disable_departments": disable_departments,
        "specs": specs,
        "agent_objects": agent_objects,
        "selected_agents": selected_agents,
        "range_valid": range_valid,
        "controls_valid": controls_valid,
        "experiment_id": experiment_id,
        "strategy_label": strategy_label,
    }


def _render_parameter_search(app_cfg, context: Dict[str, object]) -> None:
    """Render the global parameter search controls in a dedicated tab."""
    st.subheader("全局参数搜索")

    bandit_state = st.session_state.get(_DECISION_ENV_BANDIT_RESULTS_KEY)
    if bandit_state:
        _render_bandit_summary(bandit_state, app_cfg)

    specs: List[ParameterSpec] = context.get("specs") or []
    if not specs:
        st.info("请先在“策略实验管理”页配置可调节参数与动作范围。")
        return

    selected_agents = context.get("selected_agents") or []
    range_valid = context.get("range_valid", True)
    controls_valid = context.get("controls_valid", True)
    experiment_id = context.get("experiment_id")
    strategy_label = context.get("strategy_label")

    if selected_agents and not range_valid:
        st.error("请返回“策略实验管理”页，确保每个代理的最大权重大于最小权重。")
        return
    if not controls_valid:
        st.error("请先修正部门参数的取值范围后再执行搜索。")
        return

    strategy_choice = st.selectbox(
        "搜索策略",
        ["epsilon_greedy", "bayesian", "bohb"],
        format_func=lambda x: {
            "epsilon_greedy": "Epsilon-Greedy",
            "bayesian": "贝叶斯优化",
            "bohb": "BOHB/Successive Halving",
        }.get(x, x),
        key="decision_env_search_strategy",
    )

    seed_text = st.text_input(
        "随机种子（可选）",
        value="",
        key="decision_env_search_seed",
        help="填写整数可复现实验，不填写则随机。",
    ).strip()
    bandit_seed = None
    if seed_text:
        try:
            bandit_seed = int(seed_text)
        except ValueError:
            st.warning("随机种子需为整数，已忽略该值。")
            bandit_seed = None

    if strategy_choice == "epsilon_greedy":
        col_ep, col_eps = st.columns([1, 1])
        bandit_episodes = int(
            col_ep.number_input(
                "迭代次数",
                min_value=1,
                max_value=200,
                value=10,
                step=1,
                key="decision_env_bandit_episodes",
            )
        )
        bandit_epsilon = float(
            col_eps.slider(
                "探索比例 ε",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="decision_env_bandit_epsilon",
            )
        )
        bayes_iterations = bandit_episodes
        bayes_pool = 128
        bayes_explore = 0.01
        bohb_initial = 27
        bohb_eta = 3
        bohb_rounds = 3
    elif strategy_choice == "bayesian":
        col_ep, col_pool, col_xi = st.columns(3)
        bayes_iterations = int(
            col_ep.number_input(
                "迭代次数",
                min_value=3,
                max_value=200,
                value=15,
                step=1,
                key="decision_env_bayes_iterations",
            )
        )
        bayes_pool = int(
            col_pool.number_input(
                "候选采样数",
                min_value=16,
                max_value=1024,
                value=128,
                step=16,
                key="decision_env_bayes_pool",
            )
        )
        bayes_explore = float(
            col_xi.number_input(
                "探索权重 ξ",
                min_value=0.0,
                max_value=0.5,
                value=0.01,
                step=0.01,
                format="%.3f",
                key="decision_env_bayes_xi",
            )
        )
        bandit_episodes = bayes_iterations
        bandit_epsilon = 0.0
        bohb_initial = 27
        bohb_eta = 3
        bohb_rounds = 3
    else:
        col_init, col_eta, col_rounds = st.columns(3)
        bohb_initial = int(
            col_init.number_input(
                "初始候选数",
                min_value=3,
                max_value=243,
                value=27,
                step=3,
                key="decision_env_bohb_initial",
            )
        )
        bohb_eta = int(
            col_eta.number_input(
                "压缩因子 η",
                min_value=2,
                max_value=6,
                value=3,
                step=1,
                key="decision_env_bohb_eta",
            )
        )
        bohb_rounds = int(
            col_rounds.number_input(
                "最大轮次",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
                key="decision_env_bohb_rounds",
            )
        )
        bandit_episodes = bohb_initial
        bandit_epsilon = 0.0
        bayes_iterations = bandit_episodes
        bayes_pool = 128
        bayes_explore = 0.01

    start_date = context.get("start_date")
    end_date = context.get("end_date")
    if start_date is None or end_date is None:
        st.error("请先填写实验基础的开始/结束日期。")
        return

    specs_context = {
        "backtest_params": context.get("backtest_params") or {},
        "start_date": start_date,
        "end_date": end_date,
        "universe_text": context.get("universe_text", ""),
        "selected_structures": context.get("selected_structures")
        or [GameStructure.REPEATED],
        "disable_departments": context.get("disable_departments", True),
        "agent_objects": context.get("agent_objects") or [],
    }

    if st.button("执行参数搜索", key="run_decision_env_bandit"):
        universe_text = specs_context["universe_text"]
        universe_env = [
            code.strip() for code in universe_text.split(",") if code.strip()
        ]
        if not universe_env:
            st.error("请先指定至少一个股票代码。")
        else:
            baseline_weights = app_cfg.agent_weights.as_dict()
            for agent in specs_context["agent_objects"]:
                baseline_weights.setdefault(agent.name, 1.0)

            bt_cfg_env = BtConfig(
                id="decision_env_bandit",
                name="DecisionEnv Bandit",
                start_date=specs_context["start_date"],
                end_date=specs_context["end_date"],
                universe=universe_env,
                params=dict(specs_context["backtest_params"]),
                method=app_cfg.decision_method,
                game_structures=specs_context["selected_structures"],
            )
            env = DecisionEnv(
                bt_config=bt_cfg_env,
                parameter_specs=specs,
                baseline_weights=baseline_weights,
                disable_departments=specs_context["disable_departments"],
            )
            config = BanditConfig(
                experiment_id=experiment_id
                or f"bandit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy=strategy_label or strategy_choice,
                episodes=bandit_episodes,
                epsilon=bandit_epsilon,
                seed=bandit_seed,
                exploration_weight=bayes_explore,
                candidate_pool=bayes_pool,
                initial_candidates=bohb_initial,
                eta=bohb_eta,
                max_rounds=bohb_rounds,
            )
            if strategy_choice == "bayesian":
                optimizer = BayesianBandit(env, config)
            elif strategy_choice == "bohb":
                optimizer = SuccessiveHalvingOptimizer(env, config)
            else:
                optimizer = EpsilonGreedyBandit(env, config)
            with st.spinner("自动探索进行中，请稍候..."):
                summary = optimizer.run()

            episodes_dump: List[Dict[str, object]] = []
            for idx, episode in enumerate(summary.episodes, start=1):
                episodes_dump.append(
                    {
                        "序号": idx,
                        "奖励": episode.reward,
                        "动作(raw)": json.dumps(episode.action, ensure_ascii=False),
                        "参数值": json.dumps(
                            episode.resolved_action, ensure_ascii=False
                        ),
                        "总收益": episode.metrics.total_return,
                        "最大回撤": episode.metrics.max_drawdown,
                        "波动率": episode.metrics.volatility,
                        "Sharpe": episode.metrics.sharpe_like,
                        "Calmar": episode.metrics.calmar_like,
                        "权重": json.dumps(
                            episode.weights or {}, ensure_ascii=False
                        ),
                        "部门控制": json.dumps(
                            episode.department_controls or {}, ensure_ascii=False
                        ),
                    }
                )

            best_episode = summary.best_episode
            best_index = summary.episodes.index(best_episode) + 1 if best_episode else None
            st.session_state[_DECISION_ENV_BANDIT_RESULTS_KEY] = {
                "episodes": episodes_dump,
                "best_index": best_index,
                "best": {
                    "reward": best_episode.reward if best_episode else None,
                    "action": best_episode.action if best_episode else None,
                    "resolved_action": best_episode.resolved_action
                    if best_episode
                    else None,
                    "weights": best_episode.weights if best_episode else None,
                    "metrics": {
                        "total_return": best_episode.metrics.total_return
                        if best_episode
                        else None,
                        "sharpe_like": best_episode.metrics.sharpe_like
                        if best_episode
                        else None,
                        "calmar_like": best_episode.metrics.calmar_like
                        if best_episode
                        else None,
                        "max_drawdown": best_episode.metrics.max_drawdown
                        if best_episode
                        else None,
                    }
                    if best_episode
                    else None,
                    "department_controls": best_episode.department_controls
                    if best_episode
                    else None,
                },
                "experiment_id": config.experiment_id,
                "strategy": config.strategy,
            }
            st.success(f"自动探索完成，共执行 {len(episodes_dump)} 轮。")
            _render_bandit_summary(st.session_state[_DECISION_ENV_BANDIT_RESULTS_KEY], app_cfg)


def render_tuning_lab() -> None:
    st.header("实验调参")
    st.caption("统一管理强化学习、Bandit、Bayesian/BOHB 等自动调参实验，并可回写最佳参数。")

    nav_cols = st.columns([1, 1, 3])
    if nav_cols[0].button("返回回测与复盘", key="tuning_go_backtest"):
        navigate_top_menu("回测与复盘")
    nav_cols[1].info("顶部导航也可随时切换至其它视图。")
    nav_cols[2].markdown("完成实验后，记得回到回测页面验证策略表现。")

    app_cfg = get_config()
    portfolio_snapshot = get_portfolio_settings_snapshot()
    default_start, default_end = default_backtest_range(window_days=60)

    manage_tab, search_tab, log_tab, ppo_tab = st.tabs(
        ["策略实验管理", "参数搜索", "RL/BOHB 日志", "强化学习 (PPO)"]
    )

    manage_context: Dict[str, object] = {}
    with manage_tab:
        manage_context = _render_experiment_management(
            app_cfg,
            portfolio_snapshot,
            default_start,
            default_end,
        )

    with search_tab:
        _render_parameter_search(app_cfg, manage_context)

    with log_tab:
        _render_bandit_logs(
            st.session_state.get(_DECISION_ENV_BANDIT_RESULTS_KEY)
        )

    with ppo_tab:
        _render_ppo_training(
            app_cfg,
            manage_context,
            st.session_state.get(_DECISION_ENV_PPO_RESULTS_KEY),
        )
