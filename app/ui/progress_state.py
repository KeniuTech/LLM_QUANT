"""UI进度状态管理模块，用于因子计算进度同步。"""
from __future__ import annotations

from typing import Optional, Dict, Any
import streamlit as st


class FactorProgressState:
    """因子计算进度状态管理类"""
    
    def __init__(self):
        """初始化进度状态"""
        if 'factor_progress' not in st.session_state:
            st.session_state.factor_progress = {
                'current': 0,
                'total': 0,
                'percentage': 0.0,
                'current_batch': 0,
                'total_batches': 0,
                'status': 'idle',  # idle, running, completed, error
                'message': '',
                'start_time': None,
                'elapsed_time': 0
            }
    
    def start_calculation(self, total_securities: int, total_batches: int) -> None:
        """开始因子计算
        
        Args:
            total_securities: 总证券数量
            total_batches: 总批次数
        """
        st.session_state.factor_progress.update({
            'current': 0,
            'total': total_securities,
            'percentage': 0.0,
            'current_batch': 0,
            'total_batches': total_batches,
            'status': 'running',
            'message': '开始因子计算...',
            'start_time': st.session_state.get('factor_progress', {}).get('start_time'),
            'elapsed_time': 0
        })
    
    def update_progress(self, current_securities: int, current_batch: int, 
                       message: str = '') -> None:
        """更新计算进度
        
        Args:
            current_securities: 当前已处理证券数量
            current_batch: 当前批次
            message: 进度消息
        """
        progress = st.session_state.factor_progress
        
        # 计算百分比
        if progress['total'] > 0:
            percentage = (current_securities / progress['total']) * 100
        else:
            percentage = 0.0
        
        # 更新状态
        progress.update({
            'current': current_securities,
            'current_batch': current_batch,
            'percentage': percentage,
            'message': message or f'处理批次 {current_batch}/{progress["total_batches"]}',
            'status': 'running'
        })
    
    def complete_calculation(self, message: str = '因子计算完成') -> None:
        """完成因子计算
        
        Args:
            message: 完成消息
        """
        progress = st.session_state.factor_progress
        progress.update({
            'current': progress['total'],
            'percentage': 100.0,
            'status': 'completed',
            'message': message
        })
    
    def error_occurred(self, error_message: str) -> None:
        """发生错误
        
        Args:
            error_message: 错误消息
        """
        st.session_state.factor_progress.update({
            'status': 'error',
            'message': f'错误: {error_message}'
        })
    
    def get_progress_info(self) -> Dict[str, Any]:
        """获取当前进度信息
        
        Returns:
            进度信息字典
        """
        return st.session_state.factor_progress.copy()
    
    def reset(self) -> None:
        """重置进度状态"""
        st.session_state.factor_progress = {
            'current': 0,
            'total': 0,
            'percentage': 0.0,
            'current_batch': 0,
            'total_batches': 0,
            'status': 'idle',
            'message': '',
            'start_time': None,
            'elapsed_time': 0
        }


# 全局进度状态实例
factor_progress = FactorProgressState()


def render_factor_progress() -> None:
    """渲染因子计算进度组件"""
    progress_info = factor_progress.get_progress_info()
    
    # 创建进度显示区域
    with st.container():
        st.subheader("📊 因子计算进度")
        
        # 空闲状态显示提示信息
        if progress_info['status'] == 'idle':
            st.info("当前没有因子计算任务。执行因子计算时，进度将在此显示。")
            return
        
        # 进度条
        if progress_info['status'] == 'running':
            st.progress(progress_info['percentage'] / 100.0)
        
        # 进度信息
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "处理进度",
                f"{progress_info['current']}/{progress_info['total']}",
                f"{progress_info['percentage']:.1f}%"
            )
        
        with col2:
            st.metric(
                "批次进度",
                f"{progress_info['current_batch']}/{progress_info['total_batches']}",
                "批次"
            )
        
        with col3:
            status_icon = {
                'running': '🔄',
                'completed': '✅',
                'error': '❌',
                'idle': '⏸️'
            }.get(progress_info['status'], '⏸️')
            st.metric(
                "状态",
                progress_info['status'].capitalize(),
                status_icon
            )
        
        # 消息显示
        if progress_info['message']:
            st.info(progress_info['message'])
        
        # 错误状态特殊处理
        if progress_info['status'] == 'error':
            st.error("因子计算过程中发生错误，请检查日志")
        elif progress_info['status'] == 'completed':
            st.success("因子计算已完成")


def get_factor_progress_percentage() -> float:
    """获取因子计算进度百分比
    
    Returns:
        进度百分比 (0-100)
    """
    return factor_progress.get_progress_info()['percentage']


def is_factor_calculation_running() -> bool:
    """检查因子计算是否正在进行
    
    Returns:
        是否正在进行因子计算
    """
    return factor_progress.get_progress_info()['status'] == 'running'