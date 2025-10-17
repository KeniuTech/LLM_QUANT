"""Helpers for navigating between top-level Streamlit menus."""
from __future__ import annotations

import streamlit as st

TOP_NAV_STATE_KEY = "top_nav"


def navigate_top_menu(label: str) -> None:
    """Set the active top navigation label and rerun the app."""
    st.session_state[TOP_NAV_STATE_KEY] = label
    rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun is None:  # pragma: no cover - defensive guard for unexpected API changes
        raise RuntimeError("Streamlit rerun helper is unavailable")
    rerun()
