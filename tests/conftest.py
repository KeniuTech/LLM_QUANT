from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.schema import initialize_database
from app.utils.config import DataPaths, get_config


@pytest.fixture()
def isolated_db(tmp_path):
    cfg = get_config()
    original_paths = cfg.data_paths
    tmp_root = tmp_path / "data"
    tmp_root.mkdir(parents=True, exist_ok=True)
    cfg.data_paths = DataPaths(root=tmp_root)
    try:
        initialize_database()
        yield
    finally:
        cfg.data_paths = original_paths
