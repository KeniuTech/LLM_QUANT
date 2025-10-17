from __future__ import annotations

from datetime import date

from app.features.factor_audit import audit_factors
from app.features.factors import compute_factors
from app.utils.db import db_session
from tests.factor_utils import populate_sample_data


def test_audit_matches_persisted_values(isolated_db):
    ts_code = "000001.SZ"
    trade_day = date(2025, 2, 14)
    populate_sample_data(ts_code, trade_day)

    compute_factors(trade_day)
    summary = audit_factors(trade_day)

    assert summary.mismatched == 0
    assert summary.missing_persisted == 0
    assert summary.missing_recomputed == 0
    assert not summary.issues


def test_audit_detects_drift(isolated_db):
    ts_code = "000001.SZ"
    trade_day = date(2025, 2, 14)
    populate_sample_data(ts_code, trade_day)

    compute_factors(trade_day)

    trade_date_str = trade_day.strftime("%Y%m%d")
    with db_session() as conn:
        conn.execute(
            "UPDATE factors SET mom_5 = mom_5 + 0.05 WHERE ts_code = ? AND trade_date = ?",
            (ts_code, trade_date_str),
        )

    summary = audit_factors(trade_day, factors=["mom_5"], tolerance=1e-8, max_issues=5)

    assert summary.mismatched >= 1
    assert summary.issues
    first_issue = summary.issues[0]
    assert first_issue.ts_code == ts_code
    assert first_issue.factor == "mom_5"
