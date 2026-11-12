"""Leverage-aware targets and LIVE_MARGIN gating tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from binance_ai_bot.domain import RiskSettings
from binance_ai_bot.profiles import get_profile
from tests.test_cloud_api import _make_app

AUTH = {"Authorization": "Bearer owner-token"}


def test_leverage_targets_keep_stop_far_from_liquidation_and_lock_profit_sooner():
    prof = get_profile("AL_YUKSELINCE_SAT")
    base = RiskSettings(max_open_positions=5, max_position_size_usdc=Decimal("15"))
    s1 = prof.apply_to_risk_settings(base, leverage=1)
    assert s1.stop_loss_pct == Decimal("0.035")
    assert s1.min_net_profit_exit_pct == Decimal("0.0080")
    assert s1.take_profit_pct == Decimal("0.0115")

    prev_sl = s1.stop_loss_pct
    prev_min = s1.min_net_profit_exit_pct
    for lev in (2, 3, 4, 5):
        s = prof.apply_to_risk_settings(base, leverage=lev)
        # SL tightens with leverage; collateral risk stays bounded (< 5%)
        assert s.stop_loss_pct < prev_sl
        assert s.stop_loss_pct * lev <= Decimal("0.05")
        # liquidation is ~ (1 - 1/(lev*...)) far beyond the stop distance
        assert s.stop_loss_pct < Decimal("1") / Decimal(lev) * Decimal("0.5")
        # Net-lock target in price terms is smaller, but ROI on collateral rises
        assert s.min_net_profit_exit_pct <= prev_min
        assert s.min_net_profit_exit_pct * lev > Decimal("0.0080")
        assert s.take_profit_pct > s.min_net_profit_exit_pct
        prev_sl, prev_min = s.stop_loss_pct, s.min_net_profit_exit_pct


def test_live_margin_is_not_activated_without_server_keys_and_falls_back_to_paper(tmp_path: Path):
    app, _, _ = _make_app(tmp_path)
    r = app.handle(
        "POST", "/api/settings", AUTH, b'{"mode":"LIVE_MARGIN","confirm_text":"EVET","leverage":3}'
    )
    assert r.status == 400  # live trading disabled in test config
    dash = app.handle("GET", "/api/dashboard", AUTH).body
    assert dash["mode"] == "PAPER"
    assert dash["margin_confirmed"] is False


def test_paper_leverage_change_updates_effective_targets(tmp_path: Path):
    app, _, _ = _make_app(tmp_path)
    r = app.handle("POST", "/api/settings", AUTH, b'{"mode":"PAPER","leverage":5}')
    assert r.status == 200
    dash = r.body["dashboard"]
    assert dash["leverage"] == 5
    assert Decimal(dash["risk_settings"]["stop_loss_pct"]) < Decimal("1.00")
