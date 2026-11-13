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


# ---------------------------------------------------------------------------
# Trailing stop, spread filter, performance stats, dynamic leverage
# ---------------------------------------------------------------------------
import json
from datetime import datetime, timezone

from binance_ai_bot.domain import OrderBookLevel, OrderBookSnapshot
from binance_ai_bot.engine import TradingSystemEngine
from cloud.api.app import CloudBotApplication, CloudConfig
from tests.test_cloud_api import FakeMarketClient, FakeStateSync


def _open_first_position(app, market):
    r = app.handle("POST", "/api/tick", AUTH)
    assert r.status == 200
    eng = app._get_engine()
    pos = eng.storage.get_open_positions(track="MAIN")[0]
    return eng, pos


def test_trailing_stop_ratchets_then_closes_in_profit_and_stats_report_it(tmp_path: Path):
    app, _, market = _make_app(tmp_path)
    eng, pos = _open_first_position(app, market)
    bid = market.fetch_order_book(pos.symbol).best_bid

    # Pozisyonu yapay olarak kârda göster: giriş fiyatını düşür, TP'yi geçmiş say
    entry = bid * Decimal("0.98")
    eng.storage.update_position_remaining(
        pos.position_id,
        quantity=pos.quantity,
        cost_basis_usdc=pos.cost_basis_usdc,
        entry_fee_usdc=pos.entry_fee_usdc,
        entry_price=entry,
        stop_loss_price=entry * Decimal("0.97"),
        take_profit_price=entry * Decimal("1.0050"),
        update_exit_targets=True,
    )
    app.handle("POST", "/api/tick", AUTH)
    still = eng.storage.get_position(pos.position_id, track="MAIN")
    assert still is not None and still.status == "OPEN"
    assert still.stop_loss_price > entry  # stop kâra çekildi, satılmadı
    assert still.take_profit_price > bid  # hedef yukarı taşındı

    # Fiyat yeni stop'un altına düştü -> TRAILING_KAR ile kârda kapanır
    eng.storage.update_position_remaining(
        pos.position_id,
        quantity=still.quantity,
        cost_basis_usdc=still.cost_basis_usdc,
        entry_fee_usdc=still.entry_fee_usdc,
        stop_loss_price=bid * Decimal("1.01"),
        take_profit_price=bid * Decimal("1.50"),
        update_exit_targets=True,
    )
    app.handle("POST", "/api/tick", AUTH)
    closed = eng.storage.get_position(pos.position_id, track="MAIN")
    assert closed.status == "CLOSED"
    assert closed.close_reason == "TRAILING_KAR"

    dash = app.handle("GET", "/api/dashboard", AUTH).body
    assert dash["performance"]["closed_count"] >= 1
    assert any(r["reason"] == "TRAILING_KAR" for r in dash["performance"]["by_reason"])
    assert dash["risk_settings"]["trailing_stop_pct"] != "0"
    assert "cpu_usage" in dash and dash["cpu_usage"]["month_sec"] >= 0


class _WideSpreadMarket(FakeMarketClient):
    def fetch_order_book(self, symbol: str, depth_limit: int = 20) -> OrderBookSnapshot:
        px = self.price_by_symbol.get(symbol, Decimal("100.00"))
        return OrderBookSnapshot(
            symbol=symbol,
            bids=[OrderBookLevel(price=px, quantity=Decimal("500"))],
            asks=[OrderBookLevel(price=px * Decimal("1.01"), quantity=Decimal("500"))],
            fetched_at_utc=datetime.now(timezone.utc),
            source_url="https://api.binance.com/api/v3/depth",
        )


def test_wide_spread_blocks_new_entries(tmp_path: Path):
    cfg = CloudConfig(
        state_bucket="x",
        cron_secret="c" * 32,
        db_path=tmp_path / "cloud.db",
        initial_balance_usdc=Decimal("86.00"),
    )
    sync = FakeStateSync()
    eng = TradingSystemEngine(cfg.to_app_config(), market_client=_WideSpreadMarket())
    eng.storage.set_state("cloud_scan_symbols", "BTCUSDT,ETHUSDT")
    app = CloudBotApplication(cfg, engine=eng, state_sync=sync, verifier=lambda t, a: {})
    summary = app._run_scan_cycle(eng, source="TEST")
    assert summary["orders_executed"] == 0
    assert eng.storage.get_open_positions(track="MAIN") == []
    assert any("alış-satış farkı geniş" in it["note"] for it in summary["items"])


def test_dynamic_leverage_derates_when_margin_level_low(tmp_path: Path):
    cfg = CloudConfig(
        state_bucket="x",
        cron_secret="c" * 32,
        db_path=tmp_path / "cloud.db",
        initial_balance_usdc=Decimal("86.00"),
        enable_live_trading=True,
        binance_api_key="k" * 64,
        binance_api_secret="s" * 64,
    )
    eng = TradingSystemEngine(cfg.to_app_config(), market_client=FakeMarketClient())
    app = CloudBotApplication(cfg, engine=eng, state_sync=FakeStateSync(), verifier=lambda t, a: {})
    st = eng.storage
    for k, v in {
        "cloud_mode": "LIVE_MARGIN",
        "cloud_live_confirmed": "1",
        "cloud_margin_confirmed": "1",
        "cloud_leverage": "5",
        "live_margin_level": "1.20",
    }.items():
        st.set_state(k, v)
    p = app._active_cloud_params(eng)
    assert p["mode"] == "LIVE_MARGIN"
    assert p["leverage_selected"] == 5
    assert p["leverage"] == 2 and p["settings"].leverage == 2
    assert "Marjin seviyesi düşük" in p["derate_reason"]

    st.set_state("live_margin_level", "1.80")
    p2 = app._active_cloud_params(eng)
    assert p2["leverage"] == 5 and p2["derate_reason"] == ""  # seçilen değer asla aşılmaz, risk yoksa korunur
