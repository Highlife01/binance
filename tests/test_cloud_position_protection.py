"""Settings changes must preserve risk already allocated to open trades."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from binance_ai_bot.domain import Position
from tests.test_cloud_api import _make_app


def open_position(app, *, stop="98.70", target="101.80"):
    eng = app._get_engine()
    eng.storage.set_state("cloud_mode", "PAPER")
    eng.storage.set_state("cloud_leverage", "1")
    pos = Position(
        position_id="protected-trade", symbol="BTCUSDT", quantity=Decimal("0.1"),
        entry_price=Decimal("100"), cost_basis_usdc=Decimal("10"),
        entry_fee_usdc=Decimal("0.01"), opened_at_iso=datetime.now(timezone.utc).isoformat(),
        signal_key="protected-trade", strategy_source="TEST", leverage=1,
        stop_loss_price=Decimal(stop), take_profit_price=Decimal(target),
    )
    eng.storage.insert_position(pos, track="MAIN")
    return eng, pos


@pytest.mark.parametrize("payload", [
    b'{"daily_loss_limit_usdc":"1.50"}',
    b'{"max_position_size_usdc":"12.50"}',
    b'{"max_open_positions":3}',
    b'{"leverage":3}',
    b'{"mode":"PAPER"}',
    b'{"profile":"AL_YUKSELINCE_SAT"}',
])
def test_budget_edits_preserve_existing_atr_exits(tmp_path, payload):
    app, _, _ = _make_app(tmp_path)
    eng, original = open_position(app)
    result = app.handle("POST", "/api/settings", {"Authorization": "Bearer owner-token"}, payload)
    assert result.status == 200
    saved = eng.storage.get_position(original.position_id, track="MAIN")
    assert saved.stop_loss_price == original.stop_loss_price
    assert saved.take_profit_price == original.take_profit_price
    assert saved.leverage == 1


def test_profile_changes_tighten_stop_and_never_widen_it(tmp_path):
    app, _, _ = _make_app(tmp_path)
    eng, original = open_position(app)
    auth = {"Authorization": "Bearer owner-token"}
    result = app.handle("POST", "/api/settings", auth, b'{"profile":"HIZLI_30SN"}')
    assert result.status == 200
    tightened = eng.storage.get_position(original.position_id, track="MAIN")
    assert tightened.stop_loss_price == Decimal("99.6500")
    result = app.handle("POST", "/api/settings", auth, b'{"profile":"TEMKINLI"}')
    assert result.status == 200
    saved = eng.storage.get_position(original.position_id, track="MAIN")
    assert saved.stop_loss_price == tightened.stop_loss_price


def test_profile_changes_preserve_armed_trailing_exits(tmp_path):
    app, _, _ = _make_app(tmp_path)
    eng, original = open_position(app, stop="101.20", target="103")
    result = app.handle("POST", "/api/settings", {"Authorization": "Bearer owner-token"},
                        b'{"profile":"TEMKINLI"}')
    assert result.status == 200
    saved = eng.storage.get_position(original.position_id, track="MAIN")
    assert saved.stop_loss_price == original.stop_loss_price
    assert saved.take_profit_price == original.take_profit_price


@pytest.mark.parametrize("stop", ["98.70", "101.20"])
def test_legacy_target_migration_never_loosens_existing_stop(tmp_path, stop):
    app, _, _ = _make_app(tmp_path)
    eng, original = open_position(app, stop=stop)
    eng.storage.set_state("profit_target_v2_15usdt", "0")
    app._ensure_cloud_defaults(eng.storage)
    saved = eng.storage.get_position(original.position_id, track="MAIN")
    assert saved.stop_loss_price >= original.stop_loss_price
    if original.stop_loss_price > original.entry_price:
        assert saved.take_profit_price == original.take_profit_price


def test_scan_budget_does_not_skip_protection_of_later_open_positions(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from cloud.api import app as cloud_module

    app, _, _ = _make_app(tmp_path)
    eng, original = open_position(app)
    from dataclasses import replace
    eng.storage.insert_position(replace(original, position_id="second", symbol="ETHUSDT"), track="MAIN")
    eng.storage.set_state("cloud_dynamic_scan", "0")
    eng.storage.set_state("cloud_scan_symbols", "BTCUSDT,ETHUSDT,SOLUSDT")
    # After the cycle starts, emulate a stalled connection that consumes >45s.
    ticks = iter([0.0])
    monkeypatch.setattr(cloud_module.time, "monotonic", lambda: next(ticks, 100.0))
    scanned = []

    def step(**kwargs):
        scanned.append(kwargs["symbol"])
        return SimpleNamespace(symbol=kwargs["symbol"], ok=True, final_action="BEKLE",
                               executed_order=None, error_message=None, block_or_info_message="test")

    monkeypatch.setattr(eng, "run_analysis_step", step)
    app._run_scan_cycle(eng, source="test")
    assert scanned == ["BTCUSDT", "ETHUSDT"]
