"""Ağ ve gerçek hesap kullanmadan durdurma, panel ve rapor ayrımı kontrolleri."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import logging
from pathlib import Path

import httpx
import pytest

from binance_ai_bot.config import AppConfig
from binance_ai_bot.domain import Position, RiskSettings
from binance_ai_bot.engine import StepCycleSummary
from binance_ai_bot import runner
from binance_ai_bot.storage import SQLiteStorage


class _MemoryState:
    def __init__(self):
        self.values = {}

    def get_state(self, key, default=""):
        return self.values.get(key, default)

    def set_state(self, key, value):
        self.values[key] = value


def test_long_backoff_keeps_heartbeat_alive(monkeypatch):
    storage = _MemoryState()
    start = datetime(2026, 11, 12, tzinfo=timezone.utc)
    elapsed = 0.0
    sleeps = []

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return start + timedelta(seconds=elapsed)

    monkeypatch.setattr(runner, "datetime", _Clock)

    def sleep(seconds):
        nonlocal elapsed
        sleeps.append(seconds)
        elapsed += seconds
        assert runner.is_runner_alive(storage, now_utc=_Clock.now())[0]

    assert runner._wait_with_heartbeat(storage, 600, sleep)
    assert elapsed == 600
    assert max(sleeps) <= 5
    assert runner.is_runner_alive(storage, now_utc=_Clock.now())[0]


def test_stop_interrupts_backoff_within_five_seconds():
    storage = _MemoryState()
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        storage.set_state(runner.RUNNER_STOP_KEY, "1")

    assert runner._wait_with_heartbeat(storage, 600, sleep) is False
    assert sleeps == [5.0]


def test_stop_during_scan_skips_remaining_coins(tmp_path, monkeypatch):
    storage = SQLiteStorage(tmp_path / "runner.db")
    monkeypatch.setattr(runner, "_setup_logger", lambda: logging.getLogger("runner-test"))
    monkeypatch.setattr(runner, "_keep_windows_awake", lambda _: None)

    class _Engine:
        def __init__(self):
            self.storage = storage
            self.symbols = []

        def default_risk_settings(self):
            return RiskSettings(max_open_positions=5)

        def run_analysis_step(self, **kwargs):
            self.symbols.append(kwargs["symbol"])
            assert kwargs["stop_requested"]() is False
            storage.set_state(runner.RUNNER_STOP_KEY, "1")
            assert kwargs["stop_requested"]() is True
            return StepCycleSummary(
                ok=True, timestamp_iso=datetime.now(timezone.utc).isoformat(),
                symbol=kwargs["symbol"], interval=kwargs["interval"], data_source="test",
                wake_gap_detected=False, tech_signal=None, ai_decision=None,
                final_action="BEKLE", executed_order=None, block_or_info_message="test",
            )

    engine = _Engine()
    sleeps = []
    result = runner.run_bot(
        ["--mod", "sanal", "--pariteler", "BTCUSDT,ETHUSDT,SOLUSDT", "--max-dongu", "2"],
        engine=engine,
        config=AppConfig(sqlite_db_path=tmp_path / "runner.db", quote_asset="USDT"),
        sleep_fn=sleeps.append,
        lock_path=tmp_path / "runner.lock",
    )
    assert result == 0
    assert engine.symbols == ["BTCUSDT"]
    assert sleeps == []
    assert runner.is_runner_alive(storage)[0] is False


@pytest.mark.parametrize("cli_leverage, expected", [(None, 3), (2, 2)])
def test_paper_runner_accepts_launcher_leverage_and_preserves_saved_default(
    tmp_path, monkeypatch, cli_leverage, expected
):
    storage = SQLiteStorage(tmp_path / "paper-leverage.db")
    monkeypatch.setattr(runner, "_setup_logger", lambda: logging.getLogger("runner-test"))
    monkeypatch.setattr(runner, "_keep_windows_awake", lambda _: None)
    observed = []

    class _Engine:
        def __init__(self):
            self.storage = storage

        def default_risk_settings(self):
            return RiskSettings(leverage=3)

        def run_analysis_step(self, **kwargs):
            observed.append(kwargs["settings"].leverage)
            return StepCycleSummary(
                ok=True, timestamp_iso=datetime.now(timezone.utc).isoformat(),
                symbol=kwargs["symbol"], interval=kwargs["interval"], data_source="test",
                wake_gap_detected=False, tech_signal=None, ai_decision=None,
                final_action="BEKLE", executed_order=None, block_or_info_message="test",
            )

    argv = ["--mod", "sanal", "--max-dongu", "1"]
    if cli_leverage is not None:
        argv += ["--kaldirac", str(cli_leverage)]
    assert runner.run_bot(argv, engine=_Engine(), config=AppConfig(sqlite_db_path=storage.db_path),
                          lock_path=tmp_path / "paper-leverage.lock") == 0
    assert observed == [expected]


def test_live_spot_runner_rejects_leverage_before_reading_account(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(runner, "_setup_logger", lambda: logging.getLogger("runner-test"))

    class _Engine:
        def default_risk_settings(self):
            raise AssertionError("Leveraged live invocation must stop before touching the engine")

    result = runner.run_bot(
        ["--mod", "canli", "--kaldirac", "2"], engine=_Engine(),
        config=AppConfig(sqlite_db_path=tmp_path / "unused.db"),
    )
    assert result == 2
    assert "1x" in caplog.text


def test_margin_runner_remains_blocked_before_engine_or_confirmation(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(runner, "_setup_logger", lambda: logging.getLogger("runner-test"))

    class _Engine:
        def default_risk_settings(self):
            raise AssertionError("Unverified live margin must not touch the engine")

    def no_confirmation(prompt):
        raise AssertionError("Unverified live margin must not reach confirmation")

    assert runner.run_bot(
        ["--mod", "marjin", "--kaldirac", "2"], engine=_Engine(),
        config=AppConfig(sqlite_db_path=tmp_path / "unused-margin.db"), input_fn=no_confirmation,
    ) == 2
    assert "guvenlik dogrulamasina dahil degil" in caplog.text


def _dashboard(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from streamlit.testing.v1 import local_script_runner
    from binance_ai_bot import config

    # AppTest never drives this loop. Windows socketpair() cannot complete in
    # the restricted test host, so provide only its lifecycle interface here.
    class _RenderOnlyLoop(asyncio.AbstractEventLoop):
        def __init__(self):
            self._closed = False

        def is_closed(self):
            return self._closed

        def close(self):
            self._closed = True

    monkeypatch.setattr(local_script_runner.asyncio, "new_event_loop", _RenderOnlyLoop)
    cfg = AppConfig(sqlite_db_path=tmp_path / "panel.db", quote_asset="USDC", default_symbol="BTCUSDC")
    monkeypatch.setattr(config, "load_config", lambda: cfg)

    network_calls = []

    def forbid_network(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("Panel çizimi ağ isteği göndermemeli")

    monkeypatch.setattr(httpx.Client, "send", forbid_network)
    app_file = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_file), default_timeout=20)
    app._blocked_network_calls = network_calls
    return app, cfg


@pytest.mark.parametrize("runner_active", [False, True])
def test_dashboard_initial_render_is_offline_and_runner_blocks_manual_close(
    tmp_path, monkeypatch, runner_active
):
    app, cfg = _dashboard(tmp_path, monkeypatch)
    storage = SQLiteStorage(cfg.sqlite_db_path)
    if runner_active:
        storage.set_state(runner.RUNNER_HEARTBEAT_KEY, datetime.now(timezone.utc).isoformat())
        storage.set_state(runner.RUNNER_INFO_KEY, json.dumps({"mode": "PAPER", "profile": "AKTIF"}))
    app.run()
    assert not app.exception
    assert not app._blocked_network_calls
    analysis = next(b for b in app.button if "Analiz Adımını" in b.label)
    close = next(b for b in app.button if "Pozisyonları Kapat" in b.label)
    assert analysis.disabled is runner_active
    assert close.disabled is runner_active


def test_dashboard_quote_mismatch_blocks_orders_and_labels_actual_currency(tmp_path, monkeypatch):
    app, cfg = _dashboard(tmp_path, monkeypatch)
    storage = SQLiteStorage(cfg.sqlite_db_path)
    storage.ensure_quote_asset("MAIN", "USDT")
    app.run()
    assert not app.exception
    assert not app._blocked_network_calls
    cash = next(m for m in app.metric if m.label == "Kullanılabilir Nakit")
    assert cash.value.endswith("USDT")
    assert next(b for b in app.button if "Analiz Adımını" in b.label).disabled
    assert next(b for b in app.button if "Pozisyonları Kapat" in b.label).disabled
    assert any("Bu portföy USDT" in w.value for w in app.warning)


def test_open_position_render_does_not_eagerly_fetch_market_prices(tmp_path, monkeypatch):
    app, cfg = _dashboard(tmp_path, monkeypatch)
    storage = SQLiteStorage(cfg.sqlite_db_path)
    storage.insert_position(Position(
        position_id="PANEL-POS", symbol="BTCUSDC", quantity=Decimal("0.0001"),
        entry_price=Decimal("100000"), cost_basis_usdc=Decimal("10"),
        entry_fee_usdc=Decimal("0.01"), opened_at_iso=datetime.now(timezone.utc).isoformat(),
        signal_key="PANEL-SIGNAL", strategy_source="TEST", stop_loss_price=Decimal("99000"),
        take_profit_price=Decimal("102000"), status="OPEN",
    ))
    app.run()
    assert not app.exception
    assert not app._blocked_network_calls
    assert any("güncel fiyatı yok" in w.value for w in app.warning)
    labels_without_prices = {
        "Toplam Portföy Değeri", "Açık Pozisyon Net K/Z", "Açık Pozisyon Anlık K/Z",
        "Bugünkü Net K/Z (İstanbul)", "Toplam Net K/Z (Gerçekleşen + Açık)", "Anlık Portföy Değeri",
    }
    assert {m.label for m in app.metric if m.label in labels_without_prices} == labels_without_prices
    assert all(m.value == "—" for m in app.metric if m.label in labels_without_prices)


def test_manual_price_refresh_updates_top_and_live_metrics_in_one_click(tmp_path, monkeypatch):
    from binance_ai_bot.domain import OrderBookLevel, OrderBookSnapshot
    from binance_ai_bot.market_data import BinancePublicDataClient

    app, cfg = _dashboard(tmp_path, monkeypatch)
    storage = SQLiteStorage(cfg.sqlite_db_path)
    storage.insert_position(Position(
        position_id="REFRESH-POS", symbol="BTCUSDC", quantity=Decimal("0.0001"),
        entry_price=Decimal("100000"), cost_basis_usdc=Decimal("10"),
        entry_fee_usdc=Decimal("0.01"), opened_at_iso=datetime.now(timezone.utc).isoformat(),
        signal_key="REFRESH-SIGNAL", strategy_source="TEST", stop_loss_price=None,
        take_profit_price=None, status="OPEN",
    ))
    fetched = []

    def book(self, symbol, **kwargs):
        fetched.append(symbol)
        return OrderBookSnapshot(
            symbol=symbol, bids=[OrderBookLevel(Decimal("110000"), Decimal("1"))],
            asks=[OrderBookLevel(Decimal("110001"), Decimal("1"))],
            fetched_at_utc=datetime.now(timezone.utc), source_url="https://api.binance.me",
        )

    monkeypatch.setattr(BinancePublicDataClient, "fetch_order_book", book)
    app.run()
    assert not app.exception
    assert fetched == []
    next(b for b in app.button if "Anlık Fiyatları Şimdi Güncelle" in b.label).click().run()
    assert not app.exception
    assert fetched == ["BTCUSDC"]
    assert next(m for m in app.metric if m.label == "Açık Pozisyon Net K/Z").value == "+0.979 USDC"
    assert next(m for m in app.metric if m.label == "Açık Pozisyon Anlık K/Z").value == "+0.9790 USDC"
    assert not app._blocked_network_calls


def test_synthetic_demo_uses_selected_profile_and_does_not_claim_validation(tmp_path, monkeypatch):
    from binance_ai_bot import validation

    app, _ = _dashboard(tmp_path, monkeypatch)
    real_validate = validation.run_chronological_validation
    captured = {}

    def validate(candles, **kwargs):
        captured.update(kwargs)
        captured["step_ms"] = candles[1].open_time_ms - candles[0].open_time_ms
        return real_validate(candles, **kwargs)

    monkeypatch.setattr(validation, "run_chronological_validation", validate)
    app.run()
    assert not app.exception
    next(b for b in app.button if "Sentetik Veriyle" in b.label).click().run()
    assert not app.exception
    assert captured["strategy_params"].ema_fast_span == 5
    assert captured["settings"].stop_loss_pct == Decimal("0.0035")
    assert captured["step_ms"] == 60_000
    assert any("Stratejinin kârlılığını" in i.value for i in app.info)
    assert not any("Örneklem Dışı Test Özeti" in s.value for s in app.success)
    assert not app._blocked_network_calls


@pytest.mark.parametrize("exit_path", ["BUY", "STRATEGY_SELL", "STOP_LOSS", "DAILY_LOSS"])
def test_engine_forwards_stop_callback_through_order_preflight(tmp_path, monkeypatch, exit_path):
    """A stop arriving inside order/test must reach the executor from every engine path."""
    from binance_ai_bot import engine as engine_module
    from binance_ai_bot.domain import Candle, MarketBundle, OrderBookLevel, OrderBookSnapshot, TechnicalSignal
    from binance_ai_bot.engine import TradingSystemEngine
    from test_live_safety import filters, position, setup

    executor, storage, exchange = setup(tmp_path)
    stopped = {"value": False}
    exchange.after_test = lambda: stopped.update(value=True)
    px = Decimal("0.7") if exit_path == "DAILY_LOSS" else (
        Decimal("0.8") if exit_path == "STOP_LOSS" else Decimal("1")
    )
    now = datetime.now(timezone.utc)
    stamp = int(now.timestamp() * 1000) - 60_000
    candle = Candle(
        open_time_ms=stamp - 59_999, close_time_ms=stamp, open=px, high=px, low=px,
        close=px, volume=Decimal("100"), quote_volume=Decimal("100"), trade_count=10,
        is_closed=True,
    )
    book = OrderBookSnapshot(
        symbol="TESTUSDT", bids=[OrderBookLevel(px, Decimal("100"))],
        asks=[OrderBookLevel(px, Decimal("100"))], fetched_at_utc=now,
        source_url="https://api.binance.me",
    )
    market = MarketBundle(
        symbol="TESTUSDT", interval="1m", filters=filters(), closed_candles=[candle],
        order_book=book, server_time_ms=stamp + 60_000, fetched_at_utc=now, source_label="mock",
    )
    action = "AL" if exit_path == "BUY" else ("SAT" if exit_path == "STRATEGY_SELL" else "BEKLE")
    signal = TechnicalSignal(
        symbol="TESTUSDT", interval="1m", candle_open_time_ms=candle.open_time_ms,
        candle_close_time_ms=candle.close_time_ms, action=action, reason="Test signal",
        ema_fast=px, ema_slow=px, rsi_14=Decimal("50"), macd_line=Decimal("0"),
        macd_signal=Decimal("0"), macd_hist=Decimal("0"), atr_14=Decimal("0.01"),
        volume_ratio=Decimal("1"), close_price=px,
    )
    monkeypatch.setattr(engine_module, "evaluate_technical_strategy", lambda *a, **kw: (signal, []))

    class _Market:
        def fetch_verified_market_bundle(self, **kwargs):
            return market

    eng = TradingSystemEngine(executor.config, storage=storage, market_client=_Market())
    eng.live_executor = executor
    monkeypatch.setattr(eng, "_update_rule_shadow_track", lambda *a, **kw: None)
    if exit_path != "BUY":
        position(storage)
    settings = RiskSettings(
        daily_loss_limit_usdc=Decimal("2") if exit_path == "DAILY_LOSS" else Decimal("100"),
        daily_limit_exit_policy="AUTO_CLOSE" if exit_path == "DAILY_LOSS" else "HOLD_EXISTING",
    )
    result = eng.run_analysis_step(
        symbol="TESTUSDT", interval="1m", settings=settings, execution_mode="LIVE_SPOT",
        ui_live_confirmed=True, ai_enabled_by_user=False, stop_requested=lambda: stopped["value"],
    )
    assert stopped["value"] is True, "Test must actually reach the exchange preflight"
    assert exchange.posts == []
    assert result.executed_order is None
    assert "Durdurma isteği" in (result.block_or_info_message + (result.error_message or ""))
    if exit_path != "BUY":
        assert storage.get_open_positions(track="LIVE_SPOT")[0].quantity == Decimal("10")
