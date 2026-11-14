"""Unit & integration tests for Quant Desk (6-repo hedge fund core) and production stability fixes."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import gzip
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from binance_ai_bot.config import AppConfig
from binance_ai_bot.domain import (
    Candle,
    MarketBundle,
    OrderBookLevel,
    OrderBookSnapshot,
    Position,
    RiskSettings,
    SymbolFilters,
    is_effective_dust_position,
    is_substep_dust_position,
)
from binance_ai_bot.engine import TradingSystemEngine
from binance_ai_bot.live_executor import LiveExecutionError
from binance_ai_bot.market_data import BinancePublicDataClient
from binance_ai_bot.quant_desk import (
    SelfRetrainingSignalModel,
    build_research_plan,
    compute_portfolio_risk_metrics,
    evaluate_microstructure,
    forecast_ohlcv_candles,
    optimize_hrp_weights,
    process_telegram_command,
    run_quant_research,
)
from binance_ai_bot.storage import SQLiteStorage
from cloud.api.app import CloudBotApplication, CloudConfig, GcsSqliteSync, SQLITE_MAGIC


def _make_filters(symbol: str = "BTCUSDT", *, min_notional: str = "5.0") -> SymbolFilters:
    base = symbol[:-4]
    return SymbolFilters(
        symbol=symbol,
        base_asset=base,
        quote_asset="USDT",
        status="TRADING",
        is_spot_trading_allowed=True,
        min_price=Decimal("0.01"),
        max_price=Decimal("1000000"),
        tick_size=Decimal("0.01"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100000"),
        step_size=Decimal("0.001"),
        market_min_qty=Decimal("0.001"),
        market_max_qty=Decimal("100000"),
        market_step_size=Decimal("0.001"),
        min_notional=Decimal(min_notional),
        max_notional=Decimal("1000000"),
        apply_min_to_market=True,
    )


def _make_wave_candles(
    count: int = 200,
    *,
    start_price: float = 100.0,
    step_ms: int = 300_000,
    wave_amp: float = 1.6,
    drift: float = 0.04,
) -> list[Candle]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    last_close = now_ms - 5_000
    first_open = (last_close - step_ms + 1) - (count - 1) * step_ms
    candles: list[Candle] = []
    px = start_price
    for i in range(count):
        delta = drift + wave_amp * float(np.sin(i / 4.5)) + 0.35 * float(np.cos(i / 2.2))
        op = max(px, 1.0)
        cl = max(op + delta, 1.0)
        hi = max(op, cl) + 0.25 + abs(delta) * 0.15
        lo = max(min(op, cl) - 0.22 - abs(delta) * 0.12, 0.5)
        vol = Decimal(str(round(120.0 + 35.0 * abs(np.sin(i / 3.0)), 4)))
        c_dec = Decimal(str(round(cl, 4)))
        open_ms = first_open + i * step_ms
        candles.append(
            Candle(
                open_time_ms=open_ms,
                open=Decimal(str(round(op, 4))),
                high=Decimal(str(round(hi, 4))),
                low=Decimal(str(round(lo, 4))),
                close=c_dec,
                volume=vol,
                close_time_ms=open_ms + step_ms - 1,
                quote_volume=vol * c_dec,
                trade_count=250,
                is_closed=True,
            )
        )
        px = cl
    return candles


def _make_book(symbol: str, mid_price: Decimal, *, spread: Decimal = Decimal("0.02"), bid_qty: str = "400", ask_qty: str = "400") -> OrderBookSnapshot:
    half = spread / Decimal("2")
    bid = mid_price - half
    ask = mid_price + half
    return OrderBookSnapshot(
        symbol=symbol,
        bids=[OrderBookLevel(price=bid, quantity=Decimal(bid_qty)), OrderBookLevel(price=bid - Decimal("0.05"), quantity=Decimal(bid_qty))],
        asks=[OrderBookLevel(price=ask, quantity=Decimal(ask_qty)), OrderBookLevel(price=ask + Decimal("0.05"), quantity=Decimal(ask_qty))],
        fetched_at_utc=datetime.now(timezone.utc),
        source_url="https://api.binance.com/api/v3/depth",
    )


def test_kronos_forecast_probabilistic_intervals() -> None:
    candles = _make_wave_candles(120)
    fc = forecast_ohlcv_candles(candles, symbol="SOLUSDT", interval="5m", horizons=(5, 15))
    assert fc.sample_count > 0
    assert len(fc.horizons) == 2
    for h in fc.horizons:
        assert h.p10_pct <= h.p50_pct <= h.p90_pct
        assert 0.0 <= h.up_prob <= 1.0
        assert h.uncertainty_pct >= 0.0
    assert fc.regime_label != "YETERSIZ_VERI"

    short_fc = forecast_ohlcv_candles(candles[:20], symbol="SOLUSDT", interval="5m")
    assert short_fc.regime_label == "YETERSIZ_VERI"
    assert short_fc.sample_count == 0


def test_skfolio_hrp_weights_and_tail_risk() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(0.0005, 0.012, 90)
    returns = {
        "BTCUSDT": base,
        "ETHUSDT": base * 0.92 + rng.normal(0.0, 0.002, 90),
        "SOLUSDT": rng.normal(0.001, 0.025, 90),
    }
    weights = optimize_hrp_weights(returns)
    assert set(weights.keys()) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(w >= 0.0 for w in weights.values())

    report = compute_portfolio_risk_metrics(returns, candidate_symbol="ETHUSDT", open_symbols=("BTCUSDT",))
    assert report.var_95_pct >= 0.0
    assert report.cvar_95_pct >= report.var_95_pct
    assert report.max_corr_with_open > 0.75
    assert 0.45 <= report.sizing_multiplier <= 1.0


def test_hummingbot_microstructure_gates() -> None:
    candles = _make_wave_candles(80, start_price=100.0)
    last_px = candles[-1].close

    clean = evaluate_microstructure(
        _make_book("SOLUSDT", last_px, spread=Decimal("0.04"), bid_qty="500", ask_qty="480"),
        candles,
        inventory_ratio=0.1,
    )
    assert clean.ask_wall_blocked is False
    assert clean.spread_bps < 10.0

    wide_spread = evaluate_microstructure(
        _make_book("SOLUSDT", last_px, spread=last_px * Decimal("0.012"), bid_qty="500", ask_qty="500"),
        candles,
        max_spread_bps=42.0,
    )
    assert wide_spread.ask_wall_blocked is True
    assert "makas" in wide_spread.reason.lower()

    ask_wall = evaluate_microstructure(
        _make_book("SOLUSDT", last_px, spread=Decimal("0.10"), bid_qty="40", ask_qty="2500"),
        candles,
    )
    assert ask_wall.ask_wall_blocked is True
    assert ask_wall.imbalance < -0.58


def test_freqai_rolling_retrain_and_di_outlier_guard() -> None:
    candles = _make_wave_candles(160, start_price=100.0)
    model = SelfRetrainingSignalModel(horizon_bars=5, retrain_interval_bars=12, train_window_bars=100, di_threshold=2.6)

    normal_res = model.evaluate(candles, symbol="SOLUSDT", interval="5m")
    assert normal_res.do_predict is True
    assert normal_res.train_samples >= 40
    assert normal_res.last_retrain_bar >= 144

    # Extreme multi-feature flash-crash outlier should trip Dissimilarity Index gate
    prev = candles[-1]
    crash_close = (prev.close * Decimal("0.72")).quantize(Decimal("0.0001"))
    crash_open = prev.close
    crash_candle = Candle(
        open_time_ms=prev.close_time_ms + 1,
        open=crash_open,
        high=crash_open * Decimal("1.02"),
        low=crash_close * Decimal("0.92"),
        close=crash_close,
        volume=prev.volume * Decimal("600"),
        close_time_ms=prev.close_time_ms + 300_000,
        quote_volume=prev.volume * Decimal("600") * crash_close,
        trade_count=50_000,
        is_closed=True,
    )
    outlier_res = model.evaluate(candles + [crash_candle], symbol="SOLUSDT", interval="5m")
    assert outlier_res.dissimilarity_index > 2.6
    assert outlier_res.do_predict is False


def test_telegram_command_router_auth_and_actions() -> None:
    paused_calls: list[tuple[bool, str]] = []

    def dash_provider() -> dict[str, Any]:
        return {
            "mode": "LIVE_SPOT",
            "quote_asset": "USDT",
            "equity": 102.45,
            "cash": 45.10,
            "realized_pnl": 6.20,
            "unrealized_pnl": 1.15,
            "open_positions": [{"symbol": "SOLUSDT"}],
            "performance": {"closed_count": 8, "win_rate_pct": 75.0},
            "universe": [{"symbol": "SHELLUSDT", "change_pct_24h": 24.5, "opportunity_score": 88}],
        }

    # Unauthorized chat_id returns None (leaks nothing)
    assert (
        process_telegram_command(
            chat_id="999999",
            expected_chat_id="123456",
            text="/durum",
            dashboard_provider=dash_provider,
            pause_setter=lambda p, r: paused_calls.append((p, r)),
        )
        is None
    )
    assert paused_calls == []

    status_txt = process_telegram_command(
        chat_id=123456,
        expected_chat_id="123456",
        text="/durum",
        dashboard_provider=dash_provider,
        pause_setter=lambda p, r: paused_calls.append((p, r)),
    )
    assert status_txt is not None and "102.45 USDT" in status_txt and "LIVE_SPOT" in status_txt

    profit_txt = process_telegram_command(
        chat_id="123456",
        expected_chat_id="123456",
        text="/kar",
        dashboard_provider=dash_provider,
        pause_setter=lambda p, r: paused_calls.append((p, r)),
    )
    assert profit_txt is not None and "+6.20 USDT" in profit_txt and "Kapanan İşlem: 8" in profit_txt

    radar_txt = process_telegram_command(
        chat_id="123456",
        expected_chat_id="123456",
        text="/radar",
        dashboard_provider=dash_provider,
        pause_setter=lambda p, r: paused_calls.append((p, r)),
    )
    assert radar_txt is not None and "SHELLUSDT" in radar_txt

    stop_txt = process_telegram_command(
        chat_id="123456",
        expected_chat_id="123456",
        text="/durdur",
        dashboard_provider=dash_provider,
        pause_setter=lambda p, r: paused_calls.append((p, r)),
    )
    assert stop_txt is not None and "durduruldu" in stop_txt.lower()
    assert paused_calls[-1][0] is True

    start_txt = process_telegram_command(
        chat_id="123456",
        expected_chat_id="123456",
        text="/baslat",
        dashboard_provider=dash_provider,
        pause_setter=lambda p, r: paused_calls.append((p, r)),
    )
    assert start_txt is not None and "etkinleştirildi" in start_txt.lower()
    assert paused_calls[-1] == (False, "")


def test_quant_research_nlp_and_no_lookahead_backtest() -> None:
    plan = build_research_plan("SOLUSDT 5m 200 mum RSI dip tepkisi stratejisini test et")
    assert plan["symbols"][0] == "SOLUSDT"
    assert plan["interval"] == "5m"
    assert plan["lookback"] == 200
    assert plan["strategy_mode"] == "RSI_DIP"

    candles = _make_wave_candles(200, start_price=95.0, wave_amp=2.1)
    report = run_quant_research(
        question="SOLUSDT 5m 200 mum RSI dip tepkisi stratejisini test et",
        closed_candles_by_symbol={"SOLUSDT": candles},
        order_books_by_symbol={"SOLUSDT": _make_book("SOLUSDT", candles[-1].close)},
        filters_by_symbol={"SOLUSDT": _make_filters("SOLUSDT")},
        settings=RiskSettings(),
    )
    assert report["ok"] is True
    assert report["symbol"] == "SOLUSDT"
    assert "sharpe" in report["metrics"]
    assert "sortino" in report["metrics"]
    assert "profit_factor" in report["metrics"]
    assert len(report["equity_curve"]) >= 10
    assert report["quant"]["symbol"] == "SOLUSDT"


def test_storage_prune_history_and_compact_database(tmp_path: Path) -> None:
    db_path = tmp_path / "prune.db"
    st = SQLiteStorage(db_path)
    st.ensure_quote_asset("MAIN", "USDT")
    st.set_cash_balance(Decimal("86.00"), track="MAIN")
    for i in range(25):
        ts = f"2026-11-14T10:{i:02d}:00+00:00"
        st.record_decision(
            created_at_iso=ts,
            istanbul_date="2026-11-14",
            symbol="BTCUSDT",
            interval="1m",
            candle_open_time_ms=1_700_000_000_000 + i * 60_000,
            rule_action="BEKLE",
            rule_reason="Test",
            ai_action="BEKLE",
            ai_reason="Test",
            final_action="BEKLE",
            mode_label="KURAL",
            risk_notes=[],
            self_reported_score=None,
        )
        st.record_equity_snapshot(
            created_at_iso=ts,
            istanbul_date="2026-11-14",
            symbol="BTCUSDT",
            main_equity_usdc=Decimal("86.00"),
            cash_balance_usdc=Decimal("86.00"),
            unrealized_pnl_usdc=Decimal("0"),
            rule_shadow_equity_usdc=Decimal("86.00"),
            benchmark_equity_usdc=Decimal("86.00"),
        )
        st.record_ai_usage(
            created_at_iso=ts,
            istanbul_date="2026-11-14",
            cache_key=f"k-{i}",
            model_name="gemini",
            from_cache=False,
            prompt_tokens=10,
            output_tokens=5,
            estimated_cost_usd=Decimal("0"),
            response_payload={"action": "BEKLE"},
            status="OK",
        )

    removed = st.prune_history(
        max_decisions=12,
        max_snapshots=14,
        max_ai_usage=5,  # Alt sınır 10 kuralını da doğrular
        max_intents=10,
        compact_threshold_bytes=1,
    )
    assert removed["decisions"] == 13
    assert removed["equity_snapshots"] == 11
    assert removed["ai_usage"] == 15
    assert st.compact_database() > 0

    with st._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 12
        assert conn.execute("SELECT COUNT(*) FROM equity_snapshots").fetchone()[0] == 14
        assert conn.execute("SELECT COUNT(*) FROM ai_usage").fetchone()[0] == 10
        assert st.get_cash_balance("MAIN") == Decimal("86.00")


def test_gcs_sqlite_sync_gzip_persist_and_dual_restore(tmp_path: Path) -> None:
    db_path = tmp_path / "gcs_source.db"
    st = SQLiteStorage(db_path)
    st.ensure_quote_asset("MAIN", "USDT")
    st.set_cash_balance(Decimal("86.00"), track="MAIN")

    stored_payload: dict[str, bytes] = {}

    class _FakeResp:
        def __init__(self, status_code: int, content: bytes = b"") -> None:
            self.status_code = status_code
            self._content = content

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def iter_content(self, chunk_size: int = 65_536) -> list[bytes]:
            return [self._content[i : i + chunk_size] for i in range(0, len(self._content), chunk_size)]

    class _FakeSession:
        def post(self, url: str, *, data: bytes, headers: dict[str, str], timeout: Any) -> _FakeResp:
            stored_payload["bytes"] = data
            return _FakeResp(200)

        def get(self, url: str, *, timeout: Any, stream: bool = True, headers: dict[str, str] | None = None) -> _FakeResp:
            return _FakeResp(200, stored_payload["bytes"])

    sync = GcsSqliteSync("test-bucket", "sanal_sistem.db")
    sync._session = _FakeSession()  # type: ignore[assignment]

    assert sync.persist(db_path) is True
    assert stored_payload["bytes"].startswith(b"\x1f\x8b")
    raw_size = db_path.stat().st_size
    assert len(stored_payload["bytes"]) < raw_size

    # Restore from gzipped payload
    out_gz = tmp_path / "restored_gz.db"
    assert sync.restore(out_gz) is True
    assert out_gz.read_bytes()[:16] == SQLITE_MAGIC
    assert SQLiteStorage(out_gz).get_cash_balance("MAIN") == Decimal("86.00")

    # Restore from legacy uncompressed SQLite payload
    stored_payload["bytes"] = gzip.decompress(stored_payload["bytes"])
    out_raw = tmp_path / "restored_raw.db"
    assert sync.restore(out_raw) is True
    assert out_raw.read_bytes()[:16] == SQLITE_MAGIC


def test_market_data_server_time_and_book_ticker_caches() -> None:
    calls: dict[str, int] = {"/api/v3/time": 0, "/api/v3/ticker/bookTicker": 0}
    fixed_server_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    def handler(request: httpx.Request) -> httpx.Response:
        calls[request.url.path] = calls.get(request.url.path, 0) + 1
        if request.url.path == "/api/v3/time":
            return httpx.Response(200, json={"serverTime": fixed_server_ms})
        if request.url.path == "/api/v3/ticker/bookTicker":
            return httpx.Response(
                200,
                json=[
                    {"symbol": "BTCUSDT", "bidPrice": "90000.50", "askPrice": "90001.00"},
                    {"symbol": "SOLUSDT", "bidPrice": "185.25", "askPrice": "185.30"},
                ],
            )
        return httpx.Response(404)

    cfg = AppConfig()
    client = BinancePublicDataClient(cfg)
    client._shared_client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        t1 = client.fetch_server_time_ms()
        t2 = client.fetch_server_time_ms()
        assert abs(t1 - fixed_server_ms) < 5_000
        assert abs(t2 - fixed_server_ms) < 5_000
        assert calls["/api/v3/time"] == 1

        b1 = client.fetch_book_tickers(["BTCUSDT"])
        b2 = client.fetch_book_tickers(["SOLUSDT"])
        assert b1 == {"BTCUSDT": Decimal("90000.50")}
        assert b2 == {"SOLUSDT": Decimal("185.25")}
        assert calls["/api/v3/ticker/bookTicker"] == 1
    finally:
        client._reset_client()


def test_is_effective_dust_position_for_subnotional_and_depleted_wallet() -> None:
    bnb_filters = _make_filters("BNBUSDT", min_notional="5.0")
    pos = Position(
        position_id="p-bnb",
        symbol="BNBUSDT",
        quantity=Decimal("0.050"),
        entry_price=Decimal("80.00"),
        cost_basis_usdc=Decimal("4.00"),
        entry_fee_usdc=Decimal("0.004"),
        opened_at_iso="2026-11-14T10:00:00+00:00",
        signal_key="s-bnb",
        strategy_source="KURAL",
    )
    # 0.050 >= step 0.001 so substep check alone is False, but notional 4.00 < 5.0 -> effective dust!
    assert is_substep_dust_position(pos, bnb_filters) is False
    assert is_effective_dust_position(pos, bnb_filters, mark_price=Decimal("79.00")) is True

    # Normal sellable position becomes dust if live Spot wallet free balance dropped below min_qty
    big_pos = Position(
        position_id="p-btc",
        symbol="BTCUSDT",
        quantity=Decimal("0.100"),
        entry_price=Decimal("100.00"),
        cost_basis_usdc=Decimal("10.00"),
        entry_fee_usdc=Decimal("0.01"),
        opened_at_iso="2026-11-14T10:00:00+00:00",
        signal_key="s-btc",
        strategy_source="KURAL",
    )
    btc_filters = _make_filters("BTCUSDT", min_notional="5.0")
    assert is_effective_dust_position(big_pos, btc_filters, mark_price=Decimal("100.00"), free_base_qty=Decimal("0.100")) is False
    assert is_effective_dust_position(big_pos, btc_filters, mark_price=Decimal("100.00"), free_base_qty=Decimal("0.0002")) is True


def test_engine_handles_unsellable_live_spot_exit_without_crashing(tmp_path: Path) -> None:
    cfg = AppConfig(
        sqlite_db_path=tmp_path / "live_resilience.db",
        initial_balance_usdc=Decimal("86.00"),
        quote_asset="USDT",
        enable_live_trading=True,
        binance_api_key="K" * 24,
        binance_api_secret="S" * 24,
    )

    candles = _make_wave_candles(60, start_price=100.0, step_ms=60_000, wave_amp=0.2, drift=-0.15)
    last_px = candles[-1].close
    assert last_px < Decimal("96.00")

    class _DropMarket:
        def discover_spot_pairs(self, quote_asset: str = "USDT", allow_stablecoin_conversion: bool = False):
            return {"BTCUSDT": _make_filters("BTCUSDT")}

        def fetch_order_book(self, symbol: str, depth_limit: int = 20) -> OrderBookSnapshot:
            return _make_book(symbol, last_px)

        def fetch_verified_market_bundle(self, symbol: str, interval: str, **kwargs: Any) -> MarketBundle:
            return MarketBundle(
                symbol=symbol,
                interval=interval,
                filters=_make_filters(symbol),
                closed_candles=candles,
                order_book=_make_book(symbol, last_px),
                server_time_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
                fetched_at_utc=datetime.now(timezone.utc),
                source_label="MockDrop",
                wake_gap_detected=False,
            )

    eng = TradingSystemEngine(cfg, market_client=_DropMarket())
    eng.storage.ensure_quote_asset("LIVE_SPOT", "USDT")
    eng.storage.set_cash_balance(Decimal("86.00"), track="LIVE_SPOT")
    eng.storage.insert_position(
        Position(
            position_id="live-btc-1",
            symbol="BTCUSDT",
            quantity=Decimal("0.100"),
            entry_price=Decimal("105.00"),
            cost_basis_usdc=Decimal("10.50"),
            entry_fee_usdc=Decimal("0.01"),
            opened_at_iso="2026-11-14T09:00:00+00:00",
            signal_key="sig-1",
            strategy_source="KURAL",
            stop_loss_price=Decimal("98.00"),
            take_profit_price=Decimal("112.00"),
        ),
        track="LIVE_SPOT",
    )

    eng.live_executor.reconcile_pending_orders = lambda **kw: []  # type: ignore[method-assign]

    def _raise_unsellable(**kwargs: Any) -> Any:
        raise LiveExecutionError("Satılabilir BTC miktarı borsa minimumun altında; kalan miktar açık kayıtta korunuyor.")

    eng.live_executor.execute_live_market_sell = _raise_unsellable  # type: ignore[method-assign]

    summary = eng.run_analysis_step(
        symbol="BTCUSDT",
        interval="1m",
        settings=RiskSettings(),
        ai_enabled_by_user=False,
        execution_mode="LIVE_SPOT",
        ui_live_confirmed=True,
        skip_balance_prefetch=True,
        record_equity_snapshot=False,
    )
    assert summary.executed_order is None
    assert "Satılabilir BTC" in summary.block_or_info_message


def test_cloud_api_research_and_telegram_webhook_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candles_sol = _make_wave_candles(220, start_price=140.0, step_ms=300_000, wave_amp=2.2)

    class _ResearchMarket:
        def discover_spot_pairs(self, quote_asset: str = "USDT", allow_stablecoin_conversion: bool = False):
            return {"SOLUSDT": _make_filters("SOLUSDT"), "BTCUSDT": _make_filters("BTCUSDT")}

        def fetch_order_book(self, symbol: str, depth_limit: int = 20) -> OrderBookSnapshot:
            return _make_book(symbol, candles_sol[-1].close)

        def fetch_verified_market_bundle(self, symbol: str, interval: str, *, candle_limit: int = 120, **kwargs: Any) -> MarketBundle:
            seq = candles_sol[-candle_limit:] if candle_limit <= len(candles_sol) else candles_sol
            return MarketBundle(
                symbol=symbol,
                interval=interval,
                filters=_make_filters(symbol),
                closed_candles=seq,
                order_book=_make_book(symbol, seq[-1].close),
                server_time_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
                fetched_at_utc=datetime.now(timezone.utc),
                source_label="MockResearch",
                wake_gap_detected=False,
            )

    class _SyncStub:
        def restore(self, db_path: Path) -> bool:
            return False

        def persist(self, db_path: Path) -> bool:
            return True

    cfg = CloudConfig(
        project_id="adanahizlitaksi",
        owner_email="cebrailkara@gmail.com",
        state_bucket="bucket",
        cron_secret="c" * 32,
        db_path=tmp_path / "api_quant.db",
        initial_balance_usdc=Decimal("86.00"),
    )
    eng = TradingSystemEngine(cfg.to_app_config(), market_client=_ResearchMarket())

    def verifier(token: str, audience: str) -> dict[str, Any]:
        if token == "owner":
            return {
                "aud": audience,
                "iss": f"https://securetoken.google.com/{audience}",
                "sub": "uid-owner-123",
                "email": "cebrailkara@gmail.com",
                "email_verified": True,
            }
        raise ValueError("bad token")

    app = CloudBotApplication(cfg, engine=eng, state_sync=_SyncStub(), verifier=verifier)  # type: ignore[arg-type]

    class _FakeNotifier:
        enabled = True
        chat_id = "777888"

        def __init__(self) -> None:
            self.sent: list[str] = []

        def send(self, text: str) -> bool:
            self.sent.append(text)
            return True

    fake_tg = _FakeNotifier()
    app.notifier = fake_tg  # type: ignore[assignment]

    # /api/research rejects unauthenticated request
    unauth = app.handle("POST", "/api/research", {"Content-Type": "application/json"}, b'{"question":"SOLUSDT 5m test"}')
    assert unauth.status == 401

    # /api/research executes full backtest + quant report for owner
    res = app.handle(
        "POST",
        "/api/research",
        {"Authorization": "Bearer owner", "Content-Type": "application/json"},
        json.dumps({"question": "SOLUSDT 5m 220 mum RSI dip stratejisi test et"}).encode("utf-8"),
    )
    assert res.status == 200
    assert res.body is not None and res.body["ok"] is True
    assert res.body["symbol"] == "SOLUSDT"
    assert "quant" in res.body and "equity_curve" in res.body

    # /api/telegram/webhook validates secret token and chat_id
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-secret-token")
    bad_sec = app.handle("POST", "/api/telegram/webhook", {"X-Telegram-Bot-Api-Secret-Token": "wrong"}, b"{}")
    assert bad_sec.status == 401

    wrong_chat = app.handle(
        "POST",
        "/api/telegram/webhook",
        {"X-Telegram-Bot-Api-Secret-Token": "tg-secret-token"},
        json.dumps({"message": {"chat": {"id": 111}, "text": "/durdur"}}).encode("utf-8"),
    )
    assert wrong_chat.status == 200
    assert fake_tg.sent == []

    stop_ok = app.handle(
        "POST",
        "/api/telegram/webhook",
        {"X-Telegram-Bot-Api-Secret-Token": "tg-secret-token"},
        json.dumps({"message": {"chat": {"id": 777888}, "text": "/durdur"}}).encode("utf-8"),
    )
    assert stop_ok.status == 200
    assert any("durduruldu" in m.lower() for m in fake_tg.sent)
    assert eng.storage.is_new_entries_paused(track="MAIN")[0] is True

    start_ok = app.handle(
        "POST",
        "/api/telegram/webhook",
        {"X-Telegram-Bot-Api-Secret-Token": "tg-secret-token"},
        json.dumps({"message": {"chat": {"id": 777888}, "text": "/baslat"}}).encode("utf-8"),
    )
    assert start_ok.status == 200
    assert eng.storage.is_new_entries_paused(track="MAIN")[0] is False
