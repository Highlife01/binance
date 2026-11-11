"""Offline integration tests across the real engine, risk, simulator and SQLite ledger."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3
import threading
import time
import os
import subprocess
import sys
from pathlib import Path

import pytest

from binance_ai_bot.config import AppConfig
from binance_ai_bot.domain import AIDecisionRecord, Candle, MarketBundle, OrderBookLevel, OrderBookSnapshot, Position, RiskSettings, SimulatedOrderResult
from binance_ai_bot.engine import TradingSystemEngine
from binance_ai_bot.execution_safety import ExecutionBusyError
from binance_ai_bot.market_data import MarketDataError, parse_symbol_filters
from binance_ai_bot.risk import RiskManager
from binance_ai_bot.simulator import VirtualExecutionSimulator
from binance_ai_bot.storage import SQLiteStorage


D = Decimal


def test_execution_lock_is_shared_with_an_independent_process(tmp_path):
    store = SQLiteStorage(tmp_path / "process-lock.db")
    code = """
import sys
from pathlib import Path
from binance_ai_bot.execution_safety import portfolio_execution_lock, ExecutionBusyError
try:
    with portfolio_execution_lock(Path(sys.argv[1]), timeout=0.15):
        print('acquired')
except ExecutionBusyError:
    print('busy')
    raise SystemExit(7)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))

    def child():
        return subprocess.run([sys.executable, "-c", code, str(store.db_path)],
                              capture_output=True, text=True, timeout=10, env=env)

    with store.execution_lock():
        result = child()
        assert result.returncode == 7 and result.stdout.strip() == "busy", result.stderr
    result = child()
    assert result.returncode == 0 and result.stdout.strip() == "acquired", result.stderr


def risk_settings(**overrides):
    values = dict(max_open_positions=5, max_daily_new_positions=100,
                  estimated_fee_rate=D("0"), extra_slippage_bps=D("0"),
                  stop_loss_pct=D("0"), take_profit_pct=D("0"))
    values.update(overrides)
    return RiskSettings(**values)


def symbol_filters(symbol="BTCUSDT"):
    quote = "USDT" if symbol.endswith("USDT") else "USDC"
    return parse_symbol_filters({
        "symbol": symbol, "baseAsset": symbol[:-4], "quoteAsset": quote,
        "status": "TRADING", "isSpotTradingAllowed": True,
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
            {"filterType": "NOTIONAL", "minNotional": "5", "maxNotional": "1000000", "applyMinToMarket": True},
        ],
    })


def order_book(symbol="BTCUSDT", bid="100", *, age_seconds=0):
    price = D(bid)
    return OrderBookSnapshot(
        symbol=symbol, bids=[OrderBookLevel(price, D("100"))],
        asks=[OrderBookLevel(price + D("0.01"), D("100"))],
        fetched_at_utc=datetime.now(timezone.utc)-timedelta(seconds=age_seconds), source_url="offline-test",
    )


class Market:
    """Only the external market-data boundary is replaced; no network is reachable."""

    def __init__(self, prices=None, missing=(), stale=()):
        self.prices = prices or {}
        self.missing = set(missing)
        self.stale = set(stale)
        self.book_requests = []

    def fetch_order_book(self, symbol, **_kwargs):
        self.book_requests.append(symbol)
        if symbol in self.missing:
            raise MarketDataError("offline: price unavailable")
        return order_book(symbol, str(self.prices.get(symbol, "100")), age_seconds=300 if symbol in self.stale else 0)

    def fetch_verified_market_bundle(self, *, symbol, interval, **_kwargs):
        now = datetime.now(timezone.utc)
        last_open = (int(now.timestamp()*1000)//60_000-1)*60_000
        series = [Candle(
            open_time_ms=last_open-(44-i)*60_000, close_time_ms=last_open-(44-i)*60_000+59_999,
            open=D("100"), close=D("100"), high=D("100.1"), low=D("99.9"),
            volume=D("10"), quote_volume=D("1000"), trade_count=10, is_closed=True,
        ) for i in range(45)]
        return MarketBundle(
            symbol=symbol, interval=interval, filters=symbol_filters(symbol), closed_candles=series,
            order_book=self.fetch_order_book(symbol), server_time_ms=int(now.timestamp()*1000),
            fetched_at_utc=now, source_label="offline-test",
        )


class BuyAdvisor:
    def evaluate(self, **_kwargs):
        return AIDecisionRecord(
            action="AL", reason="Controlled integration signal", data_timestamp_iso="2026-01-01T00:00:00Z",
            risk_notes=[], self_reported_score=None, mode_label="OFFLINE_TEST", model_name="offline-test",
        )


def engine_at(path, market=None):
    config = AppConfig(sqlite_db_path=path)
    return TradingSystemEngine(config, market_client=market or Market(), ai_advisor=BuyAdvisor())


def seed_position(store, symbol, *, track="MAIN", stop=None, cost="10"):
    position = Position(
        position_id=f"{track}-{symbol}", symbol=symbol, quantity=D("0.1"), entry_price=D("100"),
        cost_basis_usdc=D(cost), entry_fee_usdc=D("0"), opened_at_iso=datetime.now(timezone.utc).isoformat(),
        signal_key=f"seed-{symbol}", strategy_source="OFFLINE_TEST", stop_loss_price=stop,
    )
    store.insert_position(position, track=track)
    return position


@pytest.mark.parametrize("price,expected_loss", [("90", "2"), ("89", "2.2")])
def test_engine_uses_every_held_coin_for_aggregate_daily_loss(tmp_path, price, expected_loss):
    market = Market({"BTCUSDT": price, "ETHUSDT": price})
    engine = engine_at(tmp_path / "aggregate.db", market)
    seed_position(engine.storage, "BTCUSDT")
    seed_position(engine.storage, "ETHUSDT")
    engine.storage.set_cash_balance(D("66"))
    result = engine.run_analysis_step(symbol="SOLUSDT", interval="1m", settings=risk_settings())
    assert result.executed_order is None
    assert engine.storage.is_new_entries_paused()[0]
    assert set(market.book_requests) >= {"SOLUSDT", "BTCUSDT", "ETHUSDT"}
    assert engine.get_portfolio_summary(risk_settings(), engine._latest_bid_map)["daily_net_pnl_usdc"] == -D(expected_loss)
    snapshots = engine.storage.get_equity_snapshots(track="MAIN", quote_asset="USDT")
    assert snapshots
    assert D(snapshots[-1]["main_equity_usdc"]) == D("86")-D(expected_loss)


@pytest.mark.parametrize("unavailable", ["missing", "stale"])
def test_missing_other_coin_mark_blocks_new_entry_and_omits_partial_equity_curve(tmp_path, unavailable):
    market = Market(**{unavailable: ["ETHUSDT"]})
    engine = engine_at(tmp_path / "unavailable.db", market)
    seed_position(engine.storage, "BTCUSDT")
    seed_position(engine.storage, "ETHUSDT")
    before_cash = engine.storage.get_cash_balance()
    result = engine.run_analysis_step(symbol="SOLUSDT", interval="1m", settings=risk_settings())
    assert result.executed_order is None
    assert "güncel fiyat" in result.block_or_info_message
    assert engine.storage.get_cash_balance() == before_cash
    assert {p.symbol for p in engine.storage.get_open_positions()} == {"BTCUSDT", "ETHUSDT"}
    assert engine.storage.get_equity_snapshots(track="MAIN", quote_asset="USDT") == []


def test_missing_other_mark_does_not_disable_current_positions_stop(tmp_path):
    engine = engine_at(tmp_path / "protect.db", Market({"BTCUSDT": "90"}, missing=["ETHUSDT"]))
    seed_position(engine.storage, "BTCUSDT", stop=D("95"))
    seed_position(engine.storage, "ETHUSDT")
    engine.storage.set_cash_balance(D("66"))
    result = engine.run_analysis_step(symbol="BTCUSDT", interval="1m", settings=risk_settings())
    assert result.executed_order is not None
    assert result.executed_order.side == "SELL" and result.executed_order.status == "FILLED"
    assert {p.symbol for p in engine.storage.get_open_positions()} == {"ETHUSDT"}
    assert engine.storage.get_cash_balance() == D("75")


def test_direct_risk_check_fails_closed_without_other_position_marks(tmp_path):
    store = SQLiteStorage(tmp_path / "direct-risk.db")
    seed_position(store, "BTCUSDT")
    manager = RiskManager(store)
    kwargs = dict(filters=symbol_filters("SOLUSDT"), order_book=order_book("SOLUSDT"), settings=risk_settings())
    assert not manager.check_buy_entry(**kwargs).allowed
    assert manager.check_buy_entry(**kwargs, bid_prices_by_symbol={"BTCUSDT": D("100")}).allowed


def test_price_aging_during_decision_blocks_entry(tmp_path):
    engine = engine_at(tmp_path / "aged-decision.db", Market(stale=["SOLUSDT"]))
    result = engine.run_analysis_step(symbol="SOLUSDT", interval="1m", settings=risk_settings())
    assert result.executed_order is None
    assert "piyasa fiyatı eskidi" in result.block_or_info_message
    assert engine.storage.get_open_positions() == []


def test_unreviewed_live_margin_path_cannot_contact_account_or_send_orders(tmp_path, monkeypatch):
    engine = engine_at(tmp_path / "margin-gate.db")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Unreviewed margin path must never reach the account")

    monkeypatch.setattr(engine.live_margin_executor, "fetch_live_margin_balances", forbidden)
    result = engine.run_analysis_step(
        symbol="BTCUSDT", interval="1m", settings=risk_settings(), execution_mode="LIVE_MARGIN",
        ui_live_confirmed=True, ui_margin_confirmed=True,
    )
    assert result.ok is False and result.executed_order is None
    assert "canlı marjin kapalıdır" in result.block_or_info_message
    assert calls == []


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_paper_order_failure_rolls_back_cash_position_order_and_signal(tmp_path, monkeypatch, side):
    store = SQLiteStorage(tmp_path / f"rollback-{side}.db")
    simulator = VirtualExecutionSimulator(store)
    risk = risk_settings()
    if side == "SELL":
        simulator.execute_buy(filters=symbol_filters(), order_book=order_book(), requested_qty=D("0.1"),
                              signal_key="initial-buy", strategy_source="TEST", settings=risk)
    before_cash = store.get_cash_balance()
    before_positions = store.get_all_positions()
    before_orders = store.get_orders()
    before_quote = store.get_portfolio_quote()

    def fail_record(*_args, **_kwargs):
        raise RuntimeError("injected order persistence failure")

    monkeypatch.setattr(store, "record_order", fail_record)
    with pytest.raises(RuntimeError, match="persistence failure"):
        if side == "BUY":
            simulator.execute_buy(filters=symbol_filters(), order_book=order_book(), requested_qty=D("0.1"),
                                  signal_key="rollback-signal", strategy_source="TEST", settings=risk)
        else:
            simulator.execute_sell_position(position=store.get_open_positions()[0], filters=symbol_filters(),
                                           order_book=order_book(), signal_key="rollback-signal",
                                           strategy_source="TEST", close_reason="TEST", settings=risk)
    # A separate instance must see the committed state, rather than a leaked connection's state.
    restarted = SQLiteStorage(store.db_path)
    assert restarted.get_cash_balance() == before_cash
    assert restarted.get_all_positions() == before_positions
    assert restarted.get_orders() == before_orders
    assert not restarted.has_processed_signal("rollback-signal")
    assert restarted.get_portfolio_quote() == before_quote


def test_different_instances_share_execution_lock_and_reentrancy(tmp_path):
    path = tmp_path / "shared.db"
    first, second = SQLiteStorage(path), SQLiteStorage(path)
    with first.execution_lock():
        with second.execution_lock(timeout=0.1):  # Same-thread nesting across storage objects is safe.
            assert second.get_cash_balance() == D("86")
        with ThreadPoolExecutor(max_workers=1) as pool:
            def try_lock():
                with pytest.raises(ExecutionBusyError):
                    with second.execution_lock(timeout=0.05):
                        pytest.fail("another thread entered the same portfolio")
            pool.submit(try_lock).result(timeout=2)


def test_concurrent_engines_cannot_both_pass_last_position_slot(tmp_path, monkeypatch):
    path = tmp_path / "two-engines.db"
    engines = [engine_at(path), engine_at(path)]
    for engine in engines:
        original = engine.risk_manager.check_buy_entry

        def delayed_check(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            if result.allowed and kwargs.get("track") == "MAIN":
                time.sleep(0.05)  # Exposes a check/execute race if the engine lock is absent.
            return result

        monkeypatch.setattr(engine.risk_manager, "check_buy_entry", delayed_check)
    start = threading.Barrier(2)

    def run(index):
        start.wait(timeout=2)
        return engines[index].run_analysis_step(symbol=["BTCUSDT", "ETHUSDT"][index], interval="1m",
                                                settings=risk_settings(max_open_positions=1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, index) for index in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert sum(result.executed_order is not None for result in results) == 1
    assert len(SQLiteStorage(path).get_open_positions()) == 1


def test_stale_position_from_second_instance_cannot_credit_cash_twice(tmp_path):
    path = tmp_path / "stale-sell.db"
    first, second = SQLiteStorage(path), SQLiteStorage(path)
    a, b = VirtualExecutionSimulator(first), VirtualExecutionSimulator(second)
    risk = risk_settings()
    a.execute_buy(filters=symbol_filters(), order_book=order_book(), requested_qty=D("0.1"),
                  signal_key="buy", strategy_source="TEST", settings=risk)
    stale = second.get_open_positions()[0]
    kwargs = dict(position=stale, filters=symbol_filters(), order_book=order_book(),
                  strategy_source="TEST", close_reason="TEST", settings=risk)
    assert a.execute_sell_position(**kwargs, signal_key="sell-one").status == "FILLED"
    after_first = first.get_cash_balance()
    assert b.execute_sell_position(**kwargs, signal_key="sell-two").status == "REJECTED"
    assert second.get_cash_balance() == after_first
    assert len(second.get_orders()) == 2
    assert not second.has_processed_signal("sell-two")


def test_quote_pin_prevents_cash_reinterpretation_but_tracks_are_independent(tmp_path):
    store = SQLiteStorage(tmp_path / "quotes.db")
    store.ensure_quote_asset("MAIN", "USDT")
    with pytest.raises(ValueError, match="birleştirilemez"):
        store.ensure_quote_asset("MAIN", "USDC")
    store.ensure_quote_asset("LIVE_SPOT", "USDC")
    assert store.get_portfolio_quote("MAIN") == "USDT"
    assert store.get_portfolio_quote("LIVE_SPOT") == "USDC"
    assert store.get_cash_balance("MAIN") == D("86")
    assert store.get_cash_balance("LIVE_SPOT") == 0


def test_mixed_legacy_portfolio_cannot_be_pinned_to_either_quote(tmp_path):
    store = SQLiteStorage(tmp_path / "mixed.db")
    seed_position(store, "BTCUSDT")
    seed_position(store, "ETHUSDC")
    assert store.get_portfolio_quote() == "MIXED"
    for quote in ("USDT", "USDC"):
        with pytest.raises(ValueError, match="MIXED"):
            store.ensure_quote_asset("MAIN", quote)
    assert len(store.get_open_positions()) == 2
    assert store.get_state("portfolio_quote:MAIN") == ""


def test_pause_is_track_local_and_only_daily_halts_expire(tmp_path, monkeypatch):
    store = SQLiteStorage(tmp_path / "pause.db")
    today = ["2026-01-01"]
    monkeypatch.setattr(store, "istanbul_date_str", lambda *_: today[0])
    store.set_new_entries_paused(True, "daily loss", track="MAIN", until_date=today[0])
    assert store.is_new_entries_paused("MAIN") == (True, "daily loss")
    assert store.is_new_entries_paused("LIVE_SPOT") == (False, "")
    store.set_new_entries_paused(True, "manual pause", track="LIVE_SPOT")
    today[0] = "2026-01-02"
    assert store.is_new_entries_paused("MAIN") == (False, "")
    assert store.is_new_entries_paused("LIVE_SPOT") == (True, "manual pause")


def record_views(store, track, quote, value):
    now = datetime.now(timezone.utc)
    symbol = f"BTC{quote}"
    store.record_decision(
        created_at_iso=now.isoformat(), istanbul_date=store.istanbul_date_str(now), symbol=symbol,
        interval="1m", candle_open_time_ms=int(now.timestamp()*1000), rule_action="BEKLE",
        rule_reason="test", ai_action="BEKLE", ai_reason="test", final_action="BEKLE",
        mode_label=str(value), risk_notes=[], self_reported_score=None, track=track, quote_asset=quote,
    )
    store.record_equity_snapshot(
        created_at_iso=now.isoformat(), istanbul_date=store.istanbul_date_str(now), symbol=symbol,
        main_equity_usdc=D(value), cash_balance_usdc=D(value), unrealized_pnl_usdc=D("0"),
        rule_shadow_equity_usdc=D(value), benchmark_equity_usdc=D(value), track=track, quote_asset=quote,
    )


def test_decisions_and_equity_queries_isolate_both_mode_and_currency(tmp_path):
    store = SQLiteStorage(tmp_path / "views.db")
    cases = [("MAIN", "USDT", "81"), ("MAIN", "USDC", "82"),
             ("LIVE_SPOT", "USDT", "83"), ("LIVE_SPOT", "USDC", "84")]
    for track, quote, value in cases:
        record_views(store, track, quote, value)
    for track, quote, value in cases:
        decisions = store.get_recent_decisions(track=track, quote_asset=quote)
        snapshots = store.get_equity_snapshots(track=track, quote_asset=quote)
        assert len(decisions) == len(snapshots) == 1
        assert decisions[0]["mode_label"] == value
        assert D(snapshots[0]["main_equity_usdc"]) == D(value)


def test_schema_migration_preserves_legacy_rows_outside_current_views(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at_iso TEXT, istanbul_date TEXT,
                symbol TEXT, interval TEXT, candle_open_time_ms INTEGER, rule_action TEXT,
                rule_reason TEXT, ai_action TEXT, ai_reason TEXT, final_action TEXT, mode_label TEXT,
                risk_notes_json TEXT, self_reported_score INTEGER, executed_order_id TEXT, block_reason TEXT
            );
            INSERT INTO decisions (symbol, mode_label) VALUES ('BTCUSDT', 'legacy');
            CREATE TABLE equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at_iso TEXT, istanbul_date TEXT,
                symbol TEXT, main_equity_usdc TEXT, cash_balance_usdc TEXT, unrealized_pnl_usdc TEXT,
                rule_shadow_equity_usdc TEXT, benchmark_equity_usdc TEXT
            );
            INSERT INTO equity_snapshots (symbol, main_equity_usdc) VALUES ('BTCUSDT', '999');
        """)
    store = SQLiteStorage(path)
    assert len(store.get_recent_decisions(track="LEGACY")) == 1
    assert len(store.get_equity_snapshots(track="LEGACY")) == 1
    assert store.get_recent_decisions(track="MAIN", quote_asset="USDT") == []
    assert store.get_equity_snapshots(track="LIVE_SPOT", quote_asset="USDT") == []
    record_views(store, "MAIN", "USDT", "86")
    assert len(store.get_recent_decisions(track="MAIN", quote_asset="USDT")) == 1
    assert len(store.get_equity_snapshots(track="MAIN", quote_asset="USDT")) == 1
    # Opening the database again must not reassign legacy records to the current mode.
    reopened = SQLiteStorage(path)
    assert len(reopened.get_recent_decisions(track="LEGACY")) == 1


def test_decimal_cash_and_substep_position_round_trip_without_eight_digit_truncation(tmp_path):
    path = tmp_path / "precision.db"
    store = SQLiteStorage(path)
    precise_cash = D("85.1234567890123456789012345")
    original = Position(
        position_id="microscopic-dust", symbol="BTCUSDT", quantity=D("1e-12"),
        entry_price=D("123456.7890123456789012345"),
        cost_basis_usdc=D("0.0000001234567890123456789012345"),
        entry_fee_usdc=D("0.0000000001234567890123456789012345"),
        opened_at_iso=datetime.now(timezone.utc).isoformat(), signal_key="precision-seed",
        strategy_source="TEST", stop_loss_price=D("120000.1234567890123456789"),
        take_profit_price=D("130000.1234567890123456789"),
    )
    store.set_cash_balance(precise_cash)
    store.insert_position(original)
    reopened = SQLiteStorage(path)
    assert reopened.get_cash_balance() == precise_cash
    assert reopened.get_position(original.position_id) == original
    remaining_quantity = D("3e-13")
    remaining_cost = D("0.00000003703703670370370367037035")
    remaining_fee = D("0.00000000003703703670370370367037035")
    reopened.update_position_remaining(original.position_id, quantity=remaining_quantity,
                                       cost_basis_usdc=remaining_cost, entry_fee_usdc=remaining_fee)
    final = SQLiteStorage(path).get_position(original.position_id)
    assert final.quantity == remaining_quantity
    assert final.cost_basis_usdc == remaining_cost
    assert final.entry_fee_usdc == remaining_fee


def test_order_and_closed_position_money_precision_survives_restart(tmp_path):
    path = tmp_path / "closed-precision.db"
    store = SQLiteStorage(path)
    position = seed_position(store, "BTCUSDT")
    exit_price = D("101.1234567890123456789012345")
    exit_fee = D("0.0000000001234567890123456789012345")
    pnl = D("0.1234567890123456789012345")
    now = datetime.now(timezone.utc)
    store.update_position_closed(position.position_id, closed_at_iso=now.isoformat(),
                                 closed_istanbul_date=store.istanbul_date_str(now), exit_price=exit_price,
                                 exit_fee_usdc=exit_fee, realized_pnl_usdc=pnl, close_reason="TEST")
    result = SimulatedOrderResult(
        order_id="precision-order", symbol="BTCUSDT", side="SELL", order_type="MARKET", status="FILLED",
        requested_qty=D("1e-12"), filled_qty=D("1e-12"), avg_fill_price=exit_price,
        gross_quote_usdc=D("0.0000000001011234567890123456789012345"), fee_usdc=exit_fee,
        slippage_bps=D("0"), signal_key="precision-sale", strategy_source="TEST", explanation="precision",
        created_at_iso=now.isoformat(), realized_pnl_usdc=pnl,
    )
    store.record_order(result, store.istanbul_date_str(now))
    reopened = SQLiteStorage(path)
    closed = reopened.get_position(position.position_id)
    assert closed.exit_price == exit_price
    assert closed.exit_fee_usdc == exit_fee
    assert closed.realized_pnl_usdc == pnl
    row = reopened.get_orders()[0]
    for field in ("requested_qty", "filled_qty", "avg_fill_price", "gross_quote_usdc", "fee_usdc", "realized_pnl_usdc"):
        assert D(row[field]) == getattr(result, field)


def test_partial_sale_preserves_substep_dust_without_double_counting_closed_lot(tmp_path):
    path = tmp_path / "actual-dust.db"
    store = SQLiteStorage(path)
    original = replace(seed_position(store, "BTCUSDT"), quantity=D("0.100000000001"),
                       cost_basis_usdc=D("10.0000000001"))
    # Replacing the seed row mirrors an imported exchange quantity containing sub-step dust.
    store.update_position_remaining(original.position_id, quantity=original.quantity,
                                    cost_basis_usdc=original.cost_basis_usdc, entry_fee_usdc=D("0"))
    store.set_cash_balance(D("86")-original.cost_basis_usdc)
    result = VirtualExecutionSimulator(store).execute_sell_position(
        position=original, filters=symbol_filters(), order_book=order_book(), signal_key="dust-sale",
        strategy_source="TEST", close_reason="TEST", settings=risk_settings(),
    )
    assert result.status == "PARTIALLY_FILLED"
    reopened = SQLiteStorage(path)
    remaining = reopened.get_open_positions()[0]
    assert remaining.quantity == D("1e-12")
    assert remaining.cost_basis_usdc > 0
    closed = reopened.get_position(original.position_id)
    assert closed.quantity == result.filled_qty
    all_lots = reopened.get_all_positions()
    assert sum((p.quantity for p in all_lots), D("0")) == original.quantity
    assert sum((p.cost_basis_usdc for p in all_lots), D("0")) == pytest.approx(original.cost_basis_usdc, abs=D("1e-25"))
