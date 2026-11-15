"""Offline regression coverage for configured borrowing costs and stop caps."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from binance_ai_bot.domain import OrderBookLevel, OrderBookSnapshot, Position, RiskSettings
from binance_ai_bot.expert import dynamic_exits
from binance_ai_bot.market_data import parse_symbol_filters
from binance_ai_bot.risk import RiskManager
from binance_ai_bot.storage import SQLiteStorage


D = Decimal
REFERENCE_TIME = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)


def _position(*, elapsed_hours=3, borrowed="50"):
    return Position(
        position_id="borrowed-position",
        symbol="BTCUSDT",
        quantity=D("1"),
        entry_price=D("100"),
        cost_basis_usdc=D("100"),
        entry_fee_usdc=D("0.10"),
        opened_at_iso=(REFERENCE_TIME - timedelta(hours=elapsed_hours)).isoformat(),
        signal_key="test-borrowed-entry",
        strategy_source="TEST",
        leverage=2 if D(borrowed) else 1,
        borrowed_usdc=D(borrowed),
    )


@pytest.mark.parametrize(
    ("elapsed_hours", "hourly_rate", "borrowed", "expected_loss"),
    [
        (0.5, "0.001", "50", "0.250"),
        (3, "0.001", "50", "0.400"),
        (3, "0", "50", "0.200"),
        (3, "0.001", "0", "0.200"),
    ],
)
def test_daily_pnl_uses_configured_interest_at_evaluation_time(
    tmp_path, elapsed_hours, hourly_rate, borrowed, expected_loss
):
    manager = RiskManager(SQLiteStorage(tmp_path / "daily-pnl.db"))
    pnl = manager.calculate_daily_net_pnl_usdc(
        settings=RiskSettings(hourly_margin_interest_rate=D(hourly_rate)),
        open_positions=[_position(elapsed_hours=elapsed_hours, borrowed=borrowed)],
        bid_prices_by_symbol={"BTCUSDT": D("100")},
        now_utc=REFERENCE_TIME,
    )
    # Both entry and estimated exit fees are 0.10; interest uses started hours
    # as of REFERENCE_TIME, regardless of when this historical test runs.
    assert pnl == -D(expected_loss)


def test_daily_guard_and_buy_entry_include_configured_borrowing_cost(tmp_path):
    storage = SQLiteStorage(tmp_path / "daily-guard.db")
    storage.insert_position(_position())
    manager = RiskManager(storage)
    settings = RiskSettings(
        max_open_positions=3,
        daily_loss_limit_usdc=D("0.30"),
        hourly_margin_interest_rate=D("0.001"),
        daily_limit_exit_policy="AUTO_CLOSE",
    )
    breached, loss, auto_close = manager.evaluate_daily_loss_guard(
        settings=settings,
        open_positions=storage.get_open_positions(),
        bid_prices_by_symbol={"BTCUSDT": D("100")},
        now_utc=REFERENCE_TIME,
    )
    assert breached and auto_close
    assert loss == D("0.400")

    filters = parse_symbol_filters({
        "symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT",
        "status": "TRADING", "isSpotTradingAllowed": True,
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
            {"filterType": "NOTIONAL", "minNotional": "5", "maxNotional": "1000000", "applyMinToMarket": True},
        ],
    })
    book = OrderBookSnapshot(
        symbol="ETHUSDT", bids=[OrderBookLevel(D("100"), D("100"))],
        asks=[OrderBookLevel(D("100.01"), D("100"))],
        fetched_at_utc=REFERENCE_TIME, source_url="offline-test",
    )
    check = manager.check_buy_entry(
        filters=filters, order_book=book, settings=settings,
        now_utc=REFERENCE_TIME, bid_prices_by_symbol={"BTCUSDT": D("100")},
    )
    assert not check.allowed
    assert check.trigger_daily_halt and check.trigger_protective_close
    assert check.daily_total_loss_usdc == D("0.400")

    # A zero configured borrowing rate leaves only the fees below the guard.
    breached, loss, auto_close = manager.evaluate_daily_loss_guard(
        settings=replace(settings, hourly_margin_interest_rate=D("0")),
        open_positions=storage.get_open_positions(),
        bid_prices_by_symbol={"BTCUSDT": D("100")},
        now_utc=REFERENCE_TIME,
    )
    assert not breached and not auto_close
    assert loss == D("0.200")


@pytest.mark.parametrize("base_sl", ["0.00955", "0.00349", "0.00004"])
@pytest.mark.parametrize("atr_pct", [0.0001, 0.02])
def test_dynamic_stop_rounding_preserves_positive_user_cap(base_sl, atr_pct):
    cap = D(base_sl)
    sl, _ = dynamic_exits(base_sl=cap, base_tp=D("0.0115"), atr_pct=atr_pct, leverage=1)
    assert D("0") < sl <= cap
