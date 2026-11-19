"""Unit and integration tests for the 6-layer profitability upgrade."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from binance_ai_bot.config import AppConfig
from binance_ai_bot.domain import (
    Candle,
    MarketBundle,
    OrderBookLevel,
    OrderBookSnapshot,
    Position,
    RiskSettings,
)
from binance_ai_bot.engine import TradingSystemEngine
from binance_ai_bot.expert import _iter_valid_usdt_tickers, score_universe
from binance_ai_bot.market_data import parse_symbol_filters
from binance_ai_bot.profiles import get_profile
from binance_ai_bot.risk import RiskManager
from binance_ai_bot.storage import SQLiteStorage

D = Decimal


def _filters(symbol="BTCUSDT"):
    quote = "USDT" if symbol.endswith("USDT") else "USDC"
    base = symbol[:-4] if symbol.endswith("USDT") else symbol[:-4]
    return parse_symbol_filters({
        "symbol": symbol,
        "baseAsset": base,
        "quoteAsset": quote,
        "status": "TRADING",
        "isSpotTradingAllowed": True,
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
            {"filterType": "NOTIONAL", "minNotional": "5", "maxNotional": "1000000", "applyMinToMarket": True},
        ],
    })


def _book(symbol="BTCUSDT", bid="100.00"):
    price = D(bid)
    return OrderBookSnapshot(
        symbol=symbol,
        bids=[OrderBookLevel(price, D("100"))],
        asks=[OrderBookLevel(price + D("0.01"), D("100"))],
        fetched_at_utc=datetime.now(timezone.utc),
        source_url="test",
    )


class MockMarket:
    def __init__(self, bid_price="100.00"):
        self.bid_price = D(bid_price)

    def fetch_order_book(self, symbol, **_kwargs):
        return _book(symbol, str(self.bid_price))

    def fetch_verified_market_bundle(self, *, symbol, interval="1m", **_kwargs):
        now = datetime.now(timezone.utc)
        last_open = (int(now.timestamp() * 1000) // 60_000 - 1) * 60_000
        series = [
            Candle(
                open_time_ms=last_open - (44 - i) * 60_000,
                close_time_ms=last_open - (44 - i) * 60_000 + 59_999,
                open=self.bid_price,
                close=self.bid_price,
                high=self.bid_price + D("0.1"),
                low=self.bid_price - D("0.1"),
                volume=D("10"),
                quote_volume=D("1000"),
                trade_count=10,
                is_closed=True,
            )
            for i in range(45)
        ]
        return MarketBundle(
            symbol=symbol,
            interval=interval,
            filters=_filters(symbol),
            closed_candles=series,
            order_book=_book(symbol, str(self.bid_price)),
            server_time_ms=int(now.timestamp() * 1000),
            fetched_at_utc=now,
            source_label="mock",
        )


def test_breakeven_stop_ratchet_locks_profit_above_060_pct(tmp_path: Path):
    """When best_bid reaches +0.60% above entry, stop loss ratchets to +0.18% gross (net profit)."""
    cfg = AppConfig(sqlite_db_path=tmp_path / "be.db", quote_asset="USDT")
    market = MockMarket(bid_price="100.65")  # +0.65% above 100.00
    engine = TradingSystemEngine(cfg, market_client=market)
    prof = get_profile("AL_YUKSELINCE_SAT")
    settings = prof.apply_to_risk_settings(engine.default_risk_settings(), leverage=1)

    now = datetime.now(timezone.utc)
    pos = Position(
        position_id="POS_BE_1",
        symbol="BTCUSDT",
        quantity=D("0.1"),
        entry_price=D("100.00"),
        cost_basis_usdc=D("10.00"),
        entry_fee_usdc=D("0.01"),
        opened_at_iso=now.isoformat(),
        signal_key="entry_sig",
        strategy_source="AL_YUKSELINCE_SAT",
        stop_loss_price=D("98.15"),  # -1.85% initial SL
        take_profit_price=D("101.15"),
        status="OPEN",
    )
    engine.storage.insert_position(pos, track="MAIN")

    step = engine.run_analysis_step(
        symbol="BTCUSDT",
        interval="1m",
        settings=settings,
        ai_enabled_by_user=False,
        strategy_params=prof.strategy,
    )

    # Position stop should have ratcheted to entry * 1.0018 = 100.18
    updated_pos = engine.storage.get_position("POS_BE_1", track="MAIN")
    assert updated_pos is not None
    assert updated_pos.stop_loss_price == D("100.18")
    assert updated_pos.status == "OPEN"
    assert "başabaş kâr kilidi aktif" in step.block_or_info_message

    # Now simulate price pulling back to 100.15 (below 100.18 ratchet)
    market.bid_price = D("100.15")
    step2 = engine.run_analysis_step(
        symbol="BTCUSDT",
        interval="1m",
        settings=settings,
        ai_enabled_by_user=False,
        strategy_params=prof.strategy,
    )
    assert step2.final_action == "SAT"
    assert step2.executed_order is not None
    closed_pos = engine.storage.get_position("POS_BE_1", track="MAIN")
    assert closed_pos.status == "CLOSED"
    assert closed_pos.close_reason == "TRAILING_KAR"


def test_earlier_trailing_arms_above_105_pct(tmp_path: Path):
    """When best_bid reaches +1.05%, trailing stop arms and ratchets stop upward."""
    cfg = AppConfig(sqlite_db_path=tmp_path / "trail.db", quote_asset="USDT")
    market = MockMarket(bid_price="101.10")  # +1.10% above 100.00
    engine = TradingSystemEngine(cfg, market_client=market)
    prof = get_profile("AL_YUKSELINCE_SAT")
    settings = prof.apply_to_risk_settings(engine.default_risk_settings(), leverage=1)

    now = datetime.now(timezone.utc)
    pos = Position(
        position_id="POS_TR_1",
        symbol="BTCUSDT",
        quantity=D("0.1"),
        entry_price=D("100.00"),
        cost_basis_usdc=D("10.00"),
        entry_fee_usdc=D("0.01"),
        opened_at_iso=now.isoformat(),
        signal_key="entry_sig",
        strategy_source="AL_YUKSELINCE_SAT",
        stop_loss_price=D("98.15"),
        take_profit_price=D("101.50"),
        status="OPEN",
    )
    engine.storage.insert_position(pos, track="MAIN")

    step = engine.run_analysis_step(
        symbol="BTCUSDT",
        interval="1m",
        settings=settings,
        ai_enabled_by_user=False,
        strategy_params=prof.strategy,
    )

    updated_pos = engine.storage.get_position("POS_TR_1", track="MAIN")
    assert updated_pos is not None
    # Stop loss must be ratcheted higher than entry price
    assert updated_pos.stop_loss_price > D("100.00")
    assert updated_pos.status == "OPEN"


def test_zombie_position_timeout_exits_stale_trade_in_profit(tmp_path: Path):
    """Positions older than 180m with modest profit (>= +0.20%) exit as ZOMBI_POZISYON_KARI."""
    cfg = AppConfig(sqlite_db_path=tmp_path / "zombie.db", quote_asset="USDT")
    market = MockMarket(bid_price="100.25")  # +0.25% in profit
    engine = TradingSystemEngine(cfg, market_client=market)
    prof = get_profile("AL_YUKSELINCE_SAT")
    settings = prof.apply_to_risk_settings(engine.default_risk_settings(), leverage=1)

    # 1. Stale position (> 180 min old)
    old_time = datetime.now(timezone.utc) - timedelta(minutes=200)
    pos = Position(
        position_id="POS_ZOMBIE",
        symbol="BTCUSDT",
        quantity=D("0.1"),
        entry_price=D("100.00"),
        cost_basis_usdc=D("10.00"),
        entry_fee_usdc=D("0.01"),
        opened_at_iso=old_time.isoformat(),
        signal_key="entry_sig",
        strategy_source="AL_YUKSELINCE_SAT",
        stop_loss_price=D("98.15"),
        take_profit_price=D("101.15"),
        status="OPEN",
    )
    engine.storage.insert_position(pos, track="MAIN")

    step = engine.run_analysis_step(
        symbol="BTCUSDT",
        interval="1m",
        settings=settings,
        ai_enabled_by_user=False,
        strategy_params=prof.strategy,
    )

    assert step.final_action == "SAT"
    assert step.executed_order is not None
    closed = engine.storage.get_position("POS_ZOMBIE", track="MAIN")
    assert closed.status == "CLOSED"
    assert closed.close_reason == "ZOMBI_POZISYON_KARI"


def test_zombie_timeout_does_not_exit_fresh_trade(tmp_path: Path):
    """Fresh positions (under 180m) are held normally even if in +0.25% profit."""
    cfg = AppConfig(sqlite_db_path=tmp_path / "fresh.db", quote_asset="USDT")
    market = MockMarket(bid_price="100.25")
    engine = TradingSystemEngine(cfg, market_client=market)
    prof = get_profile("AL_YUKSELINCE_SAT")
    settings = prof.apply_to_risk_settings(engine.default_risk_settings(), leverage=1)

    fresh_time = datetime.now(timezone.utc) - timedelta(minutes=15)
    pos = Position(
        position_id="POS_FRESH",
        symbol="BTCUSDT",
        quantity=D("0.1"),
        entry_price=D("100.00"),
        cost_basis_usdc=D("10.00"),
        entry_fee_usdc=D("0.01"),
        opened_at_iso=fresh_time.isoformat(),
        signal_key="entry_sig",
        strategy_source="AL_YUKSELINCE_SAT",
        stop_loss_price=D("98.15"),
        take_profit_price=D("101.15"),
        status="OPEN",
    )
    engine.storage.insert_position(pos, track="MAIN")

    step = engine.run_analysis_step(
        symbol="BTCUSDT",
        interval="1m",
        settings=settings,
        ai_enabled_by_user=False,
        strategy_params=prof.strategy,
    )
    # Fresh position is NOT exited
    assert step.final_action == "BEKLE"
    assert engine.storage.get_position("POS_FRESH", track="MAIN").status == "OPEN"


def test_max_unprotected_positions_limit(tmp_path: Path):
    """Basket dump protection: maximum 3 unprotected (stop not locked above entry) positions allowed."""
    store = SQLiteStorage(tmp_path / "risk_unprot.db")
    risk_mgr = RiskManager(store)
    store.ensure_quote_asset("MAIN", "USDT")
    store.set_cash_balance(D("100.00"), track="MAIN")

    settings = RiskSettings(
        max_open_positions=999,
        max_position_size_usdc=D("15.00"),
        min_buy_notional_usdc=D("10.00"),
        max_unprotected_positions=3,
    )

    now = datetime.now(timezone.utc)
    # Add 3 positions whose stop is <= entry_price (unprotected)
    for i in range(1, 4):
        store.insert_position(
            Position(
                position_id=f"POS_{i}",
                symbol=f"SYM{i}USDT",
                quantity=D("1.0"),
                entry_price=D("10.00"),
                cost_basis_usdc=D("10.00"),
                entry_fee_usdc=D("0.01"),
                opened_at_iso=now.isoformat(),
                signal_key="k",
                strategy_source="TEST",
                stop_loss_price=D("9.80"),  # <= entry_price, unprotected!
                take_profit_price=D("10.15"),
                status="OPEN",
            ),
            track="MAIN",
        )

    # Attempt to buy a 4th symbol
    flt = _filters("SYM4USDT")
    book = _book("SYM4USDT", "10.00")
    check = risk_mgr.check_buy_entry(
        filters=flt,
        order_book=book,
        settings=settings,
        track="MAIN",
        bid_prices_by_symbol={f"SYM{i}USDT": D("10.00") for i in range(1, 4)},
    )
    assert check.allowed is False
    assert "Maksimum korunmasız (stopu kâra kilitlenmemiş) açık pozisyon sınırına (3/3) ulaşıldı" in check.reason

    # Now simulate POS_1's stop ratcheting above entry (protected house money!)
    store.update_position_remaining(
        "POS_1",
        quantity=D("1.0"),
        cost_basis_usdc=D("10.00"),
        entry_fee_usdc=D("0.01"),
        stop_loss_price=D("10.02"),  # > entry_price (10.00) -> PROTECTED!
        take_profit_price=D("10.15"),
        update_exit_targets=True,
    )

    # Now unprotected count is 2 (POS_2 and POS_3). 4th position must be allowed!
    check2 = risk_mgr.check_buy_entry(
        filters=flt,
        order_book=book,
        settings=settings,
        track="MAIN",
        bid_prices_by_symbol={f"SYM{i}USDT": D("10.00") for i in range(1, 4)},
    )
    assert check2.allowed is True


def test_tokenized_equities_filtering():
    """Tokenized equity/ETF tokens ending in BUSDT must be excluded, while legitimate cryptos are kept."""
    raw_tickers = [
        {"symbol": "SOXSBUSDT", "lastPrice": 10.0, "highPrice": 10.2, "lowPrice": 9.8, "quoteVolume": 10_000_000, "bidPrice": 9.99, "askPrice": 10.01, "priceChangePercent": 3.0},
        {"symbol": "QQQBUSDT", "lastPrice": 500.0, "highPrice": 505.0, "lowPrice": 495.0, "quoteVolume": 20_000_000, "bidPrice": 499.9, "askPrice": 500.1, "priceChangePercent": 2.0},
        {"symbol": "INTCBUSDT", "lastPrice": 25.0, "highPrice": 25.5, "lowPrice": 24.5, "quoteVolume": 8_000_000, "bidPrice": 24.99, "askPrice": 25.01, "priceChangePercent": 1.5},
        {"symbol": "BNBUSDT", "lastPrice": 650.0, "highPrice": 660.0, "lowPrice": 640.0, "quoteVolume": 50_000_000, "bidPrice": 649.9, "askPrice": 650.1, "priceChangePercent": 2.5},
        {"symbol": "ARBUSDT", "lastPrice": 0.85, "highPrice": 0.88, "lowPrice": 0.82, "quoteVolume": 15_000_000, "bidPrice": 0.849, "askPrice": 0.851, "priceChangePercent": 3.5},
        {"symbol": "SOLUSDT", "lastPrice": 200.0, "highPrice": 205.0, "lowPrice": 195.0, "quoteVolume": 80_000_000, "bidPrice": 199.9, "askPrice": 200.1, "priceChangePercent": 4.0},
    ]
    tradable = {t["symbol"]: _filters(t["symbol"]) for t in raw_tickers}

    valid = list(_iter_valid_usdt_tickers(
        raw_tickers,
        tradable_usdt=tradable,
        min_quote_volume=1_000_000.0,
        max_spread=0.005,
        max_min_notional=10.0,
    ))
    valid_symbols = {v["symbol"] for v in valid}

    # Legitimate cryptos must be included
    assert "BNBUSDT" in valid_symbols
    assert "ARBUSDT" in valid_symbols
    assert "SOLUSDT" in valid_symbols

    # Tokenized stock/ETF synthetic assets must be strictly excluded
    assert "SOXSBUSDT" not in valid_symbols
    assert "QQQBUSDT" not in valid_symbols
    assert "INTCBUSDT" not in valid_symbols
