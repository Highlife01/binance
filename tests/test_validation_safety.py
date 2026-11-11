"""Offline regressions for the strategy actually selected and historical risk controls."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from binance_ai_bot import validation
from binance_ai_bot.ai_advisor import GeminiDecisionAdvisor
from binance_ai_bot.config import AppConfig
from binance_ai_bot.domain import Candle, MarketBundle, OrderBookLevel, OrderBookSnapshot, RiskSettings
from binance_ai_bot.market_data import parse_symbol_filters
from binance_ai_bot.profiles import get_profile
from binance_ai_bot.storage import SQLiteStorage
from binance_ai_bot.strategy import StrategyParams, evaluate_technical_strategy


D = Decimal


def candles(count=150, start=None):
    start_ms = int((start or datetime(2026, 1, 1, tzinfo=timezone.utc)).timestamp() * 1000)
    return [Candle(
        open_time_ms=start_ms + i * 60_000, close_time_ms=start_ms + (i + 1) * 60_000 - 1,
        open=D("100"), high=D("100.1"), low=D("99.9"), close=D("100"),
        volume=D("10"), quote_volume=D("1000"), trade_count=10, is_closed=True,
    ) for i in range(count)]


def settings(**kwargs):
    defaults = dict(stop_loss_pct=D("0"), take_profit_pct=D("0"),
                    extra_slippage_bps=D("0"), estimated_fee_rate=D("0"),
                    max_daily_new_positions=100, daily_loss_limit_usdc=D("2"))
    defaults.update(kwargs)
    return RiskSettings(**defaults)


def alternating_signals(monkeypatch):
    monkeypatch.setattr(validation, "evaluate_rule_signal_from_frame", lambda _df, **kw:
                        SimpleNamespace(action="AL" if kw["row_index"] % 2 else "SAT"))


def simulate(series, risk, **kwargs):
    return validation.run_no_lookahead_slice_simulation(
        series, eval_start_idx=0, eval_end_idx=len(series), slice_name="test",
        symbol="TESTUSDT", interval="1m", params=StrategyParams(), settings=risk,
        half_spread_bps=D("0"), **kwargs,
    )


def filters(min_notional="5", step="0.01"):
    return parse_symbol_filters({
        "symbol": "TESTUSDT", "baseAsset": "TEST", "quoteAsset": "USDT",
        "status": "TRADING", "isSpotTradingAllowed": True,
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "100000", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "minQty": step, "maxQty": "1000", "stepSize": step},
            {"filterType": "NOTIONAL", "minNotional": min_notional, "maxNotional": "100000", "applyMinToMarket": True},
        ],
    })


def test_selected_profile_is_never_replaced_by_optimized_candidate(monkeypatch):
    selected = get_profile("HIZLI_30SN").strategy
    seen = []
    original = validation.evaluate_rule_signal_from_frame

    def observe(frame, **kwargs):
        seen.append(kwargs["params"])
        return original(frame, **kwargs)

    monkeypatch.setattr(validation, "evaluate_rule_signal_from_frame", observe)
    report = validation.run_chronological_validation(
        candles(), symbol="TESTUSDT", interval="1m", settings=settings(), strategy_params=selected,
    )
    assert report.selected_params == selected
    assert seen and all(item == selected for item in seen)
    assert "alternatif parametre seçilmedi" in report.overall_verdict


def test_daily_entry_limit_is_enforced(monkeypatch):
    alternating_signals(monkeypatch)
    metrics = simulate(candles(), settings(max_daily_new_positions=1))
    assert metrics.trade_count == 1


def test_daily_loss_latches_entries_even_if_market_recovers(monkeypatch):
    alternating_signals(monkeypatch)
    metrics = simulate(candles(), settings(estimated_fee_rate=D("0.001"), daily_loss_limit_usdc=D("0.01")))
    assert metrics.trade_count == 1
    assert metrics.net_return_usdc == D("-0.020")


def test_daily_entry_counter_resets_at_istanbul_midnight(monkeypatch):
    alternating_signals(monkeypatch)
    series = candles(start=datetime(2026, 1, 1, 20, 23, tzinfo=timezone.utc))
    metrics = simulate(series, settings(max_daily_new_positions=1))
    assert metrics.trade_count == 2


def test_intrabar_stop_wins_when_target_also_touched(monkeypatch):
    monkeypatch.setattr(validation, "evaluate_rule_signal_from_frame", lambda _df, **kw:
                        SimpleNamespace(action="AL" if kw["row_index"] == 35 else "BEKLE"))
    series = candles()
    series[36] = replace(series[36], low=D("90"), high=D("110"))
    metrics = simulate(series, settings(stop_loss_pct=D("0.02"), take_profit_pct=D("0.03")))
    assert metrics.trade_count == 1
    assert metrics.net_return_usdc == D("-0.2")


def test_gap_exit_uses_available_open_instead_of_stop_price(monkeypatch):
    monkeypatch.setattr(validation, "evaluate_rule_signal_from_frame", lambda _df, **kw:
                        SimpleNamespace(action="AL" if kw["row_index"] == 35 else "BEKLE"))
    series = candles()
    series[37] = replace(series[37], open=D("90"), close=D("90"), high=D("91"), low=D("89"))
    metrics = simulate(series, settings(stop_loss_pct=D("0.02")))
    assert metrics.trade_count == 1
    assert metrics.net_return_usdc == D("-1")


@pytest.mark.parametrize("policy,trades,pnl", [("HOLD_EXISTING", 0, "0"), ("AUTO_CLOSE", 1, "-0.25")])
def test_daily_exit_policy_applies_to_intrabar_loss(monkeypatch, policy, trades, pnl):
    monkeypatch.setattr(validation, "evaluate_rule_signal_from_frame", lambda _df, **kw:
                        SimpleNamespace(action="AL" if kw["row_index"] == 35 else "BEKLE"))
    series = candles()
    series[36] = replace(series[36], low=D("90"))
    metrics = simulate(series, settings(daily_loss_limit_usdc=D("0.25"), daily_limit_exit_policy=policy))
    assert metrics.trade_count == trades
    assert metrics.net_return_usdc == D(pnl)


def test_exchange_minimum_never_increases_position_cap(monkeypatch):
    alternating_signals(monkeypatch)
    metrics = simulate(candles(), settings(), symbol_filters=filters(min_notional="11"))
    assert metrics.trade_count == 0
    assert metrics.total_fees_usdc == 0


def test_exchange_quantity_step_changes_the_actual_trade(monkeypatch):
    alternating_signals(monkeypatch)
    metrics = simulate(candles(), settings(max_daily_new_positions=1, estimated_fee_rate=D("0.001")),
                       symbol_filters=filters(step="0.03"))
    assert metrics.trade_count == 1
    assert metrics.total_fees_usdc == D("0.018")  # 0.09 units x 100 x 0.1% each way


@pytest.mark.parametrize("bad", ["unclosed", "duplicate", "unsorted", "ohlc"])
def test_invalid_historical_input_fails_closed(bad):
    series = candles()
    if bad == "unclosed":
        series[-1] = replace(series[-1], is_closed=False)
    elif bad == "duplicate":
        series[-1] = series[-2]
    elif bad == "unsorted":
        series[-2], series[-1] = series[-1], series[-2]
    else:
        series[-1] = replace(series[-1], high=D("NaN"))
    with pytest.raises(ValueError):
        simulate(series, settings())


def test_future_candles_do_not_change_earlier_slice():
    series = candles()
    price = D("100")
    for index, candle in enumerate(series):
        next_price = price + (D("0.04") if index % 2 else D("-0.03"))
        series[index] = replace(candle, open=price, close=next_price,
                                high=max(price, next_price)+D("0.01"),
                                low=min(price, next_price)-D("0.01"))
        price = next_price
    arguments = dict(eval_start_idx=0, eval_end_idx=100, slice_name="test", symbol="TESTUSDT",
                     interval="1m", params=get_profile("HIZLI_30SN").strategy, settings=settings())
    before = validation.run_no_lookahead_slice_simulation(series, **arguments)
    changed = series[:100] + [replace(c, open=D("1000"), close=D("1000"), high=D("1100"), low=D("900"))
                             for c in series[100:]]
    after = validation.run_no_lookahead_slice_simulation(changed, **arguments)
    assert before == after


def test_spot_validation_rejects_leverage_instead_of_reporting_spot_as_margin():
    with pytest.raises(ValueError, match="kaldıraçsız Spot"):
        simulate(candles(), settings(leverage=2))


def ai_inputs():
    series = candles()
    signal, _ = evaluate_technical_strategy(series, symbol="TESTUSDT", interval="1m")
    now = datetime.now(timezone.utc)
    book = OrderBookSnapshot(symbol="TESTUSDT", bids=[OrderBookLevel(D("100"), D("10"))],
                             asks=[OrderBookLevel(D("100.01"), D("10"))], fetched_at_utc=now, source_url="stub")
    market = MarketBundle(symbol="TESTUSDT", interval="1m", filters=filters(), closed_candles=series,
                          order_book=book, server_time_ms=int(now.timestamp()*1000), fetched_at_utc=now,
                          source_label="stub")
    return market, signal


def test_invalid_ai_schema_preserves_known_billed_usage(tmp_path):
    config = AppConfig(gemini_api_key="test-only", sqlite_db_path=tmp_path / "ai.db")
    store = SQLiteStorage(config.sqlite_db_path, initial_balance_usdc=D("86"))
    advisor = GeminiDecisionAdvisor(config, store, raw_generate_fn=lambda *_: ("invalid-json", 450, 180))
    market, signal = ai_inputs()
    result = advisor.evaluate(market=market, tech_signal=signal, recent_returns_pct=[], has_open_position=False)
    assert result.action == "BEKLE"
    assert (result.prompt_tokens, result.output_tokens) == (450, 180)
    assert result.estimated_cost_usd == D("0.000117")
    usage = store.get_ai_usage_summary(store.istanbul_date_str())
    assert usage["total_prompt_tokens"] == 450
    assert usage["total_output_tokens"] == 180
    assert usage["estimated_cost_usd"] == result.estimated_cost_usd


def test_ai_cache_separates_position_and_changed_strategy_context(tmp_path):
    config = AppConfig(gemini_api_key="test-only", sqlite_db_path=tmp_path / "cache.db")
    store = SQLiteStorage(config.sqlite_db_path, initial_balance_usdc=D("86"))
    calls = []

    def generate(*args):
        calls.append(args)
        return ('{"karar":"AL","kisa_gerekce":"Test sinyali","veri_zamani":"2026-01-01T00:00:00Z"}', 10, 5)

    advisor = GeminiDecisionAdvisor(config, store, raw_generate_fn=generate)
    market, signal = ai_inputs()
    kwargs = dict(market=market, tech_signal=signal, recent_returns_pct=[], has_open_position=False,
                  min_interval_override_sec=0)
    assert not advisor.evaluate(**kwargs).from_cache
    assert advisor.evaluate(**kwargs).from_cache
    assert not advisor.evaluate(**dict(kwargs, has_open_position=True)).from_cache
    assert not advisor.evaluate(**dict(kwargs, tech_signal=replace(signal, ema_fast=D("101")))).from_cache
    assert len(calls) == 3


def test_zero_ai_quota_override_makes_no_network_call(tmp_path):
    config = AppConfig(gemini_api_key="test-only", sqlite_db_path=tmp_path / "quota.db")
    store = SQLiteStorage(config.sqlite_db_path, initial_balance_usdc=D("86"))
    advisor = GeminiDecisionAdvisor(config, store, raw_generate_fn=lambda *_: pytest.fail("unexpected call"))
    market, signal = ai_inputs()
    result = advisor.evaluate(market=market, tech_signal=signal, recent_returns_pct=[],
                              has_open_position=False, daily_call_limit_override=0)
    assert result.mode_label == "GEMINI_KOTA_BEKLE"
