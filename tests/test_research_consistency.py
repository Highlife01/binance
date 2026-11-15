"""Research charts and profitability metrics must describe the same historical fills."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from binance_ai_bot import quant_desk, validation
from binance_ai_bot.domain import Candle, RiskSettings, SymbolFilters
from binance_ai_bot.strategy import StrategyParams


D = Decimal


def candles(count=160):
    start = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    return [Candle(
        open_time_ms=start + i * 300_000, close_time_ms=start + (i + 1) * 300_000 - 1,
        open=D("100"), high=D("100.1"), low=D("99.9"), close=D("100"),
        volume=D("100"), quote_volume=D("10000"), trade_count=50, is_closed=True,
    ) for i in range(count)]


def filters(min_notional="5"):
    return SymbolFilters(
        symbol="TESTUSDT", base_asset="TEST", quote_asset="USDT", status="TRADING",
        is_spot_trading_allowed=True, min_price=D("0.01"), max_price=D("100000"),
        tick_size=D("0.01"), min_qty=D("0.001"), max_qty=D("1000"), step_size=D("0.001"),
        market_min_qty=D("0"), market_max_qty=D("0"), market_step_size=D("0"),
        min_notional=D(min_notional), max_notional=D("100000"), apply_min_to_market=True,
    )


def research_and_source(monkeypatch, series, risk, *, symbol_filters=None):
    def latest_quant(**kwargs):
        assert kwargs["order_book"].source_url == "unavailable"
        return SimpleNamespace(
            kronos=SimpleNamespace(summary_tr="Güncel ölçüm"),
            freqai=SimpleNamespace(reason="Güncel skor"),
            to_summary_dict=lambda: {"symbol": "TESTUSDT"},
        )

    monkeypatch.setattr(quant_desk, "evaluate_quant_bundle", latest_quant)
    report = quant_desk.run_quant_research(
        question="TESTUSDT 5m", closed_candles_by_symbol={"TESTUSDT": series},
        order_books_by_symbol={}, filters_by_symbol={"TESTUSDT": symbol_filters or filters()},
        settings=risk,
    )
    source = validation.run_no_lookahead_slice_simulation(
        series, eval_start_idx=int(len(series) * 0.25), eval_end_idx=len(series),
        slice_name="source", symbol="TESTUSDT", interval="5m", params=StrategyParams(),
        settings=risk, symbol_filters=symbol_filters or filters(),
    )
    return report, source


@pytest.mark.parametrize("case", ["daily_entries", "daily_loss", "entry_bar_stop", "slippage", "minimum"])
def test_research_uses_same_fills_for_curve_profit_factor_and_metrics(monkeypatch, case):
    series = candles()
    risk = RiskSettings(stop_loss_pct=D("0"), take_profit_pct=D("0"),
                        max_daily_new_positions=100, daily_loss_limit_usdc=D("20"))
    symbol_filters = filters()
    if case == "daily_entries":
        risk = replace(risk, max_daily_new_positions=1)
    elif case == "daily_loss":
        risk = replace(risk, daily_loss_limit_usdc=D("0.01"))
    elif case == "entry_bar_stop":
        risk = replace(risk, stop_loss_pct=D("0.02"), max_daily_new_positions=1)
        series[41] = replace(series[41], high=D("110"), low=D("90"))
    elif case == "slippage":
        risk = replace(risk, extra_slippage_bps=D("40"))
    else:
        symbol_filters = filters("11")
    monkeypatch.setattr(validation, "evaluate_rule_signal_from_frame", lambda _df, **kw:
                        SimpleNamespace(action="AL" if kw["row_index"] % 2 == 0 else "SAT"))
    report, source = research_and_source(monkeypatch, series, risk, symbol_filters=symbol_filters)

    assert source.trade_count == len(source.trade_pnls_usdc)
    assert source.equity_curve[-1].strategy_equity_usdc == source.final_equity_usdc
    assert report["equity_curve"][-1]["strategy"] == round(float(source.final_equity_usdc), 4)
    assert report["equity_curve"][-1]["benchmark"] == round(float(
        source.initial_equity_usdc * (1 + source.benchmark_return_pct / 100)), 4)
    assert report["metrics"]["trade_count"] == source.trade_count
    assert report["metrics"]["total_fees_usdc"] == round(float(source.total_fees_usdc), 4)
    assert report["metrics"]["net_return_pct"] == round(float(source.net_return_pct), 2)
    assert report["insufficient_data"] == source.insufficient_data
    assert report["simulation_scope"] == "SINGLE_SYMBOL_SPOT_OHLC"
    assert report["metrics"]["profit_factor"] == (0.0 if source.trade_count else None)
    if case in {"daily_entries", "daily_loss", "entry_bar_stop"}:
        assert source.trade_count == 1
    if case == "entry_bar_stop":
        assert source.trade_pnls_usdc[0] < D("-0.19")
    if case == "minimum":
        assert source.trade_count == 0


def test_profit_factor_uses_authoritative_realized_pnls(monkeypatch):
    series = candles()
    series[42] = replace(series[42], open=D("102"), high=D("103"), low=D("101"), close=D("102"))
    series[44] = replace(series[44], open=D("98"), high=D("99"), low=D("97"), close=D("98"))
    actions = {40: "AL", 41: "SAT", 42: "AL", 43: "SAT"}
    monkeypatch.setattr(validation, "evaluate_rule_signal_from_frame", lambda _df, **kw:
                        SimpleNamespace(action=actions.get(kw["row_index"], "BEKLE")))
    report, source = research_and_source(monkeypatch, series, RiskSettings(
        stop_loss_pct=D("0"), take_profit_pct=D("0"), max_daily_new_positions=10,
    ))
    assert len(source.trade_pnls_usdc) == 2
    assert source.trade_pnls_usdc[0] > 0 > source.trade_pnls_usdc[1]
    assert report["metrics"]["profit_factor"] == round(float(
        source.trade_pnls_usdc[0] / -source.trade_pnls_usdc[1]), 2)


def test_profit_factor_without_realized_loss_is_undefined(monkeypatch):
    series = candles()
    series[42] = replace(series[42], open=D("102"), high=D("103"), low=D("101"), close=D("102"))
    actions = {40: "AL", 41: "SAT"}
    monkeypatch.setattr(validation, "evaluate_rule_signal_from_frame", lambda _df, **kw:
                        SimpleNamespace(action=actions.get(kw["row_index"], "BEKLE")))
    report, source = research_and_source(monkeypatch, series, RiskSettings(
        stop_loss_pct=D("0"), take_profit_pct=D("0"),
    ))
    assert source.trade_count == 1 and source.trade_pnls_usdc[0] > 0
    assert report["metrics"]["profit_factor"] is None


def test_research_refuses_to_silently_present_margin_as_spot():
    with pytest.raises(ValueError, match="kaldıraçsız Spot"):
        quant_desk.run_quant_research(
            question="TESTUSDT 5m", closed_candles_by_symbol={"TESTUSDT": candles()},
            order_books_by_symbol={}, filters_by_symbol={}, settings=RiskSettings(leverage=2),
        )


def test_freqai_retrains_fixed_length_rolling_window_from_absolute_candle_time():
    series = candles(200)
    model = quant_desk.SelfRetrainingSignalModel(retrain_interval_bars=12)
    first = model.evaluate(series[:160], symbol="TESTUSDT", interval="5m")
    later = model.evaluate(series[24:184], symbol="TESTUSDT", interval="5m")
    assert later.last_retrain_bar > first.last_retrain_bar
    assert len(model._model_cache) == 2
    repeat = model.evaluate(series[24:184], symbol="TESTUSDT", interval="5m")
    assert repeat == later and len(model._model_cache) == 2


def test_freqai_cache_separates_strategy_features_and_prevents_future_reuse():
    series = candles(200)
    model = quant_desk.SelfRetrainingSignalModel(retrain_interval_bars=1000)
    params = StrategyParams()
    model.evaluate(series[1:161], symbol="TESTUSDT", interval="5m", params=params)
    model.evaluate(series[1:161], symbol="TESTUSDT", interval="5m", params=replace(params, ema_fast_span=5))
    assert len(model._model_cache) == 2
    earlier = model.evaluate(series[:160], symbol="TESTUSDT", interval="5m", params=params)
    independent = quant_desk.SelfRetrainingSignalModel(retrain_interval_bars=1000).evaluate(
        series[:160], symbol="TESTUSDT", interval="5m", params=params,
    )
    assert earlier == independent

