"""Risk budgets must survive expert sizing and execution minimums (offline)."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from binance_ai_bot.config import AppConfig
from binance_ai_bot.domain import OrderBookLevel, OrderBookSnapshot, Position, RiskSettings
from binance_ai_bot.engine import TradingSystemEngine
from binance_ai_bot.profiles import get_profile
from tests.test_cloud_api import FakeMarketClient

D = Decimal


def _adjust(*, equity="1", risk_pct="0.01", min_buy="0", min_notional="5", step="0.01", market_step=None):
    engine = SimpleNamespace(
        get_portfolio_summary=lambda *_args, **_kwargs: {"total_equity_usdc": D(equity)}
    )
    filters = replace(
        FakeMarketClient()._filters("BTCUSDT"),
        min_notional=D(min_notional),
        step_size=D(step),
        market_step_size=D(step if market_step is None else market_step),
        min_qty=D("0.001"),
        market_min_qty=D("0.001"),
    )
    settings = RiskSettings(
        max_position_size_usdc=D("20"),
        stop_loss_pct=D("0.035"),
        take_profit_pct=D("0.04"),
        estimated_fee_rate=D("0.001"),
        extra_slippage_bps=D("2"),
        risk_per_trade_pct=D(risk_pct),
        min_buy_notional_usdc=D(min_buy),
    )
    return TradingSystemEngine._expert_adjust_buy(
        engine, settings,
        SimpleNamespace(filters=filters, order_book=SimpleNamespace(best_ask=D("100"))),
        SimpleNamespace(atr_14=D("1"), close_price=D("100")),
        SimpleNamespace(approved_quantity=D("0.2"), approved_notional_usdc=D("20")),
        {}, "MAIN",
    )


@pytest.mark.parametrize("min_buy,min_notional", [("0", "5"), ("10", "1")])
def test_expert_minimum_cannot_override_risk_budget(min_buy, min_notional):
    _, qty, notional = _adjust(min_buy=min_buy, min_notional=min_notional)
    assert qty == notional == 0


def test_lot_rounding_below_minimum_rejects_instead_of_restoring_full_size():
    # The budget can fund a little over 5 quote units, but one lot costs 10.
    _, qty, notional = _adjust(equity="1", risk_pct="0.1", step="0.1")
    assert qty == notional == 0


def test_market_zero_step_uses_lot_size_and_rounds_down_within_budget():
    adjusted, qty, notional = _adjust(equity="30", risk_pct="0.01", step="0.03", market_step="0")
    assert qty > 0
    assert qty % D("0.03") == 0
    assert qty <= D("0.2") and notional <= D("20")
    estimated_fill = qty * D("100") * (1 + adjusted.extra_slippage_bps / D("10000"))
    estimated_stop_loss = estimated_fill * (adjusted.stop_loss_pct + 2 * adjusted.estimated_fee_rate)
    assert estimated_stop_loss <= D("30") * D("0.01")


@pytest.mark.parametrize("equity", ["0", "-1"])
def test_nonpositive_equity_cannot_keep_pre_cap_quantity(equity):
    _, qty, notional = _adjust(equity=equity)
    assert qty == notional == 0


def test_under_minimum_risk_budget_never_reaches_executor(tmp_path, monkeypatch):
    engine = TradingSystemEngine(
        AppConfig(sqlite_db_path=tmp_path / "sizing.db", quote_asset="USDT"),
        market_client=FakeMarketClient(),
    )
    settings = replace(
        get_profile("AL_YUKSELINCE_SAT").apply_to_risk_settings(engine.default_risk_settings(), leverage=1),
        risk_per_trade_pct=D("0.00001"),
    )
    monkeypatch.setattr(engine, "_expert_entry_block", lambda *_args, **_kwargs: None)

    def unexpected_buy(**_kwargs):
        pytest.fail("An under-minimum risk budget reached the main executor")

    # The comparison portfolio has its own executor calls; disable it for this assertion.
    monkeypatch.setattr(engine, "_update_rule_shadow_track", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine.simulator, "execute_buy", unexpected_buy)
    result = engine.run_analysis_step(
        symbol="BTCUSDT", interval="1m", settings=settings, ai_enabled_by_user=False,
        strategy_params=get_profile("AL_YUKSELINCE_SAT").strategy,
    )
    assert result.executed_order is None
    assert "risk bütçesi" in result.block_or_info_message
    assert engine.storage.get_open_positions(track="MAIN") == []


@pytest.mark.parametrize("drop,expected_action", [("0.4", "BEKLE"), ("1.5", "SAT")])
def test_actual_fast_exit_signal_waits_for_full_target_but_reversal_can_exit(tmp_path, drop, expected_action):
    class FlickerMarket(FakeMarketClient):
        def fetch_verified_market_bundle(self, symbol, interval, allow_stablecoin_conversion=False):
            bundle = super().fetch_verified_market_bundle(symbol, interval, allow_stablecoin_conversion)
            candles = list(bundle.closed_candles)
            previous = candles[-2].close
            close = previous - D(drop)
            candles[-1] = replace(candles[-1], open=previous, high=previous + D("0.1"), low=close - D("0.1"), close=close)
            book = OrderBookSnapshot(
                symbol=symbol,
                bids=[OrderBookLevel(price=D("100.9"), quantity=D("500"))],
                asks=[OrderBookLevel(price=D("100.91"), quantity=D("500"))],
                fetched_at_utc=datetime.now(timezone.utc), source_url="test",
            )
            return replace(bundle, closed_candles=candles, order_book=book)

    engine = TradingSystemEngine(
        AppConfig(sqlite_db_path=tmp_path / "flicker.db", quote_asset="USDT"),
        market_client=FlickerMarket(),
    )
    profile = get_profile("AL_YUKSELINCE_SAT")
    engine.storage.insert_position(Position(
        position_id="FLICKER", symbol="BTCUSDT", quantity=D("0.1"), entry_price=D("100"),
        cost_basis_usdc=D("10"), entry_fee_usdc=D("0.01"),
        opened_at_iso=datetime.now(timezone.utc).isoformat(), signal_key="entry", strategy_source="TEST",
        stop_loss_price=D("96"), take_profit_price=D("101.15"),
    ), track="MAIN")
    result = engine.run_analysis_step(
        symbol="BTCUSDT", interval="1m",
        settings=profile.apply_to_risk_settings(engine.default_risk_settings(), leverage=1),
        ai_enabled_by_user=False, strategy_params=profile.strategy,
    )
    assert result.tech_signal.action == "SAT"
    assert result.final_action == expected_action
    if expected_action == "BEKLE":
        assert result.tech_signal.reason.startswith("Hızlı çıkış")
        assert result.executed_order is None
        assert engine.storage.get_position("FLICKER", track="MAIN").status == "OPEN"
    else:
        assert result.executed_order.side == "SELL"
