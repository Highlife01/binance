"""Uzman trader katmanı testleri (çevrimdışı)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from binance_ai_bot.domain import Candle, Position
from binance_ai_bot.expert import (
    coin_knife_block,
    dynamic_exits,
    loss_streak_cooldown,
    pick_scan_symbols,
    rank_top_gainers,
    regime_block,
    risk_capped_notional,
    score_universe,
    trend_info,
)
from tests.test_cloud_api import FakeMarketClient, _make_app

AUTH = {"Authorization": "Bearer owner-token"}


def _candles(closes: list[float]) -> list[Candle]:
    out = []
    for i, c in enumerate(closes):
        open_ms = 1_700_000_000_000 + i * 900_000
        out.append(
            Candle(
                open_time_ms=open_ms,
                open=Decimal(str(c)),
                high=Decimal(str(c * 1.001)),
                low=Decimal(str(c * 0.999)),
                close=Decimal(str(c)),
                volume=Decimal("100"),
                close_time_ms=open_ms + 899_999,
                quote_volume=Decimal("10000"),
                trade_count=100,
                is_closed=True,
            )
        )
    return out


FALLING = [120.0 - 0.4 * i for i in range(60)]
RISING = [100.0 + 0.4 * i for i in range(60)]


def test_trend_knife_and_regime_filters():
    down = trend_info(_candles(FALLING))
    up = trend_info(_candles(RISING))
    assert down.state == "DOWN" and up.state == "UP"
    assert coin_knife_block(down) is not None and coin_knife_block(up) is None
    assert regime_block(down) is not None and regime_block(up) is None
    assert trend_info(_candles([100.0] * 10)).state == "UNKNOWN"  # veri yok -> fail-open


def test_dynamic_exits_never_wider_than_cap_and_keep_reward_ratio():
    for base_sl, base_tp, lev in (("0.035", "0.0115", 1), ("0.022", "0.009", 2), ("0.0095", "0.007", 5)):
        for atr_pct in (0.0002, 0.001, 0.004, 0.02):
            sl, tp = dynamic_exits(
                base_sl=Decimal(base_sl), base_tp=Decimal(base_tp), atr_pct=atr_pct, leverage=lev
            )
            assert sl <= Decimal(base_sl)
            assert tp >= Decimal(base_tp)
            assert tp >= sl * Decimal("1.0") or tp == Decimal(base_tp) * Decimal("1.6")


def test_risk_capped_notional_only_shrinks():
    assert risk_capped_notional(
        equity=Decimal("40"), risk_pct=Decimal("0.02"), sl_pct=Decimal("0.014"),
        fee_rate=Decimal("0.001"), approved_notional=Decimal("15"),
    ) == Decimal("15")
    small = risk_capped_notional(
        equity=Decimal("5"), risk_pct=Decimal("0.02"), sl_pct=Decimal("0.02"),
        fee_rate=Decimal("0.001"), approved_notional=Decimal("30"),
    )
    assert small < Decimal("30")
    assert risk_capped_notional(
        equity=Decimal("5"), risk_pct=Decimal("0"), sl_pct=Decimal("0.02"),
        fee_rate=Decimal("0.001"), approved_notional=Decimal("30"),
    ) == Decimal("30")


def test_loss_streak_cooldown_rules():
    now = datetime.now(timezone.utc)
    iso = lambda m: (now - timedelta(minutes=m)).isoformat()  # noqa: E731
    assert loss_streak_cooldown([(iso(5), Decimal("-0.1")), (iso(20), Decimal("0.2"))], now) is None
    assert "2 zarar" in loss_streak_cooldown([(iso(5), Decimal("-0.1")), (iso(20), Decimal("-0.2"))], now)
    assert loss_streak_cooldown([(iso(45), Decimal("-0.1")), (iso(60), Decimal("-0.2"))], now) is None
    assert "3 zarar" in loss_streak_cooldown(
        [(iso(45), Decimal("-0.1")), (iso(60), Decimal("-0.2")), (iso(90), Decimal("-0.3"))], now
    )


def test_score_universe_filters_and_ranks():
    flt = lambda base, mn="5": SimpleNamespace(base_asset=base, min_notional=Decimal(mn))  # noqa: E731
    tradable = {"AAAUSDT": flt("AAA"), "BBBUSDT": flt("BBB"), "CCCUSDT": flt("CCC"), "DDDUSDT": flt("DDD", "50")}

    def tick(sym, vol, hi, lo, last, bid, ask, chg="1.0"):
        return {
            "symbol": sym, "quoteVolume": str(vol), "highPrice": str(hi), "lowPrice": str(lo),
            "lastPrice": str(last), "bidPrice": str(bid), "askPrice": str(ask), "priceChangePercent": chg,
        }

    ranked = score_universe(
        [
            tick("AAAUSDT", 400_000_000, 106, 100, 103, 102.99, 103.01),  # likit + hareketli
            tick("BBBUSDT", 30_000_000, 103, 100, 101, 100.99, 101.01),   # daha az
            tick("CCCUSDT", 1_000_000, 110, 100, 105, 104.9, 105.1),      # hacim düşük
            tick("DDDUSDT", 900_000_000, 110, 100, 105, 104.99, 105.01),  # min notional yüksek
            tick("ZZZUSDT", 900_000_000, 110, 100, 105, 104.99, 105.01),  # işlem listesinde yok
        ],
        tradable_usdt=tradable,
    )
    assert [r["symbol"] for r in ranked] == ["AAAUSDT", "BBBUSDT"]


def test_pick_scan_symbols_rotates_within_budget():
    uni = [f"C{i}USDT" for i in range(30)]
    core = ["BTCUSDT", "ETHUSDT"]
    seen: set[str] = set()
    cursor = 0
    for _ in range(6):
        picked, cursor = pick_scan_symbols(core=core, universe=uni, budget=12, cursor=cursor)
        assert len(picked) == 12
        assert picked[:4] == uni[:4]
        seen.update(picked)
    assert len(seen) > 30 - 1  # birkaç tick'te geniş evrenin tamamı taranır


class _FallingMarket(FakeMarketClient):
    def fetch_closed_candles(self, symbol, interval, limit=60):
        return _candles(FALLING)


def test_engine_blocks_entries_in_btc_risk_off(tmp_path: Path):
    app, _, market = _make_app(tmp_path)
    eng = app._get_engine()
    eng.market_client = _FallingMarket()
    summary = app._run_scan_cycle(eng, source="TEST")
    assert summary["orders_executed"] == 0
    assert eng.storage.get_open_positions(track="MAIN") == []
    assert any("Uzman filtre" in it["note"] for it in summary["items"])


def test_engine_blocks_reentry_after_two_losses(tmp_path: Path):
    app, _, market = _make_app(tmp_path)
    eng = app._get_engine()
    now = datetime.now(timezone.utc)
    for n in range(2):
        pos = Position(
            position_id=f"LOSS-{n}", symbol="BTCUSDT", quantity=Decimal("0.1"), entry_price=Decimal("100"),
            cost_basis_usdc=Decimal("10"), entry_fee_usdc=Decimal("0.01"), opened_at_iso=now.isoformat(),
            signal_key=f"sig-{n}", strategy_source="TEST", stop_loss_price=Decimal("90"),
            take_profit_price=Decimal("110"), status="OPEN",
        )
        eng.storage.insert_position(pos, track="MAIN")
        eng.storage.update_position_closed(
            pos.position_id, closed_at_iso=now.isoformat(),
            closed_istanbul_date=eng.storage.istanbul_date_str(now), exit_price=Decimal("99"),
            exit_fee_usdc=Decimal("0.01"), realized_pnl_usdc=Decimal("-0.5"), close_reason="ZARAR_KES",
        )
    summary = app._run_scan_cycle(eng, source="TEST")
    btc_note = next(it["note"] for it in summary["items"] if it["symbol"] == "BTCUSDT")
    assert "bekletme" in btc_note
    assert all(p.symbol != "BTCUSDT" for p in eng.storage.get_open_positions(track="MAIN"))


# ---------------------------------------------------------------------------
# Marjin maliyet tabanı (phantom loss) düzeltmesi
# ---------------------------------------------------------------------------
import dataclasses

from binance_ai_bot.domain import RiskSettings


def test_margin_buy_scales_cost_basis_when_base_fee_shaves_a_step(tmp_path: Path):
    app, _, market = _make_app(tmp_path)
    eng = app._get_engine()
    flt = dataclasses.replace(
        market._filters("BTCUSDT"),
        step_size=Decimal("0.00001"), market_step_size=Decimal("0.00001"),
        min_qty=Decimal("0.00001"), market_min_qty=Decimal("0.00001"),
        min_notional=Decimal("5"),
    )
    ex = eng.live_margin_executor
    ex.fetch_live_margin_balances = lambda **kw: {  # type: ignore[method-assign]
        "margin_level": Decimal("999"), "quote_free": Decimal("5"), "quote_borrowed": Decimal("0"),
        "quote_interest": Decimal("0"), "quote_net": Decimal("5"), "available_collateral": Decimal("5"),
        "trade_enabled": True, "borrow_enabled": True, "free_by_asset": {"BTC": Decimal("0.00012987")},
    }

    def fake_signed(method, path, *args, **kwargs):
        if path == "/sapi/v1/margin/maxBorrowable":
            return {"amount": "100"}
        if path == "/sapi/v1/margin/order":
            return {
                "executedQty": "0.00013", "cummulativeQuoteQty": "9.3273336", "status": "FILLED", "orderId": 1,
                "fills": [{"commission": "0.00000013", "commissionAsset": "BTC", "qty": "0.00013", "price": "71748.72"}],
            }
        return {}

    ex._request_signed = fake_signed  # type: ignore[method-assign]
    settings = RiskSettings(leverage=2, stop_loss_pct=Decimal("0.02"), take_profit_pct=Decimal("0.01"))
    ex.execute_live_margin_buy(
        filters=flt, approved_qty=Decimal("0.00013"), signal_key="PH-1", strategy_source="TEST",
        settings=settings, ui_live_confirmed=True, ui_margin_confirmed=True,
    )
    pos = eng.storage.get_open_positions(track="LIVE_MARGIN")[0]
    assert pos.quantity == Decimal("0.00012")
    # maliyet tabanı kayda alınan miktarla tutarlı olmalı (eski hata: 9.33 ~ 0.00013 BTC maliyeti)
    assert abs(pos.cost_basis_usdc - pos.quantity * pos.entry_price) < Decimal("0.02")
    assert abs(pos.borrowed_usdc - pos.cost_basis_usdc / 2) < Decimal("0.01")


def test_margin_cost_fix_migration_corrects_history(tmp_path: Path):
    app, _, _ = _make_app(tmp_path)
    eng = app._get_engine()
    now = datetime.now(timezone.utc)
    pos = Position(
        position_id="OLD-1", symbol="BTCUSDT", quantity=Decimal("0.00012"), entry_price=Decimal("71748.72"),
        cost_basis_usdc=Decimal("9.3273336"), entry_fee_usdc=Decimal("0.0093"), opened_at_iso=now.isoformat(),
        signal_key="old", strategy_source="TEST", stop_loss_price=Decimal("1"), take_profit_price=Decimal("2"),
        status="OPEN", leverage=2, borrowed_usdc=Decimal("4.6636668"),
    )
    eng.storage.insert_position(pos, track="LIVE_MARGIN")
    eng.storage.update_position_closed(
        "OLD-1", closed_at_iso=now.isoformat(), closed_istanbul_date=eng.storage.istanbul_date_str(now),
        exit_price=Decimal("70136.46"), exit_fee_usdc=Decimal("0.0084"),
        realized_pnl_usdc=Decimal("-0.9287"), close_reason="ZARAR_KES",
    )
    eng.storage.set_state("margin_cost_fix_v1", "")
    app._ensure_cloud_defaults(eng.storage)
    fixed = eng.storage.get_position("OLD-1", track="LIVE_MARGIN")
    assert abs(fixed.cost_basis_usdc - Decimal("0.00012") * Decimal("71748.72")) < Decimal("0.001")
    assert fixed.realized_pnl_usdc > Decimal("-0.25")  # sahte zarar (≈0.72 USDT) geri alındı
    assert eng.storage.get_state("margin_cost_fix_v1", "") == "1"


def test_score_universe_catches_midcap_momentum_and_rank_top_gainers_works():
    flt = lambda base, mn="5": SimpleNamespace(base_asset=base, min_notional=Decimal(mn))  # noqa: E731
    tradable = {
        "BTCUSDT": flt("BTC"),
        "SHELLUSDT": flt("SHELL"),
        "PUMPUSDT": flt("PUMP"),
        "BTCDOWNUSDT": flt("BTCDOWN"),
        "TINYUSDT": flt("TINY"),
    }

    def tick(sym, vol, hi, lo, last, bid, ask, chg):
        return {
            "symbol": sym,
            "quoteVolume": str(vol),
            "highPrice": str(hi),
            "lowPrice": str(lo),
            "lastPrice": str(last),
            "bidPrice": str(bid),
            "askPrice": str(ask),
            "priceChangePercent": str(chg),
        }

    tickers = [
        tick("BTCUSDT", 2_000_000_000, 71_000, 67_000, 68_500, 68_499, 68_501, -3.5),
        tick("SHELLUSDT", 12_000_000, 0.58, 0.45, 0.55, 0.5498, 0.5502, 18.5),  # orta hacim + ivmeli (%0.07 spread)
        tick("PUMPUSDT", 4_000_000, 2.20, 1.10, 2.00, 1.999, 2.001, 48.0),    # >%2M ama <%5M -> radar görür, evren almaz
        tick("BTCDOWNUSDT", 50_000_000, 10, 6, 9, 8.999, 9.001, 25.0),        # kaldıraçlı token -> elenir
        tick("TINYUSDT", 400_000, 1.5, 1.0, 1.4, 1.399, 1.401, 30.0),          # çok sığ (<2M) -> elenir
    ]
    ranked = score_universe(tickers, tradable_usdt=tradable)
    syms = [r["symbol"] for r in ranked]
    assert "SHELLUSDT" in syms
    assert syms[0] == "SHELLUSDT"  # pozitif ivmeli orta hacimli coin eksideki devden yüksek skor almalı
    assert "PUMPUSDT" not in syms
    assert "BTCDOWNUSDT" not in syms

    gainers = rank_top_gainers(tickers, tradable_usdt=tradable)
    assert [g["symbol"] for g in gainers] == ["PUMPUSDT", "SHELLUSDT"]
    assert gainers[0]["tag"] == "ASIRI_PUMP"
    assert gainers[1]["tag"] == "GUCLU_TREND"
