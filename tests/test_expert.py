"""Uzman trader katmanı testleri (çevrimdışı)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from binance_ai_bot.domain import Candle, OrderBookLevel, OrderBookSnapshot, Position
from binance_ai_bot.expert import (
    anomaly_outlier_block,
    candle_uncertainty_block,
    coin_knife_block,
    dynamic_exits,
    inverse_vol_capped_notional,
    loss_streak_cooldown,
    macro_bear_block,
    orderbook_microstructure_block,
    pick_scan_symbols,
    post_exit_reentry_block,
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


def test_post_exit_reentry_block_prevents_dodo_and_inj_profit_rebuy_loops():
    now = datetime(2026, 11, 14, 19, 0, 0, tzinfo=timezone.utc)

    def _closed(sym: str, exit_px: str, pnl: str, mins_ago: int, reason: str = "KAR_AL") -> Position:
        closed_dt = now - timedelta(minutes=mins_ago)
        return Position(
            position_id=f"POS-{sym}-{mins_ago}",
            symbol=sym,
            quantity=Decimal("100"),
            entry_price=Decimal(exit_px) * Decimal("0.985"),
            cost_basis_usdc=Decimal("15"),
            entry_fee_usdc=Decimal("0.015"),
            opened_at_iso=(closed_dt - timedelta(minutes=20)).isoformat(),
            signal_key="sig-1",
            strategy_source="AL_YUKSELINCE_SAT",
            stop_loss_price=Decimal("0"),
            take_profit_price=Decimal("0"),
            status="CLOSED",
            closed_at_iso=closed_dt.isoformat(),
            exit_price=Decimal(exit_px),
            exit_fee_usdc=Decimal("0.015"),
            realized_pnl_usdc=Decimal(pnl),
            close_reason=reason,
        )

    # Exact DODOUSDT scenario: took +0.165 profit @ 0.03270, bot tried rebuy 2m later @ 0.03286 -> BLOCKED
    dodo_2m = _closed("DODOUSDT", "0.03270", "0.165", mins_ago=2, reason="TRAILING_KAR")
    why_2m = post_exit_reentry_block(dodo_2m, current_price=Decimal("0.03286"), now_utc=now)
    assert why_2m is not None and "bekletme" in why_2m

    # Even 18m later (INJUSDT scenario: sold @ 7.293, tried rebuy @ 7.321) -> BLOCKED by cooldown
    inj_18m = _closed("INJUSDT", "7.293", "0.079", mins_ago=18, reason="KAR_AL")
    why_18m = post_exit_reentry_block(inj_18m, current_price=Decimal("7.321"), now_utc=now)
    assert why_18m is not None and "bekletme" in why_18m

    # After 45m cooldown (say 60m), rebuying AT or ABOVE profitable exit price is STILL blocked (top chase guard)
    dodo_60m = _closed("DODOUSDT", "0.03270", "0.165", mins_ago=60, reason="KAR_AL")
    assert post_exit_reentry_block(dodo_60m, current_price=Decimal("0.03286"), now_utc=now) is not None
    assert post_exit_reentry_block(dodo_60m, current_price=Decimal("0.03265"), now_utc=now) is not None

    # After 45m cooldown AND genuine >= 0.8% dip below exit price (0.03270 * 0.992 = 0.0324384) -> ALLOWED
    assert post_exit_reentry_block(dodo_60m, current_price=Decimal("0.03240"), now_utc=now) is None

    # After 4h pullback window elapses -> ALLOWED even above prior exit price
    dodo_5h = _closed("DODOUSDT", "0.03270", "0.165", mins_ago=250, reason="KAR_AL")
    assert post_exit_reentry_block(dodo_5h, current_price=Decimal("0.03400"), now_utc=now) is None

    # Single ZARAR_KES whipsaw guard (DODOUSDT 4m after -0.492 loss): blocked for 15m, allowed after 16m
    dodo_loss_4m = _closed("DODOUSDT", "0.03210", "-0.492", mins_ago=4, reason="ZARAR_KES")
    why_loss = post_exit_reentry_block(dodo_loss_4m, current_price=Decimal("0.03100"), now_utc=now)
    assert why_loss is not None and "zarar sonrası" in why_loss
    dodo_loss_16m = _closed("DODOUSDT", "0.03210", "-0.492", mins_ago=16, reason="ZARAR_KES")
    assert post_exit_reentry_block(dodo_loss_16m, current_price=Decimal("0.03100"), now_utc=now) is None


def test_engine_blocks_rebuy_after_profit_even_without_expert_mode(tmp_path: Path):
    app, _, _ = _make_app(tmp_path)
    eng = app._get_engine()
    # Explicitly test with expert_mode=False (e.g. AL_YUKSELINCE_SAT style non-expert mode)
    now = datetime.now(timezone.utc)
    recent = now - timedelta(minutes=3)
    pos = Position(
        position_id="WIN-BTC-1",
        symbol="BTCUSDT",
        quantity=Decimal("0.001"),
        entry_price=Decimal("99.0"),
        cost_basis_usdc=Decimal("10"),
        entry_fee_usdc=Decimal("0.01"),
        opened_at_iso=(recent - timedelta(minutes=10)).isoformat(),
        signal_key="win-1",
        strategy_source="AL_YUKSELINCE_SAT",
        stop_loss_price=Decimal("95"),
        take_profit_price=Decimal("101"),
        status="OPEN",
    )
    eng.storage.insert_position(pos, track="MAIN")
    eng.storage.update_position_closed(
        pos.position_id,
        closed_at_iso=recent.isoformat(),
        closed_istanbul_date=eng.storage.istanbul_date_str(recent),
        exit_price=Decimal("100.5"),
        exit_fee_usdc=Decimal("0.01"),
        realized_pnl_usdc=Decimal("0.20"),
        close_reason="KAR_AL",
    )
    settings = eng.default_risk_settings()
    settings = dataclasses.replace(settings, expert_mode=False)
    step = eng.run_analysis_step(symbol="BTCUSDT", interval="1m", settings=settings, ai_enabled_by_user=False)
    assert step.executed_order is None
    assert "karlı satış sonrası bekletme" in step.block_or_info_message
    assert all(p.symbol != "BTCUSDT" for p in eng.storage.get_open_positions(track="MAIN"))


def _mk_candle(i: int, op: str, hi: str, lo: str, cl: str, vol: str = "100") -> Candle:
    return Candle(
        open_time_ms=i * 60_000,
        open=Decimal(op),
        high=Decimal(hi),
        low=Decimal(lo),
        close=Decimal(cl),
        volume=Decimal(vol),
        close_time_ms=(i + 1) * 60_000 - 1,
        quote_volume=Decimal(vol) * Decimal(cl),
        trade_count=10,
        is_closed=True,
    )


def test_kronos_candle_uncertainty_and_freqai_outlier_guards():
    # 6 yüksek fitilli testere mumu (yön verimliliği düşük) -> Kronos belirsizlik filtresi engeller
    chop = [
        _mk_candle(i, "100", "105", "95", "100.2" if i % 2 == 0 else "99.9")
        for i in range(6)
    ]
    assert "belirsizliği" in (candle_uncertainty_block(chop) or "")

    # Temiz yükseliş mumları -> engel yok
    clean = [_mk_candle(i, str(100 + i), str(101.2 + i), str(99.8 + i), str(101 + i)) for i in range(6)]
    assert candle_uncertainty_block(clean) is None

    # 21 normal mum + 1 aşırı z-skorlu sıçrama mumu -> FreqAI DI koruması engeller
    normal = [_mk_candle(i, "100", "100.6", "99.7", "100.2" if i % 2 == 0 else "99.9", "100") for i in range(21)]
    spike = _mk_candle(21, "100", "112", "99.9", "110", "1200")
    assert "aykırı mum" in (anomaly_outlier_block(normal + [spike]) or "")
    assert anomaly_outlier_block(normal + [_mk_candle(21, "100", "100.5", "99.8", "100.1", "105")]) is None


def test_hummingbot_orderbook_microstructure_and_skfolio_inverse_vol():
    # Yoğun satış duvarı -> engeller
    sell_heavy = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=[OrderBookLevel(Decimal("100"), Decimal("1"))],
        asks=[OrderBookLevel(Decimal("100.05"), Decimal("20"))],
        fetched_at_utc=datetime.now(timezone.utc),
        source_url="test",
    )
    assert "satış baskısı" in (orderbook_microstructure_block(sell_heavy) or "")

    # Yüksek kaymalı (VWAP slippage) sığ defter -> engeller
    slip_book = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=[OrderBookLevel(Decimal("99.9"), Decimal("10"))],
        asks=[
            OrderBookLevel(Decimal("100.0"), Decimal("0.1")),   # 10 USDT
            OrderBookLevel(Decimal("101.0"), Decimal("10")),    # 1010 USDT (%1 yukarı)
        ],
        fetched_at_utc=datetime.now(timezone.utc),
        source_url="test",
    )
    assert "kayması yüksek" in (
        orderbook_microstructure_block(slip_book, intended_notional=Decimal("100")) or ""
    )

    # Sağlıklı derinlik -> izin verir
    healthy_book = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=[OrderBookLevel(Decimal("99.95"), Decimal("50"))],
        asks=[OrderBookLevel(Decimal("100.00"), Decimal("50"))],
        fetched_at_utc=datetime.now(timezone.utc),
        source_url="test",
    )
    assert orderbook_microstructure_block(healthy_book, intended_notional=Decimal("100")) is None

    # skfolio Ters-Volatilite ölçekleme: yüksek ATR (%3) tutarı yarıya indirir; düşük ATR (%1) büyütmez
    assert inverse_vol_capped_notional(Decimal("100"), Decimal("0.03")) == Decimal("50")
    assert inverse_vol_capped_notional(Decimal("100"), Decimal("0.01")) == Decimal("100")


def test_tradingagents_risk_manager_red_vetoes_buy_decision(tmp_path: Path):
    import json
    from binance_ai_bot.ai_advisor import GeminiDecisionAdvisor
    from binance_ai_bot.config import AppConfig
    from binance_ai_bot.domain import TechnicalSignal

    app, _, market_client = _make_app(tmp_path)
    eng = app._get_engine()
    bundle = market_client.fetch_verified_market_bundle("BTCUSDT", interval="1m")
    tech = TechnicalSignal(
        symbol="BTCUSDT",
        interval="1m",
        candle_open_time_ms=bundle.closed_candles[-1].open_time_ms,
        candle_close_time_ms=bundle.closed_candles[-1].close_time_ms,
        action="AL",
        reason="EMA + RSI al sinyali",
        close_price=Decimal("70000"),
        ema_fast=Decimal("69900"),
        ema_slow=Decimal("69500"),
        rsi_14=Decimal("58"),
        macd_line=Decimal("20"),
        macd_signal=Decimal("8"),
        macd_hist=Decimal("12"),
        volume_ratio=Decimal("1.5"),
        atr_14=Decimal("500"),
    )

    def fake_gemini(_prompt: str, _model: str, _timeout: int):
        payload = {
            "karar": "AL",
            "kisa_gerekce": "Kısa vadeli momentum yukarı yönlü.",
            "veri_zamani": "2026-11-14T12:00:00Z",
            "risk_notlari": ["Yüksek direnç yakın"],
            "model_oz_beyan_endeksi": 72,
            "boga_tezi": "EMA ve hacim yukarı kırılımı destekliyor.",
            "ayi_tezi": "Üst dirençte satış duvarı mevcut.",
            "risk_onayi": "RED",
        }
        return json.dumps(payload), 100, 50

    cfg = dataclasses.replace(eng.config, gemini_api_key="AIza" + "x" * 35)
    advisor = GeminiDecisionAdvisor(cfg, eng.storage, raw_generate_fn=fake_gemini)
    decision = advisor.evaluate(
        market=bundle,
        tech_signal=tech,
        recent_returns_pct=[0.1, 0.2],
        has_open_position=False,
        ai_enabled_by_user=True,
    )
    assert decision.action == "BEKLE"
    assert "Risk Yöneticisi vetosu" in decision.reason


def test_profile_preserves_min_buy_notional_floor():
    from binance_ai_bot.profiles import get_profile

    base = RiskSettings(
        max_open_positions=999,
        max_position_size_usdc=Decimal("15.00"),
        daily_loss_limit_usdc=Decimal("2.00"),
        min_buy_notional_usdc=Decimal("10.00"),
    )
    eff = get_profile("AL_YUKSELINCE_SAT").apply_to_risk_settings(base, leverage=1)
    assert eff.min_buy_notional_usdc == Decimal("10.00")


def test_macro_bear_block_rejects_dead_cat_bounce_and_allows_decoupled_uptrend():
    # 105-bar (26h) BTC bear market: drops from 75000 -> 65000 (-13.3%)
    btc_bear = _candles([75_000.0 - 95.0 * i for i in range(105)])
    # Dead-cat bounce altcoin: drops from 100 -> 82 (-18%) over 95 bars, then bounces +1.5% over last 10 bars
    alt_deadcat_closes = [100.0 - 0.19 * i for i in range(95)] + [81.95 + 0.14 * i for i in range(10)]
    alt_deadcat = _candles(alt_deadcat_closes)
    # Decoupled uptrend coin (like DODOUSDT): rises steadily from 0.10 -> 0.125 (+25%)
    decoupled_bull = _candles([0.10 + 0.00024 * i for i in range(105)])

    assert macro_bear_block(alt_deadcat, btc_bear, is_btc=False) is not None
    assert macro_bear_block(btc_bear, btc_bear, is_btc=True) is not None
    assert macro_bear_block(decoupled_bull, btc_bear, is_btc=False) is None


def test_trailing_armed_position_is_not_cut_early_by_1m_sat_flicker(tmp_path: Path):
    from binance_ai_bot.config import AppConfig
    from binance_ai_bot.engine import TradingSystemEngine
    from binance_ai_bot.profiles import get_profile

    prof = get_profile("AL_YUKSELINCE_SAT")

    class _FlickerMarket(FakeMarketClient):
        def fetch_verified_market_bundle(self, symbol: str, interval: str, allow_stablecoin_conversion: bool = False):
            b = super().fetch_verified_market_bundle(symbol, interval, allow_stablecoin_conversion)
            # Force RSI/EMA drop on the last 3 bars so tech_signal emits SAT while bid is still above trailing stop
            cs = list(b.closed_candles)
            last_p = cs[-4].close
            for k in (-3, -2, -1):
                p_new = last_p - Decimal("0.60")
                cs[k] = dataclasses.replace(cs[k], open=last_p, close=p_new, high=last_p, low=p_new)
                last_p = p_new
            bid_px = Decimal("102.00")
            ob = OrderBookSnapshot(
                symbol=symbol,
                bids=[OrderBookLevel(price=bid_px, quantity=Decimal("500"))],
                asks=[OrderBookLevel(price=bid_px + Decimal("0.01"), quantity=Decimal("500"))],
                fetched_at_utc=datetime.now(timezone.utc),
                source_url="test",
            )
            return dataclasses.replace(b, closed_candles=cs, order_book=ob)

    cfg = AppConfig(sqlite_db_path=tmp_path / "trail_hold.db", initial_balance_usdc=Decimal("100"), quote_asset="USDT")
    eng = TradingSystemEngine(cfg, market_client=_FlickerMarket())
    settings = prof.apply_to_risk_settings(eng.default_risk_settings(), leverage=1)
    now = datetime.now(timezone.utc)
    pos = Position(
        position_id="TRAIL-ARMED-1",
        symbol="SOLUSDT",
        quantity=Decimal("1.0"),
        entry_price=Decimal("100.00"),
        cost_basis_usdc=Decimal("100.00"),
        entry_fee_usdc=Decimal("0.08"),
        opened_at_iso=now.isoformat(),
        signal_key="s1",
        strategy_source="TEST",
        stop_loss_price=Decimal("101.20"),   # Trailing already armed above entry (100.00)!
        take_profit_price=Decimal("103.00"),
        status="OPEN",
    )
    eng.storage.insert_position(pos, track="MAIN")
    step = eng.run_analysis_step(
        symbol="SOLUSDT", interval="1m", settings=settings, ai_enabled_by_user=False, strategy_params=prof.strategy
    )
    assert step.executed_order is None
    assert step.final_action == "BEKLE"
    assert eng.storage.get_position("TRAIL-ARMED-1", track="MAIN").status == "OPEN"

