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
