"""
Uzman trader katmanı: çoklu zaman dilimi trend onayı, piyasa rejimi (BTC), ATR tabanlı dinamik
stop/hedef, risk bazlı pozisyon büyüklüğü, üst üste zarar eden coin bekletmesi ve fırsat evreni seçimi.

Tasarım ilkeleri:
- Saf fonksiyonlar (ağ erişimi yok) -> çevrimdışı test edilebilir.
- Veri alınamazsa FAIL-OPEN: filtre işlemi engellemez (mevcut işlem sıklığı korunur); yalnızca
  net olumsuz kanıt varsa (düşen bıçak / BTC risk-off / üst üste zarar) alım atlanır.
- Hiçbir fonksiyon güvenlik sınırlarını GEVŞETMEZ: stop, profil/kaldıraç tavanından geniş olamaz.
- Kâr garantisi yoktur; bunlar olasılığı iyileştirmeyi ve zararı küçültmeyi hedefler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable, Sequence

from .domain import Candle

_LEVERAGED_SUFFIXES = ("DOWN", "BULL", "BEAR")


# ---------------------------------------------------------------------------
# Trend ve rejim
# ---------------------------------------------------------------------------
def _ema(values: Sequence[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    out: list[float] = []
    prev: float | None = None
    for v in values:
        prev = v if prev is None else (alpha * v + (1.0 - alpha) * prev)
        out.append(prev)
    return out


@dataclass(frozen=True)
class TrendInfo:
    state: str  # "UP" | "FLAT" | "DOWN" | "UNKNOWN"
    ret_1h_pct: float
    ema_fast: float
    ema_slow: float
    close: float

    def label(self) -> str:
        return f"{self.state} (1sa getiri %{self.ret_1h_pct:+.2f})"


def trend_info(candles_15m: Sequence[Candle]) -> TrendInfo:
    """15 dakikalık kapanmış mumlardan trend yönü (EMA20/EMA50 + eğim + 1 saatlik getiri)."""
    if len(candles_15m) < 55:
        return TrendInfo("UNKNOWN", 0.0, 0.0, 0.0, 0.0)
    closes = [float(c.close) for c in candles_15m]
    fast = _ema(closes, 20)
    slow = _ema(closes, 50)
    close = closes[-1]
    slope_pct = (slow[-1] / slow[-5] - 1.0) * 100.0 if slow[-5] > 0 else 0.0
    ret_1h = (close / closes[-5] - 1.0) * 100.0 if closes[-5] > 0 else 0.0
    if close < slow[-1] and fast[-1] < slow[-1] and slope_pct < -0.03:
        state = "DOWN"
    elif close > slow[-1] and fast[-1] > slow[-1]:
        state = "UP"
    else:
        state = "FLAT"
    return TrendInfo(state, ret_1h, fast[-1], slow[-1], close)


def coin_knife_block(info: TrendInfo) -> str | None:
    """Net düşüş trendinde ve son 1 saatte sert düşen coini (düşen bıçak) alma."""
    if info.state == "DOWN" and info.ret_1h_pct <= -0.40:
        return f"15dk trend düşüşte ve son 1 saatte %{info.ret_1h_pct:.2f} düştü (düşen bıçak)"
    return None


def regime_block(btc: TrendInfo) -> str | None:
    """BTC net risk-off ise (düşüş trendi + sert 1 saatlik düşüş) tüm alımlar durur."""
    if btc.state == "DOWN" and btc.ret_1h_pct <= -0.80:
        return f"BTC risk-off rejimi (15dk düşüş trendi, 1 saatte %{btc.ret_1h_pct:.2f})"
    return None


# ---------------------------------------------------------------------------
# ATR tabanlı dinamik çıkış ve risk bazlı büyüklük
# ---------------------------------------------------------------------------
def atr_pct_of_price(atr_value: Decimal | float, price: Decimal | float) -> float:
    p = float(price)
    return float(atr_value) / p if p > 0 else 0.0


def dynamic_exits(
    *,
    base_sl: Decimal,
    base_tp: Decimal,
    atr_pct: float,
    leverage: int,
) -> tuple[Decimal, Decimal]:
    """
    Zarar-Kes: 4x ATR (volatiliteye uyumlu), profil/kaldıraç tavanından GENİŞ olamaz ve
    kaldıraca göre alt sınırın (komisyon+gürültü) altına inemez.
    Kâr-Al: en az profil hedefi; risk/getiri >= ~1.2 olacak şekilde stop'a oranlanır (üst sınır 1.6x).
    """
    lev = max(1, min(5, int(leverage)))
    floor_sl = max(base_sl * Decimal("0.40"), Decimal("0.0045"))
    floor_sl = min(floor_sl, base_sl)  # alt sınır asla tavanı aşamaz
    raw_sl = Decimal(str(round(4.0 * atr_pct, 6)))
    sl = min(max(raw_sl, floor_sl), base_sl)
    tp = max(base_tp, min(sl * Decimal("1.2"), base_tp * Decimal("1.6")))
    _ = lev
    return sl.quantize(Decimal("0.0001")), tp.quantize(Decimal("0.0001"))


def risk_capped_notional(
    *,
    equity: Decimal,
    risk_pct: Decimal,
    sl_pct: Decimal,
    fee_rate: Decimal,
    approved_notional: Decimal,
) -> Decimal:
    """İşlem başına kayıp bütçesi = özsermaye * risk_pct; pozisyon = bütçe / (stop + gidiş-dönüş komisyon)."""
    if risk_pct <= 0 or equity <= 0:
        return approved_notional
    loss_per_unit = sl_pct + fee_rate * Decimal("2")
    if loss_per_unit <= 0:
        return approved_notional
    return min(approved_notional, (equity * risk_pct) / loss_per_unit)


# ---------------------------------------------------------------------------
# Üst üste zarar eden coin bekletmesi
# ---------------------------------------------------------------------------
def loss_streak_cooldown(
    recent_closed: Iterable[tuple[str, Decimal]],
    now_utc: datetime,
) -> str | None:
    """
    `recent_closed`: en yeni -> en eski (kapanış zamanı ISO, net K/Z).
    2 ardışık zarar -> 30 dk, 3+ ardışık zarar -> 2 saat bekletme (son kapanıştan itibaren).
    """
    rows = list(recent_closed)
    streak = 0
    for _, pnl in rows:
        if pnl < 0:
            streak += 1
        else:
            break
    if streak < 2:
        return None
    try:
        last_close = datetime.fromisoformat(rows[0][0])
        if last_close.tzinfo is None:
            last_close = last_close.replace(tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None
    wait = timedelta(hours=2) if streak >= 3 else timedelta(minutes=30)
    until = last_close + wait
    if now_utc < until:
        mins = int((until - now_utc).total_seconds() // 60) + 1
        return f"üst üste {streak} zarar sonrası bekletme ({mins} dk kaldı)"
    return None


# ---------------------------------------------------------------------------
# Fırsat evreni ve Günün Yükselenleri radarı: likit + hareketli + ivmeli USDT
# ---------------------------------------------------------------------------
def _iter_valid_usdt_tickers(
    tickers: Iterable[dict[str, Any]],
    *,
    tradable_usdt: dict[str, Any],
    min_quote_volume: float,
    max_spread: float,
    max_min_notional: float,
) -> Iterable[dict[str, float | str]]:
    """24 saatlik ticker kayıtlarını doğrular; kaldıraçlı token, sığ hacim ve aşırı spreadi eler."""
    for t in tickers:
        sym = str(t.get("symbol", "")).upper()
        flt = tradable_usdt.get(sym)
        if flt is None:
            continue
        base = str(getattr(flt, "base_asset", ""))
        if any(base.endswith(sfx) and len(base) > len(sfx) + 1 for sfx in _LEVERAGED_SUFFIXES):
            continue
        try:
            last = float(t.get("lastPrice", 0))
            high = float(t.get("highPrice", 0))
            low = float(t.get("lowPrice", 0))
            qvol = float(t.get("quoteVolume", 0))
            bid = float(t.get("bidPrice", 0))
            ask = float(t.get("askPrice", 0))
            chg = float(t.get("priceChangePercent", 0))
        except (TypeError, ValueError):
            continue
        if last <= 0 or low <= 0 or bid <= 0 or ask <= 0:
            continue
        if qvol < min_quote_volume:
            continue
        if float(getattr(flt, "min_notional", 0)) > max_min_notional:
            continue
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / mid
        if spread > max_spread:
            continue
        range_pct = (high - low) / last * 100.0
        yield {
            "symbol": sym,
            "last": last,
            "chg": chg,
            "range_pct": range_pct,
            "qvol": qvol,
            "spread": spread,
        }


def score_universe(
    tickers: Iterable[dict[str, Any]],
    *,
    tradable_usdt: dict[str, Any],
    min_quote_volume: float = 5_000_000.0,
    max_spread: float = 0.0015,
    max_min_notional: float = 10.0,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """
    24 saatlik ticker verisinden dengeli momentum + fırsat puanı üretir:
    likidite (hacim) + günlük hareket aralığı + pozitif yükseliş ivmesi + dar spread.
    Sert düşen bıçak (<-%25), aşırı dikey mania (>+%55), kaldıraçlı token ve sığ pariteler elenir.
    """
    ranked: list[dict[str, Any]] = []
    for c in _iter_valid_usdt_tickers(
        tickers,
        tradable_usdt=tradable_usdt,
        min_quote_volume=min_quote_volume,
        max_spread=max_spread,
        max_min_notional=max_min_notional,
    ):
        chg = float(c["chg"])
        range_pct = float(c["range_pct"])
        qvol = float(c["qvol"])
        spread = float(c["spread"])
        if chg < -25.0 or chg > 55.0:
            continue
        if range_pct < 1.5 or range_pct > 55.0:
            continue
        liq = max(0.0, min((math.log10(qvol) - 6.7) / 2.8, 1.0))
        vol_score = min(range_pct, 15.0) / 15.0
        if chg <= 0.0:
            mom_score = 0.10
        elif chg <= 25.0:
            mom_score = min(chg / 15.0, 1.0)
        else:
            mom_score = max(0.35, 1.0 - (chg - 25.0) / 60.0)
        spread_score = 1.0 - min(spread / max_spread, 1.0)
        score = 0.30 * liq + 0.30 * vol_score + 0.25 * mom_score + 0.15 * spread_score
        ranked.append(
            {
                "symbol": str(c["symbol"]),
                "score": round(score, 4),
                "quote_volume_24h": round(qvol),
                "range_pct_24h": round(range_pct, 2),
                "change_pct_24h": round(chg, 2),
                "spread_pct": round(spread * 100.0, 4),
            }
        )
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[: max(1, limit)]


def rank_top_gainers(
    tickers: Iterable[dict[str, Any]],
    *,
    tradable_usdt: dict[str, Any],
    min_quote_volume: float = 2_000_000.0,
    max_spread: float = 0.0025,
    max_min_notional: float = 10.0,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """
    Binance Spot USDT çiftleri içinden son 24 saatte en çok yükselen (yeterli hacimli ve
    makul spreadli) coinleri Günün Yükselenleri Radarı için sıralar.
    """
    gainers: list[dict[str, Any]] = []
    for c in _iter_valid_usdt_tickers(
        tickers,
        tradable_usdt=tradable_usdt,
        min_quote_volume=min_quote_volume,
        max_spread=max_spread,
        max_min_notional=max_min_notional,
    ):
        chg = float(c["chg"])
        if chg < 2.0:
            continue
        if chg <= 15.0:
            tag = "ERKEN_IVME"
        elif chg <= 35.0:
            tag = "GUCLU_TREND"
        else:
            tag = "ASIRI_PUMP"
        last = float(c["last"])
        gainers.append(
            {
                "symbol": str(c["symbol"]),
                "last_price": round(last, 6 if last < 1 else 4),
                "change_pct_24h": round(chg, 2),
                "quote_volume_24h": round(float(c["qvol"])),
                "range_pct_24h": round(float(c["range_pct"]), 2),
                "spread_pct": round(float(c["spread"]) * 100.0, 4),
                "tag": tag,
            }
        )
    gainers.sort(key=lambda r: (r["change_pct_24h"], r["quote_volume_24h"]), reverse=True)
    return gainers[: max(1, limit)]


def pick_scan_symbols(
    *,
    core: Sequence[str],
    universe: Sequence[str],
    budget: int,
    cursor: int,
    always_top: int = 4,
) -> tuple[list[str], int]:
    """
    Tick başına tarama bütçesini (CPU/maliyet) sabit tutarak daha geniş evreni döngüsel tarar:
    en yüksek puanlı `always_top` coin her tick, kalanlar sırayla (round-robin).
    """
    seen: set[str] = set()
    pool: list[str] = []
    for s in list(universe) + list(core):
        if s not in seen:
            seen.add(s)
            pool.append(s)
    if len(pool) <= budget:
        return pool, 0
    top = pool[: max(0, min(always_top, budget))]
    rest = pool[len(top):]
    slots = max(0, budget - len(top))
    if slots == 0 or not rest:
        return top, 0
    start = cursor % len(rest)
    rotated = (rest + rest)[start : start + slots]
    return top + rotated, (start + slots) % len(rest)
