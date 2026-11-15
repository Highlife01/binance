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

from .domain import Candle, Position

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
    """Net 15dk düşüş trendindeki coini (veya sert düşen bıçağı) tepki mumunda alma."""
    if info.state == "DOWN":
        if info.ret_1h_pct <= -0.40:
            return f"15dk trend düşüşte ve son 1 saatte %{info.ret_1h_pct:.2f} düştü (düşen bıçak)"
        return f"15dk ana trend düşüşte ({info.label()}); düşüş trendinde tepki alımı engellendi"
    return None


def regime_block(btc: TrendInfo) -> str | None:
    """BTC net risk-off ise (düşüş trendi + sert 1 saatlik düşüş) tüm alımlar durur."""
    if btc.state == "DOWN" and btc.ret_1h_pct <= -0.80:
        return f"BTC risk-off rejimi (15dk düşüş trendi, 1 saatte %{btc.ret_1h_pct:.2f})"
    return None


def macro_bear_block(
    coin_candles_15m: Sequence[Candle],
    btc_candles_15m: Sequence[Candle] | None = None,
    *,
    is_btc: bool = False,
) -> str | None:
    """
    24 saatlik (96 x 15dk mum) makro rejim ve göreceli güç (RS) filtresi:
    - Yalnızca >=96 kapanmış 15dk mum mevcutsa devreye girer (kısa veri varsa fail-open).
    - BTC 24 saatlik ayı rejimindeyken (EMA96 altı veya EMA20<EMA50 veya 24sa <= -%1.0), korele zayıf
      coinlerin sahte tepki yükselişlerini (dead-cat bounce) engeller; yalnızca bağımsız yükseliş
      trendindeki (fiyat > EMA96, EMA20 > EMA50 >= EMA96, 24sa >= +%1.5, 6sa >= +%0.3) coinlere izin verir.
    - BTC sağlıklı/yükseliş rejimindeyken de kendi 24 saatlik trendi çökmüş coinleri eler.
    """
    if len(coin_candles_15m) < 96:
        return None
    closes = [float(c.close) for c in coin_candles_15m]
    cl = closes[-1]
    if cl <= 0 or closes[-96] <= 0 or closes[-25] <= 0:
        return None
    e20 = _ema(closes, 20)[-1]
    e50 = _ema(closes, 50)[-1]
    e96 = _ema(closes, 96)[-1]
    ret_6h = (cl / closes[-25] - 1.0) * 100.0
    ret_24h = (cl / closes[-96] - 1.0) * 100.0

    if btc_candles_15m is not None and len(btc_candles_15m) >= 96:
        b_closes = [float(c.close) for c in btc_candles_15m]
        bcl = b_closes[-1]
        if bcl > 0 and b_closes[-96] > 0 and b_closes[-25] > 0 and b_closes[-5] > 0:
            be20 = _ema(b_closes, 20)[-1]
            be50 = _ema(b_closes, 50)[-1]
            be96 = _ema(b_closes, 96)[-1]
            br1h = (bcl / b_closes[-5] - 1.0) * 100.0
            br6h = (bcl / b_closes[-25] - 1.0) * 100.0
            br24h = (bcl / b_closes[-96] - 1.0) * 100.0
            btc_bear = (bcl < be96) or (be20 < be50 * 0.999) or (br24h <= -1.0) or (br6h <= -1.2)
            if btc_bear:
                if is_btc:
                    if not (bcl > be96 and be20 > be50 and br6h >= 0.4):
                        return f"BTC 24sa düşüş rejiminde (24sa %{br24h:+.2f}, EMA96 altı)"
                else:
                    rs_bull = (
                        cl > e96
                        and e20 > e50
                        and e50 >= e96 * 0.998
                        and ret_24h >= 1.5
                        and ret_6h >= 0.3
                        and br1h > -0.80
                    )
                    if not rs_bull:
                        return (
                            f"BTC 24sa ayı rejiminde (%{br24h:+.2f}); bağımsız yükseliş trendi yok "
                            f"(coin 24sa %{ret_24h:+.2f}, 6sa %{ret_6h:+.2f})"
                        )

    if (cl < e96 * 0.995 and ret_24h <= -1.0) or (e20 < e50 * 0.999 and cl < e96) or ret_6h <= -1.5:
        return f"24sa makro trend zayıf/düşüşte (24sa %{ret_24h:+.2f}, 6sa %{ret_6h:+.2f}, EMA96 altı)"
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


_PROFIT_EXIT_REASONS = frozenset(
    {"KAR_AL", "TRAILING_KAR", "KARDA_STRATEJI_SAT", "TAKE_PROFIT", "KURAL_SAT"}
)


def post_exit_reentry_block(
    last_closed: Position | None,
    *,
    current_price: Decimal | None,
    now_utc: datetime,
    profit_cooldown_minutes: int = 45,
    price_pullback_window_minutes: int = 240,
    min_pullback_pct: Decimal = Decimal("0.008"),
    single_loss_cooldown_minutes: int = 15,
) -> str | None:
    """
    Aynı paritede karlı veya zararlı kapanıştan hemen sonra tepeden/whipsaw geri alışını engeller:
    - Kârlı / başa-baş kapanışta: `profit_cooldown_minutes` (45 dk) boyunca koşulsuz bekletir;
      ardından `price_pullback_window_minutes` (4 saat) içinde fiyat son satış fiyatının en az
      `min_pullback_pct` (%0.80) altına geri çekilmediyse tepe geri alışını durdurur.
    - Tek zararlı kapanışta: hemen tepki alışını önlemek için `single_loss_cooldown_minutes` (15 dk) bekletir.
    """
    if last_closed is None or not last_closed.closed_at_iso:
        return None
    try:
        closed_at = datetime.fromisoformat(last_closed.closed_at_iso)
        if closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=timezone.utc)
    except ValueError:
        return None

    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    pnl = last_closed.realized_pnl_usdc
    reason = (last_closed.close_reason or "").upper()
    is_profit_exit = pnl >= Decimal("0") or (reason in _PROFIT_EXIT_REASONS and pnl >= Decimal("0"))

    if is_profit_exit:
        cooldown_until = closed_at + timedelta(minutes=profit_cooldown_minutes)
        if now_utc < cooldown_until:
            mins = max(1, int((cooldown_until - now_utc).total_seconds() // 60) + 1)
            exit_str = f"{last_closed.exit_price:.6f}".rstrip("0").rstrip(".") if last_closed.exit_price else "-"
            return (
                f"karlı satış sonrası bekletme aktif "
                f"(son satış: +{pnl:.4f} USDC @ {exit_str}, {mins} dk kaldı)"
            )
        pullback_until = closed_at + timedelta(minutes=price_pullback_window_minutes)
        exit_px = last_closed.exit_price
        if (
            now_utc < pullback_until
            and exit_px is not None
            and exit_px > Decimal("0")
            and current_price is not None
            and current_price > Decimal("0")
        ):
            max_reentry_px = exit_px * (Decimal("1") - min_pullback_pct)
            if current_price > max_reentry_px:
                return (
                    f"karlı satış sonrası tepe geri alış koruması "
                    f"(güncel {current_price:.6f} > eşik {max_reentry_px:.6f}; "
                    f"son satış {exit_px:.6f} fiyatından en az %{min_pullback_pct * Decimal('100'):.2f} geri çekilme bekleniyor)"
                )
        return None

    loss_until = closed_at + timedelta(minutes=single_loss_cooldown_minutes)
    if now_utc < loss_until:
        mins = max(1, int((loss_until - now_utc).total_seconds() // 60) + 1)
        return f"zarar sonrası hızlı geri alış bekletmesi ({mins} dk kaldı)"
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
        raw_vol = min(range_pct, 15.0) / 15.0
        vol_score = raw_vol * 0.25 if chg < -1.0 else (raw_vol * 0.65 if chg < 0.0 else raw_vol)
        if chg < -1.0:
            mom_score = 0.0
        elif chg <= 0.0:
            mom_score = 0.08
        elif chg <= 25.0:
            mom_score = min(chg / 12.0, 1.0)
        else:
            mom_score = max(0.35, 1.0 - (chg - 25.0) / 60.0)
        spread_score = 1.0 - min(spread / max_spread, 1.0)
        score = 0.28 * liq + 0.27 * vol_score + 0.30 * mom_score + 0.15 * spread_score
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


# ---------------------------------------------------------------------------
# Kronos / FreqAI / Hummingbot / skfolio esinli ileri seviye nicel koruma katmanı
# ---------------------------------------------------------------------------
def candle_uncertainty_block(
    candles: Sequence[Candle],
    *,
    window: int = 6,
    max_wick_ratio: float = 0.80,
) -> str | None:
    """
    Kronos esinli OHLCV belirsizlik (gürültü/sinyal) koruması:
    Son `window` kapalı mumda fitil oranı aşırı yüksek (`> max_wick_ratio`) ve net yönlü
    ilerleme çok zayıfsa (kararsız testere/whipsaw rejimi) yeni alımı durdurur.
    """
    closed = [c for c in candles if c.is_closed]
    if len(closed) < window:
        return None
    recent = closed[-window:]
    wick_ratios: list[float] = []
    total_range = 0.0
    for c in recent:
        hi = float(c.high)
        lo = float(c.low)
        op = float(c.open)
        cl = float(c.close)
        rng = hi - lo
        if rng <= 0.0:
            continue
        body = abs(cl - op)
        wick_ratios.append(max(0.0, (rng - body) / rng))
        total_range += rng
    if len(wick_ratios) < max(3, window // 2) or total_range <= 0.0:
        return None
    avg_wick = sum(wick_ratios) / len(wick_ratios)
    net_move = abs(float(recent[-1].close) - float(recent[0].open))
    efficiency = net_move / total_range
    if avg_wick >= max_wick_ratio and efficiency < 0.15:
        return (
            f"yüksek mum belirsizliği/testere rejimi "
            f"(fitil %{avg_wick * 100:.0f}, yön verimliliği %{efficiency * 100:.0f})"
        )
    return None


def anomaly_outlier_block(
    candles: Sequence[Candle],
    *,
    lookback: int = 20,
    max_return_z: float = 3.5,
    max_volume_z: float = 4.5,
) -> str | None:
    """
    FreqAI Dissimilarity Index (DI) esinli aykırı değer / manipülasyon mumu koruması:
    Son kapalı mum, son `lookback` mumluk dağılıma göre aşırı z-skoru sıçraması
    (flash pump/dump veya dev üst fitilli tükeniş mumu) gösteriyorsa tepe kovalamayı engeller.
    """
    closed = [c for c in candles if c.is_closed]
    if len(closed) < lookback + 2:
        return None
    hist = closed[-(lookback + 1) : -1]
    latest = closed[-1]
    hist_rets: list[float] = []
    hist_vols: list[float] = []
    for c in hist:
        op = float(c.open)
        cl = float(c.close)
        if op > 0.0:
            hist_rets.append((cl - op) / op)
        hist_vols.append(float(c.volume))
    if len(hist_rets) < 10 or float(latest.open) <= 0.0:
        return None

    def _zscore(val: float, series: list[float]) -> float:
        mean = sum(series) / len(series)
        var = sum((x - mean) ** 2 for x in series) / len(series)
        std = math.sqrt(var)
        return (val - mean) / std if std > 1e-12 else 0.0

    latest_ret = (float(latest.close) - float(latest.open)) / float(latest.open)
    ret_z = abs(_zscore(latest_ret, hist_rets))
    vol_z = _zscore(float(latest.volume), hist_vols)

    rng = float(latest.high) - float(latest.low)
    upper_wick_ratio = (
        (float(latest.high) - max(float(latest.open), float(latest.close))) / rng
        if rng > 0.0
        else 0.0
    )

    if ret_z >= max_return_z:
        return f"aykırı mum hareketi (getiri z={ret_z:.1f}; FreqAI DI koruması)"
    if vol_z >= max_volume_z and upper_wick_ratio >= 0.55:
        return (
            f"anormal hacim + üst fitil tükenişi "
            f"(hacim z={vol_z:.1f}, üst fitil %{upper_wick_ratio * 100:.0f})"
        )
    return None


def orderbook_imbalance(order_book: Any, *, levels: int = 5) -> float | None:
    """
    Hummingbot esinli emir defteri ilk `levels` kademe alıcı/satıcı dengesi [-1.0, +1.0].
    Pozitif değer alış derinliğinin, negatif değer satış duvarının ağır bastığını gösterir.
    """
    bids = getattr(order_book, "bids", None) or []
    asks = getattr(order_book, "asks", None) or []
    bid_vol = sum(float(lvl.quantity) for lvl in bids[:levels] if float(lvl.quantity) > 0)
    ask_vol = sum(float(lvl.quantity) for lvl in asks[:levels] if float(lvl.quantity) > 0)
    total = bid_vol + ask_vol
    if total <= 0.0:
        return None
    return (bid_vol - ask_vol) / total


def orderbook_microstructure_block(
    order_book: Any,
    *,
    intended_notional: Decimal = Decimal("0"),
    max_slippage_pct: Decimal = Decimal("0.0025"),
    min_imbalance: float = -0.65,
    levels: int = 5,
) -> str | None:
    """
    Hummingbot / NautilusTrader esinli emir defteri derinlik ve kayma (slippage) koruması:
    1) Ağır satış duvarı dengesizliğini (`imbalance < min_imbalance`),
    2) Planlanan alım tutarı için tahmini VWAP kaymasını (`> max_slippage_pct`) ve sığ defteri engeller.
    """
    imb = orderbook_imbalance(order_book, levels=levels)
    if imb is not None and imb < min_imbalance:
        return f"emir defterinde yoğun satış baskısı (imbalance={imb:.2f})"

    if intended_notional <= Decimal("0"):
        return None
    asks = getattr(order_book, "asks", None) or []
    best_ask = getattr(order_book, "best_ask", None)
    if not asks or best_ask is None or best_ask <= Decimal("0"):
        return None

    remaining = intended_notional
    total_qty = Decimal("0")
    total_cost = Decimal("0")
    for lvl in asks[:levels]:
        px = Decimal(str(lvl.price))
        qty = Decimal(str(lvl.quantity))
        if px <= 0 or qty <= 0:
            continue
        lvl_notional = px * qty
        take_notional = min(remaining, lvl_notional)
        total_qty += take_notional / px
        total_cost += take_notional
        remaining -= take_notional
        if remaining <= Decimal("0"):
            break

    if remaining > Decimal("0") and total_cost < intended_notional * Decimal("0.95"):
        return f"emir defteri sığ (ilk {levels} kademe {intended_notional:.2f} USDT alımı karşılamıyor)"

    if total_qty > Decimal("0"):
        vwap = total_cost / total_qty
        slippage_pct = (vwap - best_ask) / best_ask
        if slippage_pct > max_slippage_pct:
            return f"tahmini emir defteri kayması yüksek (%{slippage_pct * Decimal('100'):.3f})"
    return None


def inverse_vol_capped_notional(
    approved_notional: Decimal,
    atr_pct: Decimal | None,
    *,
    target_vol_pct: Decimal = Decimal("0.015"),
    min_scale: Decimal = Decimal("0.45"),
) -> Decimal:
    """
    skfolio esinli Ters-Volatilite (Risk Parity) pozisyon ölçekleme:
    ATR volatilitesi hedef eşiğin (`%1.5`) üzerindeyse pozisyon tutarını orantılı küçültür
    (en fazla `min_scale` oranına kadar), asla büyütmez.
    """
    if approved_notional <= Decimal("0") or atr_pct is None or atr_pct <= target_vol_pct:
        return approved_notional
    scale = max(min_scale, target_vol_pct / atr_pct)
    return approved_notional * min(Decimal("1"), scale)
