"""Kural tabanlı teknik gösterge stratejisi (EMA, RSI, MACD, ATR) — yalnızca kapanmış mumlar."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from .domain import Candle, DecisionAction, TechnicalSignal


@dataclass(frozen=True)
class StrategyParams:
    """Kronolojik eğitim diliminde kalibre edilebilen teknik eşik parametreleri."""

    ema_fast_span: int = 9
    ema_slow_span: int = 21
    rsi_period: int = 14
    rsi_buy_min: float = 44.0
    rsi_buy_max: float = 68.0
    rsi_overbought_exit: float = 74.0
    min_volume_ratio: float = 0.85
    require_close_above_fast: bool = True
    require_rising_momentum: bool = True
    fast_exit: bool = False
    dip_rebound_buy: bool = False


def compute_indicator_frame(
    closed_candles: list[Candle],
    params: StrategyParams | None = None,
) -> pd.DataFrame:
    """
    Kapanmış mum dizisinden nedensel (gelecek veriye sızmayan / lookahead-free)
    teknik göstergeleri hesaplar.
    """
    cfg = params or StrategyParams()
    if any(not c.is_closed for c in closed_candles):
        raise ValueError("Güvenlik hatası: Kapanmamış mum ile teknik sinyal hesaplanamaz!")

    if len(closed_candles) < max(cfg.ema_slow_span + 10, 35):
        raise ValueError("Teknik gösterge hesabı için yeterli kapanmış mum verisi yok.")

    rows = [
        {
            "open_time_ms": c.open_time_ms,
            "close_time_ms": c.close_time_ms,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
        }
        for c in closed_candles
    ]
    df = pd.DataFrame(rows).sort_values("open_time_ms").reset_index(drop=True)

    df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast_span, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow_span, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / cfg.rsi_period, min_periods=cfg.rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / cfg.rsi_period, min_periods=cfg.rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Kayıp 0 ve kazanç 0 ise nötr 50, yalnızca kayıp 0 ise 100
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    df["rsi_14"] = rsi.fillna(50.0)

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean().bfill()

    vol_ma20 = df["volume"].rolling(window=20, min_periods=5).mean()
    df["volume_ratio"] = (df["volume"] / vol_ma20.replace(0.0, 1e-12)).fillna(1.0)
    df["return_pct"] = (df["close"].pct_change() * 100.0).fillna(0.0)
    return df


def evaluate_rule_signal_from_frame(
    df: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    row_index: int = -1,
    params: StrategyParams | None = None,
) -> TechnicalSignal:
    """Verilen satır (varsayılan son kapanmış mum) için kural tabanlı sinyal üretir."""
    cfg = params or StrategyParams()
    idx = len(df) - 1 if row_index == -1 else row_index
    if idx < 1:
        raise ValueError("Sinyal karşılaştırması için en az 2 satır gerekir.")

    curr = df.iloc[idx]
    prev = df.iloc[idx - 1]

    ema_fast = float(curr["ema_fast"])
    ema_slow = float(curr["ema_slow"])
    rsi_val = float(curr["rsi_14"])
    macd_line = float(curr["macd_line"])
    macd_sig = float(curr["macd_signal"])
    macd_hist = float(curr["macd_hist"])
    prev_macd_hist = float(prev["macd_hist"])
    atr_val = float(curr["atr_14"])
    vol_ratio = float(curr["volume_ratio"])
    close_val = float(curr["close"])

    if cfg.require_close_above_fast:
        trend_up = ema_fast > ema_slow and close_val > ema_fast
    else:
        trend_up = (ema_fast > ema_slow and close_val >= ema_slow) or (close_val > ema_fast)

    if cfg.require_rising_momentum:
        momentum_rising = macd_hist > 0.0 and macd_hist >= prev_macd_hist
    else:
        momentum_rising = (macd_hist >= 0.0) or (close_val > float(prev["close"]))

    rsi_healthy_for_buy = cfg.rsi_buy_min <= rsi_val <= cfg.rsi_buy_max
    volume_supported = vol_ratio >= cfg.min_volume_ratio
    open_val = float(curr["open"])
    prev_close_val = float(prev["close"])
    dip_bounce_up = (
        cfg.dip_rebound_buy
        and (20.0 <= rsi_val <= 58.0)
        and (close_val > prev_close_val)
        and (close_val >= open_val)
        and (macd_hist >= prev_macd_hist or close_val >= ema_fast * 0.9985)
        and volume_supported
    )

    trend_reversal_down = ema_fast < ema_slow and macd_hist < 0.0
    overbought_cooling = rsi_val >= cfg.rsi_overbought_exit and macd_hist < prev_macd_hist
    quick_momentum_loss = cfg.fast_exit and (
        (close_val < ema_fast and macd_hist <= 0.0) or (close_val < ema_slow)
    )

    action: DecisionAction = "BEKLE"
    if trend_up and momentum_rising and rsi_healthy_for_buy and volume_supported:
        action = "AL"
        reason = (
            f"Kural tabanlı yükseliş teyidi: EMA({cfg.ema_fast_span})={ema_fast:.2f}, "
            f"EMA({cfg.ema_slow_span})={ema_slow:.2f}, MACD_hist={macd_hist:.4f}, "
            f"RSI={rsi_val:.1f}, Hacim={vol_ratio:.2f}x."
        )
    elif dip_bounce_up:
        action = "AL"
        reason = (
            f"Dip dönüşü ve tepki yükselişi alımı: Kapanış ({close_val:.4g}) > Önceki ({prev_close_val:.4g}), "
            f"RSI={rsi_val:.1f}, Hacim={vol_ratio:.2f}x."
        )
    elif trend_reversal_down:
        action = "SAT"
        reason = (
            f"Kural tabanlı zayıflama sinyali: EMA({cfg.ema_fast_span}) < EMA({cfg.ema_slow_span}) "
            f"ve MACD histogram negatif ({macd_hist:.4f})."
        )
    elif overbought_cooling:
        action = "SAT"
        reason = (
            f"Kural tabanlı aşırı alım soğuması: RSI yüksek ({rsi_val:.1f}) ve "
            f"momentum ivmesi azalıyor."
        )
    elif quick_momentum_loss:
        action = "SAT"
        reason = (
            f"Hızlı çıkış (30sn/1dk aktif mod): Kapanış ({close_val:.4g}) kısa ortalama "
            f"EMA({cfg.ema_fast_span}) altına indi veya ivme negatife döndü."
        )
    else:
        action = "BEKLE"
        reason = (
            f"Net kural kesişimi yok (EMA{cfg.ema_fast_span}={ema_fast:.4g}, "
            f"EMA{cfg.ema_slow_span}={ema_slow:.4g}, RSI={rsi_val:.1f}, Hacim={vol_ratio:.2f}x); beklemede kalındı."
        )

    return TechnicalSignal(
        symbol=symbol,
        interval=interval,
        candle_open_time_ms=int(curr["open_time_ms"]),
        candle_close_time_ms=int(curr["close_time_ms"]),
        action=action,
        reason=reason,
        ema_fast=Decimal(f"{ema_fast:.8f}"),
        ema_slow=Decimal(f"{ema_slow:.8f}"),
        rsi_14=Decimal(f"{rsi_val:.4f}"),
        macd_line=Decimal(f"{macd_line:.8f}"),
        macd_signal=Decimal(f"{macd_sig:.8f}"),
        macd_hist=Decimal(f"{macd_hist:.8f}"),
        atr_14=Decimal(f"{atr_val:.8f}"),
        volume_ratio=Decimal(f"{vol_ratio:.4f}"),
        close_price=Decimal(f"{close_val:.8f}"),
    )


def evaluate_technical_strategy(
    closed_candles: list[Candle],
    *,
    symbol: str,
    interval: str,
    params: StrategyParams | None = None,
) -> tuple[TechnicalSignal, list[str]]:
    """
    Son kapanmış mum için kural tabanlı sinyali ve son 5 kapanmış mumun yüzde getiri özetini döndürür.
    Not: Bu modül saf matematiksel göstergedir; yapay zekâ olarak adlandırılmaz.
    """
    df = compute_indicator_frame(closed_candles, params=params)
    signal = evaluate_rule_signal_from_frame(df, symbol=symbol, interval=interval, params=params)
    recent_returns = [f"{float(v):+.2f}%" for v in df["return_pct"].tail(5).tolist()]
    return signal, recent_returns
