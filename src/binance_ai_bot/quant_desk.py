"""
Kantitatif Hedge-Fon Araştırma ve Yürütme Çekirdeği (6 GitHub Referans Mimarisi).

Birleştirilen açık kaynak yaklaşımlar (emir yolu her zaman LLM dışındadır):
1. Kronos (shiyu-coder/Kronos): Kapanmış OHLCV pencerelerinden çoklu ufuk (+5/+15 mum)
   olasılıksal belirsizlik bantları (`p10`, `p50`, `p90`, `up_prob`).
2. skfolio (skfolio/skfolio): Lopez de Prado Hiyerarşik Risk Paritesi (HRP) küme ağırlıkları,
   büzüştürülmüş kovaryans, %95 VaR/CVaR ve çeşitlendirme oranı.
3. NautilusTrader (nautechsystems/nautilus_trader): Geriye dönük test ile canlı motor
   arasında aynı nedensel (lookahead-free) sinyal + boyutlandırma çekirdeği eşliği.
4. Hummingbot (hummingbot/hummingbot): L2 emir defteri dengesizliği (imbalance),
   ağırlıklı mikro-fiyat (microprice), NATR dinamik makas ve Avellaneda envanter kaydırması.
5. Freqtrade / FreqAI (freqtrade/freqtrade): Kendi kendini yeniden eğiten kayan pencereli
   Ridge/Lojistik skorlayıcı + Uyumsuzluk İndeksi (Dissimilarity Index) anomali kapısı ve
   Telegram iki yönlü komut yönlendirici.
6. QuantDinger & ai-hedge-fund: Doğal dilden strateji/araştırma planlama + nedensel
   backtest + fail-loud/abstain sözleşmesi.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .domain import Candle, OrderBookSnapshot, RiskSettings, SymbolFilters
from .strategy import StrategyParams, compute_indicator_frame
from .validation import SliceBacktestMetrics, run_no_lookahead_slice_simulation


# ---------------------------------------------------------------------------
# 1. Kronos Tarzı Olasılıksal OHLCV Mum Tahmincisi
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HorizonForecast:
    horizon_bars: int
    p10_pct: float
    p50_pct: float
    p90_pct: float
    uncertainty_pct: float
    up_prob: float
    analog_similarity: float


@dataclass(frozen=True)
class KronosCandleForecast:
    symbol: str
    interval: str
    horizons: tuple[HorizonForecast, ...]
    regime_label: str
    sample_count: int
    summary_tr: str


def _candle_feature_matrix(candles: list[Candle]) -> np.ndarray:
    """Kapanmış OHLCV mumlarından normalize edilmiş nedensel şerit matrisi üretir."""
    closes = np.array([float(c.close) for c in candles], dtype=np.float64)
    opens = np.array([float(c.open) for c in candles], dtype=np.float64)
    highs = np.array([float(c.high) for c in candles], dtype=np.float64)
    lows = np.array([float(c.low) for c in candles], dtype=np.float64)
    vols = np.array([max(float(c.volume), 1e-12) for c in candles], dtype=np.float64)

    prev_c = np.concatenate(([closes[0]], closes[:-1]))
    safe_prev = np.where(prev_c > 0, prev_c, 1e-12)
    safe_o = np.where(opens > 0, opens, safe_prev)
    log_ret = np.log(np.maximum(closes, 1e-12) / safe_prev)
    body = (closes - opens) / safe_o
    hl_range = np.maximum(highs - lows, 1e-12) / np.maximum(closes, 1e-12)
    upper_wick = np.maximum(highs - np.maximum(opens, closes), 0.0) / np.maximum(closes, 1e-12)
    lower_wick = np.maximum(np.minimum(opens, closes) - lows, 0.0) / np.maximum(closes, 1e-12)

    log_vol = np.log1p(vols)
    vol_s = pd.Series(log_vol)
    vol_mean = vol_s.rolling(20, min_periods=5).mean().bfill().to_numpy()
    vol_std = vol_s.rolling(20, min_periods=5).std().fillna(1.0).to_numpy()
    vol_z = np.clip((log_vol - vol_mean) / np.maximum(vol_std, 1e-3), -3.0, 3.0)

    return np.column_stack([log_ret, body, hl_range, upper_wick, lower_wick, vol_z])


def forecast_ohlcv_candles(
    closed_candles: list[Candle],
    *,
    symbol: str,
    interval: str,
    horizons: tuple[int, ...] = (5, 15),
    window: int = 16,
    n_analogs: int = 24,
) -> KronosCandleForecast:
    """
    Kronos benzeri çoklu-örneklemli OHLCV rejim kestirimi:
    Son `window` kapanmış mum dokusuna kosinüs benzerliği en yüksek geçmiş blokların
    gerçekleşmiş ileri getirilerinden p10/p50/p90 belirsizlik bandı üretir (lookahead yok).
    """
    if len(closed_candles) < max(window + max(horizons) + 12, 35):
        empty = tuple(
            HorizonForecast(h, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0) for h in horizons
        )
        return KronosCandleForecast(
            symbol=symbol,
            interval=interval,
            horizons=empty,
            regime_label="YETERSIZ_VERI",
            sample_count=0,
            summary_tr=f"{symbol} için Kronos kestirimi adına yeterli kapanmış mum yok.",
        )

    feat = _candle_feature_matrix(closed_candles)
    closes = np.array([float(c.close) for c in closed_candles], dtype=np.float64)
    n = len(closed_candles)

    # Sorgu penceresi yalnızca son kapalı mumlara kadar uzanır
    query_mat = feat[n - window : n].copy()
    q_mean = query_mat.mean(axis=0, keepdims=True)
    q_std = np.maximum(query_mat.std(axis=0, keepdims=True), 1e-6)
    query_vec = ((query_mat - q_mean) / q_std).ravel()
    q_norm = max(float(np.linalg.norm(query_vec)), 1e-9)

    max_h = max(horizons)
    max_end = n - max_h - 1
    if max_end <= window:
        empty = tuple(HorizonForecast(h, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0) for h in horizons)
        return KronosCandleForecast(symbol, interval, empty, "YATAY_BELIRSIZ", 0, "Analog penceresi kısa.")

    # Tarihsel aday blokları vektörleştirerek puanla
    sims: list[tuple[int, float]] = []
    for end_idx in range(window, max_end + 1):
        hist_mat = feat[end_idx - window : end_idx]
        h_mean = hist_mat.mean(axis=0, keepdims=True)
        h_std = np.maximum(hist_mat.std(axis=0, keepdims=True), 1e-6)
        hist_vec = ((hist_mat - h_mean) / h_std).ravel()
        h_norm = max(float(np.linalg.norm(hist_vec)), 1e-9)
        sim = float(np.dot(query_vec, hist_vec) / (q_norm * h_norm))
        sims.append((end_idx, sim))

    sims.sort(key=lambda item: item[1], reverse=True)
    top = sims[: max(8, min(n_analogs, len(sims)))]
    raw_weights = np.array([max(s, 0.05) ** 2 for _, s in top], dtype=np.float64)
    weights = raw_weights / max(float(raw_weights.sum()), 1e-9)
    avg_sim = float(np.mean([s for _, s in top]))

    horizon_results: list[HorizonForecast] = []
    for h in horizons:
        rets_pct = np.array(
            [((closes[end_idx - 1 + h] / max(closes[end_idx - 1], 1e-12)) - 1.0) * 100.0 for end_idx, _ in top],
            dtype=np.float64,
        )
        order = np.argsort(rets_pct)
        sorted_r = rets_pct[order]
        sorted_w = weights[order]
        cdf = np.cumsum(sorted_w)
        p10 = float(np.interp(0.10, cdf, sorted_r))
        p50 = float(np.interp(0.50, cdf, sorted_r))
        p90 = float(np.interp(0.90, cdf, sorted_r))
        up_prob = float(sorted_w[sorted_r > 0.0].sum())
        horizon_results.append(
            HorizonForecast(
                horizon_bars=h,
                p10_pct=round(p10, 3),
                p50_pct=round(p50, 3),
                p90_pct=round(p90, 3),
                uncertainty_pct=round(max(p90 - p10, 0.0), 3),
                up_prob=round(min(max(up_prob, 0.0), 1.0), 3),
                analog_similarity=round(avg_sim, 3),
            )
        )

    lead = horizon_results[0]
    if lead.up_prob >= 0.62 and lead.p50_pct > 0.08:
        regime = "YUKSELIS_OLASIGI"
    elif lead.up_prob <= 0.38 and lead.p50_pct < -0.08:
        regime = "DUSUS_RISKI"
    else:
        regime = "YATAY_BELIRSIZ"

    summary_tr = (
        f"Kronos +{lead.horizon_bars}m kestirimi: p50 {lead.p50_pct:+.2f}% "
        f"[p10 {lead.p10_pct:+.2f}% .. p90 {lead.p90_pct:+.2f}%], yükselme olasılığı %{lead.up_prob * 100:.0f}"
    )
    return KronosCandleForecast(
        symbol=symbol,
        interval=interval,
        horizons=tuple(horizon_results),
        regime_label=regime,
        sample_count=len(top),
        summary_tr=summary_tr,
    )


# ---------------------------------------------------------------------------
# 2. skfolio Tarzı Hiyerarşik Risk Paritesi (HRP) & CVaR Stres Ölçütleri
# ---------------------------------------------------------------------------

def _seriate_cluster_leaves(dist: np.ndarray) -> list[int]:
    """Mesafe matrisinden açgözlü en yakın komşu + aglomeratif sıralama üretir."""
    n = dist.shape[0]
    if n <= 2:
        return list(range(n))
    # Küçük sembol kümeleri (2..24) için hızlı seriation: en düşük ortalama uzaklıktan başla
    remaining = set(range(n))
    start = int(np.argmin(dist.sum(axis=1)))
    order = [start]
    remaining.remove(start)
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda idx: float(dist[last, idx]))
        order.append(nxt)
        remaining.remove(nxt)
    return order


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    sub_cov = cov[np.ix_(indices, indices)]
    diag = np.maximum(np.diag(sub_cov), 1e-12)
    ivp = 1.0 / diag
    w = ivp / max(float(ivp.sum()), 1e-12)
    return float(w @ sub_cov @ w)


def optimize_hrp_weights(returns_by_symbol: dict[str, list[float] | np.ndarray]) -> dict[str, float]:
    """
    skfolio / Lopez de Prado Hierarchical Risk Parity (HRP):
    Ledoit-Wolf tarzı büzüştürülmüş kovaryans + korelasyon uzaklık sıralaması + özyinelemeli ikiye bölme.
    """
    clean: dict[str, np.ndarray] = {}
    for sym, series in returns_by_symbol.items():
        arr = np.asarray(series, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= 10:
            clean[sym] = arr

    symbols = list(clean.keys())
    if not symbols:
        return {}
    if len(symbols) == 1:
        return {symbols[0]: 1.0}

    min_len = min(len(v) for v in clean.values())
    mat = np.column_stack([clean[s][-min_len:] for s in symbols])
    raw_cov = np.cov(mat, rowvar=False, ddof=1)
    if np.ndim(raw_cov) < 2:
        return {s: 1.0 / len(symbols) for s in symbols}

    # Ledoit-Wolf tarzı köşegen büzüştürme (küçük örneklem kararsızlığını önler)
    shrink = 0.20 if min_len < 60 else 0.10
    diag_cov = np.diag(np.diag(raw_cov))
    cov = (1.0 - shrink) * raw_cov + shrink * diag_cov
    cov = cov + np.eye(len(symbols)) * 1e-10

    vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(vols, vols)
    corr = np.clip(corr, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)

    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    order_idx = _seriate_cluster_leaves(dist)

    weights = np.ones(len(symbols), dtype=np.float64)
    clusters: list[list[int]] = [order_idx]
    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            var_l = max(_cluster_variance(cov, left), 1e-12)
            var_r = max(_cluster_variance(cov, right), 1e-12)
            alpha = 1.0 - var_l / (var_l + var_r)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters

    total = max(float(weights.sum()), 1e-12)
    return {symbols[i]: round(float(weights[i] / total), 4) for i in range(len(symbols))}


@dataclass(frozen=True)
class PortfolioRiskReport:
    weights: dict[str, float]
    portfolio_bar_vol_pct: float
    var_95_pct: float
    cvar_95_pct: float
    diversification_ratio: float
    max_corr_with_open: float
    sizing_multiplier: float


def compute_portfolio_risk_metrics(
    returns_by_symbol: dict[str, list[float] | np.ndarray],
    *,
    candidate_symbol: str | None = None,
    open_symbols: Iterable[str] = (),
) -> PortfolioRiskReport:
    """
    skfolio risk paritesi ağırlıklarıyla portföy VaR(%95), CVaR(%95) ve aday coin için
    korelasyon yoğunlaşmasına göre daraltıcı pozisyon çarpanı (`<= 1.0`) hesaplar.
    """
    weights = optimize_hrp_weights(returns_by_symbol)
    if not weights:
        return PortfolioRiskReport({}, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)

    syms = list(weights.keys())
    min_len = min(len(returns_by_symbol[s]) for s in syms)
    mat = np.column_stack([np.asarray(returns_by_symbol[s][-min_len:], dtype=np.float64) for s in syms])
    w_vec = np.array([weights[s] for s in syms], dtype=np.float64)
    port_rets = mat @ w_vec * 100.0  # yüzde getiri

    p_vol = float(np.std(port_rets, ddof=1)) if len(port_rets) > 1 else 0.0
    var_95 = float(-np.percentile(port_rets, 5))
    tail = port_rets[port_rets <= -var_95] if var_95 > -1e9 else np.array([])
    cvar_95 = float(-np.mean(tail)) if len(tail) > 0 else max(var_95, 0.0)

    ind_vols = np.std(mat * 100.0, axis=0, ddof=1) if len(port_rets) > 1 else np.zeros(len(syms))
    weighted_ind_vol = float(np.dot(w_vec, ind_vols))
    div_ratio = round(weighted_ind_vol / max(p_vol, 1e-6), 3) if p_vol > 0 else 1.0

    max_corr = 0.0
    open_set = {s.upper() for s in open_symbols if s}
    if candidate_symbol and candidate_symbol.upper() in weights and open_set:
        c_idx = syms.index(candidate_symbol.upper())
        c_series = mat[:, c_idx]
        for o_sym in open_set:
            if o_sym == candidate_symbol.upper() or o_sym not in weights:
                continue
            o_idx = syms.index(o_sym)
            o_series = mat[:, o_idx]
            if np.std(c_series) > 1e-9 and np.std(o_series) > 1e-9:
                rho = float(np.corrcoef(c_series, o_series)[0, 1])
                if math.isfinite(rho):
                    max_corr = max(max_corr, rho)

    # Yüksek korelasyon veya aşırı CVaR varsa pozisyon büyüklüğünü %45..%100 arasına daralt (asla büyütmez)
    sizing = 1.0
    if max_corr > 0.78:
        sizing *= max(0.55, 1.0 - (max_corr - 0.78) * 1.4)
    if cvar_95 > 2.2:
        sizing *= max(0.65, 2.2 / cvar_95)
    if candidate_symbol and candidate_symbol.upper() in weights and len(syms) >= 2:
        equal_w = 1.0 / len(syms)
        rel_w = weights[candidate_symbol.upper()] / max(equal_w, 1e-6)
        if rel_w < 0.90:
            sizing *= max(0.55, rel_w)
    sizing = round(min(max(sizing, 0.45), 1.0), 3)

    return PortfolioRiskReport(
        weights=weights,
        portfolio_bar_vol_pct=round(p_vol, 3),
        var_95_pct=round(max(var_95, 0.0), 3),
        cvar_95_pct=round(max(cvar_95, 0.0), 3),
        diversification_ratio=div_ratio,
        max_corr_with_open=round(max_corr, 3),
        sizing_multiplier=sizing,
    )


# ---------------------------------------------------------------------------
# 3. Hummingbot Tarzı L2 Mikroyapı & Avellaneda Envanter Kaydırması
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MicrostructureAssessment:
    symbol: str
    spread_bps: float
    imbalance: float
    microprice_edge_bps: float
    natr_pct: float
    inventory_skew_bps: float
    ask_wall_blocked: bool
    reason: str


def evaluate_microstructure(
    order_book: OrderBookSnapshot,
    closed_candles: list[Candle],
    *,
    inventory_ratio: float = 0.0,
    max_spread_bps: float = 42.0,
    top_levels: int = 10,
) -> MicrostructureAssessment:
    """
    Hummingbot PMM Dynamic & Avellaneda yaklaşımı:
    L2 derinlik dengesizliği, hacim ağırlıklı mikro-fiyat sapması, NATR volatilitesi
    ve stok yoğunluğuna göre alış engeli/fiyat kaydırma ölçümü üretir.
    """
    best_bid = float(order_book.best_bid or 0)
    best_ask = float(order_book.best_ask or 0)
    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        return MicrostructureAssessment(order_book.symbol, 0.0, 0.0, 0.0, 0.0, 0.0, False, "Emir defteri eksik.")

    mid = (best_bid + best_ask) * 0.5
    spread_bps = ((best_ask - best_bid) / mid) * 10_000.0

    bid_w = 0.0
    ask_w = 0.0
    for idx, lvl in enumerate(order_book.bids[:top_levels]):
        decay = 1.0 / (1.0 + idx * 0.35)
        bid_w += float(lvl.price) * float(lvl.quantity) * decay
    for idx, lvl in enumerate(order_book.asks[:top_levels]):
        decay = 1.0 / (1.0 + idx * 0.35)
        ask_w += float(lvl.price) * float(lvl.quantity) * decay

    tot_w = bid_w + ask_w
    imbalance = ((bid_w - ask_w) / tot_w) if tot_w > 0 else 0.0

    b_sz = float(order_book.bids[0].quantity) if order_book.bids else 0.0
    a_sz = float(order_book.asks[0].quantity) if order_book.asks else 0.0
    if (b_sz + a_sz) > 0:
        microprice = (best_bid * a_sz + best_ask * b_sz) / (b_sz + a_sz)
        micro_edge_bps = ((microprice - mid) / mid) * 10_000.0
    else:
        micro_edge_bps = 0.0

    natr_pct = 0.0
    if len(closed_candles) >= 15:
        tail = closed_candles[-15:]
        trs = []
        for i in range(1, len(tail)):
            hi = float(tail[i].high)
            lo = float(tail[i].low)
            pc = float(tail[i - 1].close)
            trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
        atr = sum(trs) / len(trs)
        last_c = float(tail[-1].close)
        if last_c > 0:
            natr_pct = (atr / last_c) * 100.0

    # Avellaneda reservation shift: eldeki envanter arttıkça alış iştahını aşağı çeker
    inv_clipped = min(max(float(inventory_ratio), 0.0), 1.0)
    inventory_skew_bps = -inv_clipped * max(natr_pct * 25.0, 4.0)

    blocked = False
    reason = f"L2 makas {spread_bps:.1f}bps, dengesizlik {imbalance:+.2f}, mikro-fiyat {micro_edge_bps:+.1f}bps."
    if spread_bps > max_spread_bps:
        blocked = True
        reason = f"Hummingbot likidite kilidi: alış/satış makası çok geniş ({spread_bps:.1f} bps > {max_spread_bps:.0f} bps)."
    elif imbalance < -0.62 and micro_edge_bps < -1.5:
        blocked = True
        reason = f"Hummingbot satış duvarı engeli: emir defteri ağır satış baskısında (imbalance {imbalance:+.2f})."

    return MicrostructureAssessment(
        symbol=order_book.symbol,
        spread_bps=round(spread_bps, 2),
        imbalance=round(imbalance, 3),
        microprice_edge_bps=round(micro_edge_bps, 2),
        natr_pct=round(natr_pct, 3),
        inventory_skew_bps=round(inventory_skew_bps, 2),
        ask_wall_blocked=blocked,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 4. FreqAI Tarzı Kendi Kendini Yeniden Eğiten Skorlayıcı & Uyumsuzluk İndeksi
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreqAISignalResult:
    symbol: str
    interval: str
    prob_up: float
    expected_return_pct: float
    dissimilarity_index: float
    do_predict: bool
    last_retrain_bar: int
    train_samples: int
    reason: str


class SelfRetrainingSignalModel:
    """
    Freqtrade / FreqAI öz-eğitim ve Dissimilarity Index (DI) kapısı:
    - Kayan pencere üzerinde nedensel L2-Ridge modeli otomatik olarak `retrain_interval_bars`
      barlık adımlarla yeniden eğitilir.
    - Canlı özellik vektörü eğitim uzayının en yakın komşularından çok uzaklaştıysa
      (DI > di_threshold), anomali rejiminde yanlış sinyali önlemek için `do_predict=False` döndürür.
    """

    def __init__(
        self,
        *,
        horizon_bars: int = 5,
        retrain_interval_bars: int = 12,
        train_window_bars: int = 180,
        di_threshold: float = 2.6,
        ridge_alpha: float = 4.0,
    ) -> None:
        self.horizon_bars = max(2, int(horizon_bars))
        self.retrain_interval_bars = max(1, int(retrain_interval_bars))
        self.train_window_bars = max(40, int(train_window_bars))
        self.di_threshold = float(di_threshold)
        self.ridge_alpha = float(ridge_alpha)
        self._model_cache: dict[tuple[str, str, StrategyParams, int], dict[str, Any]] = {}

    @staticmethod
    def _build_features(df: pd.DataFrame) -> np.ndarray:
        close = df["close"].astype(float)
        ret_1 = close.pct_change(1).fillna(0.0)
        ret_3 = close.pct_change(3).fillna(0.0)
        ret_8 = close.pct_change(8).fillna(0.0)
        rsi_c = (df["rsi_14"].astype(float).fillna(50.0) - 50.0) / 50.0
        ema_fast = df["ema_fast"].astype(float).bfill()
        ema_slow = df["ema_slow"].astype(float).bfill()
        ema_diff = (ema_fast - ema_slow) / np.maximum(close.to_numpy(), 1e-12)
        atr_pct = df["atr_14"].astype(float).fillna(0.0) / np.maximum(close.to_numpy(), 1e-12)
        vol_ratio = np.log1p(np.maximum(df["volume_ratio"].astype(float).fillna(1.0), 0.0))
        macd_n = df["macd_hist"].astype(float).fillna(0.0) / np.maximum(close.to_numpy(), 1e-12)
        mat = np.column_stack(
            [
                ret_1.to_numpy(),
                ret_3.to_numpy(),
                ret_8.to_numpy(),
                rsi_c.to_numpy(),
                ema_diff.to_numpy(),
                atr_pct.to_numpy(),
                vol_ratio.to_numpy(),
                macd_n.to_numpy(),
            ]
        )
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    def evaluate(
        self,
        closed_candles: list[Candle],
        *,
        symbol: str,
        interval: str,
        params: StrategyParams | None = None,
        indicator_df: pd.DataFrame | None = None,
    ) -> FreqAISignalResult:
        if len(closed_candles) < 45:
            return FreqAISignalResult(
                symbol=symbol,
                interval=interval,
                prob_up=0.5,
                expected_return_pct=0.0,
                dissimilarity_index=0.0,
                do_predict=False,
                last_retrain_bar=0,
                train_samples=0,
                reason="FreqAI eğitimi için yeterli kapanmış mum yok.",
            )

        df = indicator_df if indicator_df is not None else compute_indicator_frame(closed_candles, params=params)
        X_all = self._build_features(df)
        closes = df["close"].astype(float).to_numpy()
        n = len(closes)
        last_idx = n - 1
        h = self.horizon_bars
        trainable_end = last_idx - h
        if trainable_end < 30:
            return FreqAISignalResult(
                symbol, interval, 0.5, 0.0, 0.0, False, 0, 0, "FreqAI eğitim penceresi yetersiz."
            )

        # A live client supplies a fixed-length rolling window, so its row index
        # does not advance. Anchor retraining to the actual candle time instead.
        bar_duration_ms = max(1, closed_candles[-1].close_time_ms - closed_candles[-1].open_time_ms + 1)
        absolute_bar = closed_candles[-1].open_time_ms // bar_duration_ms
        retrain_bucket = absolute_bar // self.retrain_interval_bars
        cache_key = (symbol, interval, params or StrategyParams(), retrain_bucket)
        model = self._model_cache.get(cache_key)

        if (model is None
                or int(model.get("trained_through_time_ms", 0)) > closed_candles[-1].close_time_ms):
            start_idx = max(20, trainable_end - self.train_window_bars)
            X_tr = X_all[start_idx : trainable_end + 1]
            y_tr = ((closes[start_idx + h : trainable_end + 1 + h] / np.maximum(closes[start_idx : trainable_end + 1], 1e-12)) - 1.0) * 100.0

            mean = X_tr.mean(axis=0)
            # Her özellik ekseninin doğal piyasa ölçeğiyle alt-sınır uygula (sabit hacim testlerinde z-patlamasını önler)
            min_scale = np.array([0.005, 0.010, 0.018, 0.20, 0.008, 0.006, 0.30, 0.004], dtype=np.float64)
            std = np.maximum(X_tr.std(axis=0), min_scale)
            Z_tr = np.clip((X_tr - mean) / std, -8.0, 8.0)
            # Bias sütunu ekle ve L2 Ridge kapalı form çöz
            Z_aug = np.column_stack([np.ones(len(Z_tr)), Z_tr])
            reg = np.eye(Z_aug.shape[1]) * self.ridge_alpha
            reg[0, 0] = 0.0
            try:
                beta = np.linalg.solve(Z_aug.T @ Z_aug + reg, Z_aug.T @ y_tr)
            except np.linalg.LinAlgError:
                beta = np.zeros(Z_aug.shape[1], dtype=np.float64)

            # Eğitim dağılımındaki tipik komşu & merkez uzaklığı (DI paydası)
            if len(Z_tr) >= 12:
                diff = Z_tr[:, None, :] - Z_tr[None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                mean_nn = float(np.mean(np.min(dists, axis=1)))
                med_radius = float(np.median(np.linalg.norm(Z_tr, axis=1))) * 0.65
                di_scale = max(mean_nn, med_radius, 1.0)
            else:
                di_scale = 1.25

            model = {
                "mean": mean,
                "std": std,
                "beta": beta,
                "Z_tr": Z_tr,
                "mean_nn": di_scale,
                "retrain_bar": retrain_bucket * self.retrain_interval_bars,
                "trainable_end": trainable_end,
                "trained_through_time_ms": closed_candles[-1].close_time_ms,
                "samples": len(Z_tr),
            }
            if len(self._model_cache) > 128:
                self._model_cache.clear()
            self._model_cache[cache_key] = model

        z_live = np.clip((X_all[last_idx] - model["mean"]) / model["std"], -10.0, 10.0)
        pred_pct = float(np.dot(np.concatenate(([1.0], z_live)), model["beta"]))
        pred_pct = max(min(pred_pct, 12.0), -12.0)

        nn_dist = float(np.min(np.linalg.norm(model["Z_tr"] - z_live[None, :], axis=1)))
        di = nn_dist / model["mean_nn"]
        do_predict = di <= self.di_threshold
        # Sigmoid benzeri olasılık dönüşümü
        prob_up = 1.0 / (1.0 + math.exp(-max(min(pred_pct * 1.4, 8.0), -8.0)))

        if not do_predict:
            reason = f"FreqAI uyumsuzluk kapısı (DI={di:.2f} > {self.di_threshold:.2f}): olağandışı rejim sapması."
        else:
            reason = f"FreqAI öz-eğitimli skor: +{h}m beklenen {pred_pct:+.2f}% (olasılık %{prob_up * 100:.0f}, DI={di:.2f})."

        return FreqAISignalResult(
            symbol=symbol,
            interval=interval,
            prob_up=round(prob_up, 3),
            expected_return_pct=round(pred_pct, 3),
            dissimilarity_index=round(di, 3),
            do_predict=do_predict,
            last_retrain_bar=int(model["retrain_bar"]),
            train_samples=int(model["samples"]),
            reason=reason,
        )


# ---------------------------------------------------------------------------
# 5. NautilusTarzı Birleşik Değerlendirme Çekirdeği (Backtest = Canlı)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantBundleAssessment:
    symbol: str
    interval: str
    kronos: KronosCandleForecast
    microstructure: MicrostructureAssessment
    freqai: FreqAISignalResult
    portfolio: PortfolioRiskReport
    quant_score: float
    sizing_multiplier: float
    entry_allowed: bool
    block_reason: str | None

    def to_summary_dict(self) -> dict[str, Any]:
        lead = self.kronos.horizons[0] if self.kronos.horizons else None
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "quant_score": self.quant_score,
            "sizing_multiplier": self.sizing_multiplier,
            "entry_allowed": self.entry_allowed,
            "block_reason": self.block_reason,
            "kronos_p50_pct": lead.p50_pct if lead else 0.0,
            "kronos_p10_pct": lead.p10_pct if lead else 0.0,
            "kronos_p90_pct": lead.p90_pct if lead else 0.0,
            "kronos_up_prob": lead.up_prob if lead else 0.5,
            "kronos_regime": self.kronos.regime_label,
            "freqai_prob_up": self.freqai.prob_up,
            "freqai_expected_pct": self.freqai.expected_return_pct,
            "freqai_di": self.freqai.dissimilarity_index,
            "freqai_do_predict": self.freqai.do_predict,
            "l2_spread_bps": self.microstructure.spread_bps,
            "l2_imbalance": self.microstructure.imbalance,
            "l2_microprice_bps": self.microstructure.microprice_edge_bps,
            "natr_pct": self.microstructure.natr_pct,
            "hrp_max_corr": self.portfolio.max_corr_with_open,
            "portfolio_var_95": self.portfolio.var_95_pct,
            "portfolio_cvar_95": self.portfolio.cvar_95_pct,
        }


def evaluate_quant_bundle(
    *,
    closed_candles: list[Candle],
    order_book: OrderBookSnapshot,
    symbol: str,
    interval: str,
    strategy_params: StrategyParams | None = None,
    freqai_model: SelfRetrainingSignalModel | None = None,
    returns_by_symbol: dict[str, list[float] | np.ndarray] | None = None,
    open_symbols: Iterable[str] = (),
    inventory_ratio: float = 0.0,
    strict_quant_gate: bool = False,
) -> QuantBundleAssessment:
    """
    Tüm 6 kantitatif katmanı tek bir deterministik değerlendirmede toplar.
    LLM asla emir yoluna sokulmaz; risk sınırını asla büyütmez (yalnızca küçültebilir).
    """
    kronos = forecast_ohlcv_candles(closed_candles, symbol=symbol, interval=interval)
    micro = evaluate_microstructure(order_book, closed_candles, inventory_ratio=inventory_ratio)
    model = freqai_model or SelfRetrainingSignalModel()
    freqai = model.evaluate(closed_candles, symbol=symbol, interval=interval, params=strategy_params)

    closes_arr = np.array([float(c.close) for c in closed_candles[-65:]], dtype=np.float64)
    own_rets = np.diff(np.log(np.maximum(closes_arr, 1e-12))) if len(closes_arr) > 2 else np.array([])
    ret_map = dict(returns_by_symbol or {})
    if len(own_rets) >= 10:
        ret_map[symbol.upper()] = own_rets
    portfolio = compute_portfolio_risk_metrics(
        ret_map,
        candidate_symbol=symbol.upper(),
        open_symbols=open_symbols,
    )

    lead = kronos.horizons[0] if kronos.horizons else None
    k_up = lead.up_prob if lead else 0.5
    score = (
        40.0 * k_up
        + 35.0 * (freqai.prob_up if freqai.do_predict else 0.35)
        + 25.0 * min(max(0.5 + micro.imbalance * 0.5, 0.0), 1.0)
    )
    score = round(min(max(score, 0.0), 100.0), 1)

    entry_allowed = True
    block_reason: str | None = None
    if micro.ask_wall_blocked:
        entry_allowed = False
        block_reason = micro.reason
    elif not freqai.do_predict:
        entry_allowed = False
        block_reason = freqai.reason
    elif strict_quant_gate and lead is not None and lead.up_prob < 0.40 and freqai.prob_up < 0.45:
        entry_allowed = False
        block_reason = f"Kantitatif filtre beklemeye aldı (Kronos up={lead.up_prob:.0%}, FreqAI={freqai.prob_up:.0%})."

    sizing = portfolio.sizing_multiplier
    if lead is not None and lead.uncertainty_pct > 6.5:
        sizing = round(max(0.45, sizing * 0.78), 3)

    return QuantBundleAssessment(
        symbol=symbol,
        interval=interval,
        kronos=kronos,
        microstructure=micro,
        freqai=freqai,
        portfolio=portfolio,
        quant_score=score,
        sizing_multiplier=sizing,
        entry_allowed=entry_allowed,
        block_reason=block_reason,
    )


# ---------------------------------------------------------------------------
# 6. Freqtrade Tarzı Güvenli Telegram İki Yönlü Komut Yönlendirici
# ---------------------------------------------------------------------------

def process_telegram_command(
    *,
    chat_id: str | int,
    expected_chat_id: str,
    text: str,
    dashboard_provider: Any,
    pause_setter: Any,
) -> str | None:
    """
    Telegram webhook komutlarını katı `chat_id` eşleşmesiyle doğrular ve yürütür.
    Yetkisiz sohbetlerden gelen isteklerde `None` döndürür (yanıt bile sızdırmaz).
    """
    expected = str(expected_chat_id or "").strip()
    incoming = str(chat_id or "").strip()
    if not expected or incoming != expected:
        return None

    cmd = str(text or "").strip().split()[0].lower() if str(text or "").strip() else ""
    cmd = cmd.split("@")[0]

    if cmd in ("/durum", "/status"):
        snap = dashboard_provider() or {}
        acct = snap.get("account") if isinstance(snap.get("account"), dict) else snap
        eq = float(acct.get("total_equity_usdc", acct.get("equity", 0)) or 0)
        cash = float(acct.get("cash_balance_usdc", acct.get("cash", 0)) or 0)
        pnl = float(acct.get("realized_pnl_usdc", acct.get("realized_pnl", 0)) or 0)
        q = snap.get("quote_asset", "USDT")
        pos_list = snap.get("open_positions", []) if isinstance(snap, dict) else []
        mode_lbl = snap.get("mode") or snap.get("execution_mode") or "PAPER"
        return (
            f"TradeCebrail Durum ({mode_lbl})\n"
            f"Toplam Değer: {eq:.2f} {q} | Serbest Nakit: {cash:.2f} {q}\n"
            f"Gerçekleşen K/Z: {pnl:+.2f} {q} | Açık Pozisyon: {len(pos_list)}"
        )

    if cmd in ("/kar", "/profit"):
        snap = dashboard_provider() or {}
        acct = snap.get("account") if isinstance(snap.get("account"), dict) else snap
        perf = snap.get("performance") or {}
        q = snap.get("quote_asset", "USDT")
        real_pnl = float(acct.get("realized_pnl_usdc", acct.get("realized_pnl", 0)) or 0)
        unreal_pnl = float(acct.get("unrealized_pnl_usdc", acct.get("unrealized_pnl", 0)) or 0)
        closed_cnt = int(perf.get("closed_count", perf.get("closed_trades", 0)) or 0)
        win_pct = float(perf.get("win_rate_pct", 0) or 0)
        return (
            f"Performans Özeti\n"
            f"Gerçekleşen K/Z: {real_pnl:+.2f} {q}\n"
            f"Açık K/Z: {unreal_pnl:+.2f} {q}\n"
            f"Kapanan İşlem: {closed_cnt} | "
            f"Başarı: %{win_pct:.1f}"
        )

    if cmd == "/radar":
        snap = dashboard_provider() or {}
        radar = (snap.get("universe") or snap.get("market_radar") or snap.get("gainers") or [])[:5]
        if not radar:
            return "Radar verisi henüz hazır değil."
        lines = ["Günün Fırsat Radarı (İlk 5):"]
        for item in radar:
            chg = float(item.get("change_pct_24h", item.get("change_24h_pct", 0)) or 0)
            sc = float(item.get("opportunity_score", item.get("score", 0)) or 0)
            lines.append(f"• {item.get('symbol')} | %{chg:+.2f} | Skor {sc:.0f}")
        return "\n".join(lines)

    if cmd in ("/durdur", "/stop"):
        pause_setter(True, "Telegram /durdur komutuyla yeni alımlar durduruldu.")
        return "Yeni alımlar durduruldu (açık pozisyon koruması aktif kalır)."

    if cmd in ("/baslat", "/start"):
        pause_setter(False, "")
        return "Yeni alımlar tekrar etkinleştirildi."

    return (
        "Kullanılabilir komutlar:\n"
        "/durum - Güncel bakiye ve açık pozisyon özeti\n"
        "/kar - Gerçekleşen K/Z ve başarı oranı\n"
        "/radar - En yüksek skorlu 5 fırsat\n"
        "/durdur - Yeni alımları duraklat\n"
        "/baslat - Alımları devam ettir"
    )


# ---------------------------------------------------------------------------
# 7. QuantDinger Tarzı Doğal Dil Araştırma & Backtest Motoru
# ---------------------------------------------------------------------------

_INTERVAL_ALIASES = {
    "1m": "1m", "1dk": "1m",
    "3m": "3m", "3dk": "3m",
    "5m": "5m", "5dk": "5m",
    "15m": "15m", "15dk": "15m",
    "30m": "30m", "30dk": "30m",
    "1h": "1h", "1s": "1h", "1sa": "1h",
    "4h": "4h", "4s": "4h", "4sa": "4h",
}


def build_research_plan(question: str, *, default_symbol: str = "SOLUSDT", default_interval: str = "5m") -> dict[str, Any]:
    """
    QuantDinger `ai_market_query` yaklaşımı:
    Türkçe veya İngilizce serbest araştırma sorusundan sembol, zaman dilimi, mum sayısı
    ve strateji profili çıkarır (offline/deterministik; LLM kapalıyken de %100 çalışır).
    """
    text = str(question or "").strip()
    upper = text.upper()

    # Sembol yakala (ör. SOLUSDT, BTCUSDC veya tek base "INJ", "SOL", "ETH")
    pair_matches = re.findall(r"\b([A-Z0-9]{2,10}(?:USDT|USDC))\b", upper)
    symbols: list[str] = []
    for m in pair_matches:
        if m not in symbols:
            symbols.append(m)
    if not symbols:
        token_alias = {
            "BITCOIN": "BTCUSDT", "BTC": "BTCUSDT",
            "ETHEREUM": "ETHUSDT", "ETH": "ETHUSDT",
            "SOLANA": "SOLUSDT", "SOL": "SOLUSDT",
            "INJ": "INJUSDT", "BNB": "BNBUSDT",
            "XRP": "XRPUSDT", "DOGE": "DOGEUSDT",
            "AVAX": "AVAXUSDT", "SUI": "SUIUSDT",
            "ADA": "ADAUSDT", "LINK": "LINKUSDT",
            "PEPE": "PEPEUSDT", "DODO": "DODOUSDT",
        }
        for key, pair in token_alias.items():
            if re.search(rf"\b{key}\b", upper) and pair not in symbols:
                symbols.append(pair)
    if not symbols:
        symbols = [default_symbol.upper()]

    # Zaman dilimi
    interval = default_interval
    m_tf = re.search(r"\b(1m|3m|5m|15m|30m|1h|4h|1dk|3dk|5dk|15dk|30dk|1s|4s|1sa|4sa)\b", text.lower())
    if m_tf:
        interval = _INTERVAL_ALIASES.get(m_tf.group(1), default_interval)

    # Mum sayısı
    lookback = 360
    m_bars = re.search(r"(\d{2,4})\s*(?:mum|bar|candle|kline)", text.lower())
    if m_bars:
        lookback = max(140, min(1000, int(m_bars.group(1))))

    # Strateji profili seçimi
    lower = text.lower()
    if any(k in lower for k in ("dip", "tepki", "rebound", "oversold", "düşüş")):
        strategy_mode = "RSI_DIP"
        params = StrategyParams(
            ema_fast_span=7, ema_slow_span=18, rsi_buy_min=26.0, rsi_buy_max=56.0,
            rsi_overbought_exit=72.0, min_volume_ratio=0.75, require_close_above_fast=False,
            require_rising_momentum=True, fast_exit=False, dip_rebound_buy=True,
        )
    elif any(k in lower for k in ("momentum", "breakout", "kırılım", "hızlı", "trend")):
        strategy_mode = "MOMENTUM"
        params = StrategyParams(
            ema_fast_span=9, ema_slow_span=21, rsi_buy_min=48.0, rsi_buy_max=72.0,
            rsi_overbought_exit=78.0, min_volume_ratio=1.0, require_close_above_fast=True,
            require_rising_momentum=True, fast_exit=True, dip_rebound_buy=False,
        )
    elif any(k in lower for k in ("koruma", "güvenli", "low dd", "muhafazakar", "temkin")):
        strategy_mode = "LOW_DD"
        params = StrategyParams(
            ema_fast_span=12, ema_slow_span=26, rsi_buy_min=42.0, rsi_buy_max=60.0,
            rsi_overbought_exit=68.0, min_volume_ratio=1.05, require_close_above_fast=True,
            require_rising_momentum=True, fast_exit=False, dip_rebound_buy=False,
        )
    else:
        strategy_mode = "BALANCED"
        params = StrategyParams()

    return {
        "question": text,
        "symbols": symbols[:3],
        "interval": interval,
        "lookback": lookback,
        "strategy_mode": strategy_mode,
        "params": params,
    }


def _compute_curve_and_stats(metrics: SliceBacktestMetrics) -> dict[str, Any]:
    """Derive every chart/statistic from the authoritative simulation's fills.

    Sharpe/Sortino use zero cash return and the observed sample length, rather
    than claiming annualized ratios from a short historical window.
    """
    curve = [
        {"t": point.time_ms, "strategy": round(float(point.strategy_equity_usdc), 4),
         "benchmark": round(float(point.benchmark_equity_usdc), 4)}
        for point in metrics.equity_curve
    ]
    equities = [float(metrics.initial_equity_usdc)]
    # The trace begins with the initial balance before the first executable bar.
    equities.extend(float(point.strategy_equity_usdc) for point in metrics.equity_curve[1:])
    bar_returns = [(current / previous) - 1.0 for previous, current in zip(equities, equities[1:])
                   if previous > 0]
    rets_arr = np.asarray(bar_returns, dtype=np.float64)
    sample_factor = math.sqrt(max(len(rets_arr), 1))
    mean_r = float(np.mean(rets_arr)) if len(rets_arr) else 0.0
    std_r = float(np.std(rets_arr, ddof=1)) if len(rets_arr) > 1 else 0.0
    down_std = float(np.sqrt(np.mean(np.minimum(rets_arr, 0.0) ** 2))) if len(rets_arr) else 0.0
    sharpe = round((mean_r / std_r) * sample_factor, 2) if std_r > 1e-9 else 0.0
    sortino = round((mean_r / down_std) * sample_factor, 2) if down_std > 1e-9 else 0.0

    gains = sum((p for p in metrics.trade_pnls_usdc if p > 0), Decimal("0"))
    losses = -sum((p for p in metrics.trade_pnls_usdc if p < 0), Decimal("0"))
    # No losing trades means the ratio is undefined; do not invent a capped 99.
    profit_factor = round(float(gains / losses), 2) if losses > 0 else None

    return {
        "curve": curve,
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": profit_factor,
    }


def run_quant_research(
    *,
    question: str,
    closed_candles_by_symbol: dict[str, list[Candle]],
    order_books_by_symbol: dict[str, OrderBookSnapshot],
    filters_by_symbol: dict[str, SymbolFilters],
    settings: RiskSettings,
    initial_balance: Decimal = Decimal("86.00"),
) -> dict[str, Any]:
    """
    Doğal dil araştırma sorusuna tam kantitatif backtest + Kronos kestirimi + skfolio/FreqAI raporu üretir.
    """
    if settings.leverage != 1:
        raise ValueError("Bu araştırma testi yalnızca kaldıraçsız Spot işlemleri doğrular; marjin testi desteklenmez.")
    plan = build_research_plan(question)
    primary = plan["symbols"][0]
    candles = closed_candles_by_symbol.get(primary) or []
    if len(candles) < 60:
        return {
            "ok": False,
            "error": f"{primary} için araştırma testine yetecek en az 60 kapanmış mum alınamadı.",
            "plan": {k: v for k, v in plan.items() if k != "params"},
        }

    flt = filters_by_symbol.get(primary)
    eval_start = int(len(candles) * 0.25)
    bt: SliceBacktestMetrics = run_no_lookahead_slice_simulation(
        candles,
        eval_start_idx=eval_start,
        eval_end_idx=len(candles),
        slice_name=f"HISTORICAL_SPOT_{plan['strategy_mode']}",
        symbol=primary,
        interval=plan["interval"],
        params=plan["params"],
        settings=settings,
        initial_balance_usdc=initial_balance,
        symbol_filters=flt,
    )
    curve_stats = _compute_curve_and_stats(bt)

    ret_map: dict[str, np.ndarray] = {}
    for sym, c_list in closed_candles_by_symbol.items():
        if len(c_list) >= 20:
            cls = np.array([float(c.close) for c in c_list[-120:]], dtype=np.float64)
            ret_map[sym] = np.diff(np.log(np.maximum(cls, 1e-12)))

    ob = order_books_by_symbol.get(primary) or OrderBookSnapshot(
        symbol=primary, bids=[], asks=[], fetched_at_utc=datetime.now(timezone.utc), source_url="unavailable"
    )
    quant = evaluate_quant_bundle(
        closed_candles=candles,
        order_book=ob,
        symbol=primary,
        interval=plan["interval"],
        strategy_params=plan["params"],
        returns_by_symbol=ret_map,
    )

    narrative_tr = (
        f"{primary} ({plan['interval']}, {len(candles)} mum, {plan['strategy_mode']}): "
        f"Geçmiş Spot getirisi %{float(bt.net_return_pct):+.2f} (Referans Al-Tut %{float(bt.benchmark_return_pct):+.2f}), "
        f"{bt.trade_count} işlem (%{float(bt.win_rate_pct):.1f} başarı), "
        f"Sharpe {curve_stats['sharpe']:.2f}, MaxDD %{float(bt.max_drawdown_pct):.2f}. "
        f"{quant.kronos.summary_tr}. {quant.freqai.reason} {bt.verdict_text}"
    )

    return {
        "ok": True,
        "symbol": primary,
        "interval": plan["interval"],
        "strategy_mode": plan["strategy_mode"],
        "candles_tested": len(candles),
        "simulation_scope": "SINGLE_SYMBOL_SPOT_OHLC",
        "insufficient_data": bt.insufficient_data,
        "verdict_text": bt.verdict_text,
        "ratio_basis": "SAMPLE_PERIOD_ZERO_CASH_RETURN",
        "limitations": [
            "Sabit kurallı geçmiş Spot testi; kronolojik eğitim/doğrulama ile seçilmiş örneklem dışı başarı kanıtı değildir.",
            "Çoklu zaman dilimi/quant giriş filtreleri, ATR boyutlandırması, kâr bekleme ve iz süren stop modellenmez.",
            "Kronos/FreqAI skorları güncel betimleyici ölçümlerdir; geçmiş işlem eğrisinin doğrulanmış başarı olasılığı değildir.",
        ],
        "metrics": {
            "net_return_pct": round(float(bt.net_return_pct), 2),
            "benchmark_return_pct": round(float(bt.benchmark_return_pct), 2),
            "max_drawdown_pct": round(float(bt.max_drawdown_pct), 2),
            "win_rate_pct": round(float(bt.win_rate_pct), 1),
            "trade_count": int(bt.trade_count),
            "total_fees_usdc": round(float(bt.total_fees_usdc), 4),
            "sharpe": curve_stats["sharpe"],
            "sortino": curve_stats["sortino"],
            "profit_factor": curve_stats["profit_factor"],
        },
        "quant": quant.to_summary_dict(),
        "equity_curve": curve_stats["curve"],
        "narrative_tr": narrative_tr,
    }
