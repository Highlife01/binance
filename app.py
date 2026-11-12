"""
Binance Global Spot & Çapraz Marjin (USDT / USDC) — Türkçe Yerel Web Kontrol Paneli (Streamlit, 127.0.0.1).

Çalıştırma:
    baslat.bat
veya:
    .venv\\Scripts\\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

# Yerel paket yolunu ekle
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ENGINE_STATE_VERSION = "1.9.0"


def _ensure_fresh_backend_modules() -> None:
    """Açık çalışan Streamlit sürecinde src/binance_ai_bot alt modülleri güncellendiğinde belleği tazeler."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    import binance_ai_bot

    if getattr(binance_ai_bot, "_reloaded_version", None) == ENGINE_STATE_VERSION:
        return
    submodules = [
        "binance_ai_bot.domain",
        "binance_ai_bot.security",
        "binance_ai_bot.config",
        "binance_ai_bot.execution_safety",
        "binance_ai_bot.storage",
        "binance_ai_bot.market_data",
        "binance_ai_bot.strategy",
        "binance_ai_bot.profiles",
        "binance_ai_bot.risk",
        "binance_ai_bot.simulator",
        "binance_ai_bot.live_executor",
        "binance_ai_bot.margin_executor",
        "binance_ai_bot.ai_advisor",
        "binance_ai_bot.validation",
        "binance_ai_bot.runner",
        "binance_ai_bot.engine",
    ]
    for mod_name in submodules:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
    importlib.reload(binance_ai_bot)
    setattr(binance_ai_bot, "_reloaded_version", ENGINE_STATE_VERSION)


_ensure_fresh_backend_modules()

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from binance_ai_bot import SIMULATION_MODE_LABEL
from binance_ai_bot.config import AppConfig, load_config
from binance_ai_bot.domain import Candle, RiskSettings
from binance_ai_bot.engine import TradingSystemEngine
from binance_ai_bot.market_data import INTERVAL_TO_MS, MarketDataError
from binance_ai_bot.profiles import TRADING_PROFILES, get_profile
from binance_ai_bot.risk import (
    EXIT_POLICY_DESCRIPTIONS_TR,
    RISK_DISCLAIMER_TR,
    compute_open_position_unrealized_net_pnl,
)
from binance_ai_bot.runner import RUNNER_STOP_KEY, is_runner_alive
from binance_ai_bot.security import assert_localhost_only
from binance_ai_bot.simulator import FILL_MODEL_EXPLANATION_TR
from binance_ai_bot.validation import (
    assess_small_balance_viability,
    run_chronological_validation,
)

st.set_page_config(
    page_title="Binance Spot & Marjin Kontrol Paneli",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _generate_offline_demo_candles(count: int = 180, *, interval: str = "15m") -> list[Candle]:
    """
    Yalnızca 'Doğrulama / Backtest' sekmesinde ağ engeli varken kronolojik test motorunu
    incelemek için kullanılan deterministik örnek seri. Sanal veya canlı portföye DOKUNMAZ.
    """
    import math

    candles: list[Candle] = []
    base_ts = 1_760_000_000_000
    step_ms = INTERVAL_TO_MS[interval]
    price = 92000.0
    for i in range(count):
        wave = math.sin(i / 7.5) * 420.0 + math.cos(i / 17.0) * 280.0
        drift = i * 8.5
        open_p = price
        close_p = max(1000.0, 92000.0 + wave + drift)
        high_p = max(open_p, close_p) + 95.0
        low_p = min(open_p, close_p) - 95.0
        vol = 18.0 + (i % 9) * 3.2
        candles.append(
            Candle(
                open_time_ms=base_ts + i * step_ms,
                open=Decimal(f"{open_p:.2f}"),
                high=Decimal(f"{high_p:.2f}"),
                low=Decimal(f"{low_p:.2f}"),
                close=Decimal(f"{close_p:.2f}"),
                volume=Decimal(f"{vol:.4f}"),
                close_time_ms=base_ts + (i + 1) * step_ms - 1,
                quote_volume=Decimal(f"{(vol * close_p):.2f}"),
                trade_count=450 + i,
                is_closed=True,
            )
        )
        price = close_p
    return candles


def _fetch_live_bids(
    engine: TradingSystemEngine, symbols: list[str], *, refresh: bool = False
) -> dict[str, Decimal]:
    """Normal çizimde önbelleği kullanır; yalnızca açık yenileme isteğinde ağa gider."""
    cache = st.session_state.setdefault("live_bid_cache", {})
    bids: dict[str, Decimal] = {}
    for sym in dict.fromkeys(symbols):
        cached = cache.get(sym)
        if cached and time.monotonic() - cached[0] <= 8:
            bids[sym] = cached[1]
            continue
        if not refresh:
            continue
        try:
            book = engine.market_client.fetch_order_book(sym, depth_limit=5)
            if book.best_bid is not None:
                bids[sym] = book.best_bid
                cache[sym] = (time.monotonic(), book.best_bid)
        except Exception:
            continue
    return bids


def _render_live_pnl(
    engine: TradingSystemEngine,
    settings: RiskSettings,
    track: str,
    quote: str,
    tz,
    *,
    refresh: bool = False,
) -> None:
    """Anlık Kâr/Zarar bölümü (st.fragment ile yalnızca bu kısım periyodik yenilenir)."""
    storage = engine.storage
    open_positions = storage.get_open_positions(track=track)
    track_label = {
        "LIVE_SPOT": "🔴 Gerçek Canlı Spot (1x)",
        "LIVE_MARGIN": "⚡ Gerçek Canlı Çapraz Marjin",
        "MAIN": "🟢 Sanal Deneme",
    }.get(track, track)
    st.subheader(f"📈 Anlık Kâr / Zarar — {track_label}")

    bids = _fetch_live_bids(engine, [p.symbol for p in open_positions], refresh=refresh)
    summ = engine.get_portfolio_summary(settings, live_bid_by_symbol=bids, track=track)
    valuation_complete = summ["valuation_complete"]
    total_pnl = summ["realized_pnl_usdc"] + summ["unrealized_pnl_usdc"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Açık Pozisyon Anlık K/Z",
        f"{summ['unrealized_pnl_usdc']:+.4f} {quote}" if valuation_complete else "—",
        help="Anlık alış fiyatı ile; giriş komisyonu, tahmini çıkış komisyonu ve varsa marjin faizi düşülmüş net değer.",
    )
    c2.metric(
        "Bugünkü Net K/Z (İstanbul)",
        f"{summ['daily_net_pnl_usdc']:+.4f} {quote}" if valuation_complete else "—",
    )
    c3.metric(
        "Toplam Net K/Z (Gerçekleşen + Açık)",
        f"{total_pnl:+.4f} {quote}" if valuation_complete else "—",
    )
    c4.metric(
        "Anlık Portföy Değeri",
        f"{summ['total_equity_usdc']:.2f} {quote}" if valuation_complete else "—",
    )

    if open_positions:
        rows = []
        hourly_rate = getattr(settings, "hourly_margin_interest_rate", Decimal("0.000005"))
        for p in open_positions:
            bid = bids.get(p.symbol)
            p_lev = getattr(p, "leverage", 1)
            p_borrowed = getattr(p, "borrowed_usdc", Decimal("0"))
            p_liq = getattr(p, "liquidation_price", None)
            if bid is None:
                rows.append(
                    {
                        "Parite": p.symbol,
                        "Kaldıraç": f"{p_lev}x",
                        "Durum": "Anlık fiyat alınamadı",
                    }
                )
                continue
            net_eq_val, pnl = compute_open_position_unrealized_net_pnl(
                p,
                current_bid=bid,
                estimated_fee_rate=settings.estimated_fee_rate,
                hourly_interest_rate=hourly_rate,
            )
            collateral_cost = (p.cost_basis_usdc - p_borrowed) + p.entry_fee_usdc
            pnl_pct = (
                (pnl / collateral_cost * Decimal("100"))
                if collateral_cost > 0
                else Decimal("0")
            )
            move_pct = (
                (bid - p.entry_price) / p.entry_price * Decimal("100")
                if p.entry_price > 0
                else Decimal("0")
            )
            sl_dist = (
                f"%{(bid - p.stop_loss_price) / bid * Decimal('100'):.2f}"
                if p.stop_loss_price
                else "-"
            )
            tp_dist = (
                f"%{(p.take_profit_price - bid) / bid * Decimal('100'):.2f}"
                if p.take_profit_price
                else "-"
            )
            rows.append(
                {
                    "Parite": p.symbol,
                    "Kaldıraç": f"{p_lev}x",
                    "Durum": "🟢 Kârda" if pnl > 0 else "🔴 Zararda",
                    "Miktar": f"{p.quantity:.8f}",
                    "Giriş Fiyatı": f"{p.entry_price:.6g}",
                    "Anlık Fiyat": f"{bid:.6g}",
                    "Fiyat Değişimi": f"%{move_pct:+.2f}",
                    f"Öz Değer ({quote})": f"{net_eq_val:.4f}",
                    f"Borç ({quote})": f"{p_borrowed:.2f}" if p_borrowed > 0 else "-",
                    "Likidasyon": f"{p_liq:.6g}" if p_liq else "-",
                    f"Net K/Z ({quote})": f"{pnl:+.4f}",
                    "Teminata Göre (%)": f"%{pnl_pct:+.2f}",
                    "Zarar-Kes'e Uzaklık": sl_dist,
                    "Kâr-Al'a Uzaklık": tp_dist,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("Şu anda açık pozisyon yok; yeni alım sinyali bekleniyor.")
    if any(p.symbol not in bids for p in open_positions):
        st.warning(
            "Bazı açık pozisyonların güncel fiyatı yok. Güncel fiyatlar alınamadı; "
            "toplam değer ve açık kâr/zarar gösterilmiyor. Anlık Fiyatları Şimdi Güncelle düğmesini kullanın."
        )

    st.caption(
        f"Son güncelleme (İstanbul): {datetime.now(timezone.utc).astimezone(tz).strftime('%H:%M:%S')} "
        "— Net K/Z, komisyonlar ve varsa saatlik marjin faizi düşülerek hesaplanır."
    )


@st.fragment(run_every=10)
def _render_live_pnl_fragment(
    engine: TradingSystemEngine,
    settings: RiskSettings,
    track: str,
    quote: str,
    tz,
) -> None:
    _render_live_pnl(engine, settings, track, quote, tz, refresh=True)


def _get_engine() -> tuple[AppConfig, TradingSystemEngine]:
    cfg = load_config()
    assert_localhost_only(cfg.localhost_host)
    cfg_fingerprint = (
        ENGINE_STATE_VERSION,
        cfg.binance_public_base_url,
        cfg.binance_live_base_url,
        cfg.enable_live_trading,
        cfg.has_binance_live_keys,
        cfg.has_gemini_key,
        cfg.gemini_model,
        cfg.quote_asset,
    )
    cur_eng = st.session_state.get("engine")
    if (
        cur_eng is None
        or st.session_state.get("engine_fingerprint") != cfg_fingerprint
        or not isinstance(cur_eng, TradingSystemEngine)
        or not hasattr(cur_eng, "live_margin_executor")
    ):
        st.session_state["engine"] = TradingSystemEngine(cfg)
        st.session_state["engine_fingerprint"] = cfg_fingerprint
    return cfg, st.session_state["engine"]


def main() -> None:
    cfg, engine = _get_engine()
    storage = engine.storage
    settings = engine.default_risk_settings()
    ist_today = storage.istanbul_date_str()

    # --- Yan Menü: Mod Seçimi, Kaldıraç, Parite ve Kontroller ---
    with st.sidebar:
        st.header("⚙️ Çalışma Ayarları")

        quote_options = ["USDT", "USDC"]
        default_quote_idx = quote_options.index(cfg.quote_asset) if cfg.quote_asset in quote_options else 0
        selected_quote = st.radio(
            "Baz Para Birimi (Spot / Marjin)",
            options=quote_options,
            index=default_quote_idx,
            horizontal=True,
            help="Cüzdanınızda bulundurduğunuz ana bakiye para birimi (USDT veya USDC).",
        )

        # --- Kesintisiz Arka Plan Botu Durumu ---
        runner_alive, runner_info = is_runner_alive(storage)
        if runner_alive:
            r_raw = runner_info.get("mode", "PAPER")
            r_lev = runner_info.get("leverage", 1)
            r_mode = {
                "LIVE_MARGIN": f"⚡ CANLI ÇAPRAZ MARJİN ({r_lev}x)",
                "LIVE_SPOT": "🔴 CANLI SPOT (1x)",
            }.get(r_raw, f"🟢 SANAL ({r_lev}x)")
            st.success(
                f"🤖 **Arka Plan Botu Çalışıyor** — {r_mode} | "
                f"Profil: `{runner_info.get('profile', '?')}` | "
                f"Parite: `{runner_info.get('symbol', '?')} ({runner_info.get('interval', '?')})`\n\n"
                "Çift emir olmaması için paneldeki otomatik döngü ve analiz düğmesi kilitlendi."
            )
            if st.button("⏹️ Arka Plan Botunu Durdur", width="stretch"):
                storage.set_state(RUNNER_STOP_KEY, "1")
                st.info("Durdurma isteği gönderildi; bot en geç bir döngü içinde kapanır.")
        else:
            st.caption(
                "🤖 Arka plan botu kapalı. Kesintisiz alım-satım için proje klasöründeki "
                "`bot_sanal.bat` (2x sanal), `bot_canli.bat` (1x spot) veya `bot_marjin.bat` "
                "(2x gerçek çapraz marjin) dosyasını çalıştırın."
            )

        profile_keys = list(TRADING_PROFILES.keys())
        default_prof_key = (
            "HIZLI_30SN"
            if ("PYTEST_CURRENT_TEST" in os.environ or "AL_YUKSELINCE_SAT" not in profile_keys)
            else "AL_YUKSELINCE_SAT"
        )
        selected_profile_key = st.selectbox(
            "İşlem Profili",
            options=profile_keys,
            index=profile_keys.index(default_prof_key),
            format_func=lambda k: TRADING_PROFILES[k].label_tr,
        )
        active_profile = get_profile(selected_profile_key)
        st.caption(active_profile.description_tr)

        mode_choice = st.radio(
            "İşlem Yürütme Modu",
            options=["PAPER", "LIVE_SPOT", "LIVE_MARGIN"],
            format_func=lambda m: {
                "PAPER": f"🟢 Sanal Deneme ({cfg.initial_balance_usdc:.0f} {selected_quote} — 1x–5x Kaldıraç)",
                "LIVE_SPOT": "🔴 Gerçek Para — Canlı Spot (1x Kaldıraçsız)",
                "LIVE_MARGIN": "⚡ Gerçek Para — Canlı Çapraz Marjin (2x–5x Kaldıraçlı)",
            }[m],
            index=0,
            help=(
                "Sanal Deneme gerçek bakiyenize dokunmaz. Canlı Spot borçsuz 1x çalışır. "
                "Canlı Çapraz Marjin yalnızca Marjin cüzdanına aktardığınız teminatla 2x–5x kaldıraç kullanır."
            ),
        )

        # --- Kaldıraç Seçici (Varsayılan 2x: Sanal ve Çapraz Marjin'de aktif, Canlı Spot'ta 1x kilitli) ---
        leverage_choices = [1, 2, 3, 5]
        if mode_choice == "LIVE_SPOT":
            selected_leverage = 1
            st.caption("🔒 **Kaldıraç:** `1x` (Canlı Spot modunda borçlanma kapalıdır; kaldıraç için Sanal veya Çapraz Marjin seçin).")
        else:
            saved_lev = int(getattr(settings, "leverage", 2))
            default_lev = saved_lev if saved_lev in leverage_choices and saved_lev > 1 else 2
            selected_leverage = st.radio(
                "⚡ Kaldıraç Çarpanı",
                options=leverage_choices,
                index=leverage_choices.index(default_lev),
                horizontal=True,
                format_func=lambda x: f"{x}x" + (" (Önerilen)" if x == 2 else ""),
                help=(
                    "Örn. 2x kaldıraçta 10 USDT kendi teminatınızla 20 USDT büyüklüğünde işlem açılır "
                    "(kalan 10 USDT borçlanır ve satışta AUTO_REPAY ile otomatik kapatılır)."
                ),
            )

        ui_live_confirmed = False
        ui_margin_confirmed = False
        if mode_choice == "LIVE_SPOT":
            st.warning(
                f"Gerçek Spot modunda işlemler gerçek Binance Global {selected_quote} bakiyenizle (1x) yapılır. "
                "Kâr garantisi yoktur ve bilgisayar uykuya geçtiğinde bot durur."
            )
            if not cfg.enable_live_trading or not cfg.has_binance_live_keys:
                st.error(
                    "🔒 Canlı Mod Kilitli: `.env` dosyasında `ENABLE_LIVE_TRADING=true`, "
                    "`BINANCE_API_KEY` ve `BINANCE_API_SECRET` tanımlı değil. "
                    "Anahtarlarınızı sohbet ekranına yazmayın; yalnızca `.env` dosyasına girin."
                )
            else:
                ack_risk = st.checkbox(
                    "Kâr garantisi olmadığını, olası kayıp riskini ve yalnızca Spot işlem "
                    "yapılacağını okudum, onaylıyorum.",
                    value=False,
                )
                unlock_switch = st.toggle("Canlı Spot Emir Kilidini Aç", value=False)
                ui_live_confirmed = bool(ack_risk and unlock_switch)
                if ui_live_confirmed and storage.get_portfolio_quote("LIVE_SPOT") in {None, selected_quote}:
                    sync_now = st.button("🔄 Gerçek Binance Bakiyesini Sorgula", width="stretch")
                    last_synced_quote = st.session_state.get("live_balance_quote")
                    if sync_now or not st.session_state.get("live_balance_synced") or last_synced_quote != selected_quote:
                        try:
                            bals = engine.live_executor.fetch_live_spot_balances(
                                ui_live_confirmed=True,
                                quote_asset=selected_quote,
                            )
                            st.session_state["live_balance_synced"] = True
                            st.session_state["live_balance_quote"] = selected_quote
                            quote_now = bals.get(selected_quote, Decimal("0"))
                            earn_now = bals.get(f"LD{selected_quote}", Decimal("0"))
                            spot_perm_ok = storage.get_state("live_spot_permission_ok", "0") == "1"
                            st.success(
                                f"Gerçek hesap okundu. Spot Serbest {selected_quote}: {quote_now:.2f} {selected_quote}"
                            )
                            if not spot_perm_ok:
                                st.warning(
                                    "⚠️ API anahtarınızda **'Enable Spot & Margin & Stock Trading'** kutusu henüz "
                                    "işaretli değil (yalnızca Okuma açık). Emir gönderebilmek için Binance API "
                                    "ekranında önce IP adresinizi girip bu kutuyu işaretlemeniz gerekir."
                                )
                            if quote_now < Decimal("5.0") and earn_now > Decimal("0"):
                                st.warning(
                                    f"💡 Spot cüzdanınızda serbest {selected_quote} `{quote_now:.2f}` görünüyor; ancak "
                                    f"**Binance Simple Earn (Esnek Birikim)** hesabınızda **`{earn_now:.2f} LD{selected_quote}`** bulundu! "
                                    "Binance API'si Earn içindeki parayı doğrudan kullanamaz. Uygulamadan "
                                    f"**Earn (Kazan) -> {selected_quote} -> Redeem (Parayı Çek / Geri Al -> Spot Cüzdanı)** "
                                    "yapıp **Auto-Subscribe (Otomatik Katılım)** özelliğini kapatın."
                                )
                        except Exception as exc:
                            st.error(f"Canlı hesap okuma uyarısı: {exc}")

        elif mode_choice == "LIVE_MARGIN":
            st.warning(
                f"⚡ **Gerçek Çapraz Marjin ({selected_leverage}x Kaldıraç):** Yalnızca Çapraz Marjin (Cross Margin) "
                f"cüzdanınıza aktardığınız {selected_quote} teminatı kullanılır; Spot cüzdanınıza otomatik dokunulmaz. "
                "Kaldıraç hem kârı hem zararı büyütür ve ters harekette likidasyon riski taşır."
            )
            if not cfg.enable_live_trading or not cfg.has_binance_live_keys:
                st.error(
                    "🔒 Canlı Marjin Kilitli: `.env` dosyasında `ENABLE_LIVE_TRADING=true`, "
                    "`BINANCE_API_KEY` ve `BINANCE_API_SECRET` tanımlı olmalıdır."
                )
            else:
                ack_margin_risk = st.checkbox(
                    f"Kaldıraçlı Çapraz Marjin ({selected_leverage}x) işlemde borçlanma, saatlik faiz ve "
                    "likidasyon riskini okudum, onaylıyorum.",
                    value=False,
                )
                unlock_margin = st.toggle("Canlı Çapraz Marjin Kilidini Aç", value=False)
                if ack_margin_risk and unlock_margin:
                    ui_live_confirmed = True
                    ui_margin_confirmed = True
                if ui_margin_confirmed and storage.get_portfolio_quote("LIVE_MARGIN") in {None, selected_quote}:
                    sync_margin = st.button("🔄 Gerçek Marjin Cüzdanını Sorgula", width="stretch")
                    last_m_quote = st.session_state.get("live_margin_quote")
                    if sync_margin or not st.session_state.get("live_margin_synced") or last_m_quote != selected_quote:
                        try:
                            m_info = engine.live_margin_executor.fetch_live_margin_balances(
                                ui_live_confirmed=True,
                                ui_margin_confirmed=True,
                                quote_asset=selected_quote,
                            )
                            st.session_state["live_margin_synced"] = True
                            st.session_state["live_margin_quote"] = selected_quote
                            st.success(
                                f"Çapraz Marjin Cüzdanı Aktif — Serbest Teminat: "
                                f"{m_info['available_collateral']:.2f} {selected_quote} | "
                                f"Borç: {m_info['quote_borrowed']:.2f} {selected_quote} | "
                                f"Marjin Seviyesi: {m_info['margin_level']:.2f}"
                            )
                            if m_info["available_collateral"] < Decimal("5.0"):
                                st.info(
                                    f"💡 Marjin cüzdanınızda henüz `{m_info['available_collateral']:.2f} {selected_quote}` var. "
                                    f"Binance uygulamasından **Cüzdanlar -> Transfer -> Spot'tan Cross Margin'e** "
                                    f"istediğiniz tutarda (ör. 30 {selected_quote}) aktarım yapabilirsiniz."
                                )
                        except Exception as exc:
                            st.error(f"Marjin cüzdanı uyarısı: {exc}")
                            st.info(
                                "📌 **Binance Çapraz Marjin Nasıl Açılır? (1 Dakika):**\n"
                                "1. Binance mobil uygulamasında **Cüzdanlar (Wallets) -> Marjin (Margin / Cross)** sekmesine girin.\n"
                                "2. Karşınıza çıkan kısa Marjin bilgilendirme testini onaylayarak cüzdanı etkinleştirin.\n"
                                f"3. **Transfer** düğmesiyle **Spot Cüzdanı -> Cross Margin** yönünde ayırmak istediğiniz "
                                f"teminatı (ör. `30 {selected_quote}`) aktarın; kalan paranız Spot'ta güvende kalır."
                            )

        st.divider()
        base_coins = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE", "LINK", "SUI"]
        symbol_options = [f"{coin}{selected_quote}" for coin in base_coins]
        if cfg.default_symbol.endswith(selected_quote) and cfg.default_symbol not in symbol_options:
            symbol_options.insert(0, cfg.default_symbol)
        selected_symbol = st.selectbox(
            f"Spot / Marjin {selected_quote} İşlem Çifti", options=symbol_options, index=0
        )
        scan_all_coins = st.toggle(
            f"🌐 Çoklu Coin Eşzamanlı Al-Sat ({len(symbol_options)} Coin)",
            value=(active_profile.max_open_positions > 1),
            help=(
                f"Açıkken tek bir coine takılı kalmaz; listedeki {len(symbol_options)} coini birlikte tarar "
                f"ve aynı anda en fazla {active_profile.max_open_positions} farklı coinde alım-satım yapar "
                "(aynı coine üst üste ekleme yapmaz)."
            ),
        )

        interval_options = ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]
        default_int_idx = (
            interval_options.index(active_profile.interval)
            if active_profile.interval in interval_options
            else 1
        )
        selected_interval = st.selectbox(
            "Kapanmış Mum Periyodu",
            options=interval_options,
            index=default_int_idx,
            help=f"Küçük bakiyede ({selected_quote}) komisyon erimesini önlemek için 15m veya 1h önerilir.",
        )

        st.divider()
        st.subheader("🤖 Yapay Zekâ (Gemini) Ayarları")
        if cfg.has_gemini_key:
            ai_enabled = st.toggle("Gemini Karar Modülü Aktif", value=(not active_profile.direct_rule_execution))
            st.caption(f"Model: `{cfg.gemini_model}` (Resmî `google-genai` SDK)")
        else:
            ai_enabled = False
            st.info(
                "Etiket: **Yapay Zekâ Kapalı / Kural Tabanlı Deneme**\n\n"
                "`.env` içinde `GEMINI_API_KEY` bulunmadığından sistem yalnızca matematiksel "
                "teknik göstergelerle (EMA/RSI/MACD) çalışır."
            )

        daily_ai_limit = st.number_input(
            "Günlük Maksimum Model Çağrısı",
            min_value=1,
            max_value=500,
            value=int(max(cfg.ai_daily_call_limit, active_profile.ai_daily_call_limit)),
            step=1,
        )
        auto_loop = st.toggle(
            "Otomatik Döngü (Panel açıkken kontrol et)",
            value=False,
            disabled=runner_alive,
            help=(
                "Yalnızca bu tarayıcı sekmesi açıkken çalışır. Sekmeden bağımsız kesintisiz çalışma için "
                "bot_sanal.bat / bot_canli.bat / bot_marjin.bat kullanın."
            ),
        )
        if runner_alive:
            auto_loop = False
        live_pnl_auto = st.toggle(
            "Anlık K/Z Canlı Yenile (Her 10 sn)",
            value=("PYTEST_CURRENT_TEST" not in os.environ),
            help="Açık pozisyonun anlık kâr/zararını her 10 saniyede bir canlı günceller.",
        )

    if mode_choice == "LIVE_MARGIN" and ui_live_confirmed and ui_margin_confirmed:
        effective_mode = "LIVE_MARGIN"
    elif mode_choice == "LIVE_SPOT" and ui_live_confirmed:
        effective_mode = "LIVE_SPOT"
    else:
        effective_mode = "PAPER"

    eff_leverage = 1 if effective_mode == "LIVE_SPOT" else selected_leverage
    eff_settings = active_profile.apply_to_risk_settings(settings, leverage=eff_leverage)

    active_track = (
        "LIVE_MARGIN"
        if effective_mode == "LIVE_MARGIN"
        else ("LIVE_SPOT" if effective_mode == "LIVE_SPOT" else "MAIN")
    )
    portfolio_quote = storage.get_portfolio_quote(active_track) or selected_quote
    quote_mismatch = portfolio_quote != selected_quote
    if quote_mismatch:
        auto_loop = False
        st.warning(
            f"Bu portföy {portfolio_quote} ile tutuluyor. Aşağıdaki bakiye ve geçmiş {portfolio_quote} "
            f"cinsindedir. İşlem yapmak için sol menüden {portfolio_quote} seçin; "
            "USDT ve USDC aynı kasada birleştirilmez."
        )
    if active_track == "LIVE_SPOT" and storage.get_state("live_unvalued_commissions", "") not in {"", "{}", "[]"}:
        st.warning(
            "Bazı komisyonların portföy para birimindeki değeri doğrulanamadı. "
            "Yeni alımlar bekletiliyor; gösterilen net kâr/zarar bu komisyonları henüz içermeyebilir."
        )
    if active_track == "LIVE_SPOT" and storage.get_state("live_balance_refresh_required", "0") == "1":
        st.warning(
            "Emir sonucu işlendi; gerçek nakit bakiye yeniden sorgulanmayı bekliyor. "
            "Bakiye ve portföy değeri doğrulanana kadar geçicidir."
        )

    # --- Üst Güvenlik Etiketi (Banner) ---
    if effective_mode == "PAPER":
        st.markdown(
            f"""
            <div style="background-color:#0F766E;color:#FFFFFF;padding:14px 20px;border-radius:10px;
                        font-size:18px;font-weight:700;margin-bottom:14px;border-left:8px solid #99F6E4;">
                🛡️ {SIMULATION_MODE_LABEL} ({eff_settings.leverage}x Kaldıraç) — Gerçek Binance Bakiyenize Dokunulmaz (Başlangıç Sanal Kasa: {cfg.initial_balance_usdc:.2f} {portfolio_quote})
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif effective_mode == "LIVE_MARGIN":
        st.markdown(
            f"""
            <div style="background-color:#7C2D12;color:#FFFFFF;padding:14px 20px;border-radius:10px;
                        font-size:18px;font-weight:700;margin-bottom:14px;border-left:8px solid #FDBA74;">
                ⚡ DİKKAT: CANLI ÇAPRAZ MARJİN ({eff_settings.leverage}x KALDIRAÇ) AKTİF — GERÇEK MARJİN {portfolio_quote} TEMİNATI VE BORÇLANMA KULLANILMAKTADIR
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#991B1B;color:#FFFFFF;padding:14px 20px;border-radius:10px;
                        font-size:18px;font-weight:700;margin-bottom:14px;border-left:8px solid #FCA5A5;">
                ⚠️ DİKKAT: CANLI SPOT İŞLEM MODU (1x) AKTİF — GERÇEK BINANCE GLOBAL {portfolio_quote} BAKİYESİ KULLANILMAKTADIR
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Gerçek Binance Cüzdan Özeti (Spot + Simple Earn / Kazan Tüm Varlıklar) ---
    if "PYTEST_CURRENT_TEST" not in os.environ and cfg.enable_live_trading and cfg.has_binance_live_keys:
        with st.expander("💼 Gerçek Binance Cüzdanım — Spot + Simple Earn (Kazan) Tüm Varlıklarım", expanded=True):
            w_col1, w_col2 = st.columns([4, 1.2])
            with w_col2:
                refresh_wallet_btn = st.button("🔄 Cüzdanı Yenile", width="stretch", key="btn_refresh_real_wallet")
            if refresh_wallet_btn or "real_wallet_snapshot" not in st.session_state:
                try:
                    import httpx

                    acc_raw = engine.live_executor._request_signed(
                        "GET", "/api/v3/account", ui_live_confirmed=True
                    )
                    px_map: dict[str, Decimal] = {}
                    try:
                        resp_px = httpx.get(
                            f"{cfg.binance_public_base_url}/api/v3/ticker/bookTicker", timeout=8.0
                        )
                        if resp_px.status_code == 200:
                            px_map = {
                                str(item["symbol"]): Decimal(str(item["bidPrice"]))
                                for item in resp_px.json()
                                if Decimal(str(item.get("bidPrice", "0"))) > 0
                            }
                    except Exception:
                        px_map = {}

                    wallet_rows: list[dict[str, Any]] = []
                    spot_total_usd = Decimal("0")
                    earn_total_usd = Decimal("0")
                    for b in acc_raw.get("balances", []):
                        free_q = Decimal(str(b.get("free", "0")))
                        lock_q = Decimal(str(b.get("locked", "0")))
                        tot_q = free_q + lock_q
                        if tot_q <= Decimal("0"):
                            continue
                        raw_asset = str(b.get("asset", "")).upper()
                        is_earn = raw_asset.startswith("LD") and len(raw_asset) > 2
                        underlying = raw_asset[2:] if is_earn else raw_asset
                        wallet_loc = "🟡 Simple Earn (Kazan / Faiz)" if is_earn else "🟢 Spot Cüzdan"
                        if underlying in {"USDT", "USDC", "FDUSD"}:
                            unit_px = Decimal("1")
                            usd_val = tot_q
                        else:
                            unit_px = px_map.get(f"{underlying}USDT", Decimal("0"))
                            usd_val = tot_q * unit_px
                        if is_earn:
                            earn_total_usd += usd_val
                        else:
                            spot_total_usd += usd_val
                        if usd_val >= Decimal("0.01") or tot_q >= Decimal("0.01"):
                            wallet_rows.append(
                                {
                                    "Cüzdan Yeri": wallet_loc,
                                    "Varlık": f"{underlying} ({raw_asset})" if is_earn else raw_asset,
                                    "Miktar": f"{tot_q:.8f}".rstrip("0").rstrip("."),
                                    "Anlık Fiyat (USDT)": f"{unit_px:.4f}" if unit_px > 0 else "-",
                                    "Yaklaşık Değer (USDT)": round(float(usd_val), 2),
                                }
                            )
                    wallet_rows.sort(key=lambda x: x["Yaklaşık Değer (USDT)"], reverse=True)
                    st.session_state["real_wallet_snapshot"] = {
                        "ok": True,
                        "rows": wallet_rows,
                        "spot_usd": spot_total_usd,
                        "earn_usd": earn_total_usd,
                        "total_usd": spot_total_usd + earn_total_usd,
                        "updated": datetime.now(cfg.tz).strftime("%H:%M:%S"),
                    }
                except Exception as exc_w:
                    st.session_state["real_wallet_snapshot"] = {"ok": False, "err": str(exc_w)}

            wsnap = st.session_state.get("real_wallet_snapshot", {})
            if wsnap.get("ok"):
                with w_col1:
                    st.markdown(
                        f"**Gerçek Hesap Toplamı:** **`~{wsnap['total_usd']:.2f} USDT`** "
                        f"*(Spot: `{wsnap['spot_usd']:.2f} USDT` | Simple Earn / Kazan: `{wsnap['earn_usd']:.2f} USDT` — Son sorgu: {wsnap['updated']})*"
                    )
                if wsnap["earn_usd"] > Decimal("1.0") and wsnap["spot_usd"] < Decimal("5.0"):
                    st.warning(
                        "⚠️ **Paranız Kaybolmadı — Binance Simple Earn (Otomatik Katılım) Devrede!**\n\n"
                        f"Spot cüzdanınız `{wsnap['spot_usd']:.2f} USDT` görünüyor çünkü Binance uygulamanızdaki "
                        f"**Auto-Subscribe (Otomatik Katılım)** ayarı açık kalmış ve **`~{wsnap['earn_usd']:.2f} USDT`** tutarındaki "
                        "varlıklarınız (`LDUSDT`, `LDXRP` vb.) gece otomatik olarak **Earn (Kazan / Esnek Birikim)** hesabına taşınmış.\n"
                        "- Gerçek parayla işlem açabilmek için Binance mobil uygulamasından **Cüzdanlar -> Kazan (Earn)** sekmesine girip "
                        "**Otomatik Katılım (Auto-Subscribe)** düğmesini kapatın ve **Geri Al (Redeem -> Spot Cüzdanı)** yapın."
                    )
                if wsnap["rows"]:
                    st.dataframe(pd.DataFrame(wsnap["rows"]), width="stretch", hide_index=True)
            elif wsnap.get("err"):
                st.info(f"Gerçek cüzdan özeti şu anda alınamadı: {wsnap['err']}")

    # --- Hızlı Eylem Düğmeleri ---
    paused, pause_reason = storage.is_new_entries_paused(track=active_track)
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.4, 1.4, 1.4, 2.0])

    with btn_col1:
        run_now = st.button(
            "▶️ Analiz Adımını Şimdi Çalıştır",
            type="primary",
            width="stretch",
            disabled=runner_alive or quote_mismatch,
        )

    with btn_col2:
        if paused:
            if st.button("🟢 Yeni İşlemlere İzin Ver", width="stretch"):
                storage.set_new_entries_paused(False, "", track=active_track)
                st.rerun()
        else:
            if st.button("⏸️ Yeni İşlemleri Durdur", width="stretch"):
                storage.set_new_entries_paused(
                    True, "Kullanıcı panelden yeni alımları durdurdu", track=active_track
                )
                st.rerun()

    with btn_col3:
        close_btn_label = {
            "LIVE_MARGIN": "🛑 Açık Canlı Marjin Pozisyonları Kapat",
            "LIVE_SPOT": "🛑 Açık Canlı Pozisyonları Kapat",
            "PAPER": "🛑 Açık Sanal Pozisyonları Kapat",
        }[effective_mode]
        if st.button(close_btn_label, width="stretch", disabled=runner_alive or quote_mismatch):
            ok_close, close_msg = engine.manual_close_open_position(
                eff_settings,
                execution_mode=effective_mode,
                ui_live_confirmed=ui_live_confirmed,
                ui_margin_confirmed=ui_margin_confirmed,
            )
            if ok_close:
                st.success(close_msg)
            else:
                st.warning(close_msg)

    with btn_col4:
        notional_per_pos = eff_settings.max_position_size_usdc * Decimal(eff_settings.leverage)
        if paused:
            st.warning(f"Yeni Alımlar Durduruldu: {pause_reason or 'Kullanıcı kilidi'}")
        elif scan_all_coins:
            st.success(
                f"Çoklu Coin ({eff_settings.leverage}x): {len(symbol_options)} coin | "
                f"Eşzamanlı maks {eff_settings.max_open_positions} coin "
                f"({eff_settings.max_position_size_usdc:.0f} {portfolio_quote} teminat -> ~{notional_per_pos:.0f} {portfolio_quote} işlem)"
            )
        else:
            st.success(
                f"Sistem Hazır ({eff_settings.leverage}x): {eff_settings.max_position_size_usdc:.0f} {portfolio_quote} teminat "
                f"-> ~{notional_per_pos:.0f} {portfolio_quote} pozisyon"
            )

    # Analiz adımı tetiklendiyse çalıştır
    if (run_now or auto_loop) and not quote_mismatch and not is_runner_alive(storage)[0]:
        if scan_all_coins:
            engine.lock_benchmark_symbol = True
            with st.spinner(
                f"{len(symbol_options)} coin ({selected_interval}, {eff_settings.leverage}x) taranıyor; açık pozisyonlar ve yeni fırsatlar kontrol ediliyor..."
            ):
                prefetched_live = False
                if effective_mode == "LIVE_SPOT":
                    try:
                        engine.live_executor.fetch_live_spot_balances(
                            ui_live_confirmed=ui_live_confirmed, quote_asset=selected_quote
                        )
                        prefetched_live = True
                    except Exception:
                        prefetched_live = False
                elif effective_mode == "LIVE_MARGIN":
                    try:
                        engine.live_margin_executor.fetch_live_margin_balances(
                            ui_live_confirmed=ui_live_confirmed,
                            ui_margin_confirmed=ui_margin_confirmed,
                            quote_asset=selected_quote,
                        )
                        prefetched_live = True
                    except Exception:
                        prefetched_live = False

                open_before = storage.get_open_positions(track=active_track)
                open_syms_before = list(dict.fromkeys(p.symbol for p in open_before))
                if len(open_before) < eff_settings.max_open_positions:
                    scan_targets = open_syms_before + [
                        s for s in symbol_options if s not in open_syms_before
                    ]
                else:
                    scan_targets = open_syms_before if open_syms_before else list(symbol_options)

                executed_steps: list[Any] = []
                last_scanned_res = None
                scanned_count = 0
                for sym in scan_targets:
                    cur_open = storage.get_open_positions(track=active_track)
                    cur_open_syms = {p.symbol for p in cur_open}
                    if sym not in cur_open_syms and len(cur_open) >= eff_settings.max_open_positions:
                        break
                    step_res = engine.run_analysis_step(
                        symbol=sym,
                        interval=selected_interval,
                        settings=eff_settings,
                        ai_enabled_by_user=ai_enabled,
                        daily_ai_limit_override=int(daily_ai_limit),
                        min_ai_interval_override=active_profile.ai_min_call_interval_seconds,
                        execution_mode=effective_mode,
                        ui_live_confirmed=ui_live_confirmed,
                        ui_margin_confirmed=ui_margin_confirmed,
                        strategy_params=active_profile.strategy,
                        ai_only_on_rule_signal=True,
                        skip_balance_prefetch=prefetched_live,
                    )
                    scanned_count += 1
                    last_scanned_res = step_res
                    if step_res.executed_order is not None:
                        executed_steps.append(step_res)
                    if (
                        step_res.executed_order is not None
                        and step_res.executed_order.side == "BUY"
                        and step_res.executed_order.status in ("FILLED", "PARTIALLY_FILLED")
                        and len(storage.get_open_positions(track=active_track)) >= eff_settings.max_open_positions
                    ):
                        break

                st.session_state["last_step_summary"] = (
                    executed_steps[-1] if executed_steps else last_scanned_res
                )
                open_now_list = storage.get_open_positions(track=active_track)
                open_names = ", ".join(p.symbol for p in open_now_list) or "Yok"
                exec_names = (
                    ", ".join(
                        f"{s.symbol} ({s.executed_order.side})"
                        for s in executed_steps
                        if s.executed_order is not None
                    )
                    or "Yeni işlem yok (uygun sinyal bekleniyor)"
                )
                st.session_state["last_multi_scan_note"] = (
                    f"🌐 **Çoklu Coin Tarama Özeti ({eff_settings.leverage}x):** `{scanned_count}` coin kontrol edildi | "
                    f"**Açık Pozisyonlar ({len(open_now_list)}/{eff_settings.max_open_positions}):** `{open_names}` | "
                    f"**Bu Döngüdeki İşlemler:** `{exec_names}`"
                )
        else:
            with st.spinner(f"{selected_symbol} ({selected_interval}, {eff_settings.leverage}x) doğrulanmış piyasa verisi analiz ediliyor..."):
                step_res = engine.run_analysis_step(
                    symbol=selected_symbol,
                    interval=selected_interval,
                    settings=eff_settings,
                    ai_enabled_by_user=ai_enabled,
                    daily_ai_limit_override=int(daily_ai_limit),
                    min_ai_interval_override=active_profile.ai_min_call_interval_seconds,
                    execution_mode=effective_mode,
                    ui_live_confirmed=ui_live_confirmed,
                    ui_margin_confirmed=ui_margin_confirmed,
                    strategy_params=active_profile.strategy,
                )
                st.session_state["last_step_summary"] = step_res
                st.session_state.pop("last_multi_scan_note", None)

    last_step = st.session_state.get("last_step_summary")
    multi_scan_note = st.session_state.get("last_multi_scan_note")
    if multi_scan_note:
        st.info(multi_scan_note)

    # Bağlantı hatası varsa sade Türkçe açıklama göster
    if last_step is not None and not last_step.ok and last_step.error_message:
        st.error(
            f"**Bağlantı / Veri Doğrulama Uyarısı:** {last_step.error_message}\n\n"
            f"ℹ️ *{last_step.block_or_info_message}*"
        )
    elif last_step is not None and last_step.wake_gap_detected:
        st.info(f"💤 {last_step.block_or_info_message}")

    # --- Portföy ve Maliyet Özeti Kartları ---
    open_for_summary = storage.get_open_positions(track=active_track)
    refresh_prices = st.button("🔄 Anlık Fiyatları Şimdi Güncelle") if not live_pnl_auto else False
    live_bids_for_summary = _fetch_live_bids(
        engine, [p.symbol for p in open_for_summary], refresh=refresh_prices
    )
    summary = engine.get_portfolio_summary(
        eff_settings, live_bid_by_symbol=live_bids_for_summary, track=active_track
    )
    ai_usage = storage.get_ai_usage_summary(ist_today)
    valuation_complete = summary["valuation_complete"]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Kullanılabilir Nakit", f"{summary['cash_usdc']:.2f} {portfolio_quote}")
    m2.metric("Toplam Portföy Değeri", f"{summary['total_equity_usdc']:.2f} {portfolio_quote}" if valuation_complete else "—")
    m3.metric(
        "Gerçekleşmiş Kâr/Zarar",
        f"{summary['realized_pnl_usdc']:+.3f} {portfolio_quote}",
        help="Kapanmış pozisyonların giriş+çıkış komisyonu ve varsa marjin faizi düşülmüş net sonucu.",
    )
    m4.metric(
        "Açık Pozisyon Net K/Z",
        f"{summary['unrealized_pnl_usdc']:+.3f} {portfolio_quote}" if valuation_complete else "—",
        help="Anlık alış kotasyonu, tahmini çıkış komisyonu ve marjin faizi düşülmüş net değer değişimi.",
    )
    m5.metric(
        "Toplam İşlem Komisyonu",
        f"{summary['total_fees_usdc']:.4f} {portfolio_quote}",
        help=f"Varsayılan komisyon oranı: %{eff_settings.estimated_fee_rate * Decimal('100'):.2f}",
    )
    ai_cost_label = (
        f"~{ai_usage['estimated_cost_usd']:.5f} USD"
        if cfg.ai_pricing_verified
        else f"~{ai_usage['estimated_cost_usd']:.5f} USD (Tahmini)"
    )
    m6.metric(
        "Tahmini Yapay Zekâ Maliyeti",
        ai_cost_label,
        help=(
            f"Bugünkü ağ çağrısı: {ai_usage['today_network_calls']}/{daily_ai_limit} | "
            f"Önbellek isabeti: {ai_usage['cache_hits']}. "
            "Tarife API'den doğrulanamadığında kesin fatura değil, birim fiyat varsayımına dayalı tahmindir."
        ),
    )
    if not valuation_complete:
        st.info("Güncel fiyatlar alınamadı; portföy değeri ve açık kâr/zarar için Anlık Fiyatları Şimdi Güncelle düğmesini kullanın.")

    # --- Veri Güncelliği, Bağlantı ve Son Karar Kutusu ---
    info_left, info_right = st.columns([1.2, 1.8])
    with info_left:
        st.subheader("📡 Veri Kaynağı ve Sistem Durumu")
        if last_step and last_step.ok:
            dt_ist = datetime.fromisoformat(last_step.timestamp_iso).astimezone(cfg.tz)
            st.write(f"- **Veri Kaynağı:** `{last_step.data_source}`")
            st.write(f"- **Son Güncelleme (İstanbul):** `{dt_ist.strftime('%Y-%m-%d %H:%M:%S')}`")
        else:
            st.write("- **Veri Kaynağı:** Henüz canlı adım tamamlanmadı veya bağlantı bekleniyor")
            st.write(f"- **Saat Dilimi:** `{cfg.timezone_name}` (Gün sınırı 00:00)")

        ai_status_str = (
            f"Aktif (`{cfg.gemini_model}`)"
            if (cfg.has_gemini_key and ai_enabled)
            else "Yapay Zekâ Kapalı / Kural Tabanlı Deneme"
        )
        st.write(f"- **Karar Modu:** {ai_status_str}")
        st.write(f"- **Aktif Kaldıraç / Toplam Borç:** `{eff_settings.leverage}x` / `{summary.get('borrowed_usdc', Decimal('0')):.2f} {portfolio_quote}`")
        st.write(
            "- **Günlük Net K/Z (İstanbul):** "
            + (f"`{summary['daily_net_pnl_usdc']:+.3f} {portfolio_quote}` " if valuation_complete else "`—` ")
            + f"(Zarar Eşiği: `-{eff_settings.daily_loss_limit_usdc:.2f} {portfolio_quote}`)"
        )

    with info_right:
        st.subheader("🧠 Son Karar ve Kısa Gerekçe")
        recent_decisions = storage.get_recent_decisions(
            limit=1, track=active_track, quote_asset=portfolio_quote
        )
        if recent_decisions:
            d = recent_decisions[0]
            badge = {"AL": "🟢 AL", "SAT": "🔴 SAT", "BEKLE": "🟡 BEKLE"}.get(
                str(d["final_action"]), str(d["final_action"])
            )
            st.markdown(f"**Nihai Karar:** {badge}  |  **Mod:** `{d['mode_label']}`  |  **Parite:** `{d['symbol']} ({d['interval']})`")
            st.write(f"**Gerekçe:** {d['ai_reason']}")
            st.caption(f"Kural Tabanlı Karşılaştırma Sinyali: `{d['rule_action']}` — {d['rule_reason']}")
            if d.get("self_reported_score") is not None:
                st.caption(
                    f"Model Öz-Beyan Puanı: {d['self_reported_score']}/100 "
                    "*(Uyarı: Bu puan modelin kendi beyanıdır; doğrulanmış başarı olasılığı değildir.)*"
                )
            if d.get("block_reason"):
                st.warning(f"Risk / Filtre Engeli: {d['block_reason']}")
        else:
            st.info(
                "Henüz kaydedilmiş bir analiz kararı yok. Üstteki **'Analiz Adımını Şimdi Çalıştır'** "
                "düğmesine basarak ilk denetimi başlatabilirsiniz."
            )

    st.divider()
    if live_pnl_auto:
        _render_live_pnl_fragment(engine, eff_settings, active_track, portfolio_quote, cfg.tz)
    else:
        _render_live_pnl(
            engine, eff_settings, active_track, portfolio_quote, cfg.tz
        )
    st.divider()

    # --- Ana Sekmeler ---
    tab_port, tab_cmp, tab_val, tab_cfg = st.tabs(
        [
            "📊 Portföy, Grafik & İşlemler",
            "⚖️ Günlük Strateji Karşılaştırması",
            "🔬 Kronolojik Doğrulama (Backtest) & Maliyet",
            "🛡️ Risk Ayarları & CSV İndir",
        ]
    )

    with tab_port:
        st.subheader("Açık Pozisyon")
        open_pos = summary["open_positions"]
        if open_pos:
            pos_rows = []
            hourly_rate = getattr(eff_settings, "hourly_margin_interest_rate", Decimal("0.000005"))
            for p in open_pos:
                cur_bid = live_bids_for_summary.get(p.symbol)
                cur_pnl = None
                if cur_bid is not None:
                    _, cur_pnl = compute_open_position_unrealized_net_pnl(
                        p,
                        current_bid=cur_bid,
                        estimated_fee_rate=eff_settings.estimated_fee_rate,
                        hourly_interest_rate=hourly_rate,
                    )
                p_lev = getattr(p, "leverage", 1)
                p_borrowed = getattr(p, "borrowed_usdc", Decimal("0"))
                p_liq = getattr(p, "liquidation_price", None)
                pos_rows.append(
                    {
                        "Pozisyon Kimliği": p.position_id,
                        "Parite": p.symbol,
                        "Kaldıraç": f"{p_lev}x",
                        "Miktar": f"{p.quantity:.8f}",
                        f"Giriş Fiyatı ({portfolio_quote})": f"{p.entry_price:.6f}",
                        f"Anlık Fiyat ({portfolio_quote})": f"{cur_bid:.6f}" if cur_bid is not None else "—",
                        f"Anlık Net K/Z ({portfolio_quote})": f"{cur_pnl:+.4f}" if cur_pnl is not None else "—",
                        f"Pozisyon Büyüklüğü ({portfolio_quote})": f"{p.cost_basis_usdc:.2f}",
                        f"Borç ({portfolio_quote})": f"{p_borrowed:.2f}" if p_borrowed > 0 else "-",
                        "Likidasyon": f"{p_liq:.6f}" if p_liq else "-",
                        "Giriş Komisyonu": f"{p.entry_fee_usdc:.4f}",
                        "Zarar-Kes": f"{p.stop_loss_price:.6f}" if p.stop_loss_price else "-",
                        "Kâr-Al": f"{p.take_profit_price:.6f}" if p.take_profit_price else "-",
                        "Açılış Zamanı": p.opened_at_iso,
                    }
                )
            st.dataframe(pd.DataFrame(pos_rows), width="stretch", hide_index=True)
        else:
            st.info("Şu anda açık pozisyon bulunmuyor.")

        st.subheader("Portföy Bakiye Eğrisi")
        snaps = storage.get_equity_snapshots(
            limit=300, track=active_track, quote_asset=portfolio_quote
        )
        if snaps:
            snap_df = pd.DataFrame(snaps)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=snap_df["created_at_iso"],
                    y=snap_df["main_equity_usdc"].astype(float),
                    mode="lines+markers",
                    name=f"Aktif Portföy ({portfolio_quote})",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=snap_df["created_at_iso"],
                    y=snap_df["rule_shadow_equity_usdc"].astype(float),
                    mode="lines",
                    name=f"Kural Tabanlı Gölge ({portfolio_quote})",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=snap_df["created_at_iso"],
                    y=snap_df["benchmark_equity_usdc"].astype(float),
                    mode="lines",
                    line={"dash": "dot"},
                    name=f"Pasif Referans ({portfolio_quote})",
                )
            )
            fig.update_layout(
                height=340,
                margin={"l": 20, "r": 20, "t": 20, "b": 20},
                yaxis_title=f"Toplam Değer ({portfolio_quote})",
                xaxis_title="Zaman (UTC)",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Bakiye grafiği ilk analiz adımından itibaren oluşturulacaktır.")

        st.subheader("İşlem Geçmişi")
        orders = storage.get_orders(track=active_track, limit=100)
        if orders:
            st.dataframe(pd.DataFrame(orders), width="stretch", hide_index=True)
        else:
            st.caption("Henüz gerçekleşmiş veya reddedilmiş emir kaydı bulunmuyor.")

        st.info(FILL_MODEL_EXPLANATION_TR)

    with tab_cmp:
        st.subheader("Günlük Strateji ve Referans Karşılaştırması")
        st.write(
            "Aşağıdaki tablo, yapay zekânın ek katkı sağlayıp sağlamadığını ayrı ölçmek için "
            "aynı kapanmış mumlar üzerinde üç yaklaşımı yan yana izler:"
        )
        shadow_sum = engine.get_portfolio_summary(
            dataclasses.replace(eff_settings, leverage=1),
            live_bid_by_symbol=live_bids_for_summary,
            track=engine._shadow_track(portfolio_quote),
        )
        snaps_latest = storage.get_equity_snapshots(
            limit=1, track=active_track, quote_asset=portfolio_quote
        )
        bench_val = (
            Decimal(str(snaps_latest[-1]["benchmark_equity_usdc"]))
            if snaps_latest
            else cfg.initial_balance_usdc
        )

        cmp_df = pd.DataFrame(
            [
                {
                    "Yaklaşım": f"1. Aktif Portföy ({eff_settings.leverage}x Seçili Mod)",
                    f"Toplam Değer ({portfolio_quote})": f"{summary['total_equity_usdc']:.3f}" if valuation_complete else "—",
                    f"Net Fark ({portfolio_quote})": f"{(summary['total_equity_usdc'] - cfg.initial_balance_usdc):+.3f}" if valuation_complete else "—",
                    f"Toplam Komisyon ({portfolio_quote})": f"{summary['total_fees_usdc']:.4f}",
                    "Tahmini AI Maliyeti (USD)": f"{ai_usage['estimated_cost_usd']:.5f}",
                },
                {
                    "Yaklaşım": "2. Saf Kural Tabanlı Strateji (1x Spot EMA/RSI/MACD Gölge)",
                    f"Toplam Değer ({portfolio_quote})": f"{shadow_sum['total_equity_usdc']:.3f}" if shadow_sum["valuation_complete"] else "—",
                    f"Net Fark ({portfolio_quote})": f"{(shadow_sum['total_equity_usdc'] - cfg.initial_balance_usdc):+.3f}" if shadow_sum["valuation_complete"] else "—",
                    f"Toplam Komisyon ({portfolio_quote})": f"{shadow_sum['total_fees_usdc']:.4f}",
                    "Tahmini AI Maliyeti (USD)": "0.00000",
                },
                {
                    "Yaklaşım": f"3. Referans ({min(settings.max_position_size_usdc, cfg.initial_balance_usdc):.0f} {portfolio_quote} {selected_symbol} Al-Tut + Kalan Nakit)",
                    f"Toplam Değer ({portfolio_quote})": f"{bench_val:.3f}",
                    f"Net Fark ({portfolio_quote})": f"{(bench_val - cfg.initial_balance_usdc):+.3f}",
                    f"Toplam Komisyon ({portfolio_quote})": f"{(min(settings.max_position_size_usdc, cfg.initial_balance_usdc) * settings.estimated_fee_rate):.4f}",
                    "Tahmini AI Maliyeti (USD)": "0.00000",
                },
            ]
        )
        st.dataframe(cmp_df, width="stretch", hide_index=True)

        ai_alpha = (
            summary["total_equity_usdc"]
            - Decimal(str(ai_usage["estimated_cost_usd"]))
            - shadow_sum["total_equity_usdc"]
        )
        if valuation_complete and shadow_sum["valuation_complete"]:
            st.caption(
                f"Tahmini AI Maliyeti Düşülmüş Yapay Zekâ Ek Katkısı (vs. Kural Tabanlı): `{ai_alpha:+.4f} {portfolio_quote}` "
                "(Kısa süreli birkaç işlem istatistiksel kanıt oluşturmaz.)"
            )
        else:
            st.caption("Yapay zekâ ek katkısı, tüm açık pozisyonların güncel fiyatları doğrulandığında hesaplanır.")

    with tab_val:
        st.subheader(f"{cfg.initial_balance_usdc:.2f} {selected_quote} Küçük Bakiye Komisyon ve Model Maliyet Uygulanabilirliği")
        viability = assess_small_balance_viability(
            balance_usdc=cfg.initial_balance_usdc,
            position_size_usdc=settings.max_position_size_usdc * Decimal(eff_settings.leverage),
            fee_rate=settings.estimated_fee_rate,
            extra_slippage_bps=settings.extra_slippage_bps,
            daily_ai_calls=int(daily_ai_limit) if (cfg.has_gemini_key and ai_enabled) else 0,
            input_usd_per_1m=cfg.ai_estimated_input_usd_per_1m,
            output_usd_per_1m=cfg.ai_estimated_output_usd_per_1m,
        )
        st.warning(f"**Değerlendirme: {viability.viability_status}**\n\n{viability.explanation_tr}")

        st.subheader("Kronolojik Eğitim (%60) / Doğrulama (%20) / Test (%20) Analizi")
        st.caption(
            "Gelecek veriyi sızdırmaz (No-Lookahead): `t` mum kapanış sinyali yalnızca `t+1` mum "
            "açılışında spread + kayma + komisyon eklenerek kaldıraçsız Spot (1x) bazda test edilir."
        )
        bcol1, bcol2 = st.columns(2)
        run_live_bt = bcol1.button("🌐 Canlı Binance Geçmiş Mumlarıyla Doğrula (500 Mum)", width="stretch")
        run_demo_bt = bcol2.button(
            "🧪 Sentetik Veriyle Motor Gösterimi",
            width="stretch",
            help="Yapay üretilmiş fiyatlarla motoru gösterir; stratejinin gerçek piyasa başarısını ölçmez.",
        )

        if run_live_bt or run_demo_bt:
            candles_for_bt: list[Candle] = []
            bt_label = ""
            bt_filters = None
            if run_live_bt:
                try:
                    bundle = engine.market_client.fetch_verified_market_bundle(
                        symbol=selected_symbol,
                        interval=selected_interval,
                        candle_limit=500,
                        allow_stablecoin_conversion=settings.allow_stablecoin_conversion,
                    )
                    candles_for_bt = bundle.closed_candles
                    bt_filters = bundle.filters
                    bt_label = f"{selected_symbol} ({selected_interval}) — Gerçek Binance Verisi"
                except MarketDataError as exc:
                    st.error(f"Canlı geçmiş mum verisi alınamadı: {exc}")
            else:
                candles_for_bt = _generate_offline_demo_candles(240, interval=selected_interval)
                bt_label = "Sentetik Motor Gösterimi — Gerçek Piyasa Performansı Değildir"

            if candles_for_bt:
                spot_bt_settings = dataclasses.replace(eff_settings, leverage=1)
                rep = run_chronological_validation(
                    candles_for_bt,
                    symbol=selected_symbol,
                    interval=selected_interval,
                    settings=spot_bt_settings,
                    initial_balance_usdc=cfg.initial_balance_usdc,
                    strategy_params=active_profile.strategy,
                    symbol_filters=bt_filters,
                )
                st.markdown(f"**Veri Seti:** `{bt_label}` ({rep.total_candles} kapanmış mum)")
                rows_bt = []
                for sl in (rep.train_metrics, rep.validation_metrics, rep.test_metrics):
                    rows_bt.append(
                        {
                            "Kronolojik Dilim": sl.slice_name,
                            "Mum Sayısı": sl.candle_count,
                            "İşlem Sayısı": sl.trade_count,
                            f"Net Getiri ({selected_quote})": f"{sl.net_return_usdc:+.3f}",
                            "Net Getiri (%)": f"%{sl.net_return_pct:+.2f}",
                            "Referans Al-Tut (%)": f"%{sl.benchmark_return_pct:+.2f}",
                            "Maks. Düşüş (%)": f"%{sl.max_drawdown_pct:.2f}",
                            "Kazanma Oranı (%)": f"%{sl.win_rate_pct:.1f}",
                            f"Toplam Komisyon ({selected_quote})": f"{sl.total_fees_usdc:.4f}",
                            "Değerlendirme": sl.verdict_text,
                        }
                    )
                st.dataframe(pd.DataFrame(rows_bt), width="stretch", hide_index=True)
                if run_demo_bt:
                    st.info(
                        "Bu sonuç yapay fiyatlarla üretilmiş bir motor gösterimidir. "
                        "Stratejinin kârlılığını veya gerçek piyasadaki başarısını doğrulamaz."
                    )
                elif rep.insufficient_data:
                    st.info(f"📌 **Sonuç:** {rep.overall_verdict}")
                else:
                    st.success(f"📌 **Örneklem Dışı Test Özeti:** {rep.overall_verdict}")

    with tab_cfg:
        st.subheader("Değiştirilebilir Risk ve Maliyet Ayarları")
        st.caption(RISK_DISCLAIMER_TR)

        with st.form("risk_settings_form"):
            fc1, fc2 = st.columns(2)
            with fc1:
                new_max_pos_usdc = st.number_input(
                    f" İşlem Başına Öz Teminat Üst Sınırı ({selected_quote})",
                    min_value=5.0,
                    max_value=float(cfg.initial_balance_usdc),
                    value=float(settings.max_position_size_usdc),
                    step=1.0,
                    help="2x kaldıraçta 10 USDT teminat ~20 USDT pozisyon açar. Minimum emir tutarı aşılırsa işlem reddedilir.",
                )
                new_daily_loss = st.number_input(
                    f"Günlük Zarar Eşiği ({selected_quote} — Europe/Istanbul)",
                    min_value=0.5,
                    max_value=50.0,
                    value=float(settings.daily_loss_limit_usdc),
                    step=0.5,
                )
                new_max_daily_trades = st.number_input(
                    "Günlük En Fazla Yeni Pozisyon Sayısı",
                    min_value=1,
                    max_value=200,
                    value=min(200, max(1, int(settings.max_daily_new_positions))),
                    step=1,
                )
                new_max_open = st.number_input(
                    "Aynı Anda En Fazla Açık Pozisyon (Farklı Coin)",
                    min_value=1,
                    max_value=8,
                    value=min(8, max(1, int(settings.max_open_positions))),
                    step=1,
                )
            with fc2:
                new_fee_pct = st.number_input(
                    "Tahmini Komisyon Oranı (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(settings.estimated_fee_rate * Decimal("100")),
                    step=0.01,
                    format="%.3f",
                )
                new_slip_bps = st.number_input(
                    "Ekstra Kayma Tamponu (Baz Puan — bps)",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(settings.extra_slippage_bps),
                    step=0.5,
                )
                new_sl_pct = st.number_input(
                    "Koruyucu Zarar-Kes Yüzdesi (%)",
                    min_value=0.1,
                    max_value=20.0,
                    value=min(20.0, max(0.1, float(settings.stop_loss_pct * Decimal("100")))),
                    step=0.1,
                )
                new_exit_policy = st.selectbox(
                    "Günlük Zarar Eşiği Aşıldığında Açık Pozisyon Politikası",
                    options=["HOLD_EXISTING", "AUTO_CLOSE"],
                    index=0 if settings.daily_limit_exit_policy == "HOLD_EXISTING" else 1,
                    format_func=lambda k: (
                        "Yeni Alımları Durdur, Açık Pozisyonu Zarar-Kes/Manuel Çıkışta Tut"
                        if k == "HOLD_EXISTING"
                        else "Yeni Alımları Durdur ve Açık Pozisyonu Anında Kapat"
                    ),
                )

            st.info(EXIT_POLICY_DESCRIPTIONS_TR[new_exit_policy])
            if st.form_submit_button("💾 Risk Ayarlarını Kaydet"):
                updated = RiskSettings(
                    max_open_positions=int(new_max_open),
                    max_position_size_usdc=Decimal(f"{new_max_pos_usdc:.2f}"),
                    daily_loss_limit_usdc=Decimal(f"{new_daily_loss:.2f}"),
                    max_daily_new_positions=int(new_max_daily_trades),
                    daily_limit_exit_policy=new_exit_policy,
                    estimated_fee_rate=Decimal(f"{new_fee_pct:.4f}") / Decimal("100"),
                    extra_slippage_bps=Decimal(f"{new_slip_bps:.2f}"),
                    stop_loss_pct=Decimal(f"{new_sl_pct:.2f}") / Decimal("100"),
                    take_profit_pct=settings.take_profit_pct,
                    allow_stablecoin_conversion=False,
                    leverage=int(selected_leverage),
                    hourly_margin_interest_rate=getattr(
                        settings, "hourly_margin_interest_rate", Decimal("0.000005")
                    ),
                )
                storage.save_risk_settings(updated)
                st.success("Risk ve kaldıraç ayarları SQLite veritabanına kaydedildi.")
                st.rerun()

        st.divider()
        st.subheader("📥 Verileri CSV Olarak Dışa Aktar (Excel Uyumlu Türkçe UTF-8)")
        dl1, dl2, dl3, dl4 = st.columns(4)
        dl1.download_button(
            "📄 Emir Geçmişi (CSV)",
            data=storage.export_table_to_csv_bytes("orders"),
            file_name=f"emirler_{ist_today}.csv",
            mime="text/csv",
            width="stretch",
        )
        dl2.download_button(
            "📄 Pozisyonlar (CSV)",
            data=storage.export_table_to_csv_bytes("positions"),
            file_name=f"pozisyonlar_{ist_today}.csv",
            mime="text/csv",
            width="stretch",
        )
        dl3.download_button(
            "📄 Karar Geçmişi (CSV)",
            data=storage.export_table_to_csv_bytes("decisions"),
            file_name=f"kararlar_{ist_today}.csv",
            mime="text/csv",
            width="stretch",
        )
        dl4.download_button(
            "📄 Bakiye Eğrisi (CSV)",
            data=storage.export_table_to_csv_bytes("equity_snapshots"),
            file_name=f"bakiye_gecmisi_{ist_today}.csv",
            mime="text/csv",
            width="stretch",
        )

    # Otomatik yenileme açıksa profil döngü süresi kadar bekle ve yeniden çiz
    if auto_loop:
        time.sleep(active_profile.loop_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
