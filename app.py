"""
Binance Global Spot (USDT / USDC) — Türkçe Yerel Web Kontrol Paneli (Streamlit, 127.0.0.1).

Çalıştırma:
    baslat.bat
veya:
    .venv\\Scripts\\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# Yerel paket yolunu ekle
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from binance_ai_bot import SIMULATION_MODE_LABEL
from binance_ai_bot.config import AppConfig, load_config
from binance_ai_bot.domain import Candle, RiskSettings
from binance_ai_bot.engine import TradingSystemEngine
from binance_ai_bot.market_data import MarketDataError
from binance_ai_bot.risk import EXIT_POLICY_DESCRIPTIONS_TR, RISK_DISCLAIMER_TR
from binance_ai_bot.security import assert_localhost_only
from binance_ai_bot.simulator import FILL_MODEL_EXPLANATION_TR
from binance_ai_bot.validation import (
    assess_small_balance_viability,
    run_chronological_validation,
)

st.set_page_config(
    page_title="Binance Spot Kontrol Paneli",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _generate_offline_demo_candles(count: int = 180) -> list[Candle]:
    """
    Yalnızca 'Doğrulama / Backtest' sekmesinde ağ engeli varken kronolojik test motorunu
    incelemek için kullanılan deterministik örnek seri. Sanal veya canlı portföye DOKUNMAZ.
    """
    import math

    candles: list[Candle] = []
    base_ts = 1_760_000_000_000
    step_ms = 900_000  # 15dk
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


ENGINE_STATE_VERSION = "1.4.0"


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
    if (
        "engine" not in st.session_state
        or st.session_state.get("engine_fingerprint") != cfg_fingerprint
    ):
        st.session_state["engine"] = TradingSystemEngine(cfg)
        st.session_state["engine_fingerprint"] = cfg_fingerprint
    return cfg, st.session_state["engine"]


def main() -> None:
    cfg, engine = _get_engine()
    storage = engine.storage
    settings = engine.default_risk_settings()
    ist_today = storage.istanbul_date_str()

    # --- Yan Menü: Mod Seçimi, Parite ve Kontroller ---
    with st.sidebar:
        st.header("⚙️ Çalışma Ayarları")

        quote_options = ["USDT", "USDC"]
        default_quote_idx = quote_options.index(cfg.quote_asset) if cfg.quote_asset in quote_options else 0
        selected_quote = st.radio(
            "Baz Para Birimi (Spot Cüzdan)",
            options=quote_options,
            index=default_quote_idx,
            horizontal=True,
            help="Spot cüzdanınızda bulundurduğunuz ana bakiye para birimi (USDT veya USDC).",
        )

        mode_choice = st.radio(
            "İşlem Yürütme Modu",
            options=["PAPER", "LIVE_SPOT"],
            format_func=lambda m: (
                f"🟢 Sanal Deneme ({cfg.initial_balance_usdc:.0f} {selected_quote} Sanal)"
                if m == "PAPER"
                else "🔴 Gerçek Para — Canlı Spot (Çift Kilitli)"
            ),
            index=0,
            help="Varsayılan ve önerilen mod Sanal Deneme'dir. Gerçek bakiyenize dokunmaz.",
        )

        ui_live_confirmed = False
        if mode_choice == "LIVE_SPOT":
            st.warning(
                f"Gerçek Spot modunda işlemler gerçek Binance Global {selected_quote} bakiyenizle yapılır. "
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
                if ui_live_confirmed:
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

        st.divider()
        base_coins = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX"]
        symbol_options = [f"{coin}{selected_quote}" for coin in base_coins]
        if cfg.default_symbol.endswith(selected_quote) and cfg.default_symbol not in symbol_options:
            symbol_options.insert(0, cfg.default_symbol)
        selected_symbol = st.selectbox(
            f"Spot {selected_quote} İşlem Çifti", options=symbol_options, index=0
        )

        interval_options = ["5m", "15m", "30m", "1h", "4h"]
        default_int_idx = interval_options.index(cfg.default_interval) if cfg.default_interval in interval_options else 1
        selected_interval = st.selectbox(
            "Kapanmış Mum Periyodu",
            options=interval_options,
            index=default_int_idx,
            help=f"Küçük bakiyede ({selected_quote}) komisyon erimesini önlemek için 15m veya 1h önerilir.",
        )

        st.divider()
        st.subheader("🤖 Yapay Zekâ (Gemini) Ayarları")
        if cfg.has_gemini_key:
            ai_enabled = st.toggle("Gemini Karar Modülü Aktif", value=True)
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
            max_value=200,
            value=int(cfg.ai_daily_call_limit),
            step=1,
        )
        auto_loop = st.toggle(
            "Otomatik Döngü (Her 60 sn'de Kontrol Et)",
            value=False,
            help="Bilgisayar açık ve uyanık kaldığı sürece 60 saniyede bir yeni kapanmış mum kontrolü yapar.",
        )

    effective_mode = "LIVE_SPOT" if (mode_choice == "LIVE_SPOT" and ui_live_confirmed) else "PAPER"
    active_track = "LIVE_SPOT" if effective_mode == "LIVE_SPOT" else "MAIN"

    # --- Üst Güvenlik Etiketi (Banner) ---
    if effective_mode == "PAPER":
        st.markdown(
            f"""
            <div style="background-color:#0F766E;color:#FFFFFF;padding:14px 20px;border-radius:10px;
                        font-size:18px;font-weight:700;margin-bottom:14px;border-left:8px solid #99F6E4;">
                🛡️ {SIMULATION_MODE_LABEL} — Gerçek Binance Bakiyenize Dokunulmaz (Başlangıç Sanal Kasa: {cfg.initial_balance_usdc:.2f} {selected_quote})
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#991B1B;color:#FFFFFF;padding:14px 20px;border-radius:10px;
                        font-size:18px;font-weight:700;margin-bottom:14px;border-left:8px solid #FCA5A5;">
                ⚠️ DİKKAT: CANLI SPOT İŞLEM MODU AKTİF — GERÇEK BINANCE GLOBAL {selected_quote} BAKİYESİ KULLANILMAKTADIR
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Hızlı Eylem Düğmeleri ---
    paused, pause_reason = storage.is_new_entries_paused()
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.4, 1.4, 1.4, 2.0])

    with btn_col1:
        run_now = st.button("▶️ Analiz Adımını Şimdi Çalıştır", type="primary", width="stretch")

    with btn_col2:
        if paused:
            if st.button("🟢 Yeni İşlemlere İzin Ver", width="stretch"):
                storage.set_new_entries_paused(False, "")
                st.rerun()
        else:
            if st.button("⏸️ Yeni İşlemleri Durdur", width="stretch"):
                storage.set_new_entries_paused(True, "Kullanıcı panelden yeni alımları durdurdu")
                st.rerun()

    with btn_col3:
        close_btn_label = (
            "🛑 Canlı Pozisyonu Piyasadan Sat"
            if effective_mode == "LIVE_SPOT"
            else "🛑 Sanal Pozisyonu Kapat"
        )
        if st.button(close_btn_label, width="stretch"):
            ok_close, close_msg = engine.manual_close_open_position(
                settings,
                execution_mode=effective_mode,
                ui_live_confirmed=ui_live_confirmed,
            )
            if ok_close:
                st.success(close_msg)
            else:
                st.warning(close_msg)

    with btn_col4:
        if paused:
            st.warning(f"Yeni Alımlar Durduruldu: {pause_reason or 'Kullanıcı kilidi'}")
        else:
            st.success("Sistem Durumu: Yeni sinyal denetimine hazır (Yalnızca Kaldıraçsız Spot)")

    # Analiz adımı tetiklendiyse çalıştır
    if run_now or auto_loop:
        with st.spinner(f"{selected_symbol} ({selected_interval}) doğrulanmış piyasa verisi analiz ediliyor..."):
            step_res = engine.run_analysis_step(
                symbol=selected_symbol,
                interval=selected_interval,
                settings=settings,
                ai_enabled_by_user=ai_enabled,
                daily_ai_limit_override=int(daily_ai_limit),
                execution_mode=effective_mode,
                ui_live_confirmed=ui_live_confirmed,
            )
            st.session_state["last_step_summary"] = step_res

    last_step = st.session_state.get("last_step_summary")

    # Bağlantı hatası varsa sade Türkçe açıklama göster
    if last_step is not None and not last_step.ok and last_step.error_message:
        st.error(
            f"**Bağlantı / Veri Doğrulama Uyarısı:** {last_step.error_message}\n\n"
            f"ℹ️ *{last_step.block_or_info_message}*"
        )
    elif last_step is not None and last_step.wake_gap_detected:
        st.info(f"💤 {last_step.block_or_info_message}")

    # --- Portföy ve Maliyet Özeti Kartları ---
    summary = engine.get_portfolio_summary(settings, track=active_track)
    ai_usage = storage.get_ai_usage_summary(ist_today)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Kullanılabilir Nakit", f"{summary['cash_usdc']:.2f} {selected_quote}")
    m2.metric("Toplam Portföy Değeri", f"{summary['total_equity_usdc']:.2f} {selected_quote}")
    m3.metric(
        "Gerçekleşmiş Kâr/Zarar",
        f"{summary['realized_pnl_usdc']:+.3f} {selected_quote}",
        help="Kapanmış pozisyonların giriş+çıkış komisyonu düşülmüş net sonucu.",
    )
    m4.metric(
        "Açık Pozisyon Net K/Z",
        f"{summary['unrealized_pnl_usdc']:+.3f} {selected_quote}",
        help="Anlık alış kotasyonu ve tahmini çıkış komisyonu düşülmüş net değer değişimi.",
    )
    m5.metric(
        "Toplam İşlem Komisyonu",
        f"{summary['total_fees_usdc']:.4f} {selected_quote}",
        help=f"Varsayılan komisyon oranı: %{settings.estimated_fee_rate * Decimal('100'):.2f}",
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
        st.write(
            f"- **Günlük Net K/Z (İstanbul):** `{summary['daily_net_pnl_usdc']:+.3f} {selected_quote}` "
            f"(Zarar Eşiği: `-{settings.daily_loss_limit_usdc:.2f} {selected_quote}`)"
        )

    with info_right:
        st.subheader("🧠 Son Karar ve Kısa Gerekçe")
        recent_decisions = storage.get_recent_decisions(limit=1)
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
            pos_rows = [
                {
                    "Pozisyon Kimliği": p.position_id,
                    "Parite": p.symbol,
                    "Miktar": f"{p.quantity:.8f}",
                    f"Giriş Fiyatı ({selected_quote})": f"{p.entry_price:.6f}",
                    f"Maliyet ({selected_quote})": f"{p.cost_basis_usdc:.2f}",
                    "Giriş Komisyonu": f"{p.entry_fee_usdc:.4f}",
                    "Zarar-Kes": f"{p.stop_loss_price:.6f}" if p.stop_loss_price else "-",
                    "Kâr-Al": f"{p.take_profit_price:.6f}" if p.take_profit_price else "-",
                    "Açılış Zamanı": p.opened_at_iso,
                }
                for p in open_pos
            ]
            st.dataframe(pd.DataFrame(pos_rows), width="stretch", hide_index=True)
        else:
            st.info("Şu anda açık pozisyon bulunmuyor.")

        st.subheader("Portföy Bakiye Eğrisi")
        snaps = storage.get_equity_snapshots(limit=300)
        if snaps:
            snap_df = pd.DataFrame(snaps)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=snap_df["created_at_iso"],
                    y=snap_df["main_equity_usdc"].astype(float),
                    mode="lines+markers",
                    name=f"Aktif Portföy ({selected_quote})",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=snap_df["created_at_iso"],
                    y=snap_df["rule_shadow_equity_usdc"].astype(float),
                    mode="lines",
                    name=f"Kural Tabanlı Gölge ({selected_quote})",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=snap_df["created_at_iso"],
                    y=snap_df["benchmark_equity_usdc"].astype(float),
                    mode="lines",
                    line={"dash": "dot"},
                    name=f"Pasif Referans ({selected_quote})",
                )
            )
            fig.update_layout(
                height=340,
                margin={"l": 20, "r": 20, "t": 20, "b": 20},
                yaxis_title=f"Toplam Değer ({selected_quote})",
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
        shadow_sum = engine.get_portfolio_summary(settings, track="RULE_SHADOW")
        snaps_latest = storage.get_equity_snapshots(limit=1)
        bench_val = (
            Decimal(str(snaps_latest[-1]["benchmark_equity_usdc"]))
            if snaps_latest
            else cfg.initial_balance_usdc
        )

        cmp_df = pd.DataFrame(
            [
                {
                    "Yaklaşım": "1. Aktif Portföy (Yapay Zekâ veya Seçili Mod)",
                    f"Toplam Değer ({selected_quote})": f"{summary['total_equity_usdc']:.3f}",
                    f"Net Fark ({selected_quote})": f"{(summary['total_equity_usdc'] - cfg.initial_balance_usdc):+.3f}",
                    f"Toplam Komisyon ({selected_quote})": f"{summary['total_fees_usdc']:.4f}",
                    "Tahmini AI Maliyeti (USD)": f"{ai_usage['estimated_cost_usd']:.5f}",
                },
                {
                    "Yaklaşım": "2. Saf Kural Tabanlı Strateji (EMA/RSI/MACD Gölge)",
                    f"Toplam Değer ({selected_quote})": f"{shadow_sum['total_equity_usdc']:.3f}",
                    f"Net Fark ({selected_quote})": f"{(shadow_sum['total_equity_usdc'] - cfg.initial_balance_usdc):+.3f}",
                    f"Toplam Komisyon ({selected_quote})": f"{shadow_sum['total_fees_usdc']:.4f}",
                    "Tahmini AI Maliyeti (USD)": "0.00000",
                },
                {
                    "Yaklaşım": f"3. Referans ({min(settings.max_position_size_usdc, cfg.initial_balance_usdc):.0f} {selected_quote} {selected_symbol} Al-Tut + Kalan Nakit)",
                    f"Toplam Değer ({selected_quote})": f"{bench_val:.3f}",
                    f"Net Fark ({selected_quote})": f"{(bench_val - cfg.initial_balance_usdc):+.3f}",
                    f"Toplam Komisyon ({selected_quote})": f"{(min(settings.max_position_size_usdc, cfg.initial_balance_usdc) * settings.estimated_fee_rate):.4f}",
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
        st.caption(
            f"Tahmini AI Maliyeti Düşülmüş Yapay Zekâ Ek Katkısı (vs. Kural Tabanlı): `{ai_alpha:+.4f} {selected_quote}` "
            "(Kısa süreli birkaç işlem istatistiksel kanıt oluşturmaz.)"
        )

    with tab_val:
        st.subheader(f"{cfg.initial_balance_usdc:.2f} {selected_quote} Küçük Bakiye Komisyon ve Model Maliyet Uygulanabilirliği")
        viability = assess_small_balance_viability(
            balance_usdc=cfg.initial_balance_usdc,
            position_size_usdc=settings.max_position_size_usdc,
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
            "açılışında spread + kayma + komisyon eklenerek test edilir."
        )
        bcol1, bcol2 = st.columns(2)
        run_live_bt = bcol1.button("🌐 Canlı Binance Geçmiş Mumlarıyla Doğrula (500 Mum)", width="stretch")
        run_demo_bt = bcol2.button(
            "🧪 Çevrimdışı Örnek Seri ile Doğrulama Motorunu Test Et",
            width="stretch",
            help="İnternet/İSS engeli varken doğrulama motorunun nasıl çalıştığını gösterir; portföye dokunmaz.",
        )

        if run_live_bt or run_demo_bt:
            candles_for_bt: list[Candle] = []
            bt_label = ""
            if run_live_bt:
                try:
                    bundle = engine.market_client.fetch_verified_market_bundle(
                        symbol=selected_symbol,
                        interval=selected_interval,
                        candle_limit=500,
                        allow_stablecoin_conversion=settings.allow_stablecoin_conversion,
                    )
                    candles_for_bt = bundle.closed_candles
                    bt_label = f"{selected_symbol} ({selected_interval}) — Gerçek Binance Verisi"
                except MarketDataError as exc:
                    st.error(f"Canlı geçmiş mum verisi alınamadı: {exc}")
            else:
                candles_for_bt = _generate_offline_demo_candles(240)
                bt_label = "Çevrimdışı Sentetik Doğrulama Serisi (Portföyden Bağımsız)"

            if candles_for_bt:
                rep = run_chronological_validation(
                    candles_for_bt,
                    symbol=selected_symbol,
                    interval=selected_interval,
                    settings=settings,
                    initial_balance_usdc=cfg.initial_balance_usdc,
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
                if rep.insufficient_data:
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
                    f"Pozisyon Üst Sınırı ({selected_quote})",
                    min_value=5.0,
                    max_value=float(cfg.initial_balance_usdc),
                    value=float(settings.max_position_size_usdc),
                    step=1.0,
                    help="Minimum borsa emir tutarı bu sınırı aşarsa sınır otomatik büyütülmez; işlem reddedilir.",
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
                    max_value=20,
                    value=int(settings.max_daily_new_positions),
                    step=1,
                )
                new_max_open = st.number_input(
                    "Aynı Anda En Fazla Açık Pozisyon",
                    min_value=1,
                    max_value=3,
                    value=int(settings.max_open_positions),
                    step=1,
                )
            with fc2:
                new_fee_pct = st.number_input(
                    "Tahmini Spot Komisyon Oranı (%)",
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
                    min_value=0.5,
                    max_value=20.0,
                    value=float(settings.stop_loss_pct * Decimal("100")),
                    step=0.5,
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
                )
                storage.save_risk_settings(updated)
                st.success("Risk ayarları SQLite veritabanına kaydedildi.")
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

    # Otomatik yenileme açıksa 60 sn bekle ve yeniden çiz
    if auto_loop:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
