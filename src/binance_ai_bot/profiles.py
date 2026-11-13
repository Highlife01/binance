"""
İşlem Profilleri (Hızlı 30sn / Aktif 5dk / Temkinli 15dk).

- `HIZLI_30SN`: 30 saniyede bir kontrol eden, 1 dakikalık mumlarla sürekli/aktif alım-satım profili.
- `AKTIF`     : 30 saniyede bir kontrol eden, 5 dakikalık mumlarla çalışan orta-aktif profil.
- `TEMKINLI`  : 60 saniyede bir kontrol eden, 15 dakikalık mumlarla çalışan seçici profil.

ÖNEMLİ: Hiçbir profil şu sert güvenlik sınırlarını GEVŞETMEZ:
  * İşlem başına pozisyon üst sınırı (varsayılan 10 USDT/USDC)
  * Günlük zarar eşiği (varsayılan 2 USDT/USDC, Europe/Istanbul — aşılırsa o gün yeni alımlar durur)
  * Aynı anda en fazla 1 açık pozisyon, kaldıraç/marjin/vadeli YOK
Daha sık işlem = daha fazla komisyon demektir; kâr garantisi yoktur.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from decimal import Decimal

from .domain import RiskSettings
from .strategy import StrategyParams


@dataclass(frozen=True)
class TradingProfile:
    key: str
    label_tr: str
    description_tr: str
    interval: str
    strategy: StrategyParams
    max_daily_new_positions: int
    stop_loss_pct: Decimal
    take_profit_pct: Decimal
    ai_daily_call_limit: int
    ai_min_call_interval_seconds: int
    loop_seconds: int
    direct_rule_execution: bool = False
    max_open_positions: int = 1
    min_net_profit_exit_pct: Decimal = Decimal("0")
    max_entry_spread_pct: Decimal = Decimal("0")
    expert_mode: bool = False
    risk_per_trade_pct: Decimal = Decimal("0")
    trailing_by_leverage: tuple[tuple[int, Decimal], ...] = ()

    def trailing_stop_for(self, leverage: int) -> Decimal:
        """Kaldıraca göre hareket eden stop mesafesi (fiyat yüzdesi); tanımsızsa 0 (kapalı)."""
        eff_lev = max(1, min(5, int(leverage)))
        return dict(self.trailing_by_leverage).get(eff_lev, Decimal("0"))

    def leverage_adjusted_targets(
        self, leverage: int
    ) -> tuple[Decimal, Decimal, Decimal]:
        """
        Kaldıraç çarpanına (1x..5x) göre:
        1) Likidasyon fiyatına asla yaklaştırmayan sıkı Zarar-Kes (Stop-Loss),
        2) Kaldıraçlı tam komisyon + faiz sonrasında ana para (teminat) üzerinde yüksek net kâr
           bırakırken daha küçük fiyat dalgalarını yakalayan Kâr-Al ve Kâr Kilitleme hedefleri üretir.
        """
        eff_lev = max(1, min(5, int(leverage)))
        if eff_lev <= 1 or self.key != "AL_YUKSELINCE_SAT":
            return (self.stop_loss_pct, self.take_profit_pct, self.min_net_profit_exit_pct)

        # AL_YUKSELINCE_SAT için kaldıraç kademelerine özel mühendislik tablosu:
        # - 1x: Fiyat +%0.80 kâr kilidi / +%1.15 TP / -%3.50 SL (Teminatta ~+%0.65 / +%1.00 net kâr)
        # - 2x: Fiyat +%0.65 kâr kilidi / +%0.90 TP / -%2.20 SL (Teminatta ~+%1.00 / +%1.50 net kâr, SL -%4.4 teminat)
        # - 3x: Fiyat +%0.55 kâr kilidi / +%0.80 TP / -%1.50 SL (Teminatta ~+%1.20 / +%1.95 net kâr, SL -%4.5 teminat)
        # - 4x: Fiyat +%0.50 kâr kilidi / +%0.75 TP / -%1.15 SL (Teminatta ~+%1.40 / +%2.40 net kâr, SL -%4.6 teminat)
        # - 5x: Fiyat +%0.48 kâr kilidi / +%0.70 TP / -%0.95 SL (Teminatta ~+%1.65 / +%2.75 net kâr, SL -%4.75 teminat; likidasyon -%12'den çok uzak)
        table: dict[int, tuple[Decimal, Decimal, Decimal]] = {
            2: (Decimal("0.0220"), Decimal("0.0090"), Decimal("0.0065")),
            3: (Decimal("0.0150"), Decimal("0.0080"), Decimal("0.0055")),
            4: (Decimal("0.0115"), Decimal("0.0075"), Decimal("0.0050")),
            5: (Decimal("0.0095"), Decimal("0.0070"), Decimal("0.0048")),
        }
        return table.get(eff_lev, (self.stop_loss_pct, self.take_profit_pct, self.min_net_profit_exit_pct))

    def apply_to_risk_settings(
        self, base: RiskSettings, *, leverage: int | None = None
    ) -> RiskSettings:
        """
        Profilin işlem sıklığı ve kaldıraç uyumlu çıkış seviyelerini uygular; pozisyon büyüklüğü ve
        günlük zarar eşiği gibi sert sınırlar kullanıcının kayıtlı ayarlarından AYNEN korunur.
        """
        base_lev = getattr(base, "leverage", 1)
        eff_lev = max(1, min(5, int(leverage if leverage is not None else base_lev)))
        adj_sl, adj_tp, adj_min_exit = self.leverage_adjusted_targets(eff_lev)
        return RiskSettings(
            max_open_positions=max(base.max_open_positions, self.max_open_positions),
            max_position_size_usdc=base.max_position_size_usdc,
            daily_loss_limit_usdc=base.daily_loss_limit_usdc,
            max_daily_new_positions=self.max_daily_new_positions,
            daily_limit_exit_policy=base.daily_limit_exit_policy,
            estimated_fee_rate=base.estimated_fee_rate,
            extra_slippage_bps=base.extra_slippage_bps,
            stop_loss_pct=adj_sl,
            take_profit_pct=adj_tp,
            allow_stablecoin_conversion=False,
            leverage=eff_lev,
            hourly_margin_interest_rate=getattr(
                base, "hourly_margin_interest_rate", Decimal("0.000005")
            ),
            min_net_profit_exit_pct=adj_min_exit,
            trailing_stop_pct=self.trailing_stop_for(eff_lev),
            max_entry_spread_pct=self.max_entry_spread_pct,
            expert_mode=self.expert_mode,
            risk_per_trade_pct=self.risk_per_trade_pct,
        )


TRADING_PROFILES: dict[str, TradingProfile] = {
    "AL_YUKSELINCE_SAT": TradingProfile(
        key="AL_YUKSELINCE_SAT",
        label_tr="🚀 Al & Kârı Görünce Sat (Sürekli Döngü — 1dk Mum)",
        description_tr=(
            "10 coini sürekli tarar; dip dönüşü ve yükseliş başlangıcında alır, ufak geri çekilmelerde "
            "zararına erken satış yapmaz, komisyon sonrası net kârı görünce (+%0.80–%1.15 brüt / "
            "+%0.65–%1.00 net) anında satıp nakde döner ve hemen yeni alım fırsatı arar "
            "(%3.5 acil çöküş koruması + günlük zarar sigortası aktiftir)."
        ),
        interval="1m",
        strategy=StrategyParams(
            ema_fast_span=5,
            ema_slow_span=13,
            rsi_buy_min=26.0,
            rsi_buy_max=68.0,
            rsi_overbought_exit=69.0,
            min_volume_ratio=0.12,
            require_close_above_fast=False,
            require_rising_momentum=False,
            fast_exit=True,
            dip_rebound_buy=True,
        ),
        max_daily_new_positions=120,
        stop_loss_pct=Decimal("0.035"),
        take_profit_pct=Decimal("0.0115"),
        ai_daily_call_limit=400,
        ai_min_call_interval_seconds=25,
        loop_seconds=20,
        direct_rule_execution=True,
        max_open_positions=5,
        min_net_profit_exit_pct=Decimal("0.0080"),
        max_entry_spread_pct=Decimal("0.0012"),
        expert_mode=True,
        risk_per_trade_pct=Decimal("0.02"),
        trailing_by_leverage=(
            (1, Decimal("0.0040")),
            (2, Decimal("0.0030")),
            (3, Decimal("0.0025")),
            (4, Decimal("0.0022")),
            (5, Decimal("0.0020")),
        ),
    ),
    "HIZLI_30SN": TradingProfile(
        key="HIZLI_30SN",
        label_tr="🔥 Hızlı Aktif (30 sn Döngü — 1dk Mum — Çoklu Coin)",
        description_tr=(
            "Her 30 saniyede bir 10 coini tarar; 1 dakikalık mumlarda hızlı EMA(5/13) ile aynı anda "
            "5 farklı coine kadar (her biri en fazla 10 USDT/USDC) aktif alım-satım yapar "
            "(%0.35 zarar-kes / %0.45 kâr-al + hızlı trend çıkışı). Günlük 2.00 USDT/USDC zarar sigortası aktiftir."
        ),
        interval="1m",
        strategy=StrategyParams(
            ema_fast_span=5,
            ema_slow_span=13,
            rsi_buy_min=36.0,
            rsi_buy_max=76.0,
            rsi_overbought_exit=78.0,
            min_volume_ratio=0.15,
            require_close_above_fast=False,
            require_rising_momentum=False,
            fast_exit=True,
        ),
        max_daily_new_positions=60,
        stop_loss_pct=Decimal("0.0035"),
        take_profit_pct=Decimal("0.0045"),
        ai_daily_call_limit=400,
        ai_min_call_interval_seconds=25,
        loop_seconds=30,
        direct_rule_execution=True,
        max_open_positions=5,
    ),
    "AKTIF": TradingProfile(
        key="AKTIF",
        label_tr="⚡ Aktif (30 sn Döngü — 5dk Mum)",
        description_tr=(
            "5 dakikalık kapanmış mumlar, gevşetilmiş hacim (0.5x) ve RSI (40-70) eşikleri, "
            "aynı anda 3 farklı coine kadar, günde en fazla 12 yeni pozisyon, %0.8 zarar-kes / %1.2 kâr-al."
        ),
        interval="5m",
        strategy=StrategyParams(
            rsi_buy_min=40.0,
            rsi_buy_max=70.0,
            rsi_overbought_exit=76.0,
            min_volume_ratio=0.5,
        ),
        max_daily_new_positions=12,
        stop_loss_pct=Decimal("0.008"),
        take_profit_pct=Decimal("0.012"),
        ai_daily_call_limit=300,
        ai_min_call_interval_seconds=120,
        loop_seconds=30,
        direct_rule_execution=False,
        max_open_positions=3,
    ),
    "TEMKINLI": TradingProfile(
        key="TEMKINLI",
        label_tr="🛡️ Temkinli (60 sn Döngü — 15dk Mum)",
        description_tr=(
            "15 dakikalık kapanmış mumlar, sıkı hacim teyidi, günde en fazla 3 yeni pozisyon, "
            "%2.5 zarar-kes / %4.5 kâr-al."
        ),
        interval="15m",
        strategy=StrategyParams(),
        max_daily_new_positions=3,
        stop_loss_pct=Decimal("0.025"),
        take_profit_pct=Decimal("0.045"),
        ai_daily_call_limit=25,
        ai_min_call_interval_seconds=180,
        loop_seconds=60,
        direct_rule_execution=False,
    ),
}


def get_profile(key: str | None) -> TradingProfile:
    norm = (key or "HIZLI_30SN").strip().upper()
    return TRADING_PROFILES.get(norm, TRADING_PROFILES["HIZLI_30SN"])
