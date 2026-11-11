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

    def apply_to_risk_settings(self, base: RiskSettings) -> RiskSettings:
        """
        Profilin işlem sıklığı ve çıkış seviyelerini uygular; pozisyon büyüklüğü ve günlük zarar
        eşiği gibi sert sınırlar kullanıcının kayıtlı ayarlarından AYNEN korunur.
        """
        return dataclasses.replace(
            base,
            max_open_positions=1,
            max_daily_new_positions=self.max_daily_new_positions,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            allow_stablecoin_conversion=False,
        )


TRADING_PROFILES: dict[str, TradingProfile] = {
    "HIZLI_30SN": TradingProfile(
        key="HIZLI_30SN",
        label_tr="🔥 Hızlı Aktif (30 sn Döngü — 1dk Mum)",
        description_tr=(
            "Her 30 saniyede bir kontrol eder; 1 dakikalık mumlarda hızlı EMA(5/13) ve gevşetilmiş "
            "hacim (0.15x) ile aktif alım-satım yapar (%0.35 zarar-kes / %0.45 kâr-al + hızlı trend çıkışı). "
            "Günlük 2.00 USDT/USDC zarar koruması ve 10 USDT/USDC işlem üst sınırı aynen aktiftir."
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
    ),
    "AKTIF": TradingProfile(
        key="AKTIF",
        label_tr="⚡ Aktif (30 sn Döngü — 5dk Mum)",
        description_tr=(
            "5 dakikalık kapanmış mumlar, gevşetilmiş hacim (0.5x) ve RSI (40-70) eşikleri, "
            "günde en fazla 12 yeni pozisyon, %0.8 zarar-kes / %1.2 kâr-al."
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
