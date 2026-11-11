"""
İşlem Profilleri (Temkinli / Aktif).

- `TEMKINLI`: Orijinal, düşük işlem sıklıklı varsayılan profil (15m mum, günde en fazla 3 pozisyon).
- `AKTIF`  : "Sürekli alım-satım" profili (5m mum, daha gevşek hacim/RSI eşikleri, daha dar
             zarar-kes/kâr-al, günde en fazla 12 pozisyon).

ÖNEMLİ: Hiçbir profil şu sert güvenlik sınırlarını GEVŞETMEZ:
  * İşlem başına pozisyon üst sınırı (varsayılan 10 USDT/USDC)
  * Günlük zarar eşiği (varsayılan 2 USDT/USDC, Europe/Istanbul)
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
    "TEMKINLI": TradingProfile(
        key="TEMKINLI",
        label_tr="🛡️ Temkinli (Az İşlem — 15dk)",
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
    ),
    "AKTIF": TradingProfile(
        key="AKTIF",
        label_tr="⚡ Aktif (Sürekli Alım-Satım — 5dk)",
        description_tr=(
            "5 dakikalık kapanmış mumlar, gevşetilmiş hacim (0.5x) ve RSI (40-70) eşikleri, "
            "günde en fazla 12 yeni pozisyon, %0.8 zarar-kes / %1.2 kâr-al. "
            "Daha fazla işlem = daha fazla komisyon; önce Sanal Deneme önerilir."
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
    ),
}


def get_profile(key: str | None) -> TradingProfile:
    norm = (key or "TEMKINLI").strip().upper()
    return TRADING_PROFILES.get(norm, TRADING_PROFILES["TEMKINLI"])
