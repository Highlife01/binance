"""
Bağımsız Risk Denetim Modülü.

Not: Buradaki varsayılan sınırlar (86 USDC başlangıç, en fazla 1 açık pozisyon,
10 USDC pozisyon üst sınırı, 2 USDC günlük zarar eşiği, günde en fazla 3 yeni pozisyon)
yalnızca değiştirilebilir sanal test varsayımlarıdır; yatırım tavsiyesi değildir.
Fiyat boşluğu (gap), düşük likidite veya bağlantı/uyku kesintilerinde risk sınırları
kesin kayıp garantisi sağlayamaz.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from .domain import (
    OrderBookSnapshot,
    Position,
    RiskCheckResult,
    RiskSettings,
    SymbolFilters,
    quantize_down,
)
from .security import STABLECOIN_OR_FIAT_ASSETS
from .storage import SQLiteStorage

RISK_DISCLAIMER_TR = (
    "Uyarı: Buradaki risk sınırları sanal deneme varsayımlarıdır, yatırım tavsiyesi değildir. "
    "Ani fiyat boşluğu (gap), düşük emir defteri likiditesi, yüksek makas veya bilgisayar/internet "
    "kesintilerinde gerçek piyasalarda kayıplar belirlenen eşikleri aşabilir."
)

EXIT_POLICY_DESCRIPTIONS_TR: dict[str, str] = {
    "HOLD_EXISTING": (
        "Yeni Pozisyonları Durdur / Mevcut Pozisyonu Koru: Günlük zarar eşiği aşıldığında "
        "yeni alımlar anında kilitlenir; açık pozisyon ise tanımlı sanal zarar-kes/kâr-al "
        "seviyesinden veya paneldeki 'Sanal Pozisyonu Kapat' düğmesiyle kapatılır."
    ),
    "AUTO_CLOSE": (
        "Yeni Pozisyonları Durdur ve Açık Pozisyonu Kapat: Günlük zarar eşiği aşıldığında "
        "yeni alımlar kilitlenir ve mevcut açık sanal pozisyon o anki alış kotasyonundan "
        "otomatik kapatılır."
    ),
}


def compute_open_position_unrealized_net_pnl(
    pos: Position,
    current_bid: Decimal,
    estimated_fee_rate: Decimal,
) -> tuple[Decimal, Decimal]:
    """
    Açık pozisyonun:
    1) Anlık brüt piyasa değerini (`quantity * current_bid`),
    2) Giriş komisyonu ve tahmini çıkış komisyonu düşülmüş net gerçekleşmemiş kâr/zararını döndürür.
    """
    gross_market_val = pos.quantity * current_bid
    est_exit_fee = gross_market_val * estimated_fee_rate
    net_liquidation_val = gross_market_val - est_exit_fee
    total_entry_outflow = pos.cost_basis_usdc + pos.entry_fee_usdc
    unrealized_net_pnl = net_liquidation_val - total_entry_outflow
    return gross_market_val, unrealized_net_pnl


class RiskManager:
    """
    Alım kararlarını bakiye, pozisyon sınırı, günlük zarar eşiği (`Europe/Istanbul`),
    günlük işlem adedi ve Binance `exchangeInfo` emir filtrelerine göre denetler.
    """

    def __init__(self, storage: SQLiteStorage) -> None:
        self.storage = storage

    def calculate_daily_net_pnl_usdc(
        self,
        *,
        settings: RiskSettings,
        open_positions: list[Position],
        bid_prices_by_symbol: dict[str, Decimal],
        now_utc: datetime | None = None,
        track: str = "MAIN",
    ) -> Decimal:
        """
        Europe/Istanbul gün sınırına göre:
        - Bugün kapanmış pozisyonların net gerçekleşmiş kâr/zararı (tüm komisyonlar dahil)
        + Açık pozisyonların anlık net değer değişimi (giriş ve tahmini çıkış komisyonu dahil)
        birlikte değerlendirilir.
        """
        ist_date = self.storage.istanbul_date_str(now_utc)
        realized_today, _ = self.storage.get_daily_realized_pnl_and_entry_fees(ist_date, track=track)

        unrealized_total = Decimal("0")
        for pos in open_positions:
            bid = bid_prices_by_symbol.get(pos.symbol, pos.entry_price)
            _, unreal_net = compute_open_position_unrealized_net_pnl(
                pos, current_bid=bid, estimated_fee_rate=settings.estimated_fee_rate
            )
            unrealized_total += unreal_net

        return realized_today + unrealized_total

    def evaluate_daily_loss_guard(
        self,
        *,
        settings: RiskSettings,
        open_positions: list[Position],
        bid_prices_by_symbol: dict[str, Decimal],
        now_utc: datetime | None = None,
        track: str = "MAIN",
    ) -> tuple[bool, Decimal, bool]:
        """
        Günlük zarar eşiği aşıldıysa:
        (limit_asildi_mi, gunluk_net_zarar_usdc, acik_pozisyonu_otomatik_kapat_mi) döndürür.
        """
        daily_net_pnl = self.calculate_daily_net_pnl_usdc(
            settings=settings,
            open_positions=open_positions,
            bid_prices_by_symbol=bid_prices_by_symbol,
            now_utc=now_utc,
            track=track,
        )
        daily_loss = max(Decimal("0"), -daily_net_pnl)
        breached = daily_loss >= settings.daily_loss_limit_usdc
        auto_close = breached and settings.daily_limit_exit_policy == "AUTO_CLOSE" and bool(open_positions)
        return breached, daily_loss, auto_close

    def check_buy_entry(
        self,
        *,
        filters: SymbolFilters,
        order_book: OrderBookSnapshot,
        settings: RiskSettings,
        order_type: str = "MARKET",
        limit_price: Decimal | None = None,
        now_utc: datetime | None = None,
        track: str = "MAIN",
    ) -> RiskCheckResult:
        """
        Yeni bir Spot ALIM emrini tüm risk ve borsa filtrelerinden geçirir.
        Minimum emir tutarı sınırlarla çelişirse sınırları ASLA otomatik büyütmez; işlemi reddeder.
        """
        ref_now = now_utc or datetime.now(timezone.utc)
        ist_date = self.storage.istanbul_date_str(ref_now)

        # 0. Yalnızca aktif Spot USDC çiftleri ve izinsiz stabilcoin dönüşüm engeli
        if not filters.is_tradable_usdc_spot:
            return RiskCheckResult(
                allowed=False,
                reason=f"{filters.symbol} aktif bir Spot USDC çifti değil (durum: {filters.status}).",
            )

        if not settings.allow_stablecoin_conversion and filters.base_asset in STABLECOIN_OR_FIAT_ASSETS:
            return RiskCheckResult(
                allowed=False,
                reason=f"{filters.symbol} stabilcoin/itibari dönüşüm çiftidir; USDC izinsiz dönüştürülmez.",
            )

        # 1. Kullanıcı veya sistem duraklatması (yalnızca ana portföy için kullanıcı kilidi)
        if track == "MAIN":
            paused, pause_reason = self.storage.is_new_entries_paused()
            if paused:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Yeni pozisyon açılışı durdurulmuş durumda ({pause_reason or 'Kullanıcı/Risk kilidi'}).",
                )

        open_positions = self.storage.get_open_positions(track=track)
        best_bid = order_book.best_bid
        best_ask = order_book.best_ask
        if best_bid is None or best_ask is None or best_ask <= 0:
            return RiskCheckResult(
                allowed=False,
                reason="Geçerli alış/satış kotasyonu bulunmadığından işlem reddedildi.",
            )

        # 2. Günlük zarar eşiği kontrolü (Europe/Istanbul)
        bid_map = {pos.symbol: (best_bid if pos.symbol == filters.symbol else pos.entry_price) for pos in open_positions}
        breached, daily_loss, auto_close = self.evaluate_daily_loss_guard(
            settings=settings,
            open_positions=open_positions,
            bid_prices_by_symbol=bid_map,
            now_utc=ref_now,
            track=track,
        )
        if breached:
            msg = (
                f"Günlük zarar eşiği ({settings.daily_loss_limit_usdc:.2f} USDC) aşıldı "
                f"(güncel günlük kayıp: {daily_loss:.2f} USDC, Europe/Istanbul). "
                "Yeni pozisyon açılışı durduruldu."
            )
            if track == "MAIN":
                self.storage.set_new_entries_paused(True, msg)
            return RiskCheckResult(
                allowed=False,
                reason=msg,
                daily_total_loss_usdc=daily_loss,
                trigger_daily_halt=True,
                trigger_protective_close=auto_close,
            )

        # 3. Zararda ek alım / Martingale engeli ve maksimum açık pozisyon sınırı
        for pos in open_positions:
            if pos.symbol == filters.symbol:
                return RiskCheckResult(
                    allowed=False,
                    reason=(
                        f"{filters.symbol} paritesinde zaten açık pozisyon var. "
                        "Zararda ortalama düşürme ve Martingale kesinlikle kapalıdır."
                    ),
                    daily_total_loss_usdc=daily_loss,
                )

        if len(open_positions) >= settings.max_open_positions:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"Aynı anda en fazla {settings.max_open_positions} açık pozisyon sınırına ulaşıldı."
                ),
                daily_total_loss_usdc=daily_loss,
            )

        # 4. Günlük yeni pozisyon adedi sınırı
        daily_count = self.storage.count_daily_new_positions(ist_date, track=track)
        if daily_count >= settings.max_daily_new_positions:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"Bugün için ({ist_date}, İstanbul saati) en fazla {settings.max_daily_new_positions} "
                    f"yeni pozisyon sınırına ulaşıldı ({daily_count}/{settings.max_daily_new_positions})."
                ),
                daily_total_loss_usdc=daily_loss,
            )

        # 5. Kaldıraçsız nakit bakiye ve pozisyon üst sınırı hesabı
        cash_usdc = self.storage.get_cash_balance(track=track)
        if cash_usdc <= Decimal("0"):
            return RiskCheckResult(
                allowed=False,
                reason="Kullanılabilir sanal USDC bakiyesi yetersiz (0.00 USDC).",
                daily_total_loss_usdc=daily_loss,
            )

        # Komisyon ve ekstra kayma payı dahil nakit bütçe asla aşılmaz, sınır büyütülmez
        slip_factor = Decimal("1") + (settings.extra_slippage_bps / Decimal("10000"))
        fee_factor = Decimal("1") + settings.estimated_fee_rate
        max_affordable_gross = cash_usdc / fee_factor
        target_gross_usdc = min(settings.max_position_size_usdc, max_affordable_gross)

        # Binance minimum notional ile çelişki kontrolü (sınırları ASLA otomatik büyütme!)
        if filters.min_notional > settings.max_position_size_usdc:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"{filters.symbol} minimum emir tutarı ({filters.min_notional:.2f} USDC), "
                    f"belirlenen pozisyon üst sınırını ({settings.max_position_size_usdc:.2f} USDC) aşıyor. "
                    "Risk sınırı otomatik büyütülmedi; işlem açılmadı."
                ),
                daily_total_loss_usdc=daily_loss,
            )

        if filters.min_notional > max_affordable_gross:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"Kullanılabilir bakiye ({cash_usdc:.2f} USDC, komisyon sonrası en çok "
                    f"{max_affordable_gross:.2f} USDC), {filters.symbol} minimum emir tutarını "
                    f"({filters.min_notional:.2f} USDC) karşılamıyor."
                ),
                daily_total_loss_usdc=daily_loss,
            )

        # Fiyat filtresi doğrulaması
        ref_price = limit_price if (order_type == "LIMIT" and limit_price is not None) else (best_ask * slip_factor)
        if order_type == "LIMIT" and limit_price is not None:
            if limit_price < filters.min_price or limit_price > filters.max_price:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Limit fiyatı ({limit_price}) PRICE_FILTER sınırları dışında.",
                    daily_total_loss_usdc=daily_loss,
                )
            quantized_price = quantize_down(limit_price, filters.tick_size)
            if quantized_price != limit_price:
                return RiskCheckResult(
                    allowed=False,
                    reason=(
                        f"Limit fiyatı ({limit_price}), paritenin fiyat adımına "
                        f"(tickSize={filters.tick_size}) tam uymuyor."
                    ),
                    daily_total_loss_usdc=daily_loss,
                )

        # Miktar adımı (stepSize) ve minimum miktar (minQty) kontrolü
        eff_step = filters.market_step_size if order_type == "MARKET" else filters.step_size
        eff_min_qty = max(filters.min_qty, filters.market_min_qty) if order_type == "MARKET" else filters.min_qty
        eff_max_qty = min(filters.max_qty, filters.market_max_qty) if order_type == "MARKET" else filters.max_qty

        raw_qty = target_gross_usdc / ref_price
        qty = quantize_down(raw_qty, eff_step)

        if qty < eff_min_qty or qty <= Decimal("0"):
            min_needed_usdc = eff_min_qty * ref_price
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"Miktar adımı ({eff_step}) uygulandığında hesaplanan miktar ({qty}), "
                    f"minimum emir miktarının ({eff_min_qty}, yaklaşık {min_needed_usdc:.2f} USDC) "
                    "altında kaldı; sınır büyütülmeden işlem reddedildi."
                ),
                daily_total_loss_usdc=daily_loss,
            )

        if qty > eff_max_qty:
            qty = quantize_down(eff_max_qty, eff_step)

        actual_notional = qty * best_ask if order_type == "MARKET" else qty * ref_price
        if actual_notional < filters.min_notional:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    f"Miktar adımı yuvarlaması sonrası emir tutarı ({actual_notional:.2f} USDC), "
                    f"Binance minimum emir tutarının ({filters.min_notional:.2f} USDC) altına düştü. "
                    "Pozisyon sınırı aşılmaması için işlem açılmadı."
                ),
                daily_total_loss_usdc=daily_loss,
            )

        if actual_notional > filters.max_notional:
            return RiskCheckResult(
                allowed=False,
                reason=f"Emir tutarı maksimum notional sınırını ({filters.max_notional}) aşıyor.",
                daily_total_loss_usdc=daily_loss,
            )

        return RiskCheckResult(
            allowed=True,
            reason="Tüm risk ve Binance Spot filtre kontrollerinden geçti.",
            approved_notional_usdc=actual_notional,
            approved_quantity=qty,
            daily_total_loss_usdc=daily_loss,
        )
