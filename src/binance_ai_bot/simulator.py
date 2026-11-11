"""
Gerçekçi Sanal İşlem Motoru (Paper Trading Execution Simulator).

Özellikler:
- Gerçek Binance hesabıyla hiçbir bağlantısı yoktur; gerçek emir göndermez.
- Alışta satış kotasyonlarını (`asks`), satışta alış kotasyonlarını (`bids`) ve
  emir defterindeki gerçek kademe likiditesini yürüyerek (book-walking) tüketir.
- Komisyon, spread ve fiyat kaymasını (slippage) ayrı ayrı hesaplar.
- Limit emirlerde fiyat seviyesine yalnızca dokunuldu (`ask == limit` veya `bid == limit`)
  diye %100 gerçekleşmiş SAYMAZ; kuyruk önceliğini temsil eden muhafazakâr kısmi dolum
  (`TOUCH_QUEUE_FILL_RATIO = %40`) veya gerçekleşmeme (`EXPIRED_UNFILLED`) uygular.
- Kısmi dolum (`PARTIALLY_FILLED`) ve gerçekleşmeyen (`EXPIRED_UNFILLED`) emirleri destekler.
- Güncel emir defteri verisi yokken geçmişe dönük hayalî dolum oluşturmaz.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from .domain import (
    OrderBookLevel,
    OrderBookSnapshot,
    OrderType,
    Position,
    RiskSettings,
    SimulatedOrderResult,
    SymbolFilters,
    quantize_down,
)
from .storage import SQLiteStorage

# Limit emirlerde fiyata yalnızca dokunulduğunda (delip geçmediğinde) varsayılan kuyruk dolum payı
TOUCH_QUEUE_FILL_RATIO = Decimal("0.40")

FILL_MODEL_EXPLANATION_TR = (
    "Dolum Modeli: Piyasa alımlarında satış defteri (asks), satışlarda alış defteri (bids) kademe "
    "likiditesine göre tüketilir ve üstüne muhafazakâr kayma tamponu eklenir. Limit emirlerde fiyat "
    "seviyesine yalnızca dokunulması tam dolum sayılmaz: Seviyeyi delip geçen kademeler %100, tam "
    "sınır fiyatındaki kademe ise kuyruk önceliği varsayımıyla en fazla %40 oranında (kısmi dolum) "
    "eşleştirilir; karşı likidite veya minimum tutar sağlanamazsa emir gerçekleşmeden kapanır."
)


def _new_order_id() -> str:
    return f"SIM-ORD-{uuid.uuid4().hex[:12].upper()}"


def _new_position_id() -> str:
    return f"SIM-POS-{uuid.uuid4().hex[:12].upper()}"


def _simulate_book_walk(
    levels: list[OrderBookLevel],
    requested_qty: Decimal,
    *,
    side: str,
    order_type: OrderType,
    limit_price: Decimal | None,
    step_size: Decimal,
) -> tuple[Decimal, Decimal]:
    """
    Emir defteri kademelerini yürüyerek `(filled_qty, gross_quote_usdc)` hesaplar.
    """
    remaining = requested_qty
    accumulated_qty = Decimal("0")
    accumulated_quote = Decimal("0")

    for lvl in levels:
        if remaining <= Decimal("0"):
            break

        if order_type == "LIMIT" and limit_price is not None:
            if side == "BUY":
                if lvl.price > limit_price:
                    break
                # Fiyat yalnızca dokunduysa (%40 kuyruk payı), delip geçtiyse (%100)
                avail = (
                    lvl.quantity * TOUCH_QUEUE_FILL_RATIO
                    if lvl.price == limit_price
                    else lvl.quantity
                )
            else:
                if lvl.price < limit_price:
                    break
                avail = (
                    lvl.quantity * TOUCH_QUEUE_FILL_RATIO
                    if lvl.price == limit_price
                    else lvl.quantity
                )
        else:
            avail = lvl.quantity

        take = min(remaining, avail)
        if take <= Decimal("0"):
            continue
        accumulated_qty += take
        accumulated_quote += take * lvl.price
        remaining -= take

    quantized_filled = quantize_down(accumulated_qty, step_size)
    if quantized_filled <= Decimal("0") or accumulated_qty <= Decimal("0"):
        return Decimal("0"), Decimal("0")

    avg_raw_price = accumulated_quote / accumulated_qty
    return quantized_filled, quantized_filled * avg_raw_price


class VirtualExecutionSimulator:
    """SQLite kalıcılıklı, gerçekçi maliyet ve likidite duyarlı sanal işlem yürütücüsü."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self.storage = storage

    def execute_buy(
        self,
        *,
        filters: SymbolFilters,
        order_book: OrderBookSnapshot | None,
        requested_qty: Decimal,
        signal_key: str,
        strategy_source: str,
        settings: RiskSettings,
        order_type: OrderType = "MARKET",
        limit_price: Decimal | None = None,
        now_utc: datetime | None = None,
        track: str = "MAIN",
    ) -> SimulatedOrderResult:
        """Spot sanal ALIM emrini yürütür ve SQLite'a kaydeder."""
        ref_now = now_utc or datetime.now(timezone.utc)
        now_iso = ref_now.isoformat()
        ist_date = self.storage.istanbul_date_str(ref_now)
        order_id = _new_order_id()

        # Güncel veri yokken hayalî dolum oluşturma!
        if order_book is None or not order_book.asks or order_book.best_ask is None:
            rejected = SimulatedOrderResult(
                order_id=order_id,
                symbol=filters.symbol,
                side="BUY",
                order_type=order_type,
                status="REJECTED",
                requested_qty=requested_qty,
                filled_qty=Decimal("0"),
                avg_fill_price=Decimal("0"),
                gross_quote_usdc=Decimal("0"),
                fee_usdc=Decimal("0"),
                slippage_bps=Decimal("0"),
                signal_key=signal_key,
                strategy_source=strategy_source,
                explanation="Güncel emir defteri verisi bulunmadığından hayalî dolum oluşturulmadı.",
                created_at_iso=now_iso,
            )
            self.storage.record_order(rejected, ist_date, track=track)
            return rejected

        # Mükerrer sinyal kilidi: Aynı sinyal anahtarından ikinci kez emir üretilmez
        if self.storage.has_processed_signal(signal_key, track=track):
            rejected = SimulatedOrderResult(
                order_id=order_id,
                symbol=filters.symbol,
                side="BUY",
                order_type=order_type,
                status="REJECTED",
                requested_qty=requested_qty,
                filled_qty=Decimal("0"),
                avg_fill_price=Decimal("0"),
                gross_quote_usdc=Decimal("0"),
                fee_usdc=Decimal("0"),
                slippage_bps=Decimal("0"),
                signal_key=signal_key,
                strategy_source=strategy_source,
                explanation=f"Mükerrer sinyal engellendi ({signal_key} daha önce işlendi).",
                created_at_iso=now_iso,
            )
            return rejected

        eff_step = filters.market_step_size if order_type == "MARKET" else filters.step_size
        eff_min_qty = max(filters.min_qty, filters.market_min_qty) if order_type == "MARKET" else filters.min_qty

        filled_qty, raw_quote_usdc = _simulate_book_walk(
            order_book.asks,
            requested_qty,
            side="BUY",
            order_type=order_type,
            limit_price=limit_price,
            step_size=eff_step,
        )

        if filled_qty < eff_min_qty or filled_qty <= Decimal("0"):
            status_unfilled = "EXPIRED_UNFILLED" if order_type == "LIMIT" else "REJECTED"
            reason_unfilled = (
                "Limit emri karşı likidite veya kuyruk önceliği nedeniyle gerçekleşmedi."
                if order_type == "LIMIT"
                else "Emir defterinde minimum miktarı karşılayacak satış likiditesi bulunamadı."
            )
            unfilled_res = SimulatedOrderResult(
                order_id=order_id,
                symbol=filters.symbol,
                side="BUY",
                order_type=order_type,
                status=status_unfilled,
                requested_qty=requested_qty,
                filled_qty=Decimal("0"),
                avg_fill_price=Decimal("0"),
                gross_quote_usdc=Decimal("0"),
                fee_usdc=Decimal("0"),
                slippage_bps=Decimal("0"),
                signal_key=signal_key,
                strategy_source=strategy_source,
                explanation=reason_unfilled,
                created_at_iso=now_iso,
            )
            self.storage.record_order(unfilled_res, ist_date, track=track)
            return unfilled_res

        avg_book_price = raw_quote_usdc / filled_qty
        # Piyasa emirlerinde muhafazakâr ekstra kayma tamponu uygula
        if order_type == "MARKET" and settings.extra_slippage_bps > 0:
            exec_price = avg_book_price * (Decimal("1") + settings.extra_slippage_bps / Decimal("10000"))
        else:
            exec_price = avg_book_price

        gross_usdc = filled_qty * exec_price
        if gross_usdc < filters.min_notional:
            unfilled_notional = SimulatedOrderResult(
                order_id=order_id,
                symbol=filters.symbol,
                side="BUY",
                order_type=order_type,
                status="EXPIRED_UNFILLED" if order_type == "LIMIT" else "REJECTED",
                requested_qty=requested_qty,
                filled_qty=Decimal("0"),
                avg_fill_price=Decimal("0"),
                gross_quote_usdc=Decimal("0"),
                fee_usdc=Decimal("0"),
                slippage_bps=Decimal("0"),
                signal_key=signal_key,
                strategy_source=strategy_source,
                explanation=(
                    f"Kısmi/mevcut likidite tutarı ({gross_usdc:.2f} USDC), minimum emir tutarının "
                    f"({filters.min_notional:.2f} USDC) altında kaldığı için emir gerçekleşmedi."
                ),
                created_at_iso=now_iso,
            )
            self.storage.record_order(unfilled_notional, ist_date, track=track)
            return unfilled_notional

        fee_usdc = gross_usdc * settings.estimated_fee_rate
        total_cash_needed = gross_usdc + fee_usdc
        cash_balance = self.storage.get_cash_balance(track=track)

        # Kayma sonrası nakit veya pozisyon üst sınırı aşılırsa miktarı 1 adım aşağı çek
        if total_cash_needed > cash_balance or gross_usdc > settings.max_position_size_usdc * Decimal("1.005"):
            max_gross = min(
                settings.max_position_size_usdc,
                cash_balance / (Decimal("1") + settings.estimated_fee_rate),
            )
            adjusted_qty = quantize_down(max_gross / exec_price, eff_step)
            if adjusted_qty < eff_min_qty or (adjusted_qty * exec_price) < filters.min_notional:
                rej = SimulatedOrderResult(
                    order_id=order_id,
                    symbol=filters.symbol,
                    side="BUY",
                    order_type=order_type,
                    status="REJECTED",
                    requested_qty=requested_qty,
                    filled_qty=Decimal("0"),
                    avg_fill_price=Decimal("0"),
                    gross_quote_usdc=Decimal("0"),
                    fee_usdc=Decimal("0"),
                    slippage_bps=Decimal("0"),
                    signal_key=signal_key,
                    strategy_source=strategy_source,
                    explanation="Kayma ve komisyon sonrası tutar sınırları karşılamadığı için işlem iptal edildi.",
                    created_at_iso=now_iso,
                )
                self.storage.record_order(rej, ist_date, track=track)
                return rej
            filled_qty = adjusted_qty
            gross_usdc = filled_qty * exec_price
            fee_usdc = gross_usdc * settings.estimated_fee_rate
            total_cash_needed = gross_usdc + fee_usdc

        best_ask = order_book.best_ask
        slippage_bps = ((exec_price - best_ask) / best_ask) * Decimal("10000") if best_ask > 0 else Decimal("0")
        order_status = "PARTIALLY_FILLED" if filled_qty < requested_qty else "FILLED"

        new_cash = cash_balance - total_cash_needed
        self.storage.set_cash_balance(new_cash, track=track)

        stop_price = exec_price * (Decimal("1") - settings.stop_loss_pct) if settings.stop_loss_pct > 0 else None
        tp_price = exec_price * (Decimal("1") + settings.take_profit_pct) if settings.take_profit_pct > 0 else None

        pos = Position(
            position_id=_new_position_id(),
            symbol=filters.symbol,
            quantity=filled_qty,
            entry_price=exec_price,
            cost_basis_usdc=gross_usdc,
            entry_fee_usdc=fee_usdc,
            opened_at_iso=now_iso,
            signal_key=signal_key,
            strategy_source=strategy_source,
            stop_loss_price=stop_price,
            take_profit_price=tp_price,
            status="OPEN",
        )
        self.storage.insert_position(pos, track=track)
        self.storage.mark_signal_processed(signal_key, order_id, now_iso, track=track)

        explanation = (
            f"Sanal ALIM ({order_status}): {filled_qty} {filters.base_asset} @ {exec_price:.6f} USDC "
            f"(Tutar: {gross_usdc:.2f} USDC, Komisyon: {fee_usdc:.4f} USDC, Kayma: {slippage_bps:.2f} bps)."
        )
        res = SimulatedOrderResult(
            order_id=order_id,
            symbol=filters.symbol,
            side="BUY",
            order_type=order_type,
            status=order_status,
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            avg_fill_price=exec_price,
            gross_quote_usdc=gross_usdc,
            fee_usdc=fee_usdc,
            slippage_bps=slippage_bps,
            signal_key=signal_key,
            strategy_source=strategy_source,
            explanation=explanation,
            created_at_iso=now_iso,
        )
        self.storage.record_order(res, ist_date, track=track)
        return res

    def execute_sell_position(
        self,
        *,
        position: Position,
        filters: SymbolFilters,
        order_book: OrderBookSnapshot | None,
        signal_key: str,
        strategy_source: str,
        close_reason: str,
        settings: RiskSettings,
        order_type: OrderType = "MARKET",
        limit_price: Decimal | None = None,
        now_utc: datetime | None = None,
        track: str = "MAIN",
    ) -> SimulatedOrderResult:
        """Açık sanal Spot pozisyonu emir defteri alış (`bids`) likiditesine satarak kapatır."""
        ref_now = now_utc or datetime.now(timezone.utc)
        now_iso = ref_now.isoformat()
        ist_date = self.storage.istanbul_date_str(ref_now)
        order_id = _new_order_id()

        if order_book is None or not order_book.bids or order_book.best_bid is None:
            rej = SimulatedOrderResult(
                order_id=order_id,
                symbol=position.symbol,
                side="SELL",
                order_type=order_type,
                status="REJECTED",
                requested_qty=position.quantity,
                filled_qty=Decimal("0"),
                avg_fill_price=Decimal("0"),
                gross_quote_usdc=Decimal("0"),
                fee_usdc=Decimal("0"),
                slippage_bps=Decimal("0"),
                signal_key=signal_key,
                strategy_source=strategy_source,
                explanation="Güncel alış kotasyonu bulunmadığından hayalî satış dolumu oluşturulmadı.",
                created_at_iso=now_iso,
            )
            self.storage.record_order(rej, ist_date, track=track)
            return rej

        if self.storage.has_processed_signal(signal_key, track=track):
            return SimulatedOrderResult(
                order_id=order_id,
                symbol=position.symbol,
                side="SELL",
                order_type=order_type,
                status="REJECTED",
                requested_qty=position.quantity,
                filled_qty=Decimal("0"),
                avg_fill_price=Decimal("0"),
                gross_quote_usdc=Decimal("0"),
                fee_usdc=Decimal("0"),
                slippage_bps=Decimal("0"),
                signal_key=signal_key,
                strategy_source=strategy_source,
                explanation=f"Mükerrer satış sinyali engellendi ({signal_key}).",
                created_at_iso=now_iso,
            )

        eff_step = filters.market_step_size if order_type == "MARKET" else filters.step_size
        filled_qty, raw_quote_usdc = _simulate_book_walk(
            order_book.bids,
            position.quantity,
            side="SELL",
            order_type=order_type,
            limit_price=limit_price,
            step_size=eff_step,
        )

        if filled_qty <= Decimal("0"):
            unfilled = SimulatedOrderResult(
                order_id=order_id,
                symbol=position.symbol,
                side="SELL",
                order_type=order_type,
                status="EXPIRED_UNFILLED" if order_type == "LIMIT" else "REJECTED",
                requested_qty=position.quantity,
                filled_qty=Decimal("0"),
                avg_fill_price=Decimal("0"),
                gross_quote_usdc=Decimal("0"),
                fee_usdc=Decimal("0"),
                slippage_bps=Decimal("0"),
                signal_key=signal_key,
                strategy_source=strategy_source,
                explanation="Alış likiditesi veya limit fiyat koşulu karşılanmadığından satış gerçekleşmedi.",
                created_at_iso=now_iso,
            )
            self.storage.record_order(unfilled, ist_date, track=track)
            return unfilled

        avg_book_price = raw_quote_usdc / filled_qty
        if order_type == "MARKET" and settings.extra_slippage_bps > 0:
            exec_price = avg_book_price * (Decimal("1") - settings.extra_slippage_bps / Decimal("10000"))
        else:
            exec_price = avg_book_price

        gross_proceeds_usdc = filled_qty * exec_price
        exit_fee_usdc = gross_proceeds_usdc * settings.estimated_fee_rate
        net_proceeds_usdc = gross_proceeds_usdc - exit_fee_usdc

        best_bid = order_book.best_bid
        slippage_bps = ((best_bid - exec_price) / best_bid) * Decimal("10000") if best_bid > 0 else Decimal("0")

        # Kısmi satış gerçekleştiyse pozisyonun orantılı maliyetini kapat, kalanı yeni açık pozisyon olarak tut
        fill_ratio = filled_qty / position.quantity
        allocated_cost = position.cost_basis_usdc * fill_ratio
        allocated_entry_fee = position.entry_fee_usdc * fill_ratio
        realized_pnl = net_proceeds_usdc - allocated_cost - allocated_entry_fee

        cash_balance = self.storage.get_cash_balance(track=track)
        self.storage.set_cash_balance(cash_balance + net_proceeds_usdc, track=track)

        self.storage.update_position_closed(
            position.position_id,
            closed_at_iso=now_iso,
            closed_istanbul_date=ist_date,
            exit_price=exec_price,
            exit_fee_usdc=exit_fee_usdc,
            realized_pnl_usdc=realized_pnl,
            close_reason=close_reason,
        )

        remaining_qty = quantize_down(position.quantity - filled_qty, eff_step)
        order_status = "FILLED"
        if remaining_qty > Decimal("0"):
            order_status = "PARTIALLY_FILLED"
            rem_pos = Position(
                position_id=_new_position_id(),
                symbol=position.symbol,
                quantity=remaining_qty,
                entry_price=position.entry_price,
                cost_basis_usdc=position.cost_basis_usdc - allocated_cost,
                entry_fee_usdc=position.entry_fee_usdc - allocated_entry_fee,
                opened_at_iso=position.opened_at_iso,
                signal_key=f"{position.signal_key}:REM",
                strategy_source=position.strategy_source,
                stop_loss_price=position.stop_loss_price,
                take_profit_price=position.take_profit_price,
                status="OPEN",
            )
            self.storage.insert_position(rem_pos, track=track)

        self.storage.mark_signal_processed(signal_key, order_id, now_iso, track=track)

        explanation = (
            f"Sanal SATIŞ ({order_status} - {close_reason}): {filled_qty} {filters.base_asset} "
            f"@ {exec_price:.6f} USDC (Net Hasılat: {net_proceeds_usdc:.2f} USDC, "
            f"Çıkış Komisyonu: {exit_fee_usdc:.4f} USDC, Gerçekleşmiş K/Z: {realized_pnl:+.4f} USDC)."
        )
        res = SimulatedOrderResult(
            order_id=order_id,
            symbol=position.symbol,
            side="SELL",
            order_type=order_type,
            status=order_status,
            requested_qty=position.quantity,
            filled_qty=filled_qty,
            avg_fill_price=exec_price,
            gross_quote_usdc=gross_proceeds_usdc,
            fee_usdc=exit_fee_usdc,
            slippage_bps=slippage_bps,
            signal_key=signal_key,
            strategy_source=strategy_source,
            explanation=explanation,
            created_at_iso=now_iso,
            realized_pnl_usdc=realized_pnl,
        )
        self.storage.record_order(res, ist_date, track=track)
        return res
