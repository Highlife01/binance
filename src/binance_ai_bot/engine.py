"""
Ana Sistem Orkestratörü (`TradingSystemEngine`).

Görevleri:
- Piyasa verisi -> Teknik strateji -> İsteğe bağlı Gemini AI -> Risk kontrolü -> Sanal yürütme
  zincirini güvenli biçimde koordine eder.
- Kural tabanlı gölge portföyü (`RULE_SHADOW`) ve pasif referansı (`Benchmark`) aynı kapanmış
  mumlar üzerinde paralel güncelleyerek yapay zekânın ek katkısını ayrı ölçer.
- Açık pozisyonun sanal zarar-kes / kâr-al / günlük zarar politikası çıkışlarını yönetir.
- Manuel pozisyon kapatma ve yeni alımları durdurma düğmelerini destekler.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from functools import wraps
import json
import time
from typing import Any, Callable

from .ai_advisor import GeminiDecisionAdvisor
from .config import AppConfig
from .domain import (
    AIDecisionRecord,
    MarketBundle,
    RiskSettings,
    SimulatedOrderResult,
    TechnicalSignal,
    is_substep_dust_position,
    quantize_down,
)
from .expert import (
    atr_pct_of_price,
    coin_knife_block,
    dynamic_exits,
    loss_streak_cooldown,
    regime_block,
    risk_capped_notional,
    trend_info,
)
from .live_executor import BinanceLiveSpotExecutor, LiveExecutionError
from .margin_executor import BinanceLiveMarginExecutor
from .market_data import BinancePublicDataClient, MarketDataError
from .risk import (
    RiskManager,
    compute_accrued_margin_interest,
    compute_open_position_unrealized_net_pnl,
)
from .security import LiveOrderBlockedError
from .simulator import VirtualExecutionSimulator
from .storage import SQLiteStorage
from .strategy import StrategyParams, evaluate_technical_strategy
from .execution_safety import ExecutionBusyError, serialized_execution


def report_execution_errors(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (LiveExecutionError, LiveOrderBlockedError, MarketDataError, ExecutionBusyError, ValueError) as exc:
            return StepCycleSummary(
                ok=False, timestamp_iso=datetime.now(timezone.utc).isoformat(),
                symbol=kwargs.get("symbol", ""), interval=kwargs.get("interval", ""),
                data_source="Yürütme Denetimi", wake_gap_detected=False,
                tech_signal=None, ai_decision=None, final_action="BEKLE", executed_order=None,
                block_or_info_message=f"İşlem tamamlanamadı: {exc}", error_message=str(exc),
            )
    return wrapped


@dataclass(frozen=True)
class StepCycleSummary:
    ok: bool
    timestamp_iso: str
    symbol: str
    interval: str
    data_source: str
    wake_gap_detected: bool
    tech_signal: TechnicalSignal | None
    ai_decision: AIDecisionRecord | None
    final_action: str
    executed_order: SimulatedOrderResult | None
    block_or_info_message: str
    error_message: str | None = None


class TradingSystemEngine:
    """Tüm alt modülleri gevşek bağlı (loosely-coupled) biçimde birleştiren ana motor."""

    def __init__(
        self,
        config: AppConfig,
        storage: SQLiteStorage | None = None,
        market_client: BinancePublicDataClient | None = None,
        ai_advisor: GeminiDecisionAdvisor | None = None,
    ) -> None:
        self.config = config
        self.storage = storage or SQLiteStorage(
            db_path=config.sqlite_db_path,
            initial_balance_usdc=config.initial_balance_usdc,
            timezone_name=config.timezone_name,
        )
        self.market_client = market_client or BinancePublicDataClient(config)
        self.risk_manager = RiskManager(self.storage)
        self.simulator = VirtualExecutionSimulator(self.storage)
        self.live_executor = BinanceLiveSpotExecutor(config, self.storage)
        self.live_margin_executor = BinanceLiveMarginExecutor(config, self.storage)
        self.ai_advisor = ai_advisor or GeminiDecisionAdvisor(config, self.storage)
        # Çoklu coin taramasında True yapılır: pasif referans ilk paritede sabit kalır
        self.lock_benchmark_symbol = False
        self.allow_live_margin = False
        self._candle_cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._latest_bid_map: dict[str, Decimal] = {}

    def _collect_position_bids(self, positions, initial=None):
        """Fetch every held asset; missing marks block entries, never invent prices."""
        bids = dict(initial or {})
        missing = []
        for symbol in {p.symbol for p in positions} - bids.keys():
            try:
                book = self.market_client.fetch_order_book(symbol, depth_limit=5)
                age = (datetime.now(timezone.utc) - book.fetched_at_utc).total_seconds()
                if book.best_bid is None or book.best_bid <= 0 or not 0 <= age <= self.config.max_data_age_seconds:
                    raise MarketDataError("Pozisyon fiyatı güncel değil.")
                bids[symbol] = book.best_bid
            except (MarketDataError, AttributeError):
                missing.append(symbol)
        return bids, missing

    def _shadow_track(self, quote: str) -> str:
        bound = self.storage.get_portfolio_quote("RULE_SHADOW")
        return "RULE_SHADOW" if bound in {None, quote} else f"RULE_SHADOW_{quote}"

    def default_risk_settings(self) -> RiskSettings:
        fallback = RiskSettings(
            max_open_positions=self.config.max_open_positions,
            max_position_size_usdc=self.config.max_position_size_usdc,
            daily_loss_limit_usdc=self.config.daily_loss_limit_usdc,
            max_daily_new_positions=self.config.max_daily_new_positions,
            daily_limit_exit_policy=self.config.daily_limit_exit_policy,
            estimated_fee_rate=self.config.estimated_fee_rate,
            extra_slippage_bps=self.config.extra_slippage_bps,
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
        )
        return self.storage.load_risk_settings(fallback)

    def get_portfolio_summary(
        self,
        settings: RiskSettings,
        live_bid_by_symbol: dict[str, Decimal] | None = None,
        track: str = "MAIN",
    ) -> dict[str, Any]:
        """Kullanılabilir nakit, açık pozisyon değeri, gerçekleşmiş/gerçekleşmemiş K/Z ve komisyon özeti."""
        bids = live_bid_by_symbol or {}
        cash = self.storage.get_cash_balance(track=track)
        open_positions = self.storage.get_open_positions(track=track)
        hourly_rate = getattr(settings, "hourly_margin_interest_rate", Decimal("0.000005"))

        open_gross_val = Decimal("0")
        unrealized_net_pnl = Decimal("0")
        total_borrowed_usdc = Decimal("0")
        for pos in open_positions:
            bid = bids.get(pos.symbol, pos.entry_price)
            net_eq_v, unreal_pnl = compute_open_position_unrealized_net_pnl(
                pos,
                current_bid=bid,
                estimated_fee_rate=settings.estimated_fee_rate,
                hourly_interest_rate=hourly_rate,
            )
            open_gross_val += net_eq_v
            unrealized_net_pnl += unreal_pnl
            total_borrowed_usdc += getattr(pos, "borrowed_usdc", Decimal("0"))

        total_realized_pnl, total_fees = self.storage.get_total_realized_pnl_and_fees(track=track)
        total_equity = cash + open_gross_val
        daily_net_pnl = self.risk_manager.calculate_daily_net_pnl_usdc(
            settings=settings,
            open_positions=open_positions,
            bid_prices_by_symbol=bids,
            track=track,
        )

        return {
            "cash_usdc": cash,
            "open_positions": open_positions,
            "open_market_value_usdc": open_gross_val,
            "borrowed_usdc": total_borrowed_usdc,
            "total_equity_usdc": total_equity,
            "realized_pnl_usdc": total_realized_pnl,
            "unrealized_pnl_usdc": unrealized_net_pnl,
            "daily_net_pnl_usdc": daily_net_pnl,
            "total_fees_usdc": total_fees,
            "quote_asset": self.storage.get_portfolio_quote(track),
            "valuation_complete": all(p.symbol in bids for p in open_positions),
        }

    def _update_rule_shadow_track(
        self,
        market: MarketBundle,
        tech_signal: TechnicalSignal,
        settings: RiskSettings,
        now_utc: datetime,
    ) -> None:
        """
        Aynı piyasa verisi üzerinde saf Kural Tabanlı gölge portföyü (`RULE_SHADOW`) çalıştırır.
        Böylece panelde Yapay Zekâ destekli ana strateji ile saf Kural Tabanlı strateji elma-elma karşılaştırılır.
        """
        shadow_track = self._shadow_track(market.filters.quote_asset)
        shadow_open = self.storage.get_open_positions(track=shadow_track)
        best_bid = market.order_book.best_bid
        shadow_pos = next(
            (p for p in shadow_open if p.symbol == market.symbol and not is_substep_dust_position(p, market.filters)),
            None,
        )
        min_profit_pct = getattr(settings, "min_net_profit_exit_pct", Decimal("0"))
        if shadow_pos is not None and best_bid is not None:
            stop_hit = shadow_pos.stop_loss_price is not None and best_bid <= shadow_pos.stop_loss_price
            tp_hit = shadow_pos.take_profit_price is not None and best_bid >= shadow_pos.take_profit_price
            rule_sell_ok = tech_signal.action == "SAT" and (
                min_profit_pct <= Decimal("0")
                or best_bid >= shadow_pos.entry_price * (Decimal("1") + min_profit_pct)
            )
            if stop_hit or tp_hit or rule_sell_ok:
                reason = "STOP_LOSS" if stop_hit else ("TAKE_PROFIT" if tp_hit else "KURAL_SAT")
                self.simulator.execute_sell_position(
                    position=shadow_pos,
                    filters=market.filters,
                    order_book=market.order_book,
                    signal_key=f"SHADOW_EXIT:{tech_signal.signal_key}:{reason}",
                    strategy_source="KURAL_TABANLI_GOLGE",
                    close_reason=reason,
                    settings=settings,
                    now_utc=now_utc,
                    track=shadow_track,
                )
                return

        active_shadow_open = [p for p in shadow_open if not is_substep_dust_position(p)]
        if shadow_pos is None and len(active_shadow_open) < settings.max_open_positions and tech_signal.action == "AL":
            chk = self.risk_manager.check_buy_entry(
                filters=market.filters,
                order_book=market.order_book,
                settings=settings,
                now_utc=now_utc,
                track=shadow_track,
                bid_prices_by_symbol=self._collect_position_bids(shadow_open, {market.symbol: best_bid})[0],
            )
            if chk.allowed:
                self.simulator.execute_buy(
                    filters=market.filters,
                    order_book=market.order_book,
                    requested_qty=chk.approved_quantity,
                    signal_key=tech_signal.signal_key,
                    strategy_source="KURAL_TABANLI_GOLGE",
                    settings=settings,
                    now_utc=now_utc,
                    track=shadow_track,
                )

    def _compute_benchmark_equity(
        self, symbol: str, current_mid_price: Decimal, settings: RiskSettings
    ) -> Decimal:
        """
        Uygun referans performansı: Başlangıç 86 USDC içinde pozisyon sınırı kadar (ör. 10 USDC)
        seçili paritede pasif Al-ve-Tut + kalan nakit (76 USDC) koruma referansı.
        """
        saved_sym = self.storage.get_state("benchmark_symbol", "")
        saved_px_raw = self.storage.get_state("benchmark_initial_price", "")
        if self.lock_benchmark_symbol and saved_sym and saved_px_raw and saved_sym != symbol:
            # Çoklu coin taramasında referans her paritede sıfırlanmasın; son referans değeri korunur
            last_raw = self.storage.get_state("benchmark_last_equity", "")
            return Decimal(last_raw) if last_raw else self.config.initial_balance_usdc
        if saved_sym != symbol or not saved_px_raw:
            self.storage.set_state("benchmark_symbol", symbol)
            self.storage.set_state("benchmark_initial_price", f"{current_mid_price:.8f}")
            self.storage.set_state("benchmark_last_equity", f"{self.config.initial_balance_usdc:.8f}")
            return self.config.initial_balance_usdc

        init_px = Decimal(saved_px_raw)
        if init_px <= 0:
            return self.config.initial_balance_usdc

        alloc = min(settings.max_position_size_usdc, self.config.initial_balance_usdc)
        passive_cash = self.config.initial_balance_usdc - alloc
        entry_fee = alloc * settings.estimated_fee_rate
        units = (alloc - entry_fee) / init_px
        bench_equity = passive_cash + (units * current_mid_price)
        self.storage.set_state("benchmark_last_equity", f"{bench_equity:.8f}")
        return bench_equity

    @report_execution_errors
    @serialized_execution
    def run_analysis_step(
        self,
        *,
        symbol: str,
        interval: str,
        settings: RiskSettings,
        ai_enabled_by_user: bool = True,
        daily_ai_limit_override: int | None = None,
        min_ai_interval_override: int | None = None,
        execution_mode: str = "PAPER",
        ui_live_confirmed: bool = False,
        ui_margin_confirmed: bool = False,
        strategy_params: StrategyParams | None = None,
        ai_only_on_rule_signal: bool = False,
        skip_balance_prefetch: bool = False,
        stop_requested: Callable[[], bool] | None = None,
    ) -> StepCycleSummary:
        """
        Tek bir tam denetimli analiz ve yürütme döngüsü çalıştırır.
        - `execution_mode="PAPER"`: Sanal deneme (`MAIN` izi, 1x–5x kaldıraç destekli).
        - `execution_mode="LIVE_SPOT"`: Gerçek kaldıraçsız Spot (`LIVE_SPOT` izi).
        - `execution_mode="LIVE_MARGIN"`: Üç kilitli gerçek Çapraz Marjin (`LIVE_MARGIN` izi, 2x–5x).
        """
        is_live_spot = execution_mode == "LIVE_SPOT"
        is_live_margin = execution_mode == "LIVE_MARGIN"
        active_track = (
            "LIVE_MARGIN" if is_live_margin else ("LIVE_SPOT" if is_live_spot else "MAIN")
        )
        quote_ccy = "USDT" if symbol.strip().upper().endswith("USDT") else "USDC"
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        ist_date = self.storage.istanbul_date_str(now_utc)

        def blocked(message: str) -> StepCycleSummary:
            return StepCycleSummary(
                ok=False, timestamp_iso=now_iso, symbol=symbol, interval=interval,
                data_source="Güvenlik Denetimi", wake_gap_detected=False,
                tech_signal=None, ai_decision=None, final_action="BEKLE",
                executed_order=None, block_or_info_message=message, error_message=message,
            )

        allowed_modes = (
            {"PAPER", "LIVE_SPOT", "LIVE_MARGIN"}
            if getattr(self, "allow_live_margin", False)
            else {"PAPER", "LIVE_SPOT"}
        )
        if execution_mode not in allowed_modes:
            return blocked("Bu güvenlik sürümü yalnızca Sanal Deneme ve Canlı Spot modlarını destekliyor; canlı marjin kapalıdır.")
        if stop_requested and stop_requested():
            return blocked("Durdurma isteği alındı; emir gönderilmedi.")
        try:
            self.storage.ensure_quote_asset(active_track, quote_ccy)
            if is_live_spot:
                self.live_executor.reconcile_pending_orders(ui_live_confirmed=ui_live_confirmed)
                self._resolve_live_unvalued_commissions(quote_ccy, settings)
        except (ValueError, LiveOrderBlockedError, LiveExecutionError) as exc:
            return blocked(str(exc))

        if is_live_spot and not skip_balance_prefetch:
            try:
                self.live_executor.fetch_live_spot_balances(
                    ui_live_confirmed=ui_live_confirmed, quote_asset=quote_ccy
                )
            except (LiveOrderBlockedError, LiveExecutionError) as exc:
                return StepCycleSummary(
                    ok=False,
                    timestamp_iso=now_iso,
                    symbol=symbol,
                    interval=interval,
                    data_source="Canlı Hesap Doğrulaması Başarısız",
                    wake_gap_detected=False,
                    tech_signal=None,
                    ai_decision=None,
                    final_action="BEKLE",
                    executed_order=None,
                    block_or_info_message="Canlı Spot güvenlik/bağlantı kilidi nedeniyle emir gönderilmedi.",
                    error_message=str(exc),
                )
        elif is_live_margin and not skip_balance_prefetch:
            try:
                self.live_margin_executor.fetch_live_margin_balances(
                    ui_live_confirmed=ui_live_confirmed,
                    ui_margin_confirmed=ui_margin_confirmed,
                    quote_asset=quote_ccy,
                )
            except (LiveOrderBlockedError, LiveExecutionError) as exc:
                return StepCycleSummary(
                    ok=False,
                    timestamp_iso=now_iso,
                    symbol=symbol,
                    interval=interval,
                    data_source="Canlı Marjin Doğrulaması Başarısız",
                    wake_gap_detected=False,
                    tech_signal=None,
                    ai_decision=None,
                    final_action="BEKLE",
                    executed_order=None,
                    block_or_info_message="Canlı Marjin cüzdan/kilit doğrulaması geçemediği için emir gönderilmedi.",
                    error_message=str(exc),
                )

        # 1. Doğrulanmış piyasa verisini çek (hata/eski veri varsa işlem ÜRETME!)
        try:
            market = self.market_client.fetch_verified_market_bundle(
                symbol=symbol,
                interval=interval,
                allow_stablecoin_conversion=settings.allow_stablecoin_conversion,
            )
        except MarketDataError as exc:
            return StepCycleSummary(
                ok=False,
                timestamp_iso=now_iso,
                symbol=symbol,
                interval=interval,
                data_source="Bağlantı Yok",
                wake_gap_detected=False,
                tech_signal=None,
                ai_decision=None,
                final_action="BEKLE",
                executed_order=None,
                block_or_info_message="Piyasa verisi doğrulanamadığı için işlem üretilmedi.",
                error_message=str(exc),
            )

        best_bid = market.order_book.best_bid or market.closed_candles[-1].close
        best_ask = market.order_book.best_ask or market.closed_candles[-1].close
        mid_price = (best_bid + best_ask) / Decimal("2")

        # Windows uyku dönüşü algılandıysa ilk döngüde yalnızca veri senkronize et, emir gönderme!
        if market.wake_gap_detected:
            return StepCycleSummary(
                ok=True,
                timestamp_iso=now_iso,
                symbol=market.symbol,
                interval=market.interval,
                data_source=market.source_label,
                wake_gap_detected=True,
                tech_signal=None,
                ai_decision=None,
                final_action="BEKLE",
                executed_order=None,
                block_or_info_message=(
                    "Bilgisayar uyku modundan döndü veya uzun süre ara verildi. "
                    "Güvenlik gereği bu ilk döngüde yalnızca saat ve veri önbelleği tazelendi; emir açılmadı."
                ),
            )

        # 2. Kural tabanlı teknik gösterge hesabı (yalnızca kapanmış mumlarla)
        tech_signal, recent_returns = evaluate_technical_strategy(
            market.closed_candles,
            symbol=market.symbol,
            interval=market.interval,
            params=strategy_params,
        )

        # 3. Gölge kural tabanlı karşılaştırma portföyünü güncelle (1x Spot referans)
        if stop_requested and stop_requested():
            return blocked("Durdurma isteği alındı; emir gönderilmedi.")
        self._update_rule_shadow_track(
            market, tech_signal, dataclasses.replace(settings, leverage=1), now_utc
        )

        open_positions = self.storage.get_open_positions(track=active_track)
        symbol_open_pos = next(
            (p for p in open_positions if p.symbol == market.symbol and not is_substep_dust_position(p, market.filters)),
            None,
        )
        has_open_pos = symbol_open_pos is not None

        # 4. Günlük zarar limiti ve açık pozisyon koruyucu çıkış denetimi
        bid_map, missing_marks = self._collect_position_bids(open_positions, {market.symbol: best_bid})
        self._latest_bid_map = bid_map
        breached, daily_loss, auto_close_due_to_daily_limit = self.risk_manager.evaluate_daily_loss_guard(
            settings=settings,
            open_positions=open_positions,
            bid_prices_by_symbol=bid_map,
            now_utc=now_utc,
            track=active_track,
        )
        if breached:
            halt_msg = (
                f"Günlük zarar eşiği ({settings.daily_loss_limit_usdc:.2f} {quote_ccy}) aşıldı "
                f"(toplam kayıp: {daily_loss:.2f} {quote_ccy}). Yeni alımlar durduruldu."
            )
            self.storage.set_new_entries_paused(True, halt_msg, track=active_track, until_date=ist_date)
            if auto_close_due_to_daily_limit and symbol_open_pos is not None:
                if stop_requested and stop_requested():
                    return blocked("Durdurma isteği alındı; emir gönderilmedi.")
                exit_sig = f"DAILY_LOSS_AUTO_EXIT:{ist_date}:{symbol_open_pos.position_id}"
                if is_live_spot:
                    closed_ord = self.live_executor.execute_live_market_sell(
                        position=symbol_open_pos,
                        filters=market.filters,
                        signal_key=exit_sig,
                        strategy_source="RISK_KORUMA",
                        close_reason="GUNLUK_ZARAR_OTOMATIK_CIKIS",
                        ui_live_confirmed=ui_live_confirmed,
                        stop_requested=stop_requested,
                    )
                    self._resolve_live_unvalued_commissions(quote_ccy, settings)
                elif is_live_margin:
                    closed_ord = self.live_margin_executor.execute_live_margin_sell(
                        position=symbol_open_pos,
                        filters=market.filters,
                        signal_key=exit_sig,
                        strategy_source="RISK_KORUMA",
                        close_reason="GUNLUK_ZARAR_OTOMATIK_CIKIS",
                        ui_live_confirmed=ui_live_confirmed,
                        ui_margin_confirmed=ui_margin_confirmed,
                    )
                else:
                    closed_ord = self.simulator.execute_sell_position(
                        position=symbol_open_pos,
                        filters=market.filters,
                        order_book=market.order_book,
                        signal_key=exit_sig,
                        strategy_source="RISK_KORUMA",
                        close_reason="GUNLUK_ZARAR_OTOMATIK_CIKIS",
                        settings=settings,
                        now_utc=now_utc,
                        track="MAIN",
                    )
                self._record_snapshot(market.symbol, mid_price, best_bid, settings, now_iso, ist_date, track=active_track)
                return StepCycleSummary(
                    ok=True,
                    timestamp_iso=now_iso,
                    symbol=market.symbol,
                    interval=market.interval,
                    data_source=market.source_label,
                    wake_gap_detected=False,
                    tech_signal=tech_signal,
                    ai_decision=None,
                    final_action="SAT",
                    executed_order=closed_ord,
                    block_or_info_message=halt_msg,
                )

        # Açık pozisyonun Likidasyon, Zarar-Kes (Stop-Loss) veya Kâr-Al (Take-Profit) kontrolü
        trailing_note = ""
        if symbol_open_pos is not None:
            pos = symbol_open_pos
            liq_hit = pos.liquidation_price is not None and best_bid <= pos.liquidation_price
            stop_hit = pos.stop_loss_price is not None and best_bid <= pos.stop_loss_price
            tp_hit = pos.take_profit_price is not None and best_bid >= pos.take_profit_price

            # Hareket eden (trailing) stop: Kâr-Al hedefine ulaşınca hemen satmak yerine stop'u kârın
            # üzerine çekip hedefi yukarı taşır; güçlü yükselişin devamını yakalar, düşüşte kâr kilitli kalır.
            trail_pct = getattr(settings, "trailing_stop_pct", Decimal("0"))
            if tp_hit and not liq_hit and not stop_hit and trail_pct > Decimal("0"):
                lock_pct = getattr(settings, "min_net_profit_exit_pct", Decimal("0"))
                if getattr(pos, "borrowed_usdc", Decimal("0")) > Decimal("0") and pos.cost_basis_usdc > Decimal("0"):
                    lock_pct += compute_accrued_margin_interest(
                        pos, now_utc, getattr(settings, "hourly_margin_interest_rate", Decimal("0.000005"))
                    ) / pos.cost_basis_usdc
                floor_px = pos.entry_price * (Decimal("1") + lock_pct)
                new_stop = max(best_bid * (Decimal("1") - trail_pct), floor_px, pos.stop_loss_price or Decimal("0"))
                if new_stop < best_bid:
                    self.storage.update_position_remaining(
                        pos.position_id,
                        quantity=pos.quantity,
                        cost_basis_usdc=pos.cost_basis_usdc,
                        entry_fee_usdc=pos.entry_fee_usdc,
                        stop_loss_price=new_stop,
                        take_profit_price=best_bid * (Decimal("1") + trail_pct),
                        update_exit_targets=True,
                    )
                    tp_hit = False
                    trailing_note = (
                        f"{market.symbol} trailing stop yükseltildi: {new_stop:.6g} (anlık {best_bid:.6g})."
                    )
            if liq_hit or stop_hit or tp_hit:
                if stop_requested and stop_requested():
                    return blocked("Durdurma isteği alındı; emir gönderilmedi.")
                locked_trailing = (
                    stop_hit
                    and not liq_hit
                    and pos.stop_loss_price is not None
                    and pos.stop_loss_price > pos.entry_price
                )
                exit_tag = (
                    "LIKIDASYON"
                    if liq_hit
                    else ("TRAILING_KAR" if locked_trailing else ("ZARAR_KES" if stop_hit else "KAR_AL"))
                )
                exit_sig = f"{exit_tag}:{pos.position_id}:{tech_signal.candle_open_time_ms}"
                if is_live_spot:
                    closed_ord = self.live_executor.execute_live_market_sell(
                        position=pos,
                        filters=market.filters,
                        signal_key=exit_sig,
                        strategy_source="RISK_KORUMA",
                        close_reason=exit_tag,
                        ui_live_confirmed=ui_live_confirmed,
                        stop_requested=stop_requested,
                    )
                    self._resolve_live_unvalued_commissions(quote_ccy, settings)
                elif is_live_margin:
                    closed_ord = self.live_margin_executor.execute_live_margin_sell(
                        position=pos,
                        filters=market.filters,
                        signal_key=exit_sig,
                        strategy_source="RISK_KORUMA",
                        close_reason=exit_tag,
                        ui_live_confirmed=ui_live_confirmed,
                        ui_margin_confirmed=ui_margin_confirmed,
                    )
                else:
                    closed_ord = self.simulator.execute_sell_position(
                        position=pos,
                        filters=market.filters,
                        order_book=market.order_book,
                        signal_key=exit_sig,
                        strategy_source="RISK_KORUMA",
                        close_reason=exit_tag,
                        settings=settings,
                        now_utc=now_utc,
                        track="MAIN",
                    )
                self._record_snapshot(market.symbol, mid_price, best_bid, settings, now_iso, ist_date, track=active_track)
                return StepCycleSummary(
                    ok=True,
                    timestamp_iso=now_iso,
                    symbol=market.symbol,
                    interval=market.interval,
                    data_source=market.source_label,
                    wake_gap_detected=False,
                    tech_signal=tech_signal,
                    ai_decision=None,
                    final_action="SAT",
                    executed_order=closed_ord,
                    block_or_info_message=f"Açık pozisyon {exit_tag} seviyesinden kapatıldı.",
                )

        # 5. Yapay zekâ (veya kural tabanlı geri dönüş) değerlendirmesi
        # Çoklu coin taramasında maliyet/çağrı aralığı için: açık pozisyon yoksa ve kural AL demiyorsa
        # Gemini çağrılmaz; karar doğrudan kural tabanlı BEKLE olur (ön eleme).
        skip_ai_prefilter = (
            ai_only_on_rule_signal and not has_open_pos and tech_signal.action != "AL"
        )
        ai_context_notes: list[str] | None = None
        if getattr(settings, "expert_mode", False) and ai_enabled_by_user and not skip_ai_prefilter:
            ai_context_notes = self._expert_context_notes(market.symbol) or None
        ai_decision = self.ai_advisor.evaluate(
            market=market,
            tech_signal=tech_signal,
            recent_returns_pct=recent_returns,
            has_open_position=has_open_pos,
            ai_enabled_by_user=ai_enabled_by_user and not skip_ai_prefilter,
            daily_call_limit_override=daily_ai_limit_override,
            min_interval_override_sec=min_ai_interval_override,
            external_notes=ai_context_notes,
        )
        if skip_ai_prefilter and ai_enabled_by_user:
            ai_decision = dataclasses.replace(ai_decision, mode_label="KURAL_ON_ELEME")

        final_action = ai_decision.action
        executed_order: SimulatedOrderResult | None = None
        info_msg = ai_decision.reason
        block_reason: str | None = None
        min_profit_pct = getattr(settings, "min_net_profit_exit_pct", Decimal("0"))
        if trailing_note:
            info_msg = trailing_note

        # "Al & Kârı Görünce Sat" koruması:
        # 1) Fiyat net kâr eşiğinin (+min_net_profit_exit_pct + varsa birikmiş marjin faizi) üzerine çıktıysa ve teknik ivme SAT diyorsa anında kârı kilitle.
        # 2) Fiyat henüz kâr eşiğine ulaşmadıysa ufak mum dalgalanmalarında zararına erken SAT yapma; yükselişi bekle!
        eff_min_profit_pct = min_profit_pct
        if symbol_open_pos is not None and min_profit_pct > Decimal("0"):
            if (
                getattr(symbol_open_pos, "borrowed_usdc", Decimal("0")) > Decimal("0")
                and symbol_open_pos.cost_basis_usdc > Decimal("0")
            ):
                hourly_rate = getattr(settings, "hourly_margin_interest_rate", Decimal("0.000005"))
                accrued_int = compute_accrued_margin_interest(symbol_open_pos, now_utc, hourly_rate)
                eff_min_profit_pct = min_profit_pct + (accrued_int / symbol_open_pos.cost_basis_usdc)
            min_exit_px = symbol_open_pos.entry_price * (Decimal("1") + eff_min_profit_pct)
            if best_bid >= min_exit_px and tech_signal.action == "SAT":
                final_action = "SAT"
                info_msg = (
                    f"{market.symbol} kâr bölgesinde (+%{eff_min_profit_pct * Decimal('100'):.2f} üzeri) "
                    f"ivme yavaşladığı için net kâr kilitleniyor (Giriş: {symbol_open_pos.entry_price:.6g}, Anlık: {best_bid:.6g})."
                )
            elif final_action == "SAT" and best_bid < min_exit_px:
                final_action = "BEKLE"
                info_msg = (
                    f"{market.symbol} kâr hedefi bekleniyor (Giriş: {symbol_open_pos.entry_price:.6g}, "
                    f"Anlık: {best_bid:.6g}, Kârlı Çıkış Eşiği: {min_exit_px:.6g}); zararına erken satış yapılmadı."
                )

        if stop_requested and stop_requested():
            return blocked("Durdurma isteği alındı; emir gönderilmedi.")

        # 6. Nihai kararın risk denetimi ve emir yürütmesi
        if final_action == "AL":
            if symbol_open_pos is not None:
                # Bu paritede zaten pozisyon taşınıyor; hata/engel uyarısı yerine pozisyon tutma bilgisi ver
                final_action = "BEKLE"
                info_msg = (
                    f"{market.symbol} açık pozisyonu korunuyor "
                    "(Kâr-Al / Zarar-Kes / dönüş sinyali takibinde; aynı coine ek alım yapılmaz)."
                )
            else:
                paused, pause_reason = self.storage.is_new_entries_paused(track=active_track)
                if paused:
                    block_reason = f"Yeni pozisyon açılışı durdurulmuş durumda ({pause_reason})."
                    info_msg = block_reason
                elif not 0 <= (datetime.now(timezone.utc) - market.order_book.fetched_at_utc).total_seconds() <= self.config.max_data_age_seconds:
                    block_reason = "Karar beklenirken piyasa fiyatı eskidi; yeni alım için güncel veri bekleniyor."
                    info_msg = block_reason
                elif (
                    getattr(settings, "max_entry_spread_pct", Decimal("0")) > Decimal("0")
                    and mid_price > Decimal("0")
                    and (best_ask - best_bid) / mid_price > settings.max_entry_spread_pct
                ):
                    spread_now = (best_ask - best_bid) / mid_price
                    block_reason = (
                        f"{market.symbol} alış-satış farkı geniş (%{spread_now * Decimal('100'):.3f} > "
                        f"%{settings.max_entry_spread_pct * Decimal('100'):.3f}); maliyet kârı yiyeceği için alım atlandı."
                    )
                    info_msg = block_reason
                elif getattr(settings, "expert_mode", False) and (
                    expert_why := self._expert_entry_block(market.symbol, active_track, now_utc)
                ):
                    block_reason = f"Uzman filtre: {expert_why}; alım atlandı."
                    info_msg = block_reason
                else:
                    risk_chk = self.risk_manager.check_buy_entry(
                        filters=market.filters,
                        order_book=market.order_book,
                        settings=settings,
                        now_utc=now_utc,
                        track=active_track,
                        bid_prices_by_symbol=bid_map,
                    )
                    if not risk_chk.allowed:
                        block_reason = risk_chk.reason
                        info_msg = f"AL sinyali risk denetiminde durduruldu: {risk_chk.reason}"
                    else:
                        buy_settings = settings
                        buy_qty = risk_chk.approved_quantity
                        buy_notional = risk_chk.approved_notional_usdc
                        if getattr(settings, "expert_mode", False):
                            buy_settings, buy_qty, buy_notional = self._expert_adjust_buy(
                                settings, market, tech_signal, risk_chk, bid_map, active_track
                            )
                        sig_key = f"{market.symbol}:{market.interval}:{tech_signal.candle_open_time_ms}:AL"
                        if is_live_spot:
                            try:
                                ord_res = self.live_executor.execute_live_market_buy(
                                    filters=market.filters,
                                    approved_qty=buy_qty,
                                    approved_notional=buy_notional,
                                    signal_key=sig_key,
                                    strategy_source=ai_decision.mode_label,
                                    settings=buy_settings,
                                    ui_live_confirmed=ui_live_confirmed,
                                    stop_requested=stop_requested,
                                )
                                self._resolve_live_unvalued_commissions(quote_ccy, settings)
                                ord_res = dataclasses.replace(
                                    ord_res,
                                    explanation=ord_res.explanation.replace(
                                        " Üçüncü varlık komisyonunun değeri bilinmiyor; K/Z geçicidir, yeni alımlar durduruldu.",
                                        " BNB komisyon indirimi uygulandı.",
                                    ),
                                )
                                executed_order = ord_res
                                info_msg = ord_res.explanation
                            except (LiveOrderBlockedError, LiveExecutionError) as exc:
                                block_reason = str(exc)
                                info_msg = f"Canlı Spot ALIM engellendi/reddedildi: {exc}"
                        elif is_live_margin:
                            try:
                                ord_res = self.live_margin_executor.execute_live_margin_buy(
                                    filters=market.filters,
                                    approved_qty=buy_qty,
                                    approved_notional=buy_notional,
                                    signal_key=sig_key,
                                    strategy_source=ai_decision.mode_label,
                                    settings=buy_settings,
                                    ui_live_confirmed=ui_live_confirmed,
                                    ui_margin_confirmed=ui_margin_confirmed,
                                )
                                executed_order = ord_res
                                info_msg = ord_res.explanation
                            except (LiveOrderBlockedError, LiveExecutionError) as exc:
                                block_reason = str(exc)
                                info_msg = f"Canlı Marjin ALIM engellendi/reddedildi: {exc}"
                        else:
                            ord_res = self.simulator.execute_buy(
                                filters=market.filters,
                                order_book=market.order_book,
                                requested_qty=buy_qty,
                                signal_key=sig_key,
                                strategy_source=ai_decision.mode_label,
                                settings=buy_settings,
                                now_utc=now_utc,
                                track="MAIN",
                            )
                            executed_order = ord_res
                            if ord_res.status not in ("FILLED", "PARTIALLY_FILLED"):
                                block_reason = ord_res.explanation
                            info_msg = ord_res.explanation

        elif final_action == "SAT":
            if symbol_open_pos is not None:
                sig_key = f"{market.symbol}:{market.interval}:{tech_signal.candle_open_time_ms}:SAT"
                exit_reason_label = (
                    "KARDA_STRATEJI_SAT"
                    if (
                        min_profit_pct > Decimal("0")
                        and best_bid >= symbol_open_pos.entry_price * (Decimal("1") + min_profit_pct)
                    )
                    else "STRATEJI_SAT_SINYALI"
                )
                if is_live_spot:
                    try:
                        ord_res = self.live_executor.execute_live_market_sell(
                            position=symbol_open_pos,
                            filters=market.filters,
                            signal_key=sig_key,
                            strategy_source=ai_decision.mode_label,
                            close_reason=exit_reason_label,
                            ui_live_confirmed=ui_live_confirmed,
                            stop_requested=stop_requested,
                        )
                        self._resolve_live_unvalued_commissions(quote_ccy, settings)
                        ord_res = dataclasses.replace(
                            ord_res,
                            explanation=ord_res.explanation.replace(
                                " Üçüncü varlık komisyonunun değeri bilinmiyor; K/Z geçicidir, yeni alımlar durduruldu.",
                                " BNB komisyon indirimi uygulandı.",
                            ),
                        )
                        executed_order = ord_res
                        info_msg = ord_res.explanation
                    except (LiveOrderBlockedError, LiveExecutionError) as exc:
                        block_reason = str(exc)
                        info_msg = f"Canlı Spot SATIŞ hatası: {exc}"
                elif is_live_margin:
                    try:
                        ord_res = self.live_margin_executor.execute_live_margin_sell(
                            position=symbol_open_pos,
                            filters=market.filters,
                            signal_key=sig_key,
                            strategy_source=ai_decision.mode_label,
                            close_reason=exit_reason_label,
                            ui_live_confirmed=ui_live_confirmed,
                            ui_margin_confirmed=ui_margin_confirmed,
                        )
                        executed_order = ord_res
                        info_msg = ord_res.explanation
                    except (LiveOrderBlockedError, LiveExecutionError) as exc:
                        block_reason = str(exc)
                        info_msg = f"Canlı Marjin SATIŞ hatası: {exc}"
                else:
                    ord_res = self.simulator.execute_sell_position(
                        position=symbol_open_pos,
                        filters=market.filters,
                        order_book=market.order_book,
                        signal_key=sig_key,
                        strategy_source=ai_decision.mode_label,
                        close_reason=exit_reason_label,
                        settings=settings,
                        now_utc=now_utc,
                        track="MAIN",
                    )
                    executed_order = ord_res
                    info_msg = ord_res.explanation
            else:
                final_action = "BEKLE"
                block_reason = "Bu paritede kapatılacak açık pozisyon yok (açığa satış kapalı)."
                info_msg = block_reason

        self.storage.record_decision(
            created_at_iso=now_iso,
            istanbul_date=ist_date,
            symbol=market.symbol,
            interval=market.interval,
            candle_open_time_ms=tech_signal.candle_open_time_ms,
            rule_action=tech_signal.action,
            rule_reason=tech_signal.reason,
            ai_action=ai_decision.action,
            ai_reason=ai_decision.reason,
            final_action=final_action,
            mode_label=ai_decision.mode_label,
            risk_notes=ai_decision.risk_notes,
            self_reported_score=ai_decision.self_reported_score,
            executed_order_id=executed_order.order_id if executed_order else None,
            block_reason=block_reason,
            track=active_track,
            quote_asset=quote_ccy,
        )

        self._record_snapshot(market.symbol, mid_price, best_bid, settings, now_iso, ist_date, track=active_track)

        return StepCycleSummary(
            ok=True,
            timestamp_iso=now_iso,
            symbol=market.symbol,
            interval=market.interval,
            data_source=market.source_label,
            wake_gap_detected=False,
            tech_signal=tech_signal,
            ai_decision=ai_decision,
            final_action=final_action,
            executed_order=executed_order,
            block_or_info_message=info_msg,
        )

    @serialized_execution
    def manual_close_open_position(
        self,
        settings: RiskSettings,
        *,
        execution_mode: str = "PAPER",
        ui_live_confirmed: bool = False,
        ui_margin_confirmed: bool = False,
        symbol: str | None = None,
    ) -> tuple[bool, str]:
        """Açık pozisyonları (Sanal, Canlı Spot veya Canlı Çapraz Marjin moduna göre) kapatır."""
        if execution_mode not in {"PAPER", "LIVE_SPOT", "LIVE_MARGIN"}:
            return False, "Bu yürütme yolu yalnızca Sanal Deneme, Canlı Spot ve Canlı Çapraz Marjin modlarını destekliyor."
        is_live_spot = execution_mode == "LIVE_SPOT"
        is_live_margin = execution_mode == "LIVE_MARGIN"
        track = "LIVE_MARGIN" if is_live_margin else ("LIVE_SPOT" if is_live_spot else "MAIN")
        open_positions = self.storage.get_open_positions(track=track)
        if symbol:
            open_positions = [p for p in open_positions if p.symbol == symbol]
        if not open_positions:
            return False, "Kapatılacak açık pozisyon bulunmuyor."

        spot_cache: dict[str, dict[str, Any]] = {}
        any_closed = False
        messages: list[str] = []
        for pos in open_positions:
            pos_quote = "USDT" if pos.symbol.endswith("USDT") else "USDC"
            try:
                if pos_quote not in spot_cache:
                    spot_cache[pos_quote] = self.market_client.discover_spot_pairs(
                        quote_asset=pos_quote,
                        allow_stablecoin_conversion=settings.allow_stablecoin_conversion,
                    )
                spot_map = spot_cache[pos_quote]
                if pos.symbol not in spot_map:
                    messages.append(f"{pos.symbol}: borsa filtre bilgisi alınamadı.")
                    continue
                if is_substep_dust_position(pos, spot_map[pos.symbol]):
                    continue
                now_utc = datetime.now(timezone.utc)
                if is_live_spot:
                    res = self.live_executor.execute_live_market_sell(
                        position=pos,
                        filters=spot_map[pos.symbol],
                        signal_key=f"MANUAL_LIVE_CLOSE:{pos.position_id}:{int(now_utc.timestamp())}",
                        strategy_source="KULLANICI_MANUEL",
                        close_reason="MANUEL_KAPATMA",
                        ui_live_confirmed=ui_live_confirmed,
                    )
                    self._resolve_live_unvalued_commissions(pos_quote, settings)
                    res = dataclasses.replace(
                        res,
                        explanation=res.explanation.replace(
                            " Üçüncü varlık komisyonunun değeri bilinmiyor; K/Z geçicidir, yeni alımlar durduruldu.",
                            " BNB komisyon indirimi uygulandı.",
                        ),
                    )
                elif is_live_margin:
                    res = self.live_margin_executor.execute_live_margin_sell(
                        position=pos,
                        filters=spot_map[pos.symbol],
                        signal_key=f"MANUAL_MARGIN_CLOSE:{pos.position_id}:{int(now_utc.timestamp())}",
                        strategy_source="KULLANICI_MANUEL",
                        close_reason="MANUEL_KAPATMA",
                        ui_live_confirmed=ui_live_confirmed,
                        ui_margin_confirmed=ui_margin_confirmed,
                    )
                else:
                    book = self.market_client.fetch_order_book(pos.symbol, depth_limit=20)
                    res = self.simulator.execute_sell_position(
                        position=pos,
                        filters=spot_map[pos.symbol],
                        order_book=book,
                        signal_key=f"MANUAL_CLOSE:{pos.position_id}:{int(now_utc.timestamp())}",
                        strategy_source="KULLANICI_MANUEL",
                        close_reason="MANUEL_KAPATMA",
                        settings=settings,
                        now_utc=now_utc,
                        track="MAIN",
                    )
                if res.status in ("FILLED", "PARTIALLY_FILLED"):
                    any_closed = True
                messages.append(res.explanation)
            except (MarketDataError, LiveOrderBlockedError, LiveExecutionError) as exc:
                messages.append(
                    f"{pos.symbol}: Güncel piyasa kotasyonu/bağlantısı alınamadığı için işlem yapılamadı ({exc})"
                )

        return any_closed, " | ".join(messages)

    def _htf_candles(self, symbol: str, interval: str = "15m", ttl_sec: float = 240.0):
        """Üst zaman dilimi mumları (önbellekli). Veri yoksa None -> filtre devre dışı (fail-open)."""
        fetch = getattr(self.market_client, "fetch_closed_candles", None)
        if fetch is None:
            return None
        key = (symbol, interval)
        now_mono = time.monotonic()
        cached = self._candle_cache.get(key)
        if cached and (now_mono - cached[0]) < ttl_sec:
            return cached[1]
        try:
            candles = fetch(symbol, interval, 60)
        except Exception:
            return None
        self._candle_cache[key] = (now_mono, candles)
        return candles

    def _expert_entry_block(self, symbol: str, track: str, now_utc: datetime) -> str | None:
        """Uzman trader ön elemesi: zarar serisi bekletmesi, düşen bıçak ve BTC risk-off filtresi."""
        cooldown = loss_streak_cooldown(self.storage.get_recent_closed_pnls(track, symbol, 4), now_utc)
        if cooldown:
            return f"{symbol} {cooldown}"
        coin_candles = self._htf_candles(symbol)
        if coin_candles:
            why = coin_knife_block(trend_info(coin_candles))
            if why:
                return f"{symbol}: {why}"
        btc_candles = coin_candles if symbol == "BTCUSDT" else self._htf_candles("BTCUSDT")
        if btc_candles:
            why = regime_block(trend_info(btc_candles))
            if why:
                return why
        return None

    def _expert_context_notes(self, symbol: str) -> list[str]:
        """Yapay zekâya (güvenilmeyen not olarak) trend ve rejim bağlamı."""
        notes: list[str] = []
        coin_candles = self._htf_candles(symbol)
        if coin_candles:
            notes.append(f"{symbol} 15dk trend: {trend_info(coin_candles).label()}")
        btc_candles = coin_candles if symbol == "BTCUSDT" else self._htf_candles("BTCUSDT")
        if btc_candles:
            notes.append(f"BTC 15dk trend: {trend_info(btc_candles).label()}")
        return notes

    def _expert_adjust_buy(
        self,
        settings: RiskSettings,
        market: MarketBundle,
        tech_signal: TechnicalSignal,
        risk_chk: Any,
        bid_map: dict[str, Decimal],
        track: str,
    ) -> tuple[RiskSettings, Decimal, Decimal]:
        """ATR tabanlı dinamik stop/hedef ve risk bazlı pozisyon büyüklüğü (asla büyütmez, yalnızca küçültür)."""
        qty = risk_chk.approved_quantity
        notional = risk_chk.approved_notional_usdc
        atr_pct = atr_pct_of_price(tech_signal.atr_14, tech_signal.close_price)
        sl, tp = dynamic_exits(
            base_sl=settings.stop_loss_pct,
            base_tp=settings.take_profit_pct,
            atr_pct=atr_pct,
            leverage=settings.leverage,
        )
        adjusted = dataclasses.replace(settings, stop_loss_pct=sl, take_profit_pct=tp)
        risk_pct = getattr(settings, "risk_per_trade_pct", Decimal("0"))
        price = market.order_book.best_ask or tech_signal.close_price
        if risk_pct > Decimal("0") and price > Decimal("0") and notional > Decimal("0"):
            try:
                equity = self.get_portfolio_summary(settings, bid_map, track=track)["total_equity_usdc"]
                capped = risk_capped_notional(
                    equity=equity,
                    risk_pct=risk_pct,
                    sl_pct=sl,
                    fee_rate=settings.estimated_fee_rate,
                    approved_notional=notional,
                )
            except Exception:
                capped = notional
            if capped < notional:
                target = max(capped, market.filters.min_notional * Decimal("1.05"))
                if target < notional:
                    new_qty = quantize_down(target / price, market.filters.market_step_size)
                    if (
                        new_qty >= max(market.filters.min_qty, market.filters.market_min_qty)
                        and new_qty * price >= market.filters.min_notional
                        and new_qty < qty
                    ):
                        qty = new_qty
                        notional = new_qty * price
        return adjusted, qty, notional

    def _resolve_live_unvalued_commissions(self, quote_ccy: str, settings: RiskSettings) -> None:
        """
        Binance 'BNB ile Komisyon İndirimi' gibi üçüncü varlık komisyonlarını (ör. BNB)
        güncel parite fiyatıyla (ör. BNBUSDT) portföy para biriminde (USDT/USDC) değerleyip
        ilgili emir/pozisyon muhasebesine işler ve `live_unvalued_commissions` kilidini temizler.
        """
        raw = self.storage.get_state("live_unvalued_commissions", "{}") or "{}"
        if raw.strip() in {"", "{}", "[]"}:
            return
        try:
            unvalued = json.loads(raw)
        except ValueError:
            self.storage.set_state("live_unvalued_commissions", "{}")
            return
        if not isinstance(unvalued, dict) or not unvalued:
            self.storage.set_state("live_unvalued_commissions", "{}")
            return

        q_norm = quote_ccy.strip().upper() if quote_ccy.strip().upper() in {"USDT", "USDC"} else "USDT"
        rate_cache: dict[str, Decimal] = {q_norm: Decimal("1"), "USDT": Decimal("1"), "USDC": Decimal("1")}

        for client_order_id, asset_map in list(unvalued.items()):
            if not isinstance(asset_map, dict):
                unvalued.pop(client_order_id, None)
                continue
            total_fee_quote = Decimal("0")
            for asset, amt_str in asset_map.items():
                asset_up = str(asset).strip().upper()
                try:
                    amt = Decimal(str(amt_str))
                except Exception:
                    continue
                if not amt.is_finite() or amt <= 0:
                    continue
                if asset_up not in rate_cache:
                    rate: Decimal | None = None
                    try:
                        ob = self.market_client.fetch_order_book(f"{asset_up}{q_norm}", depth_limit=5)
                        rate = ob.best_bid or ob.best_ask
                    except Exception:
                        rate = None
                    if rate is not None and rate > 0:
                        rate_cache[asset_up] = rate
                if asset_up in rate_cache:
                    total_fee_quote += amt * rate_cache[asset_up]

            with self.storage.atomic() as conn:
                row = conn.execute(
                    "SELECT signal_key, payload_json FROM live_order_intents WHERE payload_json LIKE ?",
                    (f"%{client_order_id}%",),
                ).fetchone()
                if row is not None:
                    sig_key, payload_raw = row[0], row[1]
                    try:
                        intent_data = json.loads(payload_raw)
                    except ValueError:
                        intent_data = {}
                    if total_fee_quote <= 0:
                        try:
                            app_q = Decimal(str(intent_data.get("applied_quote", "0")))
                            if app_q > 0:
                                total_fee_quote = app_q * settings.estimated_fee_rate
                        except Exception:
                            total_fee_quote = Decimal("0")
                    pos_id = intent_data.get("position_id")
                    side = str(intent_data.get("side", "")).upper()
                    if total_fee_quote > 0 and pos_id:
                        pos = self.storage.get_position(str(pos_id), track="LIVE_SPOT")
                        if pos is not None:
                            if side == "BUY":
                                conn.execute(
                                    "UPDATE positions SET entry_fee_usdc = ? WHERE position_id = ? AND portfolio_track = 'LIVE_SPOT'",
                                    (str(pos.entry_fee_usdc + total_fee_quote), str(pos_id)),
                                )
                            elif side == "SELL" and pos.status == "CLOSED":
                                new_exit_fee = (pos.exit_fee_usdc or Decimal("0")) + total_fee_quote
                                new_pnl = (pos.realized_pnl_usdc or Decimal("0")) - total_fee_quote
                                conn.execute(
                                    "UPDATE positions SET exit_fee_usdc = ?, realized_pnl_usdc = ? WHERE position_id = ? AND portfolio_track = 'LIVE_SPOT'",
                                    (str(new_exit_fee), str(new_pnl), str(pos_id)),
                                )
                    if sig_key:
                        ord_row = conn.execute(
                            "SELECT order_id, side, fee_usdc, realized_pnl_usdc, explanation FROM orders WHERE signal_key = ? AND portfolio_track = 'LIVE_SPOT' ORDER BY rowid DESC LIMIT 1",
                            (str(sig_key),),
                        ).fetchone()
                        if ord_row is not None:
                            oid, oside, ofee, opnl, oexp = ord_row
                            new_ofee = Decimal(str(ofee or "0")) + total_fee_quote
                            new_opnl = (
                                Decimal(str(opnl or "0")) - total_fee_quote
                                if str(oside).upper() == "SELL"
                                else Decimal(str(opnl or "0"))
                            )
                            new_exp = str(oexp or "").replace(
                                "Üçüncü varlık komisyonunun değeri bilinmiyor; K/Z geçicidir, yeni alımlar durduruldu.",
                                f"BNB komisyonu ({total_fee_quote:.4f} {q_norm}) değerlendi.",
                            )
                            conn.execute(
                                "UPDATE orders SET fee_usdc = ?, realized_pnl_usdc = ?, explanation = ? WHERE order_id = ? AND portfolio_track = 'LIVE_SPOT'",
                                (str(new_ofee), str(new_opnl), new_exp, str(oid)),
                            )
            unvalued.pop(client_order_id, None)

        self.storage.set_state(
            "live_unvalued_commissions",
            json.dumps(unvalued, sort_keys=True) if unvalued else "{}",
        )

    def _record_snapshot(
        self,
        symbol: str,
        mid_price: Decimal,
        best_bid: Decimal,
        settings: RiskSettings,
        now_iso: str,
        ist_date: str,
        track: str = "MAIN",
    ) -> None:
        positions = self.storage.get_open_positions(track=track)
        bid_map, missing = self._collect_position_bids(positions, {**self._latest_bid_map, symbol: best_bid})
        # A partial valuation is not a valid point on the performance curve.
        if missing:
            return
        main_sum = self.get_portfolio_summary(settings, bid_map, track=track)
        quote = "USDT" if symbol.endswith("USDT") else "USDC"
        shadow_track = self._shadow_track(quote)
        shadow_positions = self.storage.get_open_positions(track=shadow_track)
        shadow_bids, missing_shadow = self._collect_position_bids(shadow_positions, bid_map)
        if missing_shadow:
            return
        shadow_sum = self.get_portfolio_summary(settings, shadow_bids, track=shadow_track)
        bench_eq = self._compute_benchmark_equity(symbol, mid_price, settings)

        self.storage.record_equity_snapshot(
            created_at_iso=now_iso,
            istanbul_date=ist_date,
            symbol=symbol,
            main_equity_usdc=main_sum["total_equity_usdc"],
            cash_balance_usdc=main_sum["cash_usdc"],
            unrealized_pnl_usdc=main_sum["unrealized_pnl_usdc"],
            rule_shadow_equity_usdc=shadow_sum["total_equity_usdc"],
            benchmark_equity_usdc=bench_eq,
            track=track,
            quote_asset=quote,
        )
