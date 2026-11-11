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
from typing import Any

from .ai_advisor import GeminiDecisionAdvisor
from .config import AppConfig
from .domain import (
    AIDecisionRecord,
    MarketBundle,
    RiskSettings,
    SimulatedOrderResult,
    TechnicalSignal,
)
from .live_executor import BinanceLiveSpotExecutor, LiveExecutionError
from .market_data import BinancePublicDataClient, MarketDataError
from .risk import RiskManager, compute_open_position_unrealized_net_pnl
from .security import LiveOrderBlockedError
from .simulator import VirtualExecutionSimulator
from .storage import SQLiteStorage
from .strategy import StrategyParams, evaluate_technical_strategy


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
        self.ai_advisor = ai_advisor or GeminiDecisionAdvisor(config, self.storage)
        # Çoklu coin taramasında True yapılır: pasif referans ilk paritede sabit kalır
        self.lock_benchmark_symbol = False

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

        open_gross_val = Decimal("0")
        unrealized_net_pnl = Decimal("0")
        for pos in open_positions:
            bid = bids.get(pos.symbol, pos.entry_price)
            gross_v, unreal_pnl = compute_open_position_unrealized_net_pnl(
                pos, current_bid=bid, estimated_fee_rate=settings.estimated_fee_rate
            )
            open_gross_val += gross_v
            unrealized_net_pnl += unreal_pnl

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
            "total_equity_usdc": total_equity,
            "realized_pnl_usdc": total_realized_pnl,
            "unrealized_pnl_usdc": unrealized_net_pnl,
            "daily_net_pnl_usdc": daily_net_pnl,
            "total_fees_usdc": total_fees,
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
        shadow_open = self.storage.get_open_positions(track="RULE_SHADOW")
        best_bid = market.order_book.best_bid
        shadow_pos = next((p for p in shadow_open if p.symbol == market.symbol), None)
        if shadow_pos is not None and best_bid is not None:
            stop_hit = shadow_pos.stop_loss_price is not None and best_bid <= shadow_pos.stop_loss_price
            tp_hit = shadow_pos.take_profit_price is not None and best_bid >= shadow_pos.take_profit_price
            if stop_hit or tp_hit or tech_signal.action == "SAT":
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
                    track="RULE_SHADOW",
                )
                return

        if shadow_pos is None and len(shadow_open) < settings.max_open_positions and tech_signal.action == "AL":
            chk = self.risk_manager.check_buy_entry(
                filters=market.filters,
                order_book=market.order_book,
                settings=settings,
                now_utc=now_utc,
                track="RULE_SHADOW",
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
                    track="RULE_SHADOW",
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
        strategy_params: StrategyParams | None = None,
        ai_only_on_rule_signal: bool = False,
        skip_balance_prefetch: bool = False,
    ) -> StepCycleSummary:
        """
        Tek bir tam denetimli analiz ve yürütme döngüsü çalıştırır.
        Varsayılan `execution_mode="PAPER"` sanal denemedir (`MAIN` izi).
        `execution_mode="LIVE_SPOT"` yalnızca tüm 5 güvenlik kilidi açıksa gerçek Spot emir iletir (`LIVE_SPOT` izi).
        """
        is_live_mode = execution_mode == "LIVE_SPOT"
        active_track = "LIVE_SPOT" if is_live_mode else "MAIN"
        quote_ccy = "USDT" if symbol.strip().upper().endswith("USDT") else "USDC"
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        ist_date = self.storage.istanbul_date_str(now_utc)

        if is_live_mode and not skip_balance_prefetch:
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

        # 3. Gölge kural tabanlı karşılaştırma portföyünü güncelle
        self._update_rule_shadow_track(market, tech_signal, settings, now_utc)

        open_positions = self.storage.get_open_positions(track=active_track)
        symbol_open_pos = next((p for p in open_positions if p.symbol == market.symbol), None)
        has_open_pos = symbol_open_pos is not None

        # 4. Günlük zarar limiti ve açık pozisyon koruyucu çıkış denetimi
        bid_map = {pos.symbol: (best_bid if pos.symbol == market.symbol else pos.entry_price) for pos in open_positions}
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
            self.storage.set_new_entries_paused(True, halt_msg)
            if auto_close_due_to_daily_limit and symbol_open_pos is not None:
                if is_live_mode:
                    closed_ord = self.live_executor.execute_live_market_sell(
                        position=symbol_open_pos,
                        filters=market.filters,
                        signal_key=f"DAILY_LOSS_AUTO_EXIT:{ist_date}:{symbol_open_pos.position_id}",
                        strategy_source="RISK_KORUMA",
                        close_reason="GUNLUK_ZARAR_OTOMATIK_CIKIS",
                        ui_live_confirmed=ui_live_confirmed,
                    )
                else:
                    closed_ord = self.simulator.execute_sell_position(
                        position=symbol_open_pos,
                        filters=market.filters,
                        order_book=market.order_book,
                        signal_key=f"DAILY_LOSS_AUTO_EXIT:{ist_date}:{symbol_open_pos.position_id}",
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

        # Açık pozisyonun Zarar-Kes (Stop-Loss) veya Kâr-Al (Take-Profit) kontrolü
        if symbol_open_pos is not None:
            pos = symbol_open_pos
            stop_hit = pos.stop_loss_price is not None and best_bid <= pos.stop_loss_price
            tp_hit = pos.take_profit_price is not None and best_bid >= pos.take_profit_price
            if stop_hit or tp_hit:
                exit_tag = "ZARAR_KES" if stop_hit else "KAR_AL"
                if is_live_mode:
                    closed_ord = self.live_executor.execute_live_market_sell(
                        position=pos,
                        filters=market.filters,
                        signal_key=f"{exit_tag}:{pos.position_id}:{tech_signal.candle_open_time_ms}",
                        strategy_source="RISK_KORUMA",
                        close_reason=exit_tag,
                        ui_live_confirmed=ui_live_confirmed,
                    )
                else:
                    closed_ord = self.simulator.execute_sell_position(
                        position=pos,
                        filters=market.filters,
                        order_book=market.order_book,
                        signal_key=f"{exit_tag}:{pos.position_id}:{tech_signal.candle_open_time_ms}",
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
        ai_decision = self.ai_advisor.evaluate(
            market=market,
            tech_signal=tech_signal,
            recent_returns_pct=recent_returns,
            has_open_position=has_open_pos,
            ai_enabled_by_user=ai_enabled_by_user and not skip_ai_prefilter,
            daily_call_limit_override=daily_ai_limit_override,
            min_interval_override_sec=min_ai_interval_override,
        )
        if skip_ai_prefilter and ai_enabled_by_user:
            ai_decision = dataclasses.replace(ai_decision, mode_label="KURAL_ON_ELEME")

        final_action = ai_decision.action
        executed_order: SimulatedOrderResult | None = None
        info_msg = ai_decision.reason
        block_reason: str | None = None

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
                paused, pause_reason = self.storage.is_new_entries_paused()
                if paused:
                    block_reason = f"Yeni pozisyon açılışı durdurulmuş durumda ({pause_reason})."
                    info_msg = block_reason
                else:
                    risk_chk = self.risk_manager.check_buy_entry(
                        filters=market.filters,
                        order_book=market.order_book,
                        settings=settings,
                        now_utc=now_utc,
                        track=active_track,
                    )
                    if not risk_chk.allowed:
                        block_reason = risk_chk.reason
                        info_msg = f"AL sinyali risk denetiminde durduruldu: {risk_chk.reason}"
                    else:
                        sig_key = f"{market.symbol}:{market.interval}:{tech_signal.candle_open_time_ms}:AL"
                        if is_live_mode:
                            try:
                                ord_res = self.live_executor.execute_live_market_buy(
                                    filters=market.filters,
                                    approved_qty=risk_chk.approved_quantity,
                                    signal_key=sig_key,
                                    strategy_source=ai_decision.mode_label,
                                    settings=settings,
                                    ui_live_confirmed=ui_live_confirmed,
                                )
                                executed_order = ord_res
                                info_msg = ord_res.explanation
                            except (LiveOrderBlockedError, LiveExecutionError) as exc:
                                block_reason = str(exc)
                                info_msg = f"Canlı Spot ALIM engellendi/reddedildi: {exc}"
                        else:
                            ord_res = self.simulator.execute_buy(
                                filters=market.filters,
                                order_book=market.order_book,
                                requested_qty=risk_chk.approved_quantity,
                                signal_key=sig_key,
                                strategy_source=ai_decision.mode_label,
                                settings=settings,
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
                if is_live_mode:
                    try:
                        ord_res = self.live_executor.execute_live_market_sell(
                            position=symbol_open_pos,
                            filters=market.filters,
                            signal_key=sig_key,
                            strategy_source=ai_decision.mode_label,
                            close_reason="STRATEJI_SAT_SINYALI",
                            ui_live_confirmed=ui_live_confirmed,
                        )
                        executed_order = ord_res
                        info_msg = ord_res.explanation
                    except (LiveOrderBlockedError, LiveExecutionError) as exc:
                        block_reason = str(exc)
                        info_msg = f"Canlı Spot SATIŞ hatası: {exc}"
                else:
                    ord_res = self.simulator.execute_sell_position(
                        position=symbol_open_pos,
                        filters=market.filters,
                        order_book=market.order_book,
                        signal_key=sig_key,
                        strategy_source=ai_decision.mode_label,
                        close_reason="STRATEJI_SAT_SINYALI",
                        settings=settings,
                        now_utc=now_utc,
                        track="MAIN",
                    )
                    executed_order = ord_res
                    info_msg = ord_res.explanation
            else:
                final_action = "BEKLE"
                block_reason = "Bu paritede kapatılacak açık spot pozisyon yok (açığa satış kapalı)."
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

    def manual_close_open_position(
        self,
        settings: RiskSettings,
        *,
        execution_mode: str = "PAPER",
        ui_live_confirmed: bool = False,
        symbol: str | None = None,
    ) -> tuple[bool, str]:
        """Açık pozisyonları (Sanal veya Canlı Spot moduna göre) anlık piyasa alış fiyatından kapatır."""
        is_live = execution_mode == "LIVE_SPOT"
        track = "LIVE_SPOT" if is_live else "MAIN"
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
                now_utc = datetime.now(timezone.utc)
                if is_live:
                    res = self.live_executor.execute_live_market_sell(
                        position=pos,
                        filters=spot_map[pos.symbol],
                        signal_key=f"MANUAL_LIVE_CLOSE:{pos.position_id}:{int(now_utc.timestamp())}",
                        strategy_source="KULLANICI_MANUEL",
                        close_reason="MANUEL_KAPATMA",
                        ui_live_confirmed=ui_live_confirmed,
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
        bid_map = {symbol: best_bid}
        main_sum = self.get_portfolio_summary(settings, bid_map, track=track)
        shadow_sum = self.get_portfolio_summary(settings, bid_map, track="RULE_SHADOW")
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
        )
