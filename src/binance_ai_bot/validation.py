"""
Kronolojik Doğrulama (Backtest) ve Küçük Bakiye Maliyet Uygulanabilirlik Modülü.

İlkeler:
- Gelecek veriyi sızdırmaz (No Lookahead): `t` mumunun kapanışında üretilen sinyal,
  yalnızca `t+1` mumunun açılış fiyatına spread + kayma + komisyon eklenerek yürütülür.
- Kronolojik Eğitim (%60) / Doğrulama (%20) / Test (%20) ayrımı uygular.
- Birkaç kazanan işlemi başarı kanıtı saymaz; yetersiz mum veya işlem sayısında
  açıkça "Sonuç çıkarmak için yetersiz veri" döndürür.
- 86 USDC küçük bakiyede komisyon ve model maliyetinin başabaş etkisini ayrıca hesaplar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from .domain import Candle, RiskSettings, SymbolFilters, quantize_down
from .security import STABLECOIN_OR_FIAT_ASSETS
from .strategy import (
    StrategyParams,
    compute_indicator_frame,
    evaluate_rule_signal_from_frame,
)

MIN_CANDLES_FOR_SPLIT = 120
MIN_TRADES_FOR_STATISTICAL_NOTE = 15


def _validate_candles(candles: list[Candle]) -> None:
    """Reject ambiguous historical input instead of silently changing its chronology."""
    previous_close = -1
    for candle in candles:
        values = (candle.open, candle.high, candle.low, candle.close)
        if not candle.is_closed:
            raise ValueError("Geçmiş test yalnızca kapanmış mumları kabul eder.")
        if (candle.open_time_ms <= previous_close
                or candle.close_time_ms < candle.open_time_ms):
            raise ValueError("Mumlar kronolojik, benzersiz ve çakışmasız olmalıdır.")
        if (any(not value.is_finite() or value <= 0 for value in values)
                or candle.high < max(candle.open, candle.close)
                or candle.low > min(candle.open, candle.close)
                or candle.low > candle.high
                or not candle.volume.is_finite() or candle.volume < 0):
            raise ValueError("Geçmiş test verisinde geçersiz OHLC/hacim değeri var.")
        previous_close = candle.close_time_ms


@dataclass(frozen=True)
class SliceBacktestMetrics:
    slice_name: str
    candle_count: int
    start_time_ms: int
    end_time_ms: int
    initial_equity_usdc: Decimal
    final_equity_usdc: Decimal
    net_return_usdc: Decimal
    net_return_pct: Decimal
    benchmark_return_pct: Decimal
    max_drawdown_pct: Decimal
    max_drawdown_usdc: Decimal
    trade_count: int
    winning_trades: int
    win_rate_pct: Decimal
    total_fees_usdc: Decimal
    estimated_ai_cost_usd: Decimal
    verdict_text: str
    insufficient_data: bool


@dataclass(frozen=True)
class ChronologicalValidationReport:
    symbol: str
    interval: str
    total_candles: int
    selected_params: StrategyParams
    train_metrics: SliceBacktestMetrics
    validation_metrics: SliceBacktestMetrics
    test_metrics: SliceBacktestMetrics
    overall_verdict: str
    insufficient_data: bool


@dataclass(frozen=True)
class SmallBalanceViabilityReport:
    balance_usdc: Decimal
    position_size_usdc: Decimal
    roundtrip_fee_usdc: Decimal
    roundtrip_spread_slippage_usdc: Decimal
    breakeven_move_pct_without_ai: Decimal
    monthly_estimated_ai_cost_usd: Decimal
    monthly_ai_drag_on_balance_pct: Decimal
    viability_status: str
    explanation_tr: str


def run_no_lookahead_slice_simulation(
    candles: list[Candle],
    *,
    eval_start_idx: int,
    eval_end_idx: int,
    slice_name: str,
    symbol: str,
    interval: str,
    params: StrategyParams,
    settings: RiskSettings,
    initial_balance_usdc: Decimal = Decimal("86.00"),
    half_spread_bps: Decimal = Decimal("1.5"),
    estimated_cost_per_ai_eval_usd: Decimal = Decimal("0"),
    symbol_filters: SymbolFilters | None = None,
) -> SliceBacktestMetrics:
    """
    Nedensel (event-driven, lookahead-free) dilim simülasyonu:
    - Sinyal `t` indeksinde YALNIZCA `candles[:t+1]` kullanılarak hesaplanır.
    - Giriş/sinyal çıkışı t+1 açılışında; koruyucu çıkışlar OHLC temas modeliyle uygulanır.
    - Mum içindeki sıra bilinmediğinden uzun pozisyonda olumsuz temas önce işlenir.
    """
    _validate_candles(candles)
    if settings.leverage != 1:
        raise ValueError("Bu geçmiş test yalnızca kaldıraçsız Spot işlemleri doğrular.")
    if not 0 <= eval_start_idx <= eval_end_idx <= len(candles):
        raise ValueError("Geçmiş test dilim sınırları geçersiz.")
    if not initial_balance_usdc.is_finite() or initial_balance_usdc <= 0:
        raise ValueError("Başlangıç bakiyesi sonlu ve pozitif olmalıdır.")
    if symbol_filters is not None and symbol_filters.symbol != symbol:
        raise ValueError("Borsa filtrelerinin paritesi test edilen pariteyle eşleşmiyor.")
    slice_candles = candles[eval_start_idx:eval_end_idx]
    if len(slice_candles) < 10 or eval_end_idx - eval_start_idx < 10:
        return SliceBacktestMetrics(
            slice_name=slice_name,
            candle_count=len(slice_candles),
            start_time_ms=slice_candles[0].open_time_ms if slice_candles else 0,
            end_time_ms=slice_candles[-1].close_time_ms if slice_candles else 0,
            initial_equity_usdc=initial_balance_usdc,
            final_equity_usdc=initial_balance_usdc,
            net_return_usdc=Decimal("0"),
            net_return_pct=Decimal("0"),
            benchmark_return_pct=Decimal("0"),
            max_drawdown_pct=Decimal("0"),
            max_drawdown_usdc=Decimal("0"),
            trade_count=0,
            winning_trades=0,
            win_rate_pct=Decimal("0"),
            total_fees_usdc=Decimal("0"),
            estimated_ai_cost_usd=Decimal("0"),
            verdict_text="Sonuç çıkarmak için yetersiz veri (dilim çok kısa).",
            insufficient_data=True,
        )

    # Gösterge serisi nedensel EWM/rolling kullandığı için `t` satırı yalnızca `0..t` mumlarına bağlıdır.
    full_df = compute_indicator_frame(candles[:eval_end_idx], params=params)

    cash = initial_balance_usdc
    open_qty = Decimal("0")
    entry_total_outflow = Decimal("0")
    entry_price = Decimal("0")
    total_fees = Decimal("0")
    trade_count = 0
    winning_trades = 0
    ai_cost_total = Decimal("0")
    daily_entries = 0
    daily_realized = Decimal("0")
    current_day = ""
    entries_paused = False
    istanbul = ZoneInfo("Europe/Istanbul")

    peak_equity = initial_balance_usdc
    max_dd_usdc = Decimal("0")
    max_dd_pct = Decimal("0")

    friction_bps = half_spread_bps + settings.extra_slippage_bps
    buy_mult = Decimal("1") + (friction_bps / Decimal("10000"))
    sell_mult = Decimal("1") - (friction_bps / Decimal("10000"))
    if not 0 < sell_mult <= 1 or not 0 <= settings.estimated_fee_rate < 1:
        raise ValueError("Komisyon/kayma varsayımları geçersiz.")

    def mark_equity(raw_price: Decimal) -> None:
        nonlocal peak_equity, max_dd_usdc, max_dd_pct
        value = cash + open_qty * raw_price * sell_mult * (Decimal("1") - settings.estimated_fee_rate)
        peak_equity = max(peak_equity, value)
        drawdown = peak_equity - value
        max_dd_usdc = max(max_dd_usdc, drawdown)
        if peak_equity > 0:
            max_dd_pct = max(max_dd_pct, drawdown / peak_equity * Decimal("100"))

    def check_daily_guard(raw_price: Decimal) -> bool:
        nonlocal entries_paused
        unrealized = (
            open_qty * raw_price * sell_mult * (Decimal("1") - settings.estimated_fee_rate)
            - entry_total_outflow if open_qty > 0 else Decimal("0")
        )
        breached = max(Decimal("0"), -(daily_realized + unrealized)) >= settings.daily_loss_limit_usdc
        entries_paused = entries_paused or breached
        return breached

    def permitted_quantity(qty: Decimal, price: Decimal) -> Decimal:
        if symbol_filters is None:
            return qty
        filt = symbol_filters
        step = filt.market_step_size if filt.market_step_size > 0 else filt.step_size
        positive_caps = [value for value in (filt.max_qty, filt.market_max_qty) if value > 0]
        if positive_caps:
            qty = min(qty, min(positive_caps))
        qty = quantize_down(qty, step)
        minimum = max(filt.min_qty, filt.market_min_qty)
        notional = qty * price
        if (qty <= 0 or qty < minimum
                or (filt.apply_min_to_market and notional < filt.min_notional)
                or (filt.max_notional > 0 and notional > filt.max_notional)):
            return Decimal("0")
        return qty

    def close_position(raw_price: Decimal) -> bool:
        nonlocal cash, open_qty, entry_total_outflow, entry_price
        nonlocal total_fees, trade_count, winning_trades, daily_realized
        execution_price = raw_price * sell_mult
        quantity = permitted_quantity(open_qty, execution_price)
        # A partial/dust remainder stays open; historical fills cannot bypass exchange rules.
        if quantity <= 0:
            return False
        allocated_cost = entry_total_outflow * quantity / open_qty
        gross = quantity * execution_price
        fee = gross * settings.estimated_fee_rate
        pnl = gross - fee - allocated_cost
        cash += gross - fee
        total_fees += fee
        daily_realized += pnl
        trade_count += 1
        winning_trades += int(pnl > 0)
        open_qty -= quantity
        entry_total_outflow -= allocated_cost
        if open_qty <= 0:
            entry_price = Decimal("0")
            entry_total_outflow = Decimal("0")
        check_daily_guard(raw_price)
        return True

    # t mumunun kapanışında sinyal üretilir -> t+1 mumunun açılışında uygulanır (eval_end_idx - 1 son uygulanabilir t'dir)
    start_t = max(eval_start_idx, 35)
    for t in range(start_t, eval_end_idx - 1):
        sig = evaluate_rule_signal_from_frame(
            full_df, symbol=symbol, interval=interval, row_index=t, params=params
        )
        ai_cost_total += estimated_cost_per_ai_eval_usd
        next_bar = candles[t + 1]
        bar_day = datetime.fromtimestamp(next_bar.open_time_ms / 1000, tz=timezone.utc).astimezone(istanbul).date().isoformat()
        if bar_day != current_day:
            current_day = bar_day
            daily_entries = 0
            daily_realized = Decimal("0")
            entries_paused = False

        # Existing positions are protected even when new entries have been paused.
        breached_at_open = check_daily_guard(next_bar.open)
        exited_at_open = False
        if open_qty > Decimal("0"):
            stop_hit = (
                settings.stop_loss_pct > 0
                and next_bar.open <= entry_price * (Decimal("1") - settings.stop_loss_pct)
            )
            tp_hit = (
                settings.take_profit_pct > 0
                and next_bar.open >= entry_price * (Decimal("1") + settings.take_profit_pct)
            )
            if (sig.action == "SAT" or stop_hit or tp_hit
                    or (breached_at_open and settings.daily_limit_exit_policy == "AUTO_CLOSE")):
                mark_equity(next_bar.open)
                exited_at_open = close_position(next_bar.open)

        elif (sig.action == "AL" and not entries_paused
                and daily_entries < settings.max_daily_new_positions
                and settings.max_open_positions >= 1):
            exec_buy_price = next_bar.open * buy_mult
            max_gross = min(
                settings.max_position_size_usdc,
                cash / (Decimal("1") + settings.estimated_fee_rate),
            )
            filters_allow = (
                symbol_filters is None or (
                    symbol_filters.is_tradable_spot_for_quote()
                    and (settings.allow_stablecoin_conversion
                         or symbol_filters.base_asset not in STABLECOIN_OR_FIAT_ASSETS)
                )
            )
            if max_gross > 0 and exec_buy_price > 0 and filters_allow:
                qty = permitted_quantity(max_gross / exec_buy_price, exec_buy_price)
                gross_cost = qty * exec_buy_price
                entry_fee = gross_cost * settings.estimated_fee_rate
                total_out = gross_cost + entry_fee
                if qty > 0 and total_out <= cash:
                    cash -= total_out
                    open_qty = qty
                    entry_price = exec_buy_price
                    entry_total_outflow = total_out
                    total_fees += entry_fee
                    daily_entries += 1

        if open_qty > 0 and not exited_at_open:
            # Model the adverse excursion first. A gap fills at the open, not at an
            # unavailable stop price; simultaneous stop/target touches choose the stop.
            adverse_levels: list[Decimal] = []
            if settings.stop_loss_pct > 0:
                stop_level = entry_price * (Decimal("1") - settings.stop_loss_pct)
                if next_bar.low <= stop_level:
                    adverse_levels.append(min(next_bar.open, stop_level))
            if settings.daily_limit_exit_policy == "AUTO_CLOSE":
                loss_level = (entry_total_outflow - daily_realized - settings.daily_loss_limit_usdc) / (
                    open_qty * sell_mult * (Decimal("1") - settings.estimated_fee_rate)
                )
                if loss_level > 0 and next_bar.low <= loss_level:
                    adverse_levels.append(min(next_bar.open, loss_level))
            if adverse_levels:
                exit_reference = max(adverse_levels)
                check_daily_guard(exit_reference)
                mark_equity(exit_reference)
                if not close_position(exit_reference):
                    check_daily_guard(next_bar.low)
                    mark_equity(next_bar.low)
            else:
                check_daily_guard(next_bar.low)
                target = entry_price * (Decimal("1") + settings.take_profit_pct)
                if settings.take_profit_pct > 0 and next_bar.high >= target:
                    mark_equity(next_bar.low)
                    close_position(max(next_bar.open, target))
                elif open_qty > 0:
                    # With no target exit, high-before-low gives the conservative
                    # drawdown bound for an OHLC bar whose event order is unknown.
                    mark_equity(next_bar.high)
                    mark_equity(next_bar.low)
        mark_equity(next_bar.close)

    # Dilim sonunda açık pozisyon kaldıysa son kapanış alış değerinden değerle
    last_close = slice_candles[-1].close
    final_pos_val = (
        open_qty * (last_close * sell_mult) * (Decimal("1") - settings.estimated_fee_rate)
        if open_qty > 0
        else Decimal("0")
    )
    final_equity = cash + final_pos_val
    net_ret_usdc = final_equity - initial_balance_usdc
    net_ret_pct = (net_ret_usdc / initial_balance_usdc) * Decimal("100")

    # Eşdeğer riskli referans (10 USDC pariteye yatırılmış, 76 USDC nakitte tutulmuş Al-ve-Tut)
    first_open = slice_candles[0].open
    alloc_bench = min(settings.max_position_size_usdc, initial_balance_usdc)
    if first_open > 0:
        asset_ret_ratio = (last_close - first_open) / first_open
        bench_net_usdc = (alloc_bench * asset_ret_ratio) - (alloc_bench * settings.estimated_fee_rate * Decimal("2"))
        bench_ret_pct = (bench_net_usdc / initial_balance_usdc) * Decimal("100")
    else:
        bench_ret_pct = Decimal("0")

    win_rate = (
        (Decimal(winning_trades) / Decimal(trade_count)) * Decimal("100")
        if trade_count > 0
        else Decimal("0")
    )

    insufficient = trade_count < MIN_TRADES_FOR_STATISTICAL_NOTE
    if insufficient:
        verdict = (
            f"Sonuç çıkarmak için yetersiz veri: Bu dilimde yalnızca {trade_count} tamamlanmış işlem "
            f"oluştu (anlamlı değerlendirme için en az {MIN_TRADES_FOR_STATISTICAL_NOTE} işlem gerekir). "
            "Birkaç kazanan veya kaybeden işlem başarı/başarısızlık kanıtı değildir."
        )
    else:
        verdict = (
            f"{slice_name} diliminde {trade_count} işlem tamamlandı (Net: {net_ret_usdc:+.2f} USDC / "
            f"%{net_ret_pct:+.2f}, Maks. Düşüş: %{max_dd_pct:.2f}, Komisyon: {total_fees:.3f} USDC). "
            "Geçmiş performans gelecekteki getiriyi garanti etmez."
        )
    verdict += (
        " Tek parite Spot simülasyonu; mum içi temaslarda zarar-kes önce uygulanır."
        " Günlük giriş/risk kilidi İstanbul gününde sıfırlanır; manuel duraklatmalar modellenmez."
    )
    if symbol_filters is None:
        verdict += " Borsa filtreleri sağlanmadı; minimum tutar/miktar ve adım uygunluğu doğrulanmadı."
    else:
        verdict += " Sağlanan borsa filtreleri uygulanır; geçmiş dönemdeki filtre değişiklikleri modellenmez."

    return SliceBacktestMetrics(
        slice_name=slice_name,
        candle_count=len(slice_candles),
        start_time_ms=slice_candles[0].open_time_ms,
        end_time_ms=slice_candles[-1].close_time_ms,
        initial_equity_usdc=initial_balance_usdc,
        final_equity_usdc=final_equity,
        net_return_usdc=net_ret_usdc,
        net_return_pct=net_ret_pct,
        benchmark_return_pct=bench_ret_pct,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_usdc=max_dd_usdc,
        trade_count=trade_count,
        winning_trades=winning_trades,
        win_rate_pct=win_rate,
        total_fees_usdc=total_fees,
        estimated_ai_cost_usd=ai_cost_total,
        verdict_text=verdict,
        insufficient_data=insufficient,
    )


def run_chronological_validation(
    closed_candles: list[Candle],
    *,
    symbol: str,
    interval: str,
    settings: RiskSettings,
    initial_balance_usdc: Decimal = Decimal("86.00"),
    strategy_params: StrategyParams | None = None,
    symbol_filters: SymbolFilters | None = None,
) -> ChronologicalValidationReport:
    """
    Veriyi kronolojik olarak %60 Eğitim / %20 Doğrulama / %20 Test dilimlerine böler.
    strategy_params verilirse seçili strateji değiştirilmeden üç dilimde doğrulanır.
    Verilmezse parametre seçimi Eğitim+Doğrulama dilimindedir; Test dilimi el değmemiş kalır.
    """
    _validate_candles(closed_candles)
    if settings.leverage != 1:
        raise ValueError("Bu geçmiş test yalnızca kaldıraçsız Spot işlemleri doğrular.")
    clean_candles = list(closed_candles)
    n = len(clean_candles)

    if n < MIN_CANDLES_FOR_SPLIT:
        empty_m = SliceBacktestMetrics(
            slice_name="Yetersiz Veri",
            candle_count=n,
            start_time_ms=clean_candles[0].open_time_ms if clean_candles else 0,
            end_time_ms=clean_candles[-1].close_time_ms if clean_candles else 0,
            initial_equity_usdc=initial_balance_usdc,
            final_equity_usdc=initial_balance_usdc,
            net_return_usdc=Decimal("0"),
            net_return_pct=Decimal("0"),
            benchmark_return_pct=Decimal("0"),
            max_drawdown_pct=Decimal("0"),
            max_drawdown_usdc=Decimal("0"),
            trade_count=0,
            winning_trades=0,
            win_rate_pct=Decimal("0"),
            total_fees_usdc=Decimal("0"),
            estimated_ai_cost_usd=Decimal("0"),
            verdict_text=(
                f"Sonuç çıkarmak için yetersiz veri (mevcut mum: {n}, "
                f"kronolojik üçlü ayrım için gereken en az: {MIN_CANDLES_FOR_SPLIT})."
            ),
            insufficient_data=True,
        )
        return ChronologicalValidationReport(
            symbol=symbol,
            interval=interval,
            total_candles=n,
            selected_params=strategy_params or StrategyParams(),
            train_metrics=empty_m,
            validation_metrics=empty_m,
            test_metrics=empty_m,
            overall_verdict="Sonuç çıkarmak için yetersiz veri.",
            insufficient_data=True,
        )

    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    candidates = [strategy_params] if strategy_params is not None else [
        StrategyParams(ema_fast_span=9, ema_slow_span=21, rsi_buy_min=44.0, rsi_buy_max=68.0),
        StrategyParams(ema_fast_span=12, ema_slow_span=26, rsi_buy_min=46.0, rsi_buy_max=66.0),
        StrategyParams(ema_fast_span=8, ema_slow_span=24, rsi_buy_min=45.0, rsi_buy_max=70.0),
    ]

    best_params = candidates[0]
    best_score = Decimal("-999999")
    best_train_m: SliceBacktestMetrics | None = None
    best_val_m: SliceBacktestMetrics | None = None

    for cand in candidates:
        tr_m = run_no_lookahead_slice_simulation(
            clean_candles,
            eval_start_idx=0,
            eval_end_idx=train_end,
            slice_name="Eğitim (%60)",
            symbol=symbol,
            interval=interval,
            params=cand,
            settings=settings,
            initial_balance_usdc=initial_balance_usdc,
            symbol_filters=symbol_filters,
        )
        va_m = run_no_lookahead_slice_simulation(
            clean_candles,
            eval_start_idx=train_end,
            eval_end_idx=val_end,
            slice_name="Doğrulama (%20)",
            symbol=symbol,
            interval=interval,
            params=cand,
            settings=settings,
            initial_balance_usdc=initial_balance_usdc,
            symbol_filters=symbol_filters,
        )
        # Doğrulama diliminde düşüş cezalı net getiri skoru (Test dilimine ASLA bakılmaz!)
        score = (tr_m.net_return_usdc * Decimal("0.4")) + (va_m.net_return_usdc * Decimal("0.6")) - va_m.max_drawdown_usdc
        if score > best_score:
            best_score = score
            best_params = cand
            best_train_m = tr_m
            best_val_m = va_m

    assert best_train_m is not None and best_val_m is not None

    test_m = run_no_lookahead_slice_simulation(
        clean_candles,
        eval_start_idx=val_end,
        eval_end_idx=n,
        slice_name="Örneklem Dışı Test (%20)",
        symbol=symbol,
        interval=interval,
        params=best_params,
        settings=settings,
        initial_balance_usdc=initial_balance_usdc,
        symbol_filters=symbol_filters,
    )

    overall_insufficient = test_m.insufficient_data
    if overall_insufficient:
        overall_verdict = (
            f"Sonuç çıkarmak için yetersiz veri: Örneklem dışı test diliminde {test_m.trade_count} "
            f"işlem oluştu (en az {MIN_TRADES_FOR_STATISTICAL_NOTE} işlem önerilir). "
            "Kısa dönemli birkaç işlem stratejinin kârlı olduğunu kanıtlamaz."
        )
    else:
        overall_verdict = (
            f"Örneklem dışı test sonucu: Net {test_m.net_return_usdc:+.2f} USDC "
            f"(%{test_m.net_return_pct:+.2f}), Referans Al-Tut: %{test_m.benchmark_return_pct:+.2f}, "
            f"Maksimum Düşüş: %{test_m.max_drawdown_pct:.2f}, Toplam Komisyon: {test_m.total_fees_usdc:.3f} USDC."
        )
    overall_verdict += (
        " Seçili profil sabit tutularak doğrulandı; alternatif parametre seçilmedi."
        if strategy_params is not None
        else " Parametreler eğitim/doğrulama dilimlerinde seçildi; test dilimi seçime katılmadı."
    )
    overall_verdict += " Tek parite Spot testi; OHLC temas sırası belirsizse zarar-kes önce gelir."
    if symbol_filters is None:
        overall_verdict += " Borsa filtreleri sağlanmadığından emir uygunluğu doğrulanmadı."

    return ChronologicalValidationReport(
        symbol=symbol,
        interval=interval,
        total_candles=n,
        selected_params=best_params,
        train_metrics=best_train_m,
        validation_metrics=best_val_m,
        test_metrics=test_m,
        overall_verdict=overall_verdict,
        insufficient_data=overall_insufficient,
    )


def assess_small_balance_viability(
    *,
    balance_usdc: Decimal,
    position_size_usdc: Decimal,
    fee_rate: Decimal,
    extra_slippage_bps: Decimal,
    daily_ai_calls: int,
    input_usd_per_1m: Decimal,
    output_usd_per_1m: Decimal,
    avg_prompt_tokens: int = 450,
    avg_output_tokens: int = 180,
) -> SmallBalanceViabilityReport:
    """
    86 USDC gibi küçük bir bakiyede komisyon, spread/kayma ve ücretli yapay zekâ çağrılarının
    portföy üzerindeki gerçek yükünü açık biçimde değerlendirir.
    """
    roundtrip_fee = position_size_usdc * fee_rate * Decimal("2")
    # Tipik likit paritede 2 bps spread + 2 * extra_slippage_bps gidiş-dönüş maliyeti
    est_friction_bps = Decimal("2.0") + (extra_slippage_bps * Decimal("2"))
    roundtrip_slip = position_size_usdc * (est_friction_bps / Decimal("10000"))
    total_trade_friction = roundtrip_fee + roundtrip_slip
    breakeven_pct = (
        (total_trade_friction / position_size_usdc) * Decimal("100")
        if position_size_usdc > 0
        else Decimal("0")
    )

    cost_per_call = (
        (Decimal(avg_prompt_tokens) / Decimal("1000000")) * input_usd_per_1m
        + (Decimal(avg_output_tokens) / Decimal("1000000")) * output_usd_per_1m
    )
    monthly_ai_cost = cost_per_call * Decimal(daily_ai_calls) * Decimal("30")
    monthly_drag_pct = (
        (monthly_ai_cost / balance_usdc) * Decimal("100") if balance_usdc > 0 else Decimal("0")
    )

    if breakeven_pct > Decimal("0.60") or monthly_drag_pct > Decimal("1.0"):
        status = "YÜKSEK MALİYET BASKISI — DİKKATLİ OLUNMALI"
    elif breakeven_pct > Decimal("0.30") or monthly_drag_pct > Decimal("0.30"):
        status = "ORTA MALİYET DUYARLILIĞI — DÜŞÜK İŞLEM SIKLIĞI GEREKLİ"
    else:
        status = "KONTROLLÜ MALİYET — ÖNBELLEK VE ÜCRETSİZ KATMAN ÖNERİLİR"

    explanation = (
        f"{position_size_usdc:.2f} USDC büyüklüğündeki her sanal al-sat turunda tahmini "
        f"{roundtrip_fee:.4f} USDC komisyon + {roundtrip_slip:.4f} USDC makas/kayma olmak üzere "
        f"yaklaşık {total_trade_friction:.4f} USDC maliyet oluşur; yani işlemin yalnızca masrafını "
        f"çıkarması için fiyatın en az %{breakeven_pct:.2f} lehinize hareket etmesi gerekir. "
        f"Günde {daily_ai_calls} ücretli model çağrısı yapılırsa aylık tahmini ~{monthly_ai_cost:.4f} USD "
        f"(%{monthly_drag_pct:.2f} kasa yükü) ek maliyet doğabilir. 86 USDC bakiyede çok sık al-sat "
        f"(scalping) ve yoğun ücretli API kullanımı komisyon erimesine yol açar; 15m/1h gibi daha "
        f"sakin periyotlar, önbellek ve ücretsiz katman kullanımı daha uygundur."
    )

    return SmallBalanceViabilityReport(
        balance_usdc=balance_usdc,
        position_size_usdc=position_size_usdc,
        roundtrip_fee_usdc=roundtrip_fee,
        roundtrip_spread_slippage_usdc=roundtrip_slip,
        breakeven_move_pct_without_ai=breakeven_pct,
        monthly_estimated_ai_cost_usd=monthly_ai_cost,
        monthly_ai_drag_on_balance_pct=monthly_drag_pct,
        viability_status=status,
        explanation_tr=explanation,
    )
