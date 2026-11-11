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
from decimal import Decimal

from .domain import Candle, RiskSettings
from .strategy import (
    StrategyParams,
    compute_indicator_frame,
    evaluate_rule_signal_from_frame,
)

MIN_CANDLES_FOR_SPLIT = 120
MIN_TRADES_FOR_STATISTICAL_NOTE = 15


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
) -> SliceBacktestMetrics:
    """
    Nedensel (event-driven, lookahead-free) dilim simülasyonu:
    - Sinyal `t` indeksinde YALNIZCA `candles[:t+1]` kullanılarak hesaplanır.
    - İşlem dolumu `t+1` mumunun açılış fiyatında (`candles[t+1].open`) gerçekleşir!
    """
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

    peak_equity = initial_balance_usdc
    max_dd_usdc = Decimal("0")
    max_dd_pct = Decimal("0")

    friction_bps = half_spread_bps + settings.extra_slippage_bps
    buy_mult = Decimal("1") + (friction_bps / Decimal("10000"))
    sell_mult = Decimal("1") - (friction_bps / Decimal("10000"))

    # t mumunun kapanışında sinyal üretilir -> t+1 mumunun açılışında uygulanır (eval_end_idx - 1 son uygulanabilir t'dir)
    start_t = max(eval_start_idx, 35)
    for t in range(start_t, eval_end_idx - 1):
        sig = evaluate_rule_signal_from_frame(
            full_df, symbol=symbol, interval=interval, row_index=t, params=params
        )
        ai_cost_total += estimated_cost_per_ai_eval_usd
        next_bar = candles[t + 1]

        # Açık pozisyon varsa önce t+1 açılışında zarar-kes / kâr-al / SAT kontrolü
        if open_qty > Decimal("0"):
            exec_sell_price = next_bar.open * sell_mult
            stop_hit = (
                settings.stop_loss_pct > 0
                and next_bar.open <= entry_price * (Decimal("1") - settings.stop_loss_pct)
            )
            tp_hit = (
                settings.take_profit_pct > 0
                and next_bar.open >= entry_price * (Decimal("1") + settings.take_profit_pct)
            )
            if sig.action == "SAT" or stop_hit or tp_hit:
                gross_proceeds = open_qty * exec_sell_price
                exit_fee = gross_proceeds * settings.estimated_fee_rate
                net_proceeds = gross_proceeds - exit_fee
                pnl = net_proceeds - entry_total_outflow
                cash += net_proceeds
                total_fees += exit_fee
                trade_count += 1
                if pnl > Decimal("0"):
                    winning_trades += 1
                open_qty = Decimal("0")
                entry_total_outflow = Decimal("0")
                entry_price = Decimal("0")

        elif sig.action == "AL":
            exec_buy_price = next_bar.open * buy_mult
            max_gross = min(
                settings.max_position_size_usdc,
                cash / (Decimal("1") + settings.estimated_fee_rate),
            )
            if max_gross >= Decimal("5.0") and exec_buy_price > Decimal("0"):
                qty = max_gross / exec_buy_price
                gross_cost = qty * exec_buy_price
                entry_fee = gross_cost * settings.estimated_fee_rate
                total_out = gross_cost + entry_fee
                if total_out <= cash:
                    cash -= total_out
                    open_qty = qty
                    entry_price = exec_buy_price
                    entry_total_outflow = total_out
                    total_fees += entry_fee

        # t+1 kapanışında anlık özsermaye ve maksimum düşüş (drawdown) takibi
        mark_val = open_qty * (next_bar.close * sell_mult) * (Decimal("1") - settings.estimated_fee_rate)
        equity = cash + mark_val
        if equity > peak_equity:
            peak_equity = equity
        dd_usdc = peak_equity - equity
        dd_pct = (dd_usdc / peak_equity) * Decimal("100") if peak_equity > 0 else Decimal("0")
        if dd_usdc > max_dd_usdc:
            max_dd_usdc = dd_usdc
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

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
) -> ChronologicalValidationReport:
    """
    Veriyi kronolojik olarak %60 Eğitim / %20 Doğrulama / %20 Test dilimlerine böler.
    Parametre seçimi yalnızca Eğitim+Doğrulama diliminde yapılır; Test dilimi el değmemiş kalır.
    """
    clean_candles = sorted([c for c in closed_candles if c.is_closed], key=lambda c: c.open_time_ms)
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
            selected_params=StrategyParams(),
            train_metrics=empty_m,
            validation_metrics=empty_m,
            test_metrics=empty_m,
            overall_verdict="Sonuç çıkarmak için yetersiz veri.",
            insufficient_data=True,
        )

    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    candidates = [
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
