"""
Kesintisiz Arka Plan Alım-Satım Botu (tarayıcıdan bağımsız).

Kullanım (proje klasöründen):
    .venv\\Scripts\\python.exe bot_calistir.py --mod sanal --profil aktif --parite BTCUSDT
    .venv\\Scripts\\python.exe bot_calistir.py --mod canli --profil aktif --parite BTCUSDT

Güvenlik:
- Canlı modda `.env` içinde ENABLE_LIVE_TRADING=true + API anahtarları ZORUNLUDUR ve
  başlangıçta konsola elle "EVET" yazılarak onay verilmesi gerekir.
- Aynı anda yalnızca TEK bir arka plan botu çalışabilir (dosya kilidi).
- Panel bu bot çalışırken kendi otomatik döngüsünü ve analiz düğmesini kilitler (çift emir önlemi).
- Pozisyon büyüklüğü ve günlük zarar eşiği profillerden bağımsız korunur.
- Bot çalıştığı sürece Windows'un uykuya geçmesi engellenir (ekran kapanabilir).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .config import PROJECT_ROOT, AppConfig, load_config
from .domain import is_substep_dust_position
from .engine import StepCycleSummary, TradingSystemEngine
from .profiles import TradingProfile, get_profile
from .risk import compute_open_position_unrealized_net_pnl

RUNNER_HEARTBEAT_KEY = "runner_heartbeat_iso"
RUNNER_INFO_KEY = "runner_info_json"
RUNNER_STOP_KEY = "runner_stop_requested"
RUNNER_STALE_AFTER_SECONDS = 150

DEFAULT_SCAN_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "SUIUSDT",
]

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def is_runner_alive(storage, now_utc: datetime | None = None) -> tuple[bool, dict]:
    """Panelin arka plan botunun canlı olup olmadığını kalp atışından anlaması için yardımcı."""
    raw_hb = storage.get_state(RUNNER_HEARTBEAT_KEY, "")
    if not raw_hb:
        return False, {}
    try:
        hb = datetime.fromisoformat(raw_hb)
    except ValueError:
        return False, {}
    now = now_utc or datetime.now(timezone.utc)
    alive = (now - hb).total_seconds() <= RUNNER_STALE_AFTER_SECONDS
    try:
        info = json.loads(storage.get_state(RUNNER_INFO_KEY, "") or "{}")
    except json.JSONDecodeError:
        info = {}
    info["heartbeat_iso"] = raw_hb
    return alive, info


def _stop_requested(storage) -> bool:
    return storage.get_state(RUNNER_STOP_KEY, "0") == "1"


def _wait_with_heartbeat(storage, seconds: float, sleep_fn=time.sleep) -> bool:
    """Beklerken canlılık kaydını yeniler; durdurma isteğinde en geç 5 sn'de döner."""
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        storage.set_state(RUNNER_HEARTBEAT_KEY, datetime.now(timezone.utc).isoformat())
        if _stop_requested(storage):
            return False
        chunk = min(remaining, 5.0)
        sleep_fn(chunk)
        remaining -= chunk
    storage.set_state(RUNNER_HEARTBEAT_KEY, datetime.now(timezone.utc).isoformat())
    return not _stop_requested(storage)


def _keep_windows_awake(enable: bool) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        flags = (_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED) if enable else _ES_CONTINUOUS
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


class _SingleInstanceLock:
    """Aynı anda iki arka plan botunun çalışıp çift emir göndermesini engeller."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            self._fh.close()
            self._fh = None
            return False

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self._fh.close()
        self._fh = None


def _setup_logger() -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("binance_bot_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(
        log_dir / f"bot_{datetime.now().strftime('%Y-%m-%d')}.log", encoding="utf-8"
    )
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Kesintisiz Binance Spot / Çapraz Marjin arka plan botu")
    ap.add_argument("--mod", choices=["sanal", "canli", "marjin"], default="sanal")
    ap.add_argument(
        "--profil",
        choices=["al_yukselince_sat", "hizli_30sn", "aktif", "temkinli"],
        default="hizli_30sn",
    )
    ap.add_argument("--parite", default="", help="Örn. BTCUSDT (boşsa .env DEFAULT_SYMBOL)")
    ap.add_argument(
        "--pariteler",
        default="",
        help=(
            "Çoklu coin taraması: virgülle ayrılmış liste veya 'varsayilan' "
            f"({','.join(DEFAULT_SCAN_SYMBOLS)})."
        ),
    )
    ap.add_argument("--ai", choices=["acik", "kapali", "zorla"], default="acik")
    ap.add_argument(
        "--kaldirac", type=int, choices=range(1, 6), default=None,
        help="Sanal veya Çapraz Marjin kaldıracı (1-5). Canlı Spot yalnızca 1x çalışır.",
    )
    ap.add_argument("--max-dongu", type=int, default=0, help="Test amaçlı: 0 = sınırsız")
    ap.add_argument(
        "--onay",
        default="",
        help="Yalnızca otomasyon/test içindir; canlı modda normalde konsoldan EVET yazılır.",
    )
    return ap.parse_args(argv)


def _summarize_step(res: StepCycleSummary) -> str:
    parts = [f"{res.symbol} {res.interval}", f"Karar={res.final_action}"]
    if res.ai_decision is not None:
        parts.append(f"Mod={res.ai_decision.mode_label}")
    if res.tech_signal is not None:
        ts = res.tech_signal
        parts.append(
            f"Kapanis={ts.close_price:.6g} RSI={ts.rsi_14:.1f} Hacim={ts.volume_ratio:.2f}x"
        )
    if res.executed_order is not None:
        parts.append(f"EMIR: {res.executed_order.explanation}")
    elif res.block_or_info_message:
        parts.append(f"Not: {res.block_or_info_message[:180]}")
    if res.error_message:
        parts.append(f"HATA: {res.error_message[:200]}")
    return " | ".join(parts)


def run_bot(
    argv: list[str] | None = None,
    *,
    engine: TradingSystemEngine | None = None,
    config: AppConfig | None = None,
    sleep_fn=time.sleep,
    input_fn=input,
    lock_path: Path | None = None,
) -> int:
    args = _parse_args(argv)
    log = _setup_logger()
    cfg = config or load_config()
    profile: TradingProfile = get_profile(args.profil)
    symbol = (args.parite or cfg.default_symbol).strip().upper()
    is_live_spot = args.mod == "canli"
    is_live_margin = args.mod == "marjin"
    is_any_live = is_live_spot or is_live_margin
    execution_mode = "LIVE_MARGIN" if is_live_margin else ("LIVE_SPOT" if is_live_spot else "PAPER")
    track = "LIVE_MARGIN" if is_live_margin else ("LIVE_SPOT" if is_live_spot else "MAIN")
    if is_live_margin:
        log.error("Canli marjin henuz bu guvenlik dogrulamasina dahil degil; hesap istegi veya emir gonderilmedi.")
        return 2
    if is_live_spot and args.kaldirac not in {None, 1}:
        log.error("Canli Spot yalnizca kaldiracsiz (1x) calisir; kaldiracli deneme icin yalnizca --mod sanal kullanin.")
        return 2

    raw_list = args.pariteler.strip()
    if raw_list.lower() in {"varsayilan", "default", "hepsi"}:
        scan_symbols = list(DEFAULT_SCAN_SYMBOLS)
    elif raw_list:
        scan_symbols = [s.strip().upper() for s in raw_list.split(",") if s.strip()]
    else:
        scan_symbols = [symbol]
    multi_scan = len(scan_symbols) > 1

    eng = engine or TradingSystemEngine(cfg)
    if multi_scan:
        eng.lock_benchmark_symbol = True

    leverage_override = (
        1
        if is_live_spot
        else (args.kaldirac if args.kaldirac is not None else (2 if is_live_margin else None))
    )
    base_settings = profile.apply_to_risk_settings(
        eng.default_risk_settings(), leverage=leverage_override
    )

    mode_banner = (
        f"CANLI CAPRAZ MARJIN ({base_settings.leverage}x KALDIRAC)"
        if is_live_margin
        else ("CANLI SPOT (1x GERCEK PARA)" if is_live_spot else f"SANAL DENEME ({base_settings.leverage}x)")
    )
    log.info("=" * 78)
    log.info(
        "ARKA PLAN BOTU | Mod=%s | Profil=%s | Parite(ler)=%s | Mum=%s | Dongu=%ss | MaksAcik=%s | AI=%s",
        mode_banner,
        profile.key,
        ",".join(scan_symbols),
        profile.interval,
        profile.loop_seconds,
        base_settings.max_open_positions,
        args.ai,
    )
    log.info("Profil: %s", profile.description_tr)
    if multi_scan:
        log.info(
            "Coklu coin tarama aktif: %s parite her %s sn'de taranir; ayni anda en fazla %s farkli "
            "coinde acik pozisyon tutulur (ayni coine ekleme yapilmaz).",
            len(scan_symbols),
            profile.loop_seconds,
            base_settings.max_open_positions,
        )

    quote = "USDT" if scan_symbols[0].endswith("USDT") else "USDC"
    if is_any_live:
        if not cfg.enable_live_trading or not cfg.has_binance_live_keys:
            log.error(
                "Canli mod kilitli: .env icinde ENABLE_LIVE_TRADING=true ve BINANCE_API_KEY/SECRET gerekli."
            )
            return 2
        try:
            if is_live_margin:
                minfo = eng.live_margin_executor.fetch_live_margin_balances(
                    ui_live_confirmed=True, ui_margin_confirmed=True, quote_asset=quote
                )
                log.info(
                    "Gercek Capraz Marjin serbest teminat: %.2f %s | Borc: %.2f %s | Marjin Seviyesi: %.2f",
                    minfo["available_collateral"],
                    quote,
                    minfo["quote_borrowed"],
                    quote,
                    minfo["margin_level"],
                )
            else:
                bals = eng.live_executor.fetch_live_spot_balances(
                    ui_live_confirmed=True, quote_asset=quote
                )
                log.info("Gercek Spot serbest bakiye: %s %s", bals.get(quote, 0), quote)
        except Exception as exc:
            log.error("Gercek hesap okunamadi, bot baslatilmadi: %s", exc)
            return 3
        print()
        print("!" * 78)
        if is_live_margin:
            print(f"  DIKKAT: GERCEK CAPRAZ MARJIN ({base_settings.leverage}x KALDIRAC) OTOMATIK ISLEM BASLAYACAK.")
            print(
                f"  Islem basina en fazla {base_settings.max_position_size_usdc:.2f} {quote} teminat "
                f"(~{base_settings.max_position_size_usdc * Decimal(base_settings.leverage):.2f} {quote} pozisyon buyuklugu), "
                f"ayni anda en fazla {base_settings.max_open_positions} farkli coin, "
                f"gunluk zarar esigi {base_settings.daily_loss_limit_usdc:.2f} {quote}."
            )
        else:
            print("  DIKKAT: GERCEK PARA ILE OTOMATIK SPOT ALIM-SATIM BASLAYACAK.")
            print(
                f"  Islem basina en fazla {base_settings.max_position_size_usdc:.2f} {quote}, "
                f"ayni anda en fazla {base_settings.max_open_positions} farkli coin, "
                f"gunluk zarar esigi {base_settings.daily_loss_limit_usdc:.2f} {quote}, "
                f"gunde en fazla {base_settings.max_daily_new_positions} yeni pozisyon."
            )
        print("  Kar garantisi yoktur. Durdurmak icin bu pencerede Ctrl+C yapin veya pencereyi kapatin.")
        print("!" * 78)
        answer = args.onay or input_fn("  Onayliyorsaniz buyuk harflerle EVET yazip Enter'a basin: ")
        if answer.strip() != "EVET":
            log.warning("Canli mod onayi verilmedi; bot baslatilmadi.")
            return 4

    lock = _SingleInstanceLock(lock_path or (PROJECT_ROOT / "data" / "bot_runner.lock"))
    if not lock.acquire():
        log.error("Baska bir arka plan botu zaten calisiyor. Ayni anda yalnizca bir bot calisabilir.")
        return 5

    storage = eng.storage
    started_iso = datetime.now(timezone.utc).isoformat()
    storage.set_state(RUNNER_STOP_KEY, "0")
    storage.set_state(
        RUNNER_INFO_KEY,
        json.dumps(
            {
                "mode": execution_mode,
                "profile": profile.key,
                "symbol": ",".join(scan_symbols) if multi_scan else symbol,
                "interval": profile.interval,
                "leverage": base_settings.leverage,
                "pid": os.getpid(),
                "started_iso": started_iso,
            }
        ),
    )
    _keep_windows_awake(True)

    loops = 0
    consecutive_errors = 0
    last_logged_key: dict[str, str] = {}
    last_alive_log = 0.0
    exit_code = 0
    try:
        while True:
            loops += 1
            now_iso = datetime.now(timezone.utc).isoformat()
            storage.set_state(RUNNER_HEARTBEAT_KEY, now_iso)

            if _stop_requested(storage):
                log.info("Panelden durdurma istegi alindi; bot guvenli sekilde kapatiliyor.")
                break

            settings = profile.apply_to_risk_settings(
                eng.default_risk_settings(), leverage=leverage_override
            )

            # Canlı modda döngü başında bakiyeyi 1 kez senkronize et (10 paritede 10 ayrı hesap çağrısını önler)
            prefetched_live = False
            if is_live_spot and getattr(eng, "live_executor", None) is not None:
                try:
                    eng.live_executor.fetch_live_spot_balances(
                        ui_live_confirmed=True, quote_asset=quote
                    )
                    prefetched_live = True
                except Exception as exc:
                    log.warning("Canli bakiye on-okuma uyarisi: %s", exc)
            elif is_live_margin and getattr(eng, "live_margin_executor", None) is not None:
                try:
                    eng.live_margin_executor.fetch_live_margin_balances(
                        ui_live_confirmed=True, ui_margin_confirmed=True, quote_asset=quote
                    )
                    prefetched_live = True
                except Exception as exc:
                    log.warning("Canli marjin bakiye on-okuma uyarisi: %s", exc)

            # Önce tüm açık pozisyonları kontrol et (zarar-kes / kâr-al / SAT), boş yer varsa diğer coinleri tara
            open_now = [p for p in storage.get_open_positions(track=track) if not is_substep_dust_position(p)]
            open_syms = list(dict.fromkeys(p.symbol for p in open_now))
            if len(open_now) < settings.max_open_positions:
                cycle_symbols = open_syms + [s for s in scan_symbols if s not in open_syms]
            else:
                cycle_symbols = open_syms if open_syms else list(scan_symbols)

            cycle_ok = True
            cycle_summaries: list[str] = []
            for sym in cycle_symbols:
                storage.set_state(RUNNER_HEARTBEAT_KEY, datetime.now(timezone.utc).isoformat())
                if _stop_requested(storage):
                    break
                # Açık olmayan bir adaya bakarken kapasite dolduysa yeni alım aramayı kes
                cur_open = [p for p in storage.get_open_positions(track=track) if not is_substep_dust_position(p)]
                cur_open_syms = {p.symbol for p in cur_open}
                if sym not in cur_open_syms and len(cur_open) >= settings.max_open_positions:
                    break
                try:
                    use_ai_now = (
                        True
                        if args.ai == "zorla"
                        else (False if (args.ai == "kapali" or profile.direct_rule_execution) else True)
                    )
                    step_kwargs = dict(
                        symbol=sym,
                        interval=profile.interval,
                        settings=settings,
                        ai_enabled_by_user=use_ai_now,
                        daily_ai_limit_override=profile.ai_daily_call_limit,
                        min_ai_interval_override=profile.ai_min_call_interval_seconds,
                        execution_mode=execution_mode,
                        ui_live_confirmed=is_any_live,
                        strategy_params=profile.strategy,
                        ai_only_on_rule_signal=multi_scan,
                        skip_balance_prefetch=prefetched_live,
                        stop_requested=lambda: _stop_requested(storage),
                    )
                    if is_live_margin:
                        step_kwargs["ui_margin_confirmed"] = True
                    res = eng.run_analysis_step(**step_kwargs)
                except Exception as exc:  # Ağ/geçici hata: botu çökertme, bekle ve yeniden dene
                    cycle_ok = False
                    log.exception("Dongu hatasi (%s): %s", sym, exc)
                    continue

                candle_key = (
                    str(res.tech_signal.candle_open_time_ms) if res.tech_signal is not None else ""
                )
                log_key = f"{candle_key}:{res.final_action}:{bool(res.executed_order)}:{bool(res.error_message)}"
                if (
                    res.executed_order is not None
                    or res.error_message
                    or log_key != last_logged_key.get(sym, "")
                ):
                    level = logging.WARNING if (res.error_message or not res.ok) else logging.INFO
                    log.log(level, _summarize_step(res))
                    last_logged_key[sym] = log_key
                else:
                    cycle_summaries.append(f"{sym}={res.final_action}")
                if not res.ok:
                    cycle_ok = False

                # Yeni alım gerçekleştiyse ve maksimum açık pozisyon kapasitesi dolduysa kalan adayları atla
                if (
                    res.executed_order is not None
                    and res.executed_order.side == "BUY"
                    and res.executed_order.status in ("FILLED", "PARTIALLY_FILLED")
                    and len([p for p in storage.get_open_positions(track=track) if not is_substep_dust_position(p)]) >= settings.max_open_positions
                ):
                    break

            if _stop_requested(storage):
                log.info("Durdurma istegi alindi; kalan coinler taranmadan bot kapatiliyor.")
                break

            # Açık pozisyonların tamamı için her döngüde anlık Kâr/Zarar durumunu yazdır
            open_after = [p for p in storage.get_open_positions(track=track) if not is_substep_dust_position(p)]
            mc = getattr(eng, "market_client", None)
            if open_after and mc is not None:
                est_settings = profile.apply_to_risk_settings(
                    eng.default_risk_settings(), leverage=leverage_override
                )
                for p_open in open_after:
                    if _stop_requested(storage):
                        break
                    try:
                        ob = mc.fetch_order_book(p_open.symbol, depth_limit=5)
                        if ob.best_bid is not None:
                            _, net_pnl = compute_open_position_unrealized_net_pnl(
                                p_open,
                                current_bid=ob.best_bid,
                                estimated_fee_rate=est_settings.estimated_fee_rate,
                            )
                            cost_tot = p_open.cost_basis_usdc + p_open.entry_fee_usdc
                            pnl_pct = (net_pnl / cost_tot * Decimal("100")) if cost_tot > 0 else Decimal("0")
                            q_lbl = "USDT" if p_open.symbol.endswith("USDT") else "USDC"
                            sl_txt = f"{p_open.stop_loss_price:.6g}" if p_open.stop_loss_price else "-"
                            tp_txt = f"{p_open.take_profit_price:.6g}" if p_open.take_profit_price else "-"
                            log.info(
                                "ANLIK K/Z | %s | Giris=%.6g Anlik=%.6g | Net=%+.4f %s (%%%+.2f) | Zarar-Kes=%s Kar-Al=%s",
                                p_open.symbol,
                                p_open.entry_price,
                                ob.best_bid,
                                net_pnl,
                                q_lbl,
                                pnl_pct,
                                sl_txt,
                                tp_txt,
                            )
                    except Exception:
                        pass

            if cycle_summaries and time.monotonic() - last_alive_log > 300:
                log.info("Bot calisiyor (yeni mum bekleniyor) | %s", " ".join(cycle_summaries))
                last_alive_log = time.monotonic()
            consecutive_errors = 0 if cycle_ok else consecutive_errors + 1

            if args.max_dongu and loops >= args.max_dongu:
                break

            delay = profile.loop_seconds
            if consecutive_errors >= 3:
                delay = min(profile.loop_seconds * (2 ** min(consecutive_errors - 2, 4)), 600)
                log.warning("Ust uste %s hatali dongu; %s sn beklenecek.", consecutive_errors, delay)
            if not _wait_with_heartbeat(storage, delay, sleep_fn):
                log.info("Bekleme sirasinda durdurma istegi alindi; bot kapatiliyor.")
                break
    except KeyboardInterrupt:
        log.info("Ctrl+C alindi; bot durduruluyor. (Acik pozisyon varsa panelden yonetebilirsiniz.)")
        exit_code = 0
    finally:
        storage.set_state(RUNNER_HEARTBEAT_KEY, "")
        storage.set_state(RUNNER_STOP_KEY, "0")
        _keep_windows_awake(False)
        lock.release()
        open_pos = storage.get_open_positions(track=track)
        if open_pos:
            pos_desc = ", ".join(f"{p.symbol} ({p.quantity} @ {p.entry_price})" for p in open_pos)
            log.warning(
                "Bot durdu ancak %s ACIK POZISYON var: %s. Zarar-kes/kar-al takibi bot "
                "yeniden baslatilana kadar YAPILMAZ.",
                len(open_pos),
                pos_desc,
            )
        log.info("Arka plan botu kapandi. Toplam dongu: %s", loops)
    return exit_code


def main() -> None:
    sys.exit(run_bot())


if __name__ == "__main__":
    main()
