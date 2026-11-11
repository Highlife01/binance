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
- Pozisyon üst sınırı, günlük zarar eşiği ve 1 açık pozisyon kuralı profillerden bağımsız korunur.
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
from pathlib import Path

from .config import PROJECT_ROOT, AppConfig, load_config
from .engine import StepCycleSummary, TradingSystemEngine
from .profiles import TradingProfile, get_profile

RUNNER_HEARTBEAT_KEY = "runner_heartbeat_iso"
RUNNER_INFO_KEY = "runner_info_json"
RUNNER_STOP_KEY = "runner_stop_requested"
RUNNER_STALE_AFTER_SECONDS = 150

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
    ap = argparse.ArgumentParser(description="Kesintisiz Binance Spot arka plan botu")
    ap.add_argument("--mod", choices=["sanal", "canli"], default="sanal")
    ap.add_argument("--profil", choices=["aktif", "temkinli"], default="aktif")
    ap.add_argument("--parite", default="", help="Örn. BTCUSDT (boşsa .env DEFAULT_SYMBOL)")
    ap.add_argument("--ai", choices=["acik", "kapali"], default="acik")
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
            f"Kapanis={ts.close_price:.2f} RSI={ts.rsi_14:.1f} Hacim={ts.volume_ratio:.2f}x"
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
    is_live = args.mod == "canli"
    execution_mode = "LIVE_SPOT" if is_live else "PAPER"
    eng = engine or TradingSystemEngine(cfg)

    log.info("=" * 78)
    log.info(
        "ARKA PLAN BOTU | Mod=%s | Profil=%s | Parite=%s | Mum=%s | Dongu=%ss | AI=%s",
        "CANLI SPOT (GERCEK PARA)" if is_live else "SANAL DENEME",
        profile.key,
        symbol,
        profile.interval,
        profile.loop_seconds,
        args.ai,
    )
    log.info("Profil: %s", profile.description_tr)

    if is_live:
        if not cfg.enable_live_trading or not cfg.has_binance_live_keys:
            log.error(
                "Canli mod kilitli: .env icinde ENABLE_LIVE_TRADING=true ve BINANCE_API_KEY/SECRET gerekli."
            )
            return 2
        quote = "USDT" if symbol.endswith("USDT") else "USDC"
        try:
            bals = eng.live_executor.fetch_live_spot_balances(
                ui_live_confirmed=True, quote_asset=quote
            )
            log.info("Gercek Spot serbest bakiye: %s %s", bals.get(quote, 0), quote)
        except Exception as exc:
            log.error("Gercek hesap okunamadi, bot baslatilmadi: %s", exc)
            return 3
        base_settings = profile.apply_to_risk_settings(eng.default_risk_settings())
        print()
        print("!" * 78)
        print("  DIKKAT: GERCEK PARA ILE OTOMATIK SPOT ALIM-SATIM BASLAYACAK.")
        print(
            f"  Islem basina en fazla {base_settings.max_position_size_usdc:.2f} {quote}, "
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
                "symbol": symbol,
                "interval": profile.interval,
                "pid": os.getpid(),
                "started_iso": started_iso,
            }
        ),
    )
    _keep_windows_awake(True)

    loops = 0
    consecutive_errors = 0
    last_logged_key = ""
    last_alive_log = 0.0
    exit_code = 0
    try:
        while True:
            loops += 1
            now_iso = datetime.now(timezone.utc).isoformat()
            storage.set_state(RUNNER_HEARTBEAT_KEY, now_iso)

            if storage.get_state(RUNNER_STOP_KEY, "0") == "1":
                log.info("Panelden durdurma istegi alindi; bot guvenli sekilde kapatiliyor.")
                break

            try:
                settings = profile.apply_to_risk_settings(eng.default_risk_settings())
                res = eng.run_analysis_step(
                    symbol=symbol,
                    interval=profile.interval,
                    settings=settings,
                    ai_enabled_by_user=(args.ai == "acik"),
                    daily_ai_limit_override=profile.ai_daily_call_limit,
                    min_ai_interval_override=profile.ai_min_call_interval_seconds,
                    execution_mode=execution_mode,
                    ui_live_confirmed=is_live,
                    strategy_params=profile.strategy,
                )
                candle_key = (
                    str(res.tech_signal.candle_open_time_ms) if res.tech_signal is not None else ""
                )
                log_key = f"{candle_key}:{res.final_action}:{bool(res.executed_order)}:{bool(res.error_message)}"
                if res.executed_order is not None or res.error_message or log_key != last_logged_key:
                    level = logging.WARNING if (res.error_message or not res.ok) else logging.INFO
                    log.log(level, _summarize_step(res))
                    last_logged_key = log_key
                elif time.monotonic() - last_alive_log > 300:
                    log.info("Bot calisiyor (yeni mum bekleniyor) | %s", _summarize_step(res))
                    last_alive_log = time.monotonic()
                consecutive_errors = 0 if res.ok else consecutive_errors + 1
            except Exception as exc:  # Ağ/geçici hata: botu çökertme, bekle ve yeniden dene
                consecutive_errors += 1
                log.exception("Dongu hatasi (#%s): %s", consecutive_errors, exc)

            if args.max_dongu and loops >= args.max_dongu:
                break

            delay = profile.loop_seconds
            if consecutive_errors >= 3:
                delay = min(profile.loop_seconds * (2 ** min(consecutive_errors - 2, 4)), 600)
                log.warning("Ust uste %s hata; %s sn beklenecek.", consecutive_errors, delay)
            sleep_fn(delay)
    except KeyboardInterrupt:
        log.info("Ctrl+C alindi; bot durduruluyor. (Acik pozisyon varsa panelden yonetebilirsiniz.)")
        exit_code = 0
    finally:
        storage.set_state(RUNNER_HEARTBEAT_KEY, "")
        storage.set_state(RUNNER_STOP_KEY, "0")
        _keep_windows_awake(False)
        lock.release()
        open_pos = storage.get_open_positions(track="LIVE_SPOT" if is_live else "MAIN")
        if open_pos:
            p = open_pos[0]
            log.warning(
                "Bot durdu ancak ACIK POZISYON var: %s %s @ %s. Zarar-kes/kar-al takibi bot "
                "yeniden baslatilana kadar YAPILMAZ.",
                p.symbol,
                p.quantity,
                p.entry_price,
            )
        log.info("Arka plan botu kapandi. Toplam dongu: %s", loops)
    return exit_code


def main() -> None:
    sys.exit(run_bot())


if __name__ == "__main__":
    main()
