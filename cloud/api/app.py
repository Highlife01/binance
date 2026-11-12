"""
TradeCebrail 7/24 Bulut Alım-Satım Motoru ve Sahip Doğrulamalı Kontrol API'si (Cloud Run).

Özellikler:
- Yalnızca hesap sahibine (`cebrailkara@gmail.com`, doğrulanmış Firebase Google ID Token) açık `/api/*` uç noktaları.
- Cloud Scheduler tarafından dakikada 1 kez `X-Cron-Secret` (sabit zamanlı karşılaştırma) ile tetiklenen `/api/cron/tick`.
- Özel GCS bucket (`STATE_BUCKET`) üzerinde atomik SQLite yedekleme (`sqlite3.Connection.backup`) ile kalıcı durum.
- Varsayılan güvenli başlangıç: 7/24 Otomatik Sanal Deneme (`PAPER`), `2x` Kaldıraç, `HIZLI_30SN`, 10 USDT paritesi ve Gemini AI.
- Yanıtlarda ve günlüklerde gizli anahtar / token sızıntısını önleyen çok katmanlı maskeleme (`redact_secrets`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlsplit

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import google.auth
from google.auth import exceptions as auth_exceptions
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2 import id_token
import requests

from binance_ai_bot.config import AppConfig
from binance_ai_bot.domain import RiskSettings, is_substep_dust_position
from binance_ai_bot.engine import StepCycleSummary, TradingSystemEngine
from binance_ai_bot.live_executor import LiveExecutionError
from binance_ai_bot.market_data import MarketDataError
from binance_ai_bot.profiles import TRADING_PROFILES, TradingProfile, get_profile
from binance_ai_bot.risk import compute_open_position_unrealized_net_pnl
from binance_ai_bot.runner import DEFAULT_SCAN_SYMBOLS
from binance_ai_bot.security import LiveOrderBlockedError, redact_secrets
from binance_ai_bot.storage import SQLiteStorage


ALLOWED_ORIGINS = frozenset(
    {
        "https://tradecebrail.web.app",
        "https://tradecebrail.firebaseapp.com",
    }
)
MAX_DB_BYTES = 32 * 1024 * 1024
MAX_BODY_BYTES = 64 * 1024
SQLITE_MAGIC = b"SQLite format 3\x00"
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{4,20}$")
_SECRET_RE = re.compile(
    r"(?i)(?:api[_ -]?key|secret|password|authorization|access[_ -]?token|private[_ -]?key)\s*[:=]"
    r"|AIza[0-9A-Za-z_-]{20,}|\b[A-Za-z0-9]{64}\b"
)


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "evet", "on"}


def _safe_text(value: Any, length: int = 420) -> str:
    if value is None:
        return ""
    raw = " ".join(str(value).split())
    if not raw:
        return ""
    cleaned = redact_secrets(raw)
    if _SECRET_RE.search(cleaned):
        return "Gizli bilgi içerebilecek metin güvenlik gereği maskelendi."
    return cleaned[:length]


def _dec_str(value: Any, places: str | None = None) -> str:
    try:
        d = value if isinstance(value, Decimal) else Decimal(str(value))
        if not d.is_finite():
            return "0"
        if places is not None:
            return str(d.quantize(Decimal(places)))
        return format(d.normalize(), "f") if d != 0 else "0"
    except (InvalidOperation, ValueError, TypeError):
        return "0"


@dataclass(frozen=True)
class CloudConfig:
    project_id: str = "adanahizlitaksi"
    owner_email: str = "cebrailkara@gmail.com"
    state_bucket: str = ""
    state_object: str = "sanal_sistem.db"
    cron_secret: str = ""
    db_path: Path = Path("/tmp/tradecebrail/sanal_sistem.db")
    stale_after_seconds: int = 180
    initial_balance_usdc: Decimal = Decimal("86.00")
    enable_live_trading: bool = False
    binance_api_key: str = ""
    binance_api_secret: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    binance_public_base_url: str = "https://api.binance.com"
    binance_fallback_base_url: str = "https://data-api.binance.vision"
    binance_live_base_url: str = "https://api.binance.com"

    @classmethod
    def from_env(cls) -> "CloudConfig":
        raw_bal = os.getenv("INITIAL_BALANCE_USDC", "86.00").strip()
        try:
            init_bal = Decimal(raw_bal)
            if init_bal < Decimal("1"):
                init_bal = Decimal("86.00")
        except InvalidOperation:
            init_bal = Decimal("86.00")

        return cls(
            project_id=os.getenv("FIREBASE_PROJECT_ID", "adanahizlitaksi").strip(),
            owner_email=os.getenv("OWNER_EMAIL", "cebrailkara@gmail.com").strip().casefold(),
            state_bucket=os.getenv("STATE_BUCKET", os.getenv("STATUS_BUCKET", "")).strip(),
            state_object=os.getenv("STATE_OBJECT", "sanal_sistem.db").strip() or "sanal_sistem.db",
            cron_secret=os.getenv("TRADECEBRAIL_CRON_SECRET", "").strip(),
            db_path=Path(os.getenv("SQLITE_DB_PATH", "/tmp/tradecebrail/sanal_sistem.db").strip()),
            stale_after_seconds=max(60, int(os.getenv("STALE_AFTER_SECONDS", "180"))),
            initial_balance_usdc=init_bal,
            enable_live_trading=_parse_bool(os.getenv("ENABLE_LIVE_TRADING"), default=False),
            binance_api_key=(
                os.getenv("TRADECEBRAIL_BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY") or ""
            ).strip(),
            binance_api_secret=(
                os.getenv("TRADECEBRAIL_BINANCE_API_SECRET") or os.getenv("BINANCE_API_SECRET") or ""
            ).strip(),
            gemini_api_key=(
                os.getenv("TRADECEBRAIL_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
            ).strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite",
            binance_public_base_url=os.getenv("BINANCE_PUBLIC_BASE_URL", "https://api.binance.com").strip().rstrip("/"),
            binance_fallback_base_url=os.getenv(
                "BINANCE_FALLBACK_BASE_URL", "https://data-api.binance.vision"
            ).strip().rstrip("/"),
            binance_live_base_url=os.getenv("BINANCE_LIVE_BASE_URL", "https://api.binance.com").strip().rstrip("/"),
        )

    def to_app_config(self) -> AppConfig:
        return AppConfig(
            initial_balance_usdc=self.initial_balance_usdc,
            max_open_positions=5,
            max_position_size_usdc=Decimal("15.00"),
            daily_loss_limit_usdc=Decimal("2.00"),
            max_daily_new_positions=60,
            daily_limit_exit_policy="HOLD_EXISTING",
            binance_public_base_url=self.binance_public_base_url,
            binance_fallback_base_url=self.binance_fallback_base_url,
            quote_asset="USDT",
            default_symbol="BTCUSDT",
            default_interval="1m",
            enable_live_trading=self.enable_live_trading,
            binance_api_key=self.binance_api_key,
            binance_api_secret=self.binance_api_secret,
            binance_live_base_url=self.binance_live_base_url,
            gemini_api_key=self.gemini_api_key,
            gemini_model=self.gemini_model,
            ai_daily_call_limit=400,
            ai_min_call_interval_seconds=25,
            timezone_name="Europe/Istanbul",
            sqlite_db_path=self.db_path,
        )


@dataclass(frozen=True)
class Response:
    status: int
    body: dict[str, Any] | None
    headers: dict[str, str] = field(default_factory=dict)


class _BoundedRequest(Request):
    def __call__(self, *args: Any, **kwargs: Any):
        kwargs["timeout"] = 10
        return super().__call__(*args, **kwargs)


def verify_token(token: str, project_id: str) -> Mapping[str, Any]:
    with requests.Session() as session:
        return id_token.verify_firebase_token(token, _BoundedRequest(session=session), audience=project_id)


def snapshot_sqlite_bytes(db_path: Path) -> bytes:
    """SQLite WAL sayfalarını tek bir tutarlı dosya görüntüsünde birleştirerek bayt dizisi üretir."""
    backup_path = db_path.with_suffix(".upload_snapshot.sqlite")
    if backup_path.exists():
        backup_path.unlink()
    src = sqlite3.connect(str(db_path), timeout=15.0)
    dst = sqlite3.connect(str(backup_path), timeout=15.0)
    try:
        src.backup(dst)
        dst.commit()
    finally:
        dst.close()
        src.close()
    try:
        data = backup_path.read_bytes()
        if len(data) > MAX_DB_BYTES:
            raise RuntimeError("SQLite durum dosyası izin verilen boyutu aştı.")
        return data
    finally:
        if backup_path.exists():
            backup_path.unlink()


class GcsSqliteSync:
    """Cloud Run ölçeklenmesinde `/tmp` SQLite veritabanını özel GCS bucket ile senkronize eder."""

    def __init__(self, bucket: str, object_name: str = "sanal_sistem.db") -> None:
        self.bucket = bucket.strip()
        self.object_name = object_name.strip() or "sanal_sistem.db"

    def restore(self, db_path: Path) -> bool:
        if not self.bucket:
            return False
        db_path.parent.mkdir(parents=True, exist_ok=True)
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
        )
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{quote(self.bucket, safe='')}"
            f"/o/{quote(self.object_name, safe='')}?alt=media"
        )
        tmp_path = db_path.with_suffix(".restore.tmp")
        with AuthorizedSession(credentials) as session:
            with session.get(
                url, timeout=(4.0, 20.0), stream=True, headers={"Cache-Control": "no-cache"}
            ) as resp:
                if resp.status_code == 404:
                    return False
                if resp.status_code != 200:
                    raise RuntimeError(f"GCS durum okuma hatası: HTTP {resp.status_code}")
                buf = bytearray()
                for chunk in resp.iter_content(chunk_size=65_536):
                    buf.extend(chunk)
                    if len(buf) > MAX_DB_BYTES:
                        raise RuntimeError("GCS durum nesnesi izin verilen boyutu aştı.")
        if len(buf) < 100 or bytes(buf[:16]) != SQLITE_MAGIC:
            raise RuntimeError("GCS üzerindeki durum nesnesi geçerli bir SQLite veritabanı değil.")
        tmp_path.write_bytes(buf)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{db_path}{suffix}")
            if sidecar.exists():
                sidecar.unlink()
        os.replace(tmp_path, db_path)
        return True

    def persist(self, db_path: Path) -> bool:
        if not self.bucket or not db_path.exists():
            return False
        payload = snapshot_sqlite_bytes(db_path)
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
        )
        url = (
            f"https://storage.googleapis.com/upload/storage/v1/b/{quote(self.bucket, safe='')}"
            f"/o?uploadType=media&name={quote(self.object_name, safe='')}"
        )
        with AuthorizedSession(credentials) as session:
            resp = session.post(
                url,
                data=payload,
                headers={"Content-Type": "application/x-sqlite3"},
                timeout=(4.0, 25.0),
            )
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"GCS durum yazma hatası: HTTP {resp.status_code}")
        return True


class CloudBotApplication:
    """Sahip doğrulamalı web paneli ve 7/24 zamanlayıcı tetikleyicisini yöneten ana bulut uygulaması."""

    def __init__(
        self,
        config: CloudConfig,
        *,
        engine: TradingSystemEngine | None = None,
        state_sync: GcsSqliteSync | None = None,
        verifier: Callable[[str, str], Mapping[str, Any]] = verify_token,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.state_sync = state_sync or GcsSqliteSync(config.state_bucket, config.state_object)
        self.verifier = verifier
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._op_lock = threading.RLock()
        self._init_lock = threading.Lock()
        self._engine: TradingSystemEngine | None = engine
        self._initialized = engine is not None
        self._bid_cache: dict[str, tuple[float, Decimal]] = {}

        if self._engine is not None:
            self._engine.lock_benchmark_symbol = True
            self._ensure_cloud_defaults(self._engine.storage)

    def _get_engine(self) -> TradingSystemEngine:
        if self._initialized and self._engine is not None:
            return self._engine
        with self._init_lock:
            if self._initialized and self._engine is not None:
                return self._engine
            self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                restored = self.state_sync.restore(self.config.db_path)
                logging.info("GCS durum yukleme tamamlandi (restored=%s)", restored)
            except Exception as exc:
                logging.error("GCS durum yuklenemedi: %s", _safe_text(exc))
                raise
            app_cfg = self.config.to_app_config()
            eng = TradingSystemEngine(app_cfg)
            eng.lock_benchmark_symbol = True
            created_defaults = self._ensure_cloud_defaults(eng.storage)
            if created_defaults:
                try:
                    self.state_sync.persist(self.config.db_path)
                except Exception as exc:
                    logging.warning("Ilk varsayilan GCS kaydi uyarisi: %s", _safe_text(exc))
            self._engine = eng
            self._initialized = True
            return eng

    def _ensure_cloud_defaults(self, storage: SQLiteStorage) -> bool:
        changed = False
        defaults = {
            "cloud_auto_enabled": "1",
            "cloud_mode": "PAPER",
            "cloud_profile": "AL_YUKSELINCE_SAT",
            "cloud_leverage": "2",
            "cloud_ai_enabled": "1" if bool(self.config.gemini_api_key) else "0",
            "cloud_ai_force": "0",
            "cloud_live_confirmed": "0",
            "cloud_scan_symbols": ",".join(DEFAULT_SCAN_SYMBOLS),
        }
        for key, val in defaults.items():
            if storage.get_state(key, "") == "":
                storage.set_state(key, val)
                changed = True
        if storage.get_portfolio_quote("MAIN") is None:
            storage.ensure_quote_asset("MAIN", "USDT")
            changed = True
        if storage.get_state("risk_settings_json", "") == "":
            prof = get_profile("AL_YUKSELINCE_SAT")
            base = RiskSettings(
                max_open_positions=5,
                max_position_size_usdc=Decimal("15.00"),
                daily_loss_limit_usdc=Decimal("2.00"),
                max_daily_new_positions=prof.max_daily_new_positions,
                stop_loss_pct=prof.stop_loss_pct,
                take_profit_pct=prof.take_profit_pct,
                leverage=2,
                min_net_profit_exit_pct=prof.min_net_profit_exit_pct,
            )
            storage.save_risk_settings(base)
            changed = True
        if storage.get_state("profit_target_v2_15usdt", "") != "1":
            prof = get_profile(storage.get_state("cloud_profile", "AL_YUKSELINCE_SAT") or "AL_YUKSELINCE_SAT")
            cur = storage.load_risk_settings(
                RiskSettings(max_open_positions=5, max_position_size_usdc=Decimal("15.00"), daily_loss_limit_usdc=Decimal("2.00"))
            )
            new_pos_size = Decimal("15.00") if cur.max_position_size_usdc == Decimal("10.00") else cur.max_position_size_usdc
            upgraded = RiskSettings(
                max_open_positions=max(1, min(10, int(cur.max_open_positions))),
                max_position_size_usdc=new_pos_size,
                daily_loss_limit_usdc=cur.daily_loss_limit_usdc,
                max_daily_new_positions=prof.max_daily_new_positions,
                daily_limit_exit_policy=cur.daily_limit_exit_policy,
                estimated_fee_rate=cur.estimated_fee_rate,
                extra_slippage_bps=cur.extra_slippage_bps,
                stop_loss_pct=prof.stop_loss_pct,
                take_profit_pct=prof.take_profit_pct,
                leverage=cur.leverage,
                hourly_margin_interest_rate=cur.hourly_margin_interest_rate,
                min_net_profit_exit_pct=prof.min_net_profit_exit_pct,
            )
            storage.save_risk_settings(upgraded)
            for trk in ("LIVE_SPOT", "MAIN"):
                for pos in storage.get_open_positions(track=trk):
                    storage.update_position_remaining(
                        pos.position_id,
                        quantity=pos.quantity,
                        cost_basis_usdc=pos.cost_basis_usdc,
                        entry_fee_usdc=pos.entry_fee_usdc,
                        stop_loss_price=pos.entry_price * (Decimal("1") - prof.stop_loss_pct),
                        take_profit_price=pos.entry_price * (Decimal("1") + prof.take_profit_pct),
                        update_exit_targets=True,
                    )
            storage.set_state("profit_target_v2_15usdt", "1")
            changed = True
        return changed

    def _active_cloud_params(self, eng: TradingSystemEngine) -> dict[str, Any]:
        st = eng.storage
        raw_mode = st.get_state("cloud_mode", "PAPER").strip().upper()
        live_confirmed = st.get_state("cloud_live_confirmed", "0") == "1"
        can_live = (
            raw_mode == "LIVE_SPOT"
            and live_confirmed
            and eng.config.enable_live_trading
            and eng.config.has_binance_live_keys
        )
        mode = "LIVE_SPOT" if can_live else "PAPER"
        track = "LIVE_SPOT" if mode == "LIVE_SPOT" else "MAIN"
        profile: TradingProfile = get_profile(st.get_state("cloud_profile", "AL_YUKSELINCE_SAT"))
        try:
            lev = max(1, min(5, int(st.get_state("cloud_leverage", "2") or "2")))
        except ValueError:
            lev = 2
        if mode == "LIVE_SPOT":
            lev = 1

        saved_risk = eng.default_risk_settings()
        eff_settings = profile.apply_to_risk_settings(saved_risk, leverage=lev)
        # Kullanicinin kaydettigi acik pozisyon ust sinirini (1..10) koru
        eff_settings = RiskSettings(
            max_open_positions=max(1, min(10, int(saved_risk.max_open_positions))),
            max_position_size_usdc=saved_risk.max_position_size_usdc,
            daily_loss_limit_usdc=saved_risk.daily_loss_limit_usdc,
            max_daily_new_positions=profile.max_daily_new_positions,
            daily_limit_exit_policy=saved_risk.daily_limit_exit_policy,
            estimated_fee_rate=saved_risk.estimated_fee_rate,
            extra_slippage_bps=saved_risk.extra_slippage_bps,
            stop_loss_pct=profile.stop_loss_pct,
            take_profit_pct=profile.take_profit_pct,
            allow_stablecoin_conversion=False,
            leverage=lev,
            hourly_margin_interest_rate=saved_risk.hourly_margin_interest_rate,
            min_net_profit_exit_pct=profile.min_net_profit_exit_pct,
        )
        raw_syms = st.get_state("cloud_scan_symbols", ",".join(DEFAULT_SCAN_SYMBOLS))
        symbols = [
            s.strip().upper()
            for s in raw_syms.split(",")
            if _SYMBOL_RE.fullmatch(s.strip().upper()) and s.strip().upper().endswith("USDT")
        ]
        if not symbols:
            symbols = list(DEFAULT_SCAN_SYMBOLS)
        ai_enabled = (st.get_state("cloud_ai_enabled", "1") == "1") and eng.config.has_gemini_key
        ai_force = (st.get_state("cloud_ai_force", "0") == "1") and ai_enabled
        ai_mode = "2" if ai_force else ("1" if ai_enabled else "0")
        auto_enabled = st.get_state("cloud_auto_enabled", "1") == "1"
        return {
            "mode": mode,
            "track": track,
            "profile": profile,
            "leverage": lev,
            "settings": eff_settings,
            "symbols": symbols,
            "ai_enabled": ai_enabled,
            "ai_force": ai_force,
            "ai_mode": ai_mode,
            "auto_enabled": auto_enabled,
            "live_confirmed": live_confirmed,
        }

    def _get_live_bids_for_positions(self, eng: TradingSystemEngine, positions: list[Any]) -> dict[str, Decimal]:
        bids: dict[str, Decimal] = {}
        now_mono = time.monotonic()
        for pos in positions:
            cached = self._bid_cache.get(pos.symbol)
            if cached and (now_mono - cached[0]) <= 6.0:
                bids[pos.symbol] = cached[1]
                continue
            try:
                ob = eng.market_client.fetch_order_book(pos.symbol, depth_limit=5)
                if ob.best_bid is not None and ob.best_bid > 0:
                    bids[pos.symbol] = ob.best_bid
                    self._bid_cache[pos.symbol] = (now_mono, ob.best_bid)
            except Exception:
                pass
        return bids

    def _run_scan_cycle(self, eng: TradingSystemEngine, *, source: str) -> dict[str, Any]:
        params = self._active_cloud_params(eng)
        st = eng.storage
        mode: str = params["mode"]
        track: str = params["track"]
        profile: TradingProfile = params["profile"]
        settings: RiskSettings = params["settings"]
        scan_symbols: list[str] = params["symbols"]
        ai_enabled: bool = params["ai_enabled"]
        ai_force: bool = params["ai_force"]
        use_ai_now = ai_force if profile.direct_rule_execution else ai_enabled
        is_live_spot = mode == "LIVE_SPOT"

        prefetched_live = False
        last_error = ""
        if is_live_spot:
            try:
                eng.live_executor.fetch_live_spot_balances(ui_live_confirmed=True, quote_asset="USDT")
                prefetched_live = True
            except Exception as exc:
                last_error = self._friendly_live_error(str(exc))
                st.set_state("cloud_last_error", last_error)

        open_now = [p for p in st.get_open_positions(track=track) if not is_substep_dust_position(p)]
        open_syms = list(dict.fromkeys(p.symbol for p in open_now))
        paused, _ = st.is_new_entries_paused(track=track)
        if len(open_now) < settings.max_open_positions and not paused:
            cycle_symbols = open_syms + [s for s in scan_symbols if s not in open_syms]
        else:
            cycle_symbols = open_syms if open_syms else scan_symbols[:3]

        scanned: list[dict[str, Any]] = []
        orders_executed = 0
        for sym in cycle_symbols:
            cur_open = [p for p in st.get_open_positions(track=track) if not is_substep_dust_position(p)]
            cur_open_syms = {p.symbol for p in cur_open}
            if sym not in cur_open_syms and (len(cur_open) >= settings.max_open_positions or paused):
                break
            res: StepCycleSummary = eng.run_analysis_step(
                symbol=sym,
                interval=profile.interval,
                settings=settings,
                ai_enabled_by_user=use_ai_now,
                daily_ai_limit_override=profile.ai_daily_call_limit,
                min_ai_interval_override=profile.ai_min_call_interval_seconds,
                execution_mode=mode,
                ui_live_confirmed=is_live_spot,
                strategy_params=profile.strategy,
                ai_only_on_rule_signal=(len(scan_symbols) > 1),
                skip_balance_prefetch=prefetched_live,
            )
            if res.symbol and sym in eng._latest_bid_map:
                self._bid_cache[sym] = (time.monotonic(), eng._latest_bid_map[sym])
            if res.executed_order is not None and res.executed_order.status in ("FILLED", "PARTIALLY_FILLED"):
                orders_executed += 1
            if res.error_message and not last_error:
                last_error = self._friendly_live_error(res.error_message)
            scanned.append(
                {
                    "symbol": sym,
                    "ok": res.ok,
                    "action": res.final_action,
                    "order_side": res.executed_order.side if res.executed_order else None,
                    "note": _safe_text(res.block_or_info_message, 180),
                }
            )
            if (
                res.executed_order is not None
                and res.executed_order.side == "BUY"
                and res.executed_order.status in ("FILLED", "PARTIALLY_FILLED")
                and len([p for p in st.get_open_positions(track=track) if not is_substep_dust_position(p)]) >= settings.max_open_positions
            ):
                break

        now_iso = self.now().astimezone(timezone.utc).isoformat()
        summary_obj = {
            "timestamp": now_iso,
            "source": source,
            "mode": mode,
            "profile": profile.key,
            "leverage": settings.leverage,
            "scanned_count": len(scanned),
            "orders_executed": orders_executed,
            "items": scanned[:12],
        }
        st.set_state("cloud_last_tick_iso", now_iso)
        st.set_state("cloud_last_tick_summary_json", json.dumps(summary_obj, ensure_ascii=False))
        st.set_state("cloud_last_error", last_error)
        self.state_sync.persist(self.config.db_path)
        return summary_obj

    @staticmethod
    def _friendly_live_error(raw: str) -> str:
        clean = _safe_text(raw, 360)
        if "-2015" in clean or "Invalid API-key, IP" in clean:
            return (
                "Binance API anahtarınızda ev IP kısıtlaması açık olduğu için bulut sunucusundan gerçek hesap "
                "erişimi reddedildi (-2015). 7/24 Sanal (2x Kaldıraç) mod sorunsuz çalışır. Bulutta gerçek Spot "
                "işlem için Binance API ayarlarında Para Çekme (Withdrawal) kapalı ayrı bir anahtar kullanabilirsiniz."
            )
        return clean

    def _fetch_real_wallet_summary(self, eng: TradingSystemEngine) -> dict[str, Any]:
        if not eng.config.enable_live_trading or not eng.config.has_binance_live_keys:
            return {
                "ok": False,
                "error": "Sunucuda gerçek Binance API anahtarı tanımlı değil.",
                "rows": [],
            }
        try:
            import httpx

            acc_raw = eng.live_executor._request_signed(
                "GET", "/api/v3/account", ui_live_confirmed=True
            )
            px_map: dict[str, Decimal] = {}
            try:
                resp_px = httpx.get(
                    f"{eng.config.binance_public_base_url}/api/v3/ticker/bookTicker", timeout=8.0
                )
                if resp_px.status_code == 200:
                    px_map = {
                        str(item["symbol"]): Decimal(str(item["bidPrice"]))
                        for item in resp_px.json()
                        if Decimal(str(item.get("bidPrice", "0"))) > 0
                    }
            except Exception:
                px_map = {}

            rows: list[dict[str, Any]] = []
            spot_usd = Decimal("0")
            earn_usd = Decimal("0")
            for b in acc_raw.get("balances", []):
                tot = Decimal(str(b.get("free", "0"))) + Decimal(str(b.get("locked", "0")))
                if tot <= Decimal("0"):
                    continue
                raw_asset = str(b.get("asset", "")).upper()
                if not _SYMBOL_RE.fullmatch(raw_asset) and len(raw_asset) < 2:
                    continue
                is_earn = raw_asset.startswith("LD") and len(raw_asset) > 2
                underlying = raw_asset[2:] if is_earn else raw_asset
                unit_px = Decimal("1") if underlying in {"USDT", "USDC", "FDUSD"} else px_map.get(f"{underlying}USDT", Decimal("0"))
                usd_val = tot * unit_px
                if is_earn:
                    earn_usd += usd_val
                else:
                    spot_usd += usd_val
                if usd_val >= Decimal("0.01") or tot >= Decimal("0.01"):
                    rows.append(
                        {
                            "location": "Simple Earn (Kazan)" if is_earn else "Spot Cüzdan",
                            "is_earn": is_earn,
                            "asset": f"{underlying} ({raw_asset})" if is_earn else raw_asset,
                            "quantity": _dec_str(tot),
                            "price_usdt": _dec_str(unit_px, "0.0001") if unit_px > 0 else None,
                            "value_usdt": _dec_str(usd_val, "0.01"),
                        }
                    )
            rows.sort(key=lambda r: Decimal(r["value_usdt"]), reverse=True)
            return {
                "ok": True,
                "spot_usdt": _dec_str(spot_usd, "0.01"),
                "earn_usdt": _dec_str(earn_usd, "0.01"),
                "total_usdt": _dec_str(spot_usd + earn_usd, "0.01"),
                "rows": rows,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": self._friendly_live_error(str(exc)),
                "rows": [],
            }

    def _build_dashboard(self, eng: TradingSystemEngine) -> dict[str, Any]:
        st = eng.storage
        params = self._active_cloud_params(eng)
        mode: str = params["mode"]
        track: str = params["track"]
        profile: TradingProfile = params["profile"]
        settings: RiskSettings = params["settings"]
        now_dt = self.now().astimezone(timezone.utc)

        open_positions = st.get_open_positions(track=track)
        bids = self._get_live_bids_for_positions(eng, open_positions)
        summary = eng.get_portfolio_summary(settings, live_bid_by_symbol=bids, track=track)
        paused, pause_reason = st.is_new_entries_paused(track=track)
        ist_today = st.istanbul_date_str(now_dt)
        ai_summary = st.get_ai_usage_summary(ist_today)

        pos_rows: list[dict[str, Any]] = []
        for pos in open_positions:
            if is_substep_dust_position(pos):
                continue
            cur_bid = bids.get(pos.symbol)
            unreal_pnl: Decimal | None = None
            unreal_pct: Decimal | None = None
            if cur_bid is not None:
                _, unreal_pnl = compute_open_position_unrealized_net_pnl(
                    pos,
                    current_bid=cur_bid,
                    estimated_fee_rate=settings.estimated_fee_rate,
                    hourly_interest_rate=settings.hourly_margin_interest_rate,
                )
                denom = pos.cost_basis_usdc + pos.entry_fee_usdc
                if denom > 0:
                    unreal_pct = (unreal_pnl / denom) * Decimal("100")
            pos_rows.append(
                {
                    "position_id": pos.position_id,
                    "symbol": pos.symbol,
                    "quantity": _dec_str(pos.quantity),
                    "entry_price": _dec_str(pos.entry_price),
                    "current_bid": _dec_str(cur_bid) if cur_bid is not None else None,
                    "cost_basis": _dec_str(pos.cost_basis_usdc, "0.01"),
                    "borrowed": _dec_str(pos.borrowed_usdc, "0.01"),
                    "leverage": int(pos.leverage),
                    "liquidation_price": _dec_str(pos.liquidation_price) if pos.liquidation_price else None,
                    "stop_loss_price": _dec_str(pos.stop_loss_price) if pos.stop_loss_price else None,
                    "take_profit_price": _dec_str(pos.take_profit_price) if pos.take_profit_price else None,
                    "unrealized_pnl": _dec_str(unreal_pnl, "0.0001") if unreal_pnl is not None else None,
                    "unrealized_pnl_pct": _dec_str(unreal_pct, "0.01") if unreal_pct is not None else None,
                    "opened_at_iso": pos.opened_at_iso,
                    "strategy_source": _safe_text(pos.strategy_source, 60),
                }
            )

        order_rows = [
            {
                "order_id": str(r.get("order_id", "")),
                "created_at_iso": str(r.get("created_at_iso", "")),
                "symbol": str(r.get("symbol", "")),
                "side": str(r.get("side", "")),
                "status": str(r.get("status", "")),
                "filled_qty": _dec_str(r.get("filled_qty", "0")),
                "avg_fill_price": _dec_str(r.get("avg_fill_price", "0")),
                "gross_quote_usdc": _dec_str(r.get("gross_quote_usdc", "0"), "0.01"),
                "fee_usdc": _dec_str(r.get("fee_usdc", "0"), "0.0001"),
                "realized_pnl_usdc": _dec_str(r.get("realized_pnl_usdc", "0"), "0.0001"),
                "strategy_source": _safe_text(r.get("strategy_source"), 60),
                "explanation": _safe_text(r.get("explanation"), 240),
            }
            for r in st.get_orders(track=track, limit=30)
        ]

        decision_rows = [
            {
                "created_at_iso": str(r.get("created_at_iso", "")),
                "symbol": str(r.get("symbol", "")),
                "interval": str(r.get("interval", "")),
                "rule_action": str(r.get("rule_action", "BEKLE")),
                "ai_action": str(r.get("ai_action", "BEKLE")),
                "final_action": str(r.get("final_action", "BEKLE")),
                "mode_label": _safe_text(r.get("mode_label"), 60),
                "ai_reason": _safe_text(r.get("ai_reason"), 240),
                "block_reason": _safe_text(r.get("block_reason"), 200),
                "executed_order_id": r.get("executed_order_id"),
            }
            for r in st.get_recent_decisions(limit=30, track=track, quote_asset="USDT")
        ]

        eq_rows = [
            {
                "created_at_iso": str(r.get("created_at_iso", "")),
                "main_equity": _dec_str(r.get("main_equity_usdc", "0"), "0.01"),
                "rule_shadow_equity": _dec_str(r.get("rule_shadow_equity_usdc", "0"), "0.01"),
                "benchmark_equity": _dec_str(r.get("benchmark_equity_usdc", "0"), "0.01"),
            }
            for r in st.get_equity_snapshots(limit=90, track=track, quote_asset="USDT")
        ]

        last_tick_iso = st.get_state("cloud_last_tick_iso", "")
        age_sec: float | None = None
        if last_tick_iso:
            try:
                age_sec = round((now_dt - datetime.fromisoformat(last_tick_iso)).total_seconds(), 1)
            except ValueError:
                age_sec = None
        try:
            last_tick_summary = json.loads(st.get_state("cloud_last_tick_summary_json", "") or "null")
        except ValueError:
            last_tick_summary = None

        stale = (
            not params["auto_enabled"]
            or age_sec is None
            or not (-30 <= age_sec <= self.config.stale_after_seconds)
        )

        quote_ccy = summary.get("quote_asset") or "USDT"
        return {
            "schema_version": 2,
            "environment": "cloud",
            "server_time": now_dt.isoformat(),
            "owner_email": self.config.owner_email,
            "mode": mode,
            "track": track,
            "quote_asset": quote_ccy,
            "cloud_auto_enabled": params["auto_enabled"],
            "runner_alive": params["auto_enabled"] and not stale,
            "stale": stale,
            "heartbeat_at": last_tick_iso or None,
            "heartbeat_age_seconds": age_sec,
            "last_tick_summary": last_tick_summary,
            "last_error": _safe_text(st.get_state("cloud_last_error", ""), 360) or None,
            "new_entries_paused": paused,
            "pause_reason": _safe_text(pause_reason, 220),
            "profile": profile.key,
            "profile_label": profile.label_tr,
            "profile_description": profile.description_tr,
            "interval": profile.interval,
            "leverage": settings.leverage,
            "ai_enabled": params["ai_enabled"],
            "ai_force": params["ai_force"],
            "ai_mode": params["ai_mode"],
            "has_gemini_key": eng.config.has_gemini_key,
            "gemini_model": eng.config.gemini_model,
            "has_binance_live_keys": eng.config.has_binance_live_keys,
            "live_env_enabled": eng.config.enable_live_trading,
            "live_confirmed": params["live_confirmed"],
            "scan_symbols": params["symbols"],
            "valuation_complete": bool(summary["valuation_complete"]),
            # Eski /api/status uyumluluk alanlari + detayli portfoy
            "cash": _dec_str(summary["cash_usdc"], "0.01"),
            "open_market_value": _dec_str(summary["open_market_value_usdc"], "0.01"),
            "borrowed": _dec_str(summary["borrowed_usdc"], "0.01"),
            "equity": _dec_str(summary["total_equity_usdc"], "0.01"),
            "realized_pnl": _dec_str(summary["realized_pnl_usdc"], "0.0001"),
            "unrealized_pnl": _dec_str(summary["unrealized_pnl_usdc"], "0.0001"),
            "daily_pnl": _dec_str(summary["daily_net_pnl_usdc"], "0.0001"),
            "total_fees": _dec_str(summary["total_fees_usdc"], "0.0001"),
            "daily_new_positions": st.count_daily_new_positions(ist_today, track=track),
            "risk_settings": {
                "max_open_positions": settings.max_open_positions,
                "max_position_size_usdc": _dec_str(settings.max_position_size_usdc, "0.01"),
                "daily_loss_limit_usdc": _dec_str(settings.daily_loss_limit_usdc, "0.01"),
                "max_daily_new_positions": settings.max_daily_new_positions,
                "stop_loss_pct": _dec_str(settings.stop_loss_pct * Decimal("100"), "0.01"),
                "take_profit_pct": _dec_str(settings.take_profit_pct * Decimal("100"), "0.01"),
                "min_net_profit_exit_pct": _dec_str(settings.min_net_profit_exit_pct * Decimal("100"), "0.02"),
            },
            "ai_usage": {
                "today_network_calls": ai_summary["today_network_calls"],
                "total_network_calls": ai_summary["total_network_calls"],
                "cache_hits": ai_summary["cache_hits"],
                "estimated_cost_usd": _dec_str(ai_summary["estimated_cost_usd"], "0.0001"),
            },
            "open_positions": pos_rows,
            "recent_orders": order_rows,
            "recent_decisions": decision_rows,
            "equity_history": eq_rows,
            "profiles_catalog": [
                {
                    "key": p.key,
                    "label": p.label_tr,
                    "description": p.description_tr,
                    "interval": p.interval,
                }
                for p in TRADING_PROFILES.values()
            ],
        }

    def _apply_settings_update(self, eng: TradingSystemEngine, payload: dict[str, Any]) -> str:
        st = eng.storage
        notes: list[str] = []

        if "cloud_auto_enabled" in payload:
            enabled = bool(payload["cloud_auto_enabled"])
            st.set_state("cloud_auto_enabled", "1" if enabled else "0")
            notes.append("7/24 Bulut Otomatik Tarama açıldı." if enabled else "7/24 Bulut Otomatik Tarama duraklatıldı.")

        if "profile" in payload and isinstance(payload["profile"], str):
            pkey = payload["profile"].strip().upper()
            if pkey in TRADING_PROFILES:
                st.set_state("cloud_profile", pkey)
                notes.append(f"Profil '{pkey}' olarak güncellendi.")

        if "leverage" in payload:
            try:
                lev = max(1, min(5, int(payload["leverage"])))
                st.set_state("cloud_leverage", str(lev))
                notes.append(f"Kaldıraç {lev}x olarak ayarlandı.")
            except (TypeError, ValueError):
                pass

        if "ai_mode" in payload:
            req_ai_mode = str(payload["ai_mode"]).strip()
            has_key = eng.config.has_gemini_key
            if req_ai_mode == "2" and has_key:
                st.set_state("cloud_ai_enabled", "1")
                st.set_state("cloud_ai_force", "1")
            elif req_ai_mode == "1" and has_key:
                st.set_state("cloud_ai_enabled", "1")
                st.set_state("cloud_ai_force", "0")
            else:
                st.set_state("cloud_ai_enabled", "0")
                st.set_state("cloud_ai_force", "0")
        elif "ai_enabled" in payload:
            ai_on = bool(payload["ai_enabled"]) and eng.config.has_gemini_key
            st.set_state("cloud_ai_enabled", "1" if ai_on else "0")
            if not ai_on:
                st.set_state("cloud_ai_force", "0")

        cur_risk = eng.default_risk_settings()
        max_open = cur_risk.max_open_positions
        pos_size = cur_risk.max_position_size_usdc
        daily_loss = cur_risk.daily_loss_limit_usdc
        risk_changed = False

        if "max_open_positions" in payload:
            try:
                max_open = max(1, min(10, int(payload["max_open_positions"])))
                risk_changed = True
            except (TypeError, ValueError):
                pass

        if "max_position_size_usdc" in payload:
            try:
                val = Decimal(str(payload["max_position_size_usdc"]))
                if Decimal("5") <= val <= Decimal("250"):
                    pos_size = val
                    risk_changed = True
            except InvalidOperation:
                pass

        if "daily_loss_limit_usdc" in payload:
            try:
                val = Decimal(str(payload["daily_loss_limit_usdc"]))
                if Decimal("0.5") <= val <= Decimal("100"):
                    daily_loss = val
                    risk_changed = True
            except InvalidOperation:
                pass

        if risk_changed:
            lev_now = max(1, min(5, int(st.get_state("cloud_leverage", "2") or "2")))
            prof_now = get_profile(st.get_state("cloud_profile", "AL_YUKSELINCE_SAT"))
            updated_risk = RiskSettings(
                max_open_positions=max_open,
                max_position_size_usdc=pos_size,
                daily_loss_limit_usdc=daily_loss,
                max_daily_new_positions=prof_now.max_daily_new_positions,
                daily_limit_exit_policy=cur_risk.daily_limit_exit_policy,
                estimated_fee_rate=cur_risk.estimated_fee_rate,
                extra_slippage_bps=cur_risk.extra_slippage_bps,
                stop_loss_pct=prof_now.stop_loss_pct,
                take_profit_pct=prof_now.take_profit_pct,
                leverage=lev_now,
                min_net_profit_exit_pct=prof_now.min_net_profit_exit_pct,
            )
            st.save_risk_settings(updated_risk)
            notes.append("Risk ve pozisyon limitleri kaydedildi.")

        if "mode" in payload and isinstance(payload["mode"], str):
            req_mode = payload["mode"].strip().upper()
            if req_mode == "PAPER":
                st.set_state("cloud_mode", "PAPER")
                st.set_state("cloud_live_confirmed", "0")
                notes.append("Sanal Deneme (PAPER) moduna geçildi.")
            elif req_mode == "LIVE_SPOT":
                confirm_txt = str(payload.get("confirm_text", "")).strip().upper()
                if not eng.config.enable_live_trading or not eng.config.has_binance_live_keys:
                    raise ValueError(
                        "Canlı Spot mod kilitli: Sunucuda ENABLE_LIVE_TRADING=true ve Binance API anahtarları tanımlı olmalıdır."
                    )
                if confirm_txt != "EVET" and st.get_state("cloud_live_confirmed", "0") != "1":
                    raise ValueError(
                        "Canlı Spot modunu açmak için onay kutusuna büyük harflerle EVET yazmanız gerekir."
                    )
                # Gercek hesabi hemen dogrula; IP engeli (-2015) varsa PAPER modda guvende tut
                try:
                    st.ensure_quote_asset("LIVE_SPOT", "USDT")
                    bals = eng.live_executor.fetch_live_spot_balances(ui_live_confirmed=True, quote_asset="USDT")
                except (LiveExecutionError, LiveOrderBlockedError, ValueError) as exc:
                    msg = self._friendly_live_error(str(exc))
                    st.set_state("cloud_last_error", msg)
                    raise ValueError(msg) from exc
                st.set_state("cloud_live_confirmed", "1")
                st.set_state("cloud_mode", "LIVE_SPOT")
                st.set_state("cloud_leverage", "1")
                st.set_state("cloud_last_error", "")
                free_usdt = bals.get("USDT", Decimal("0"))
                notes.append(f"Canlı Spot (1x) doğrulandı ve açıldı (Serbest: {free_usdt:.2f} USDT).")

        if "new_entries_paused" in payload:
            params = self._active_cloud_params(eng)
            pause_flag = bool(payload["new_entries_paused"])
            st.set_new_entries_paused(
                pause_flag,
                "Web panelinden yeni alımlar duraklatıldı." if pause_flag else "",
                track=params["track"],
            )
            notes.append("Yeni alımlar duraklatıldı." if pause_flag else "Yeni alım kilidi kaldırıldı.")

        self.state_sync.persist(self.config.db_path)
        return " ".join(notes) if notes else "Ayarlar güncellendi."

    def _reset_paper_portfolio(self, eng: TradingSystemEngine, initial_balance: Decimal) -> str:
        st = eng.storage
        paper_tracks = ("MAIN", "RULE_SHADOW", "RULE_SHADOW_USDT", "RULE_SHADOW_USDC")
        placeholders = ",".join("?" for _ in paper_tracks)
        with st.execution_lock(), st.atomic() as conn:
            conn.execute(f"DELETE FROM positions WHERE portfolio_track IN ({placeholders})", paper_tracks)
            conn.execute(f"DELETE FROM orders WHERE portfolio_track IN ({placeholders})", paper_tracks)
            conn.execute(f"DELETE FROM processed_signals WHERE portfolio_track IN ({placeholders})", paper_tracks)
            conn.execute(f"DELETE FROM decisions WHERE portfolio_track IN ({placeholders})", paper_tracks)
            conn.execute(f"DELETE FROM equity_snapshots WHERE portfolio_track IN ({placeholders})", paper_tracks)
        st.set_cash_balance(initial_balance, track="MAIN")
        st.set_cash_balance(initial_balance, track="RULE_SHADOW")
        st.set_state("initial_balance_usdc", str(initial_balance))
        st.set_state("portfolio_quote:MAIN", "USDT")
        st.set_new_entries_paused(False, "", track="MAIN")
        st.set_state("benchmark_symbol", "")
        st.set_state("benchmark_initial_price", "")
        st.set_state("benchmark_last_equity", "")
        st.set_state("cloud_last_error", "")
        self.state_sync.persist(self.config.db_path)
        return f"Sanal portföy {initial_balance:.2f} USDT başlangıç bakiyesiyle sıfırlandı."

    def _authenticate_owner(
        self, normalized_headers: Mapping[str, str], response_headers: dict[str, str]
    ) -> Response | None:
        authorization = normalized_headers.get("authorization", "")
        parts = authorization.split()
        if len(parts) != 2 or parts[0].casefold() != "bearer" or len(parts[1]) > 16_384:
            response_headers["WWW-Authenticate"] = "Bearer"
            return Response(401, {"error": "Görüntülemek için Google hesabınızla giriş yapın."}, dict(response_headers))
        if not self.config.project_id or not self.config.owner_email:
            return Response(503, {"error": "Panel erişim ayarları henüz hazır değil."}, dict(response_headers))
        try:
            claims = self.verifier(parts[1], self.config.project_id)
        except (auth_exceptions.TransportError, requests.RequestException):
            return Response(503, {"error": "Giriş doğrulanamıyor. Biraz sonra tekrar deneyin."}, dict(response_headers))
        except Exception:
            response_headers["WWW-Authenticate"] = "Bearer"
            return Response(401, {"error": "Oturum doğrulanamadı. Yeniden giriş yapın."}, dict(response_headers))

        if (
            claims.get("aud") != self.config.project_id
            or claims.get("iss") != f"https://securetoken.google.com/{self.config.project_id}"
            or not isinstance(claims.get("sub"), str)
            or not (0 < len(claims["sub"]) <= 128)
        ):
            return Response(401, {"error": "Oturum doğrulanamadı. Yeniden giriş yapın."}, dict(response_headers))

        email = claims.get("email")
        if (
            claims.get("email_verified") is not True
            or not isinstance(email, str)
            or email.strip().casefold() != self.config.owner_email.casefold()
        ):
            return Response(403, {"error": "Bu panel yalnızca hesap sahibine açıktır."}, dict(response_headers))
        return None

    def handle(
        self, method: str, path: str, headers: Mapping[str, str], body_bytes: bytes = b""
    ) -> Response:
        normalized = {k.casefold(): v for k, v in headers.items()}
        route = urlsplit(path).path.rstrip("/") or "/"
        origin = normalized.get("origin")
        response_headers = {
            "Cache-Control": "no-store, private, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Vary": "Origin",
            "X-Content-Type-Options": "nosniff",
        }
        if origin in ALLOWED_ORIGINS:
            response_headers["Access-Control-Allow-Origin"] = origin

        def respond(status: int, payload: dict[str, Any] | None) -> Response:
            return Response(status, payload, dict(response_headers))

        if route in {"/healthz", "/api/health"} and method == "GET":
            return respond(200, {"ok": True, "service": "tradecebrail-bot"})

        if origin is not None and origin not in ALLOWED_ORIGINS:
            return respond(403, {"error": "Bu adresten erişime izin verilmiyor."})

        if method == "OPTIONS" and route.startswith("/api/"):
            response_headers.update(
                {
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Access-Control-Max-Age": "600",
                }
            )
            return respond(204, None)

        # Cloud Scheduler 7/24 zamanlayici uc noktasi
        if route == "/api/cron/tick":
            if method != "POST":
                return respond(405, {"error": "Yalnızca POST desteklenir."})
            if len(self.config.cron_secret) < 16:
                return respond(503, {"error": "Zamanlayıcı gizli anahtarı yapılandırılmamış."})
            provided = normalized.get("x-cron-secret", "")
            if not provided or not hmac.compare_digest(
                provided.encode("utf-8"), self.config.cron_secret.encode("utf-8")
            ):
                return respond(401, {"error": "Yetkisiz zamanlayıcı çağrısı."})
            if not self._op_lock.acquire(blocking=False):
                return respond(200, {"ok": True, "skipped": True, "reason": "Önceki tarama döngüsü devam ediyor."})
            try:
                eng = self._get_engine()
                if eng.storage.get_state("cloud_auto_enabled", "1") != "1":
                    return respond(200, {"ok": True, "skipped": True, "reason": "7/24 otomatik tarama kullanıcı tarafından duraklatılmış."})
                tick_summary = self._run_scan_cycle(eng, source="CRON")
            except Exception as exc:
                logging.exception("Cron tick hatasi")
                return respond(500, {"error": _safe_text(exc, 240)})
            finally:
                self._op_lock.release()

            # 30 saniyede bir kâr-al ve yeni alım fırsatlarını yakalamak için kilidi bırakarak ikinci geçiş
            double_pass = (
                "PYTEST_CURRENT_TEST" not in os.environ
                and _parse_bool(os.getenv("CLOUD_CRON_DOUBLE_PASS", "0"), default=False)
            )
            if double_pass:
                time.sleep(25.0)
                if self._op_lock.acquire(blocking=False):
                    try:
                        eng2 = self._get_engine()
                        if eng2.storage.get_state("cloud_auto_enabled", "1") == "1":
                            tick_summary_2 = self._run_scan_cycle(eng2, source="CRON_30SN")
                            tick_summary["orders_executed"] = int(tick_summary.get("orders_executed", 0)) + int(
                                tick_summary_2.get("orders_executed", 0)
                            )
                            tick_summary["second_pass"] = True
                    except Exception:
                        logging.exception("Cron 2. gecis uyarisi")
                    finally:
                        self._op_lock.release()
            return respond(200, {"ok": True, "skipped": False, "summary": tick_summary})

        known_owner_routes = {
            "/status",
            "/api/status",
            "/api/dashboard",
            "/api/wallet",
            "/api/tick",
            "/api/settings",
            "/api/close-positions",
            "/api/reset-paper",
        }
        if route not in known_owner_routes:
            return respond(404, {"error": "Adres bulunamadı."})

        auth_err = self._authenticate_owner(normalized, response_headers)
        if auth_err is not None:
            return auth_err

        if len(body_bytes) > MAX_BODY_BYTES:
            return respond(413, {"error": "İstek gövdesi çok büyük."})

        payload: dict[str, Any] = {}
        if method == "POST" and body_bytes.strip():
            try:
                decoded = json.loads(body_bytes.decode("utf-8"))
                if not isinstance(decoded, dict):
                    return respond(400, {"error": "Geçersiz JSON nesnesi."})
                payload = decoded
            except (UnicodeDecodeError, ValueError):
                return respond(400, {"error": "Geçersiz JSON biçimi."})

        try:
            eng = self._get_engine()
        except Exception:
            return respond(503, {"error": "Bulut durum deposu şu anda yüklenemiyor. Birkaç saniye sonra tekrar deneyin."})

        if route == "/api/wallet":
            if method != "GET":
                return respond(405, {"error": "Yalnızca GET desteklenir."})
            with self._op_lock:
                return respond(200, self._fetch_real_wallet_summary(eng))

        if route in {"/status", "/api/status", "/api/dashboard"}:
            if method != "GET":
                return respond(405, {"error": "Yalnızca GET desteklenir."})
            with self._op_lock:
                return respond(200, self._build_dashboard(eng))

        if method != "POST":
            return respond(405, {"error": "Yalnızca POST desteklenir."})

        with self._op_lock:
            try:
                if route == "/api/tick":
                    tick_summary = self._run_scan_cycle(eng, source="MANUEL")
                    return respond(
                        200,
                        {
                            "ok": True,
                            "message": f"{tick_summary['scanned_count']} parite tarandı ({tick_summary['orders_executed']} işlem).",
                            "summary": tick_summary,
                            "dashboard": self._build_dashboard(eng),
                        },
                    )
                if route == "/api/settings":
                    msg = self._apply_settings_update(eng, payload)
                    return respond(200, {"ok": True, "message": msg, "dashboard": self._build_dashboard(eng)})
                if route == "/api/close-positions":
                    params = self._active_cloud_params(eng)
                    sym = payload.get("symbol")
                    target_sym = sym.strip().upper() if isinstance(sym, str) and _SYMBOL_RE.fullmatch(sym.strip().upper()) else None
                    closed_ok, close_msg = eng.manual_close_open_position(
                        params["settings"],
                        execution_mode=params["mode"],
                        ui_live_confirmed=(params["mode"] == "LIVE_SPOT"),
                        symbol=target_sym,
                    )
                    self.state_sync.persist(self.config.db_path)
                    return respond(
                        200,
                        {
                            "ok": closed_ok,
                            "message": _safe_text(close_msg, 320),
                            "dashboard": self._build_dashboard(eng),
                        },
                    )
                if route == "/api/reset-paper":
                    if str(payload.get("confirm", "")).strip().upper() != "SIFIRLA":
                        return respond(400, {"error": "Sıfırlama onayı için 'SIFIRLA' gönderilmelidir."})
                    init_bal = self.config.initial_balance_usdc
                    if "initial_balance_usdc" in payload:
                        try:
                            cand = Decimal(str(payload["initial_balance_usdc"]))
                            if Decimal("10") <= cand <= Decimal("100000"):
                                init_bal = cand
                        except InvalidOperation:
                            pass
                    msg = self._reset_paper_portfolio(eng, init_bal)
                    return respond(200, {"ok": True, "message": msg, "dashboard": self._build_dashboard(eng)})
            except ValueError as exc:
                return respond(400, {"error": _safe_text(exc, 340), "dashboard": self._build_dashboard(eng)})
            except (MarketDataError, LiveExecutionError, LiveOrderBlockedError) as exc:
                return respond(502, {"error": _safe_text(exc, 340), "dashboard": self._build_dashboard(eng)})
            except Exception as exc:
                logging.exception("API islem hatasi (%s)", route)
                return respond(500, {"error": _safe_text(exc, 260)})

        return respond(404, {"error": "Adres bulunamadı."})


# Geriye donuk adlandirma uyumlulugu
Config = CloudConfig
StatusApplication = CloudBotApplication


def make_handler(application: CloudBotApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TradeCebrailCloud"
        sys_version = ""
        timeout = 55

        def _handle(self) -> None:
            content_len = 0
            raw_len = self.headers.get("Content-Length")
            if raw_len:
                try:
                    content_len = max(0, int(raw_len))
                except ValueError:
                    content_len = 0
            body_bytes = self.rfile.read(min(content_len, MAX_BODY_BYTES + 1)) if content_len > 0 else b""
            response = application.handle(self.command, self.path, self.headers, body_bytes)
            payload = (
                json.dumps(response.body, ensure_ascii=False, allow_nan=False).encode("utf-8")
                if response.body is not None
                else b""
            )
            self.send_response(response.status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD" and payload:
                self.wfile.write(payload)

        do_GET = do_OPTIONS = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = _handle

        def log_message(self, format: str, *args: Any) -> None:
            # Asla baslik, token, sorgu dizgisi veya govde loglama.
            logging.info("%s %s", self.command, urlsplit(self.path).path)

    return Handler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    app = CloudBotApplication(CloudConfig.from_env())
    port = int(os.getenv("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(app))
    server.daemon_threads = True
    logging.info("TradeCebrail Cloud API dinleniyor: 0.0.0.0:%s", port)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
