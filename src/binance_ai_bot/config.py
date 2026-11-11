"""Uygulama yapılandırması ve güvenli varsayılan sanal test parametreleri."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_decimal(raw: str | None, default: str, min_val: Decimal | None = None) -> Decimal:
    if raw is None or not raw.strip():
        value = Decimal(default)
    else:
        try:
            value = Decimal(raw.strip())
        except InvalidOperation:
            value = Decimal(default)
    if min_val is not None and value < min_val:
        return Decimal(default)
    return value


def _parse_int(raw: str | None, default: int, min_val: int = 1) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw.strip())
        return parsed if parsed >= min_val else default
    except ValueError:
        return default


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "evet", "on"}


@dataclass(frozen=True)
class AppConfig:
    """Tüm modüllerin kullandığı doğrulanmış yerel yapılandırma."""

    # Bakiye ve risk varsayimlari (Yatirim tavsiyesi degildir)
    initial_balance_usdc: Decimal = Decimal("86.00")
    max_open_positions: int = 1
    max_position_size_usdc: Decimal = Decimal("10.00")
    daily_loss_limit_usdc: Decimal = Decimal("2.00")
    max_daily_new_positions: int = 3
    daily_limit_exit_policy: str = "HOLD_EXISTING"  # HOLD_EXISTING | AUTO_CLOSE
    stop_loss_pct: Decimal = Decimal("0.025")       # %2.5 koruyucu sanal zarar-kes
    take_profit_pct: Decimal = Decimal("0.045")     # %4.5 sanal kar-al hedefi

    # Maliyet ve dolum modeli varsayimlari
    estimated_fee_rate: Decimal = Decimal("0.0010")  # %0.10 standart spot komisyon varsayimi
    extra_slippage_bps: Decimal = Decimal("2.0")     # 2 baz puan (%0.02) muhafazakar kayma tamponu

    # Binance herkese acik veri katmani (Turkiye resmi erisilebilir uc nokta: api.binance.me)
    binance_public_base_url: str = "https://api.binance.me"
    binance_fallback_base_url: str = "https://data-api.binance.vision"
    default_symbol: str = "BTCUSDC"
    default_interval: str = "15m"
    max_data_age_seconds: int = 120

    # Canli Spot Islem Ayarlari (Varsayilan kapali; 5 katmanli guvenlik kilidi)
    enable_live_trading: bool = False
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_live_base_url: str = "https://api.binance.me"
    binance_recv_window_ms: int = 5000

    # Gemini yapay zeka ayarlari (Istege bagli)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    ai_daily_call_limit: int = 25
    ai_min_call_interval_seconds: int = 180
    ai_timeout_seconds: int = 20
    ai_estimated_input_usd_per_1m: Decimal = Decimal("0.10")
    ai_estimated_output_usd_per_1m: Decimal = Decimal("0.40")
    ai_pricing_verified: bool = False

    # Sistem ve veritabani
    timezone_name: str = "Europe/Istanbul"
    sqlite_db_path: Path = PROJECT_ROOT / "data" / "sanal_sistem.db"
    localhost_host: str = "127.0.0.1"
    localhost_port: int = 8501

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    @property
    def has_gemini_key(self) -> bool:
        cleaned = self.gemini_api_key.strip()
        return bool(cleaned) and cleaned not in {"BURAYA_YAZIN", "YOUR_API_KEY"}

    @property
    def has_binance_live_keys(self) -> bool:
        k = self.binance_api_key.strip()
        s = self.binance_api_secret.strip()
        placeholders = {"", "BURAYA_YAZIN", "YOUR_API_KEY", "YOUR_SECRET_KEY"}
        return k not in placeholders and s not in placeholders


def load_config(env_path: Path | None = None) -> AppConfig:
    """`.env` dosyasını güvenli biçimde yükler ve doğrulanmış AppConfig döndürür."""
    target_env = env_path if env_path is not None else (PROJECT_ROOT / ".env")
    if target_env.exists():
        # Pytest ortaminda monkeypatch degiskenlerini koru; normal calismada .env guncellemelerini aninda oku
        should_override = "PYTEST_CURRENT_TEST" not in os.environ
        load_dotenv(dotenv_path=target_env, override=should_override)

    exit_policy = os.getenv("DAILY_LIMIT_EXIT_POLICY", "HOLD_EXISTING").strip().upper()
    if exit_policy not in {"HOLD_EXISTING", "AUTO_CLOSE"}:
        exit_policy = "HOLD_EXISTING"

    db_raw = os.getenv("SQLITE_DB_PATH", "data/sanal_sistem.db").strip()
    db_path = Path(db_raw)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    tz_name = os.getenv("SYSTEM_TIMEZONE", "Europe/Istanbul").strip() or "Europe/Istanbul"
    try:
        ZoneInfo(tz_name)
    except Exception:
        tz_name = "Europe/Istanbul"

    return AppConfig(
        initial_balance_usdc=_parse_decimal(
            os.getenv("INITIAL_BALANCE_USDC"), "86.00", min_val=Decimal("1")
        ),
        max_open_positions=_parse_int(os.getenv("MAX_OPEN_POSITIONS"), 1, min_val=1),
        max_position_size_usdc=_parse_decimal(
            os.getenv("MAX_POSITION_SIZE_USDC"), "10.00", min_val=Decimal("1")
        ),
        daily_loss_limit_usdc=_parse_decimal(
            os.getenv("DAILY_LOSS_LIMIT_USDC"), "2.00", min_val=Decimal("0.10")
        ),
        max_daily_new_positions=_parse_int(os.getenv("MAX_DAILY_NEW_POSITIONS"), 3, min_val=1),
        daily_limit_exit_policy=exit_policy,
        estimated_fee_rate=_parse_decimal(
            os.getenv("ESTIMATED_FEE_RATE"), "0.0010", min_val=Decimal("0")
        ),
        extra_slippage_bps=_parse_decimal(
            os.getenv("EXTRA_SLIPPAGE_BPS"), "2.0", min_val=Decimal("0")
        ),
        binance_public_base_url=(
            os.getenv("BINANCE_PUBLIC_BASE_URL", "https://api.binance.me").strip().rstrip("/")
        ),
        binance_fallback_base_url=(
            os.getenv("BINANCE_FALLBACK_BASE_URL", "https://data-api.binance.vision").strip().rstrip("/")
        ),
        default_symbol=os.getenv("DEFAULT_SYMBOL", "BTCUSDC").strip().upper() or "BTCUSDC",
        default_interval=os.getenv("DEFAULT_INTERVAL", "15m").strip() or "15m",
        max_data_age_seconds=_parse_int(os.getenv("MAX_DATA_AGE_SECONDS"), 120, min_val=15),
        enable_live_trading=_parse_bool(os.getenv("ENABLE_LIVE_TRADING"), default=False),
        binance_api_key=os.getenv("BINANCE_API_KEY", "").strip(),
        binance_api_secret=os.getenv("BINANCE_API_SECRET", "").strip(),
        binance_live_base_url=(
            os.getenv("BINANCE_LIVE_BASE_URL", "https://api.binance.me").strip().rstrip("/")
        ),
        binance_recv_window_ms=_parse_int(os.getenv("BINANCE_RECV_WINDOW_MS"), 5000, min_val=1000),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite",
        ai_daily_call_limit=_parse_int(os.getenv("AI_DAILY_CALL_LIMIT"), 25, min_val=1),
        ai_min_call_interval_seconds=_parse_int(
            os.getenv("AI_MIN_CALL_INTERVAL_SECONDS"), 180, min_val=10
        ),
        ai_timeout_seconds=_parse_int(os.getenv("AI_TIMEOUT_SECONDS"), 20, min_val=5),
        ai_estimated_input_usd_per_1m=_parse_decimal(
            os.getenv("AI_ESTIMATED_INPUT_USD_PER_1M"), "0.10", min_val=Decimal("0")
        ),
        ai_estimated_output_usd_per_1m=_parse_decimal(
            os.getenv("AI_ESTIMATED_OUTPUT_USD_PER_1M"), "0.40", min_val=Decimal("0")
        ),
        ai_pricing_verified=_parse_bool(os.getenv("AI_PRICING_VERIFIED"), default=False),
        timezone_name=tz_name,
        sqlite_db_path=db_path,
    )
