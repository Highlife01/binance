"""Finansal veri modelleri, emir filtreleri ve yapılandırılmış karar şemaları."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DecisionAction = Literal["AL", "SAT", "BEKLE"]
OrderType = Literal["MARKET", "LIMIT"]
OrderSide = Literal["BUY", "SELL"]
OrderStatus = Literal["FILLED", "PARTIALLY_FILLED", "EXPIRED_UNFILLED", "REJECTED"]


def quantize_down(value: Decimal, step: Decimal) -> Decimal:
    """Binance stepSize / tickSize uyumu için aşağı yönlü hassas basamak yuvarlama."""
    if step <= Decimal("0"):
        return value
    units = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return (units * step).normalize() if (units * step) != 0 else Decimal("0")


@dataclass(frozen=True)
class SymbolFilters:
    """Binance `/api/v3/exchangeInfo` üzerinden dinamik çekilen parite kuralları."""

    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    is_spot_trading_allowed: bool
    min_price: Decimal
    max_price: Decimal
    tick_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal
    market_min_qty: Decimal
    market_max_qty: Decimal
    market_step_size: Decimal
    min_notional: Decimal
    max_notional: Decimal
    apply_min_to_market: bool = True

    @property
    def is_tradable_usdc_spot(self) -> bool:
        return self.is_tradable_spot_for_quote("USDC")

    def is_tradable_spot_for_quote(self, quote_asset: str | None = None) -> bool:
        allowed_quotes = {quote_asset.upper()} if quote_asset else {"USDC", "USDT"}
        return (
            self.quote_asset in allowed_quotes
            and self.status == "TRADING"
            and self.is_spot_trading_allowed
        )


@dataclass(frozen=True)
class Candle:
    """Tek bir mum verisi (açılış ve kapanış zamanları UTC milisaniye)."""

    open_time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time_ms: int
    quote_volume: Decimal
    trade_count: int
    is_closed: bool


@dataclass(frozen=True)
class OrderBookLevel:
    price: Decimal
    quantity: Decimal


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    fetched_at_utc: datetime
    source_url: str

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def spread_pct(self) -> Decimal | None:
        if self.best_bid and self.best_ask and self.best_bid > 0:
            return ((self.best_ask - self.best_bid) / self.best_bid) * Decimal("100")
        return None


@dataclass(frozen=True)
class MarketBundle:
    """Bir analiz adımında toplanan doğrulanmış piyasa verisi paketi."""

    symbol: str
    interval: str
    filters: SymbolFilters
    closed_candles: list[Candle]
    order_book: OrderBookSnapshot
    server_time_ms: int
    fetched_at_utc: datetime
    source_label: str
    wake_gap_detected: bool = False
    used_weight_1m: int | None = None


@dataclass(frozen=True)
class TechnicalSignal:
    """Kural tabanlı teknik gösterge stratejisi çıktısı."""

    symbol: str
    interval: str
    candle_open_time_ms: int
    candle_close_time_ms: int
    action: DecisionAction
    reason: str
    ema_fast: Decimal
    ema_slow: Decimal
    rsi_14: Decimal
    macd_line: Decimal
    macd_signal: Decimal
    macd_hist: Decimal
    atr_14: Decimal
    volume_ratio: Decimal
    close_price: Decimal

    @property
    def signal_key(self) -> str:
        return f"{self.symbol}:{self.interval}:{self.candle_open_time_ms}:{self.action}"


class GeminiStructuredResponse(BaseModel):
    """Gemini `response_schema` için katı doğrulamalı çıktı şeması."""

    karar: Literal["AL", "SAT", "BEKLE"] = Field(
        description="Yalnızca AL, SAT veya BEKLE değerlerinden biri."
    )
    kisa_gerekce: str = Field(
        min_length=5,
        max_length=400,
        description="Piyasa verilerine dayalı kısa, net Türkçe gerekçe.",
    )
    veri_zamani: str = Field(
        min_length=4,
        max_length=64,
        description="Analiz edilen son kapanmış mumun ISO zaman damgası.",
    )
    risk_notlari: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Kısa Türkçe risk uyarıları listesi.",
    )
    model_oz_beyan_endeksi: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Modelin kendi öz-beyan puanı (doğrulanmış başarı olasılığı değildir).",
    )

    @field_validator("kisa_gerekce")
    @classmethod
    def clean_reason(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Gerekçe boş olamaz.")
        return cleaned


@dataclass(frozen=True)
class AIDecisionRecord:
    """Uygulama içinde kullanılan nihai yapay zekâ veya geri dönüş (fallback) kararı."""

    action: DecisionAction
    reason: str
    data_timestamp_iso: str
    risk_notes: list[str]
    self_reported_score: int | None
    mode_label: str  # "GEMINI_AKTIF" | "YAPAY_ZEKA_KAPALI_KURAL_TABANLI" | "GEMINI_HATA_BEKLE" | "ONBELLEK"
    model_name: str
    from_cache: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: Decimal = Decimal("0")
    pricing_verified: bool = False
    error_detail: str | None = None


@dataclass(frozen=True)
class RiskSettings:
    """Kullanıcı tarafından panelden güncellenebilen risk sınırları."""

    max_open_positions: int = 1
    max_position_size_usdc: Decimal = Decimal("10.00")
    daily_loss_limit_usdc: Decimal = Decimal("2.00")
    max_daily_new_positions: int = 3
    daily_limit_exit_policy: str = "HOLD_EXISTING"  # HOLD_EXISTING | AUTO_CLOSE
    estimated_fee_rate: Decimal = Decimal("0.0010")
    extra_slippage_bps: Decimal = Decimal("2.0")
    stop_loss_pct: Decimal = Decimal("0.025")
    take_profit_pct: Decimal = Decimal("0.045")
    allow_stablecoin_conversion: bool = False


@dataclass(frozen=True)
class RiskCheckResult:
    allowed: bool
    reason: str
    approved_notional_usdc: Decimal = Decimal("0")
    approved_quantity: Decimal = Decimal("0")
    daily_total_loss_usdc: Decimal = Decimal("0")
    trigger_daily_halt: bool = False
    trigger_protective_close: bool = False


@dataclass
class Position:
    position_id: str
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    cost_basis_usdc: Decimal       # Komisyon hariç net varlık maliyeti
    entry_fee_usdc: Decimal        # Girişte ödenen komisyon
    opened_at_iso: str
    signal_key: str
    strategy_source: str
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    status: str = "OPEN"
    closed_at_iso: str | None = None
    exit_price: Decimal | None = None
    exit_fee_usdc: Decimal = Decimal("0")
    realized_pnl_usdc: Decimal = Decimal("0")
    close_reason: str | None = None


@dataclass(frozen=True)
class SimulatedOrderResult:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    requested_qty: Decimal
    filled_qty: Decimal
    avg_fill_price: Decimal
    gross_quote_usdc: Decimal
    fee_usdc: Decimal
    slippage_bps: Decimal
    signal_key: str
    strategy_source: str
    explanation: str
    created_at_iso: str
    realized_pnl_usdc: Decimal = Decimal("0")
