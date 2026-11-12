"""Offline unit tests for TradeCebrail 24/7 Cloud Run API (`cloud/api/app.py`)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

import pytest

from binance_ai_bot.domain import (
    Candle,
    MarketBundle,
    OrderBookLevel,
    OrderBookSnapshot,
    SymbolFilters,
)
from binance_ai_bot.engine import TradingSystemEngine
from cloud.api.app import CloudBotApplication, CloudConfig, snapshot_sqlite_bytes


OWNER_CLAIMS = {
    "aud": "adanahizlitaksi",
    "iss": "https://securetoken.google.com/adanahizlitaksi",
    "sub": "uid-owner-123",
    "email": "cebrailkara@gmail.com",
    "email_verified": True,
}


class FakeStateSync:
    """In-memory GCS SQLite object simulator for testing cold-start persistence."""

    def __init__(self) -> None:
        self.blob: bytes | None = None
        self.persist_calls = 0
        self.restore_calls = 0

    def restore(self, db_path: Path) -> bool:
        self.restore_calls += 1
        if self.blob is None:
            return False
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_bytes(self.blob)
        return True

    def persist(self, db_path: Path) -> bool:
        self.persist_calls += 1
        self.blob = snapshot_sqlite_bytes(db_path)
        return True


class FakeMarketClient:
    """Deterministic bullish candle + orderbook generator for offline cloud scans."""

    def __init__(self) -> None:
        self.price_by_symbol: dict[str, Decimal] = {}

    def _filters(self, symbol: str) -> SymbolFilters:
        base = symbol[:-4]
        return SymbolFilters(
            symbol=symbol,
            base_asset=base,
            quote_asset="USDT",
            status="TRADING",
            is_spot_trading_allowed=True,
            min_price=Decimal("0.0001"),
            max_price=Decimal("1000000"),
            tick_size=Decimal("0.01"),
            min_qty=Decimal("0.001"),
            max_qty=Decimal("100000"),
            step_size=Decimal("0.001"),
            market_min_qty=Decimal("0.001"),
            market_max_qty=Decimal("100000"),
            market_step_size=Decimal("0.001"),
            min_notional=Decimal("5.0"),
            max_notional=Decimal("1000000"),
            apply_min_to_market=True,
        )

    def discover_spot_pairs(self, quote_asset: str = "USDT", allow_stablecoin_conversion: bool = False):
        return {sym: self._filters(sym) for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT")}

    def fetch_order_book(self, symbol: str, depth_limit: int = 20) -> OrderBookSnapshot:
        px = self.price_by_symbol.get(symbol, Decimal("100.00"))
        now = datetime.now(timezone.utc)
        return OrderBookSnapshot(
            symbol=symbol,
            bids=[OrderBookLevel(price=px, quantity=Decimal("500"))],
            asks=[OrderBookLevel(price=px + Decimal("0.02"), quantity=Decimal("500"))],
            fetched_at_utc=now,
            source_url="https://api.binance.com/api/v3/depth",
        )

    def fetch_verified_market_bundle(
        self, symbol: str, interval: str, allow_stablecoin_conversion: bool = False
    ) -> MarketBundle:
        now = datetime.now(timezone.utc)
        now_ms = int(now.timestamp() * 1000)
        step_ms = 60_000
        count = 60
        last_close = now_ms - 5_000
        last_open = last_close - step_ms + 1
        first_open = last_open - (count - 1) * step_ms
        candles: list[Candle] = []
        px = Decimal("100.00")
        for i in range(count):
            step = Decimal("0.35") if (i % 2 == 0) else Decimal("-0.25")
            if i >= count - 2:
                step = Decimal("0.45")
            open_p = px
            close_p = px + step
            vol = Decimal("25.0") if i >= count - 3 else Decimal("15.0")
            open_ms = first_open + i * step_ms
            candles.append(
                Candle(
                    open_time_ms=open_ms,
                    open=open_p,
                    high=max(open_p, close_p) + Decimal("0.30"),
                    low=min(open_p, close_p) - Decimal("0.30"),
                    close=close_p,
                    volume=vol,
                    close_time_ms=open_ms + step_ms - 1,
                    quote_volume=vol * close_p,
                    trade_count=200,
                    is_closed=True,
                )
            )
            px = close_p
        last_px = candles[-1].close
        self.price_by_symbol[symbol] = last_px
        return MarketBundle(
            symbol=symbol,
            interval=interval,
            filters=self._filters(symbol),
            closed_candles=candles,
            order_book=self.fetch_order_book(symbol),
            server_time_ms=now_ms,
            fetched_at_utc=now,
            source_label="MockCloudTest",
            wake_gap_detected=False,
        )


def _make_app(tmp_path: Path, state_sync: FakeStateSync | None = None) -> tuple[CloudBotApplication, FakeStateSync, FakeMarketClient]:
    sync = state_sync or FakeStateSync()
    cfg = CloudConfig(
        project_id="adanahizlitaksi",
        owner_email="cebrailkara@gmail.com",
        state_bucket="tradecebrail-bot-state-873885134851",
        cron_secret="c" * 32,
        db_path=tmp_path / "cloud.db",
        initial_balance_usdc=Decimal("86.00"),
        enable_live_trading=False,
    )
    sync.restore(cfg.db_path)
    market = FakeMarketClient()
    eng = TradingSystemEngine(cfg.to_app_config(), market_client=market)
    eng.storage.set_state("cloud_scan_symbols", "BTCUSDT,ETHUSDT,SOLUSDT")

    def fake_verifier(token: str, audience: str):
        if token == "owner-token":
            return dict(OWNER_CLAIMS)
        if token == "stranger-token":
            return {**OWNER_CLAIMS, "email": "other@example.com"}
        if token == "unverified-token":
            return {**OWNER_CLAIMS, "email_verified": False}
        raise ValueError("Invalid token")

    app = CloudBotApplication(cfg, engine=eng, state_sync=sync, verifier=fake_verifier)
    return app, sync, market


def test_public_health_and_auth_boundaries(tmp_path: Path):
    app, _, _ = _make_app(tmp_path)

    # Public health endpoints require no auth and leak nothing
    r_health = app.handle("GET", "/api/health", {})
    assert r_health.status == 200
    assert r_health.body == {"ok": True, "service": "tradecebrail-bot"}

    # Missing Bearer token -> 401
    r_unauth = app.handle("GET", "/api/dashboard", {})
    assert r_unauth.status == 401

    # Verified Google user who is NOT cebrailkara@gmail.com -> 403
    r_stranger = app.handle("GET", "/api/dashboard", {"Authorization": "Bearer stranger-token"})
    assert r_stranger.status == 403

    # Unverified email -> 403
    r_unverified = app.handle("GET", "/api/dashboard", {"Authorization": "Bearer unverified-token"})
    assert r_unverified.status == 403

    # Evil cross-origin request -> 403 even with valid token
    r_evil_origin = app.handle(
        "GET",
        "/api/dashboard",
        {"Authorization": "Bearer owner-token", "Origin": "https://attacker.example"},
    )
    assert r_evil_origin.status == 403

    # Allowed Firebase Hosting origin -> 200 + CORS header
    r_ok = app.handle(
        "GET",
        "/api/dashboard",
        {"Authorization": "Bearer owner-token", "Origin": "https://tradecebrail.web.app"},
    )
    assert r_ok.status == 200
    assert r_ok.headers["Access-Control-Allow-Origin"] == "https://tradecebrail.web.app"
    assert r_ok.body is not None
    assert r_ok.body["mode"] == "PAPER"
    assert r_ok.body["leverage"] == 2
    assert r_ok.body["cash"] == "86.00"


def test_cron_tick_executes_2x_paper_trades_and_survives_cold_restart(tmp_path: Path):
    app, sync, _ = _make_app(tmp_path / "instance1")

    # Wrong cron secret -> 401
    r_bad = app.handle("POST", "/api/cron/tick", {"X-Cron-Secret": "wrong"})
    assert r_bad.status == 401

    # Valid cron secret -> scans symbols, opens 2x leveraged paper positions, persists to GCS
    r_cron = app.handle("POST", "/api/cron/tick", {"X-Cron-Secret": "c" * 32})
    assert r_cron.status == 200
    assert r_cron.body is not None
    assert r_cron.body["ok"] is True
    assert r_cron.body["skipped"] is False
    assert r_cron.body["summary"]["orders_executed"] >= 1
    assert sync.persist_calls >= 1
    assert sync.blob is not None

    # Simulate Cloud Run scale-to-zero and cold boot on a brand-new directory!
    app2, _, _ = _make_app(tmp_path / "instance2_cold_boot", state_sync=sync)
    dash = app2.handle("GET", "/api/dashboard", {"Authorization": "Bearer owner-token"})
    assert dash.status == 200
    body: dict[str, Any] = dash.body or {}
    assert len(body["open_positions"]) >= 1
    pos0 = body["open_positions"][0]
    assert pos0["leverage"] == 2
    assert Decimal(pos0["borrowed"]) > Decimal("0")
    assert pos0["liquidation_price"] is not None


def test_settings_manual_close_reset_and_secret_redaction(tmp_path: Path):
    app, sync, _ = _make_app(tmp_path)
    headers = {"Authorization": "Bearer owner-token"}

    # Trigger manual scan -> opens positions
    r_tick = app.handle("POST", "/api/tick", headers)
    assert r_tick.status == 200
    assert len(r_tick.body["dashboard"]["open_positions"]) >= 1

    # Change leverage to 3x and pause new entries
    r_set = app.handle(
        "POST",
        "/api/settings",
        headers,
        json.dumps({"leverage": 3, "new_entries_paused": True, "max_position_size_usdc": "12.50"}).encode("utf-8"),
    )
    assert r_set.status == 200
    assert r_set.body["dashboard"]["leverage"] == 3
    assert r_set.body["dashboard"]["new_entries_paused"] is True
    assert r_set.body["dashboard"]["risk_settings"]["max_position_size_usdc"] == "12.50"

    # Attempting LIVE_SPOT when server ENABLE_LIVE_TRADING=False stays safely blocked (400)
    r_live_block = app.handle(
        "POST",
        "/api/settings",
        headers,
        json.dumps({"mode": "LIVE_SPOT", "confirm_text": "EVET"}).encode("utf-8"),
    )
    assert r_live_block.status == 400
    assert r_live_block.body["dashboard"]["mode"] == "PAPER"

    # Close all open positions manually
    r_close = app.handle("POST", "/api/close-positions", headers, b"{}")
    assert r_close.status == 200
    assert r_close.body["ok"] is True
    assert r_close.body["dashboard"]["open_positions"] == []

    # Inject a fake secret into cloud_last_error and verify dashboard masks it
    app._get_engine().storage.set_state(
        "cloud_last_error", "Hata olustu api_key=AIzaSyFakeSecretKey1234567890abcdef"
    )
    r_masked = app.handle("GET", "/api/dashboard", headers)
    assert r_masked.status == 200
    assert "AIzaSyFakeSecretKey" not in json.dumps(r_masked.body)

    # Reset paper portfolio back to 86.00 USDT
    r_reset = app.handle(
        "POST",
        "/api/reset-paper",
        headers,
        json.dumps({"confirm": "SIFIRLA"}).encode("utf-8"),
    )
    assert r_reset.status == 200
    assert r_reset.body["dashboard"]["cash"] == "86.00"
    assert r_reset.body["dashboard"]["recent_orders"] == []
