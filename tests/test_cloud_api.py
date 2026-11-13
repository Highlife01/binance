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


def test_al_yukselince_sat_holds_pullback_locks_net_profit_and_rebuys_continuously(tmp_path: Path):
    """Verify continuous Buy -> Hold slight dip -> Sell in net profit -> Buy again cycle."""
    from binance_ai_bot.config import AppConfig
    from binance_ai_bot.profiles import get_profile
    from binance_ai_bot.strategy import evaluate_technical_strategy

    prof = get_profile("AL_YUKSELINCE_SAT")
    assert prof.min_net_profit_exit_pct == Decimal("0.0080")
    assert prof.take_profit_pct == Decimal("0.0115")
    assert prof.direct_rule_execution is True

    # 1. Dip-rebound candle series: steady decline (EMA5 < EMA13, RSI ~36) followed by a green bounce candle
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    step_ms = 60_000
    count = 45
    first_open = (now_ms - 5_000 - step_ms + 1) - (count - 1) * step_ms
    dip_candles: list[Candle] = []
    px = Decimal("100.00")
    for i in range(count):
        if i < count - 1:
            delta = Decimal("-0.18") if (i % 3 != 0) else Decimal("0.08")
        else:
            delta = Decimal("0.22")  # Fresh green bounce candle from the dip!
        open_p = px
        close_p = px + delta
        open_ms = first_open + i * step_ms
        dip_candles.append(
            Candle(
                open_time_ms=open_ms,
                open=open_p,
                high=max(open_p, close_p) + Decimal("0.10"),
                low=min(open_p, close_p) - Decimal("0.10"),
                close=close_p,
                volume=Decimal("20.0"),
                close_time_ms=open_ms + step_ms - 1,
                quote_volume=Decimal("20.0") * close_p,
                trade_count=100,
                is_closed=True,
            )
        )
        px = close_p

    sig_dip, _ = evaluate_technical_strategy(
        dip_candles, symbol="SOLUSDT", interval="1m", params=prof.strategy
    )
    assert sig_dip.action == "AL"

    # 2. Full engine cycle: Buy -> Pullback (-0.20%, SAT ignored!) -> Rise (+0.65%, sold in net profit!) -> Rebuy!
    class SteppedMarket(FakeMarketClient):
        def __init__(self) -> None:
            super().__init__()
            self.mode = "DIP_BUY"
            self.candle_shift = 0

        def fetch_verified_market_bundle(
            self, symbol: str, interval: str, allow_stablecoin_conversion: bool = False
        ) -> MarketBundle:
            now_dt = datetime.now(timezone.utc)
            nms = int(now_dt.timestamp() * 1000) + (self.candle_shift * 60_000)
            f_open = (nms - 5_000 - step_ms + 1) - (count - 1) * step_ms
            c_list: list[Candle] = []
            p = Decimal("100.00")
            for idx in range(count):
                if self.mode in ("DIP_BUY", "REBUY"):
                    d = Decimal("-0.15") if (idx % 3 != 0) else Decimal("0.07")
                    if idx == count - 1:
                        d = Decimal("0.25")
                elif self.mode == "TINY_PULLBACK":
                    d = Decimal("-0.20") if idx >= count - 3 else Decimal("0.05")
                else:  # PROFIT_SPIKE
                    d = Decimal("0.10")
                op = p
                cp = p + d
                om = f_open + idx * step_ms
                c_list.append(
                    Candle(
                        open_time_ms=om,
                        open=op,
                        high=max(op, cp) + Decimal("0.10"),
                        low=min(op, cp) - Decimal("0.10"),
                        close=cp,
                        volume=Decimal("25.0"),
                        close_time_ms=om + step_ms - 1,
                        quote_volume=Decimal("25.0") * cp,
                        trade_count=120,
                        is_closed=True,
                    )
                )
                p = cp
            bid_px = self.price_by_symbol.get(symbol, c_list[-1].close)
            ob = OrderBookSnapshot(
                symbol=symbol,
                bids=[OrderBookLevel(price=bid_px, quantity=Decimal("500"))],
                asks=[OrderBookLevel(price=bid_px + Decimal("0.01"), quantity=Decimal("500"))],
                fetched_at_utc=now_dt,
                source_url="https://api.binance.com/api/v3/depth",
            )
            return MarketBundle(
                symbol=symbol,
                interval=interval,
                filters=self._filters(symbol),
                closed_candles=c_list,
                order_book=ob,
                server_time_ms=nms,
                fetched_at_utc=now_dt,
                source_label="MockStep",
                wake_gap_detected=False,
            )

    sm = SteppedMarket()
    cfg = AppConfig(sqlite_db_path=tmp_path / "cycle.db", initial_balance_usdc=Decimal("86.00"), quote_asset="USDT")
    eng = TradingSystemEngine(cfg, market_client=sm)
    settings = prof.apply_to_risk_settings(eng.default_risk_settings(), leverage=1)
    settings = settings.__class__(**{**settings.__dict__, "extra_slippage_bps": Decimal("0")})

    # Step A: Buys on dip rebound at ~96.95
    sm.price_by_symbol["SOLUSDT"] = Decimal("100.00")
    s1 = eng.run_analysis_step(
        symbol="SOLUSDT", interval="1m", settings=settings, ai_enabled_by_user=False, strategy_params=prof.strategy
    )
    assert s1.executed_order is not None and s1.executed_order.side == "BUY"
    entry_px = eng.storage.get_open_positions()[0].entry_price

    # Step B: Tiny -0.20% pullback triggers technical SAT, but engine holds (no lossy sale!)
    sm.candle_shift = 1
    sm.mode = "TINY_PULLBACK"
    sm.price_by_symbol["SOLUSDT"] = (entry_px * Decimal("0.998")).quantize(Decimal("0.01"))
    s2 = eng.run_analysis_step(
        symbol="SOLUSDT", interval="1m", settings=settings, ai_enabled_by_user=False, strategy_params=prof.strategy
    )
    assert s2.executed_order is None
    assert s2.final_action == "BEKLE"
    assert "zararına erken satış yapılmadı" in s2.block_or_info_message
    assert len(eng.storage.get_open_positions()) == 1

    # Step C: Price rises +1.30% above entry -> immediately sells in net profit!
    sm.candle_shift = 2
    sm.mode = "PROFIT_SPIKE"
    sm.price_by_symbol["SOLUSDT"] = (entry_px * Decimal("1.013")).quantize(Decimal("0.01"))
    s3 = eng.run_analysis_step(
        symbol="SOLUSDT", interval="1m", settings=settings, ai_enabled_by_user=False, strategy_params=prof.strategy
    )
    assert s3.executed_order is not None and s3.executed_order.side == "SELL"
    assert s3.executed_order.realized_pnl_usdc > Decimal("0")
    assert len(eng.storage.get_open_positions()) == 0
    assert eng.storage.get_cash_balance() > Decimal("86.00")

    # Step D: Next dip rebound immediately buys again continuously!
    sm.candle_shift = 3
    sm.mode = "REBUY"
    sm.price_by_symbol["SOLUSDT"] = Decimal("100.00")
    s4 = eng.run_analysis_step(
        symbol="SOLUSDT", interval="1m", settings=settings, ai_enabled_by_user=False, strategy_params=prof.strategy
    )
    assert s4.executed_order is not None and s4.executed_order.side == "BUY"


def test_live_spot_substep_dust_never_blocks_rebuy_and_recycles_cleanly(tmp_path: Path):
    """When Binance deducts 0.1% fee in base coin leaving sub-step dust after SELL, next BUY recycles it."""
    import httpx
    from binance_ai_bot.config import AppConfig
    from binance_ai_bot.domain import RiskSettings, is_substep_dust_position
    from binance_ai_bot.live_executor import BinanceLiveSpotExecutor
    from binance_ai_bot.market_data import parse_symbol_filters
    from binance_ai_bot.storage import SQLiteStorage

    xrp_filters = parse_symbol_filters(
        {
            "symbol": "XRPUSDT",
            "baseAsset": "XRP",
            "quoteAsset": "USDT",
            "status": "TRADING",
            "isSpotTradingAllowed": True,
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.1", "minQty": "0.1", "maxQty": "100000"},
                {"filterType": "NOTIONAL", "minNotional": "5.0"},
            ],
        }
    )

    wallet = {"USDT": "85.00", "XRP": "0.00"}
    next_fill: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/time"):
            return httpx.Response(200, json={"serverTime": int(datetime.now(timezone.utc).timestamp() * 1000)})
        if req.url.path.endswith("/account"):
            return httpx.Response(
                200,
                json={
                    "canTrade": True,
                    "permissions": ["SPOT"],
                    "balances": [
                        {"asset": "USDT", "free": wallet["USDT"]},
                        {"asset": "XRP", "free": wallet["XRP"]},
                    ],
                },
            )
        if req.url.path.endswith("/order/test"):
            return httpx.Response(200, json={})
        if req.url.path.endswith("/order") and req.method == "POST":
            return httpx.Response(200, json=next_fill["resp"])
        raise AssertionError(f"Unexpected {req.method} {req.url.path}")

    cfg = AppConfig(
        sqlite_db_path=tmp_path / "dust_recycle.db",
        enable_live_trading=True,
        binance_api_key="k",
        binance_api_secret="s",
    )
    st = SQLiteStorage(cfg.sqlite_db_path)
    ex = BinanceLiveSpotExecutor(cfg, st, httpx.Client(transport=httpx.MockTransport(handler)))
    rset = RiskSettings(max_open_positions=1, max_position_size_usdc=Decimal("10.00"), stop_loss_pct=Decimal("0.035"), take_profit_pct=Decimal("0.0055"))

    # Trade 1 BUY: 7.1 XRP bought, 0.0071 XRP fee -> net 7.0929 XRP
    next_fill["resp"] = {
        "orderId": 1001,
        "status": "FILLED",
        "executedQty": "7.1",
        "cummulativeQuoteQty": "9.94",
        "fills": [{"qty": "7.1", "price": "1.40", "commission": "0.0071", "commissionAsset": "XRP"}],
    }
    wallet["XRP"] = "7.0929"
    wallet["USDT"] = "75.06"
    b1 = ex.execute_live_market_buy(
        filters=xrp_filters,
        approved_qty=Decimal("7.1"),
        approved_notional=Decimal("10.00"),
        signal_key="XRP-BUY-1",
        strategy_source="TEST",
        settings=rset,
        ui_live_confirmed=True,
    )
    assert b1.status == "FILLED"
    pos1 = st.get_open_positions(track="LIVE_SPOT")[0]
    assert pos1.quantity == Decimal("7.0929")

    # Trade 1 SELL at profit: sells 7.0 XRP, leaving 0.0929 XRP sub-step dust
    next_fill["resp"] = {
        "orderId": 1002,
        "status": "FILLED",
        "executedQty": "7.0",
        "cummulativeQuoteQty": "9.94",
        "fills": [{"qty": "7.0", "price": "1.42", "commission": "0.00994", "commissionAsset": "USDT"}],
    }
    s1 = ex.execute_live_market_sell(
        position=pos1,
        filters=xrp_filters,
        signal_key="XRP-SELL-1",
        strategy_source="TEST",
        close_reason="KAR_AL",
        ui_live_confirmed=True,
    )
    assert s1.status == "FILLED"
    dust_pos = st.get_open_positions(track="LIVE_SPOT")[0]
    assert dust_pos.quantity == Decimal("0.0929")
    assert is_substep_dust_position(dust_pos, xrp_filters) is True

    # Trade 2 BUY (even with max_open_positions=1!): dust does NOT block rebuy and merges cleanly!
    wallet["XRP"] = "0.0929"
    wallet["USDT"] = "84.99"
    next_fill["resp"] = {
        "orderId": 1003,
        "status": "FILLED",
        "executedQty": "7.1",
        "cummulativeQuoteQty": "9.94",
        "fills": [{"qty": "7.1", "price": "1.40", "commission": "0.0071", "commissionAsset": "XRP"}],
    }
    b2 = ex.execute_live_market_buy(
        filters=xrp_filters,
        approved_qty=Decimal("7.1"),
        approved_notional=Decimal("10.00"),
        signal_key="XRP-BUY-2",
        strategy_source="TEST",
        settings=rset,
        ui_live_confirmed=True,
    )
    assert b2.status == "FILLED"
    merged_open = st.get_open_positions(track="LIVE_SPOT")
    assert len(merged_open) == 1
    assert merged_open[0].quantity == Decimal("7.1858")  # 0.0929 + 7.0929 -> next sell will sell full 7.1 XRP!


def test_engine_automatically_values_bnb_commission_and_unblocks_next_live_buy(tmp_path: Path):
    """Binance BNB komisyon indirimiyle kesilen komisyonlar otomatik değerlenir ve sonraki alımı engellemez."""
    import httpx
    from binance_ai_bot.config import AppConfig
    from binance_ai_bot.domain import RiskSettings
    from binance_ai_bot.live_executor import BinanceLiveSpotExecutor
    from binance_ai_bot.market_data import parse_symbol_filters
    from binance_ai_bot.storage import SQLiteStorage

    eth_filters = parse_symbol_filters(
        {
            "symbol": "ETHUSDT",
            "baseAsset": "ETH",
            "quoteAsset": "USDT",
            "status": "TRADING",
            "isSpotTradingAllowed": True,
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.0001", "maxQty": "1000"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
            ],
        }
    )

    cfg = AppConfig(
        sqlite_db_path=tmp_path / "bnb_fee.db",
        enable_live_trading=True,
        binance_api_key="test-key",
        binance_api_secret="test-secret",
        quote_asset="USDT",
    )
    st = SQLiteStorage(cfg.sqlite_db_path)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/time"):
            return httpx.Response(200, json={"serverTime": int(datetime.now(timezone.utc).timestamp() * 1000)})
        if req.url.path.endswith("/account"):
            return httpx.Response(
                200,
                json={
                    "canTrade": True,
                    "permissions": ["SPOT"],
                    "balances": [{"asset": "USDT", "free": "80.00"}, {"asset": "BNB", "free": "0.01"}],
                },
            )
        if req.url.path.endswith("/order/test"):
            return httpx.Response(200, json={})
        if req.url.path.endswith("/order") and req.method == "POST":
            return httpx.Response(
                200,
                json={
                    "orderId": 2001,
                    "status": "FILLED",
                    "executedQty": "0.0048",
                    "cummulativeQuoteQty": "9.60",
                    "fills": [{"qty": "0.0048", "price": "2000.00", "commission": "0.00001100", "commissionAsset": "BNB"}],
                },
            )
        raise AssertionError(req.url.path)

    ex = BinanceLiveSpotExecutor(cfg, st, httpx.Client(transport=httpx.MockTransport(handler)))
    rset = RiskSettings(max_open_positions=5, max_position_size_usdc=Decimal("10.00"), leverage=1)
    ex.execute_live_market_buy(
        filters=eth_filters,
        approved_qty=Decimal("0.0048"),
        approved_notional=Decimal("10.00"),
        signal_key="ETH-BNB-1",
        strategy_source="TEST",
        settings=rset,
        ui_live_confirmed=True,
    )
    assert st.get_state("live_unvalued_commissions") not in {"", "{}"}

    class _BnbBookMarket:
        def fetch_order_book(self, symbol: str, depth_limit: int = 5) -> OrderBookSnapshot:
            return OrderBookSnapshot(
                symbol=symbol,
                bids=[OrderBookLevel(Decimal("650.00"), Decimal("10"))],
                asks=[OrderBookLevel(Decimal("650.10"), Decimal("10"))],
                fetched_at_utc=datetime.now(timezone.utc),
                source_url="https://api.binance.com/api/v3/depth",
            )

    eng = TradingSystemEngine(cfg, storage=st, market_client=_BnbBookMarket())
    eng._resolve_live_unvalued_commissions("USDT", rset)
    assert st.get_state("live_unvalued_commissions") == "{}"
    pos = st.get_open_positions(track="LIVE_SPOT")[0]
    assert pos.entry_fee_usdc == Decimal("0.00001100") * Decimal("650.00")


def test_radar_refresh_and_scan_symbol_watchlist(tmp_path: Path):
    app, sync, market = _make_app(tmp_path)
    market.discover_spot_pairs = lambda **kw: {  # type: ignore[method-assign]
        sym: market._filters(sym) for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "SHELLUSDT")
    }
    market.fetch_market_radar = lambda **kw: (  # type: ignore[method-assign]
        [{"symbol": "SHELLUSDT", "score": 0.72, "quote_volume_24h": 12_000_000, "range_pct_24h": 14.0, "change_pct_24h": 18.5, "spread_pct": 0.05}],
        [{"symbol": "SHELLUSDT", "last_price": 0.55, "change_pct_24h": 18.5, "quote_volume_24h": 12_000_000, "range_pct_24h": 14.0, "spread_pct": 0.05, "tag": "GUCLU_TREND"}],
    )
    headers = {"Authorization": "Bearer owner-token", "Origin": "https://tradecebrail.web.app"}

    r_rad = app.handle("POST", "/api/radar-refresh", headers, b"{}")
    assert r_rad.status == 200
    assert r_rad.body["dashboard"]["gainers"][0]["symbol"] == "SHELLUSDT"
    assert r_rad.body["dashboard"]["universe"][0]["symbol"] == "SHELLUSDT"

    r_add = app.handle("POST", "/api/settings", headers, json.dumps({"add_scan_symbol": "SHELL"}).encode())
    assert r_add.status == 200
    assert "SHELLUSDT" in r_add.body["dashboard"]["scan_symbols"]

    r_bad = app.handle("POST", "/api/settings", headers, json.dumps({"add_scan_symbol": "NOPEUSDT"}).encode())
    assert r_bad.status == 400

    r_rem = app.handle("POST", "/api/settings", headers, json.dumps({"remove_scan_symbol": "SHELLUSDT"}).encode())
    assert r_rem.status == 200
    assert "SHELLUSDT" not in r_rem.body["dashboard"]["scan_symbols"]
    assert sync.persist_calls >= 3

