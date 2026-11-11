"""Live execution regressions using only an in-memory HTTP transport and temp DBs."""

from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path

import httpx
import pytest

from binance_ai_bot.config import AppConfig
from binance_ai_bot.domain import Position, RiskSettings
from binance_ai_bot.live_executor import BinanceLiveSpotExecutor, LiveExecutionError
from binance_ai_bot.market_data import parse_symbol_filters
from binance_ai_bot.risk import compute_open_position_unrealized_net_pnl
from binance_ai_bot.storage import SQLiteStorage


def filters():
    return parse_symbol_filters({
        "symbol": "TESTUSDT", "baseAsset": "TEST", "quoteAsset": "USDT",
        "status": "TRADING", "isSpotTradingAllowed": True,
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "10000"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
        ],
    })


def fill(qty="10", commission="0", asset="USDT", price="1"):
    return {"qty": qty, "price": price, "commission": commission, "commissionAsset": asset}


def response(qty="10", quote="10", status="FILLED", fills=None):
    return {"orderId": 123, "status": status, "executedQty": qty,
            "cummulativeQuoteQty": quote, "fills": fills if fills is not None else [fill(qty)]}


class Exchange:
    def __init__(self):
        self.result = response()
        self.free_quote = "86"
        self.free_base = "10"
        self.lose_post_response = False
        self.query_missing = False
        self.posts = []
        self.account_reads = 0
        self.my_trades = None
        self.before_post = None
        self.after_test = None
        self.after_post = None
        self.account_failure = False
        self.post_error = None
        self.order_queries = 0

    def handle(self, request):
        path = request.url.path
        if path.endswith("/time"):
            return httpx.Response(200, json={"serverTime": int(datetime.now(timezone.utc).timestamp() * 1000)})
        if path.endswith("/account"):
            self.account_reads += 1
            if self.account_failure:
                return httpx.Response(503, json={"code": -1001, "msg": "Disconnected"})
            return httpx.Response(200, json={"canTrade": True, "permissions": ["SPOT"], "balances": [
                {"asset": "USDT", "free": self.free_quote},
                {"asset": "TEST", "free": self.free_base},
            ]})
        if path.endswith("/order/test"):
            if self.after_test:
                self.after_test()
            return httpx.Response(200, json={})
        if path.endswith("/myTrades"):
            return httpx.Response(200, json=self.my_trades if self.my_trades is not None else self.result["fills"])
        if path.endswith("/order") and request.method == "GET":
            self.order_queries += 1
            if self.query_missing:
                return httpx.Response(400, json={"code": -2013, "msg": "Order does not exist."})
            return httpx.Response(200, json={k: v for k, v in self.result.items() if k != "fills"})
        if path.endswith("/order") and request.method == "POST":
            if self.before_post:
                self.before_post(request)
            self.posts.append(dict(request.url.params))
            if self.after_post:
                self.after_post()
            if self.post_error:
                status, code, message = self.post_error
                return httpx.Response(status, json={"code": code, "msg": message})
            if self.lose_post_response:
                raise httpx.ReadTimeout("Lost response after exchange acceptance", request=request)
            return httpx.Response(200, json=self.result)
        raise AssertionError(f"Unexpected mocked request: {request.method} {path}")


def setup(tmp_path: Path):
    config = AppConfig(sqlite_db_path=tmp_path / "live.db", enable_live_trading=True,
                       binance_api_key="test-key", binance_api_secret="test-secret")
    storage = SQLiteStorage(config.sqlite_db_path)
    exchange = Exchange()
    client = httpx.Client(transport=httpx.MockTransport(exchange.handle))
    executor = BinanceLiveSpotExecutor(config, storage, client)
    return executor, storage, exchange


def buy(executor, signal="BUY-1", **kwargs):
    return executor.execute_live_market_buy(
        filters=filters(), approved_qty=D("10"), signal_key=signal,
        strategy_source="TEST", settings=RiskSettings(max_open_positions=5),
        ui_live_confirmed=True, **kwargs,
    )


def position(storage, qty="10", cost="10", entry_fee="0.10"):
    pos = Position(position_id="LIVE-POS-ORIGINAL", symbol="TESTUSDT", quantity=D(qty),
                   entry_price=D(1), cost_basis_usdc=D(cost), entry_fee_usdc=D(entry_fee),
                   opened_at_iso=datetime.now(timezone.utc).isoformat(), signal_key="ORIGINAL",
                   strategy_source="TEST", stop_loss_price=D("0.9"), take_profit_price=D("1.1"))
    storage.insert_position(pos, track="LIVE_SPOT")
    return pos


def sell(executor, pos, signal="SELL-1"):
    return executor.execute_live_market_sell(
        position=pos, filters=filters(), signal_key=signal, strategy_source="TEST",
        close_reason="TEST_EXIT", ui_live_confirmed=True,
    )


def test_lost_response_is_queried_without_another_post_and_intent_precedes_post(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.lose_post_response = True

    def before_post(request):
        # An independent connection sees the intent: it has already committed.
        other = SQLiteStorage(storage.db_path)
        intent = other.get_live_intent("BUY-1")
        assert intent["status"] == "SUBMITTED"
        assert intent["client_order_id"] == request.url.params["newClientOrderId"]

    exchange.before_post = before_post
    result = buy(executor)
    assert result.status == "FILLED"
    assert len(exchange.posts) == 1
    assert storage.get_live_intent("BUY-1")["status"] == "APPLIED"
    assert len(storage.get_open_positions(track="LIVE_SPOT")) == 1
    with pytest.raises(LiveExecutionError, match="Mükerrer"):
        buy(executor)
    assert len(exchange.posts) == 1


def test_unknown_order_survives_restart_blocks_buys_and_reconciles_once(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.lose_post_response = True
    exchange.query_missing = True
    with pytest.raises(LiveExecutionError, match="belirsiz"):
        buy(executor)
    assert storage.get_live_intent("BUY-1")["status"] == "UNKNOWN"
    with pytest.raises(LiveExecutionError, match="doğrulanmamış"):
        buy(executor, signal="BUY-2")
    assert len(exchange.posts) == 1
    restarted_storage = SQLiteStorage(storage.db_path)
    restarted = BinanceLiveSpotExecutor(executor.config, restarted_storage, executor._external_client)
    exchange.query_missing = False
    restarted.reconcile_pending_orders(ui_live_confirmed=True)
    restarted.reconcile_pending_orders(ui_live_confirmed=True)
    assert restarted_storage.list_pending_live_intents() == []
    assert len(restarted_storage.get_open_positions(track="LIVE_SPOT")) == 1
    assert len(restarted_storage.get_orders(track="LIVE_SPOT")) == 1
    assert len(exchange.posts) == 1


def test_partial_sell_allocates_cost_and_tracks_remaining_inventory(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    pos = position(storage)
    exchange.result = response("4", "4", "EXPIRED", [fill("4", "0.004")])
    result = sell(executor, pos)
    remaining = storage.get_position(pos.position_id)
    assert remaining.status == "OPEN"
    assert remaining.quantity == D("6")
    assert remaining.cost_basis_usdc == D("6")
    assert remaining.entry_fee_usdc == D("0.06")
    assert result.status == "PARTIALLY_FILLED"
    assert result.realized_pnl_usdc == D("-0.044")
    realized, _ = storage.get_daily_realized_pnl_and_entry_fees(storage.istanbul_date_str(), track="LIVE_SPOT")
    assert realized == result.realized_pnl_usdc


def test_active_partial_sell_reconciliation_only_applies_new_fills(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    pos = position(storage, entry_fee="0")
    exchange.result = response("4", "4", "PARTIALLY_FILLED", [fill("4")])
    sell(executor, pos)
    assert storage.get_position(pos.position_id).quantity == D("6")
    executor.reconcile_pending_orders(ui_live_confirmed=True)
    assert storage.get_position(pos.position_id).quantity == D("6")
    assert len(storage.get_orders(track="LIVE_SPOT")) == 1
    exchange.result = response("10", "10", "FILLED", [fill("4"), fill("6")])
    executor.reconcile_pending_orders(ui_live_confirmed=True)
    assert storage.get_position(pos.position_id).status == "CLOSED"
    assert len(storage.get_orders(track="LIVE_SPOT")) == 2
    assert storage.list_pending_live_intents() == []


def test_zero_fill_never_closes_position_or_records_a_realized_loss(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    pos = position(storage)
    exchange.result = response("0", "0", "EXPIRED", [])
    result = sell(executor, pos)
    assert result.status == "EXPIRED_UNFILLED"
    assert result.realized_pnl_usdc == 0
    remaining = storage.get_position(pos.position_id)
    assert remaining.status == "OPEN" and remaining.quantity == D(10)


def test_sell_rereads_current_position_and_rejects_stale_closed_request(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    stale = position(storage)
    storage.update_position_remaining(stale.position_id, quantity=D(6), cost_basis_usdc=D(6), entry_fee_usdc=D("0.06"))
    exchange.result = response("6", "6", "FILLED", [fill("6")])
    sell(executor, stale)
    assert exchange.posts[0]["quantity"] == "6"
    with pytest.raises(LiveExecutionError, match="artık açık değil"):
        sell(executor, stale, signal="SELL-2")
    assert len(exchange.posts) == 1


def test_sell_keeps_sub_step_dust_open(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    pos = position(storage, qty="10.0005", cost="10.0005", entry_fee="0")
    exchange.free_base = "10.0005"
    sell(executor, pos)
    remaining = storage.get_position(pos.position_id)
    assert remaining.status == "OPEN" and remaining.quantity == D("0.0005")
    with pytest.raises(LiveExecutionError, match="minimumun altında"):
        sell(executor, remaining, signal="SELL-2")
    assert len(exchange.posts) == 1


def test_base_asset_buy_fee_reduces_inventory_without_double_expense(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.result = response(fills=[fill("10", "0.01", "TEST")])
    result = buy(executor)
    pos = storage.get_open_positions(track="LIVE_SPOT")[0]
    assert pos.quantity == D("9.99") and pos.cost_basis_usdc == D(10)
    assert pos.entry_fee_usdc == 0
    assert result.fee_usdc == D("0.01")
    assert compute_open_position_unrealized_net_pnl(pos, D(1), D(0))[1] == D("-0.01")


def test_third_asset_fee_is_persisted_as_unknown_and_blocks_new_buys(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.result = response(fills=[fill("10", "0.001", "BNB")])
    result = buy(executor)
    assert storage.get_live_intent("BUY-1")["unknown_commissions"] == {"BNB": "0.001"}
    assert "K/Z geçicidir" in result.explanation
    with pytest.raises(LiveExecutionError, match="değerlenmeden"):
        buy(executor, signal="BUY-2")
    assert len(exchange.posts) == 1


def test_live_spot_rejects_leverage_before_any_http_request(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    with pytest.raises(LiveExecutionError, match="kaldıraçsız"):
        executor.execute_live_market_buy(filters=filters(), approved_qty=D(10), signal_key="LEV",
            strategy_source="TEST", settings=RiskSettings(leverage=2), ui_live_confirmed=True)
    assert exchange.account_reads == 0 and exchange.posts == []
    assert storage.get_live_intent("LEV") is None


def test_inventory_and_signal_updates_roll_back_together_and_recover(tmp_path, monkeypatch):
    executor, storage, exchange = setup(tmp_path)
    original = storage.record_order

    def fail_record(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(storage, "record_order", fail_record)
    with pytest.raises(RuntimeError, match="disk failure"):
        buy(executor)
    assert storage.get_open_positions(track="LIVE_SPOT") == []
    assert not storage.has_processed_signal("BUY-1", track="LIVE_SPOT")
    assert storage.list_pending_live_intents()
    monkeypatch.setattr(storage, "record_order", original)
    executor.reconcile_pending_orders(ui_live_confirmed=True)
    assert len(storage.get_open_positions(track="LIVE_SPOT")) == 1
    assert len(exchange.posts) == 1


def test_missing_trade_history_leaves_order_pending_without_inventing_fees(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.lose_post_response = True
    exchange.my_trades = []
    with pytest.raises(LiveExecutionError, match="belirsiz"):
        buy(executor)
    assert storage.get_open_positions(track="LIVE_SPOT") == []
    assert storage.list_pending_live_intents()


def test_quote_budget_caps_spending_and_accepts_more_quantity_when_price_falls(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.result = response("20", "10", "FILLED", [fill("20", price="0.5")])
    buy(executor, approved_notional=D("50"))
    params = exchange.posts[0]
    assert "quantity" not in params and params["quoteOrderQty"] == "10"
    assert storage.get_open_positions(track="LIVE_SPOT")[0].quantity == D(20)
    assert exchange.account_reads == 2


def test_quote_budget_reserves_fee_and_rejects_insufficient_fresh_balance(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.free_quote = "4.999"
    with pytest.raises(LiveExecutionError, match="minimum emir tutarı"):
        buy(executor, approved_notional=D(10))
    assert exchange.posts == []
    assert storage.get_live_intent("BUY-1") is None


def test_active_partial_buy_adds_only_incremental_inventory(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.result = response("4", "4", "PARTIALLY_FILLED", [fill("4", "0.004", "TEST")])
    buy(executor)
    assert storage.get_open_positions(track="LIVE_SPOT")[0].quantity == D("3.996")
    exchange.result = response("10", "10", "FILLED", [fill("4", "0.004", "TEST"), fill("6", "0.006", "TEST")])
    executor.reconcile_pending_orders(ui_live_confirmed=True)
    executor.reconcile_pending_orders(ui_live_confirmed=True)
    pos = storage.get_open_positions(track="LIVE_SPOT")[0]
    assert pos.quantity == D("9.99") and pos.cost_basis_usdc == D(10)
    assert pos.entry_fee_usdc == 0
    assert storage.count_daily_new_positions(storage.istanbul_date_str(), track="LIVE_SPOT") == 1


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_stop_during_order_test_blocks_real_post_for_buys_and_sells(tmp_path, side):
    executor, storage, exchange = setup(tmp_path)
    stopped = {"value": False}
    exchange.after_test = lambda: stopped.update(value=True)
    stop_requested = lambda: stopped["value"]
    with pytest.raises(LiveExecutionError, match="Durdurma isteği"):
        if side == "BUY":
            buy(executor, stop_requested=stop_requested)
        else:
            executor.execute_live_market_sell(
                position=position(storage), filters=filters(), signal_key="SELL-1",
                strategy_source="TEST", close_reason="TEST", ui_live_confirmed=True,
                stop_requested=stop_requested,
            )
    assert exchange.posts == []
    assert storage.get_live_intent(f"{side}-1")["status"] == "REJECTED"
    if side == "SELL":
        assert storage.get_open_positions(track="LIVE_SPOT")[0].quantity == D(10)


def test_stop_does_not_misclassify_previous_unknown_order_as_rejected(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.lose_post_response = True
    exchange.query_missing = True
    with pytest.raises(LiveExecutionError):
        buy(executor)
    with pytest.raises(LiveExecutionError, match="doğrulanmamış"):
        buy(executor, signal="BUY-2", stop_requested=lambda: True)
    assert storage.get_live_intent("BUY-1")["status"] == "UNKNOWN"
    assert len(exchange.posts) == 1


def test_restart_balance_refresh_before_reconciliation_does_not_double_debit_cash(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.lose_post_response = True
    exchange.query_missing = True
    exchange.after_post = lambda: setattr(exchange, "free_quote", "76")
    with pytest.raises(LiveExecutionError):
        buy(executor)
    # Runner startup reads a wallet that already includes the accepted order.
    executor.fetch_live_spot_balances(ui_live_confirmed=True, quote_asset="USDT")
    assert storage.get_cash_balance(track="LIVE_SPOT") == D(76)
    exchange.query_missing = False
    executor.reconcile_pending_orders(ui_live_confirmed=True)
    assert storage.get_cash_balance(track="LIVE_SPOT") == D(76)
    assert storage.get_state("live_balance_refresh_required") == "0"


def test_partial_sell_prefetched_wallet_does_not_double_credit_cash(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    pos = position(storage, entry_fee="0")
    exchange.result = response("4", "4", "PARTIALLY_FILLED", [fill("4")])
    exchange.after_post = lambda: setattr(exchange, "free_quote", "90")
    sell(executor, pos)
    assert storage.get_cash_balance(track="LIVE_SPOT") == D(90)
    # The next account snapshot includes the final six units before ledger replay.
    exchange.free_quote = "96"
    executor.fetch_live_spot_balances(ui_live_confirmed=True, quote_asset="USDT")
    exchange.result = response("10", "10", "FILLED", [fill("4"), fill("6")])
    executor.reconcile_pending_orders(ui_live_confirmed=True)
    assert storage.get_cash_balance(track="LIVE_SPOT") == D(96)


def test_post_fill_balance_read_failure_preserves_success_and_blocks_unfunded_submission(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.after_post = lambda: setattr(exchange, "account_failure", True)
    result = buy(executor)
    assert result.status == "FILLED"
    assert storage.get_live_intent("BUY-1")["status"] == "APPLIED"
    assert storage.get_state("live_balance_refresh_required") == "1"
    other_filters = parse_symbol_filters({
        "symbol": "OTHERUSDT", "baseAsset": "OTHER", "quoteAsset": "USDT",
        "status": "TRADING", "isSpotTradingAllowed": True, "filters": [],
    })
    with pytest.raises(LiveExecutionError):
        executor.execute_live_market_buy(filters=other_filters, approved_qty=D(10), signal_key="BUY-2",
            strategy_source="TEST", settings=RiskSettings(max_open_positions=5), ui_live_confirmed=True)
    assert len(exchange.posts) == 1
    exchange.account_failure = False
    executor.fetch_live_spot_balances(ui_live_confirmed=True, quote_asset="USDT")
    assert storage.get_state("live_balance_refresh_required") == "0"


@pytest.mark.parametrize("field,value", [("symbol", "OTHERUSDT"), ("clientOrderId", "someone-elses-order")])
def test_response_identity_mismatch_never_creates_inventory(tmp_path, field, value):
    executor, storage, exchange = setup(tmp_path)
    exchange.result[field] = value
    with pytest.raises(LiveExecutionError, match="doğrulanamadı"):
        buy(executor)
    assert storage.get_open_positions(track="LIVE_SPOT") == []
    assert storage.get_live_intent("BUY-1")["status"] == "UNKNOWN"


@pytest.mark.parametrize("status,code,message", [
    (503, -1001, "Disconnected"), (400, -1000, "Unknown error"),
    (400, -1006, "Unexpected response"), (400, -1007, "Execution status unknown"),
    (408, -9999, "Request timeout"), (400, -2010, "Duplicate order sent."),
])
def test_ambiguous_api_errors_query_existing_order_instead_of_marking_rejected(tmp_path, status, code, message):
    executor, storage, exchange = setup(tmp_path)
    exchange.post_error = (status, code, message)
    result = buy(executor)
    assert result.status == "FILLED"
    assert len(exchange.posts) == 1 and exchange.order_queries == 1
    assert storage.get_live_intent("BUY-1")["status"] == "APPLIED"


def test_definitive_api_rejection_does_not_create_inventory_or_retry(tmp_path):
    executor, storage, exchange = setup(tmp_path)
    exchange.post_error = (400, -2010, "Account has insufficient balance for requested action.")
    with pytest.raises(LiveExecutionError):
        buy(executor)
    assert storage.get_live_intent("BUY-1")["status"] == "REJECTED"
    assert storage.get_open_positions(track="LIVE_SPOT") == []
    assert exchange.order_queries == 0 and len(exchange.posts) == 1
