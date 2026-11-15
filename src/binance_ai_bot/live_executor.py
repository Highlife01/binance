"""
Çift Kilitli Gerçek Binance Global Spot Emir Yürütücüsü (`BinanceLiveSpotExecutor`).

ÖNEMLİ GÜVENLİK MİMARİSİ:
- Varsayılan olarak KAPALIDIR (`ENABLE_LIVE_TRADING=false`).
- Sanal Deneme (`MAIN`) veritabanı kayıtlarıyla ASLA karışmaz (`LIVE_SPOT` izinde tutulur).
- Yalnızca kaldıraçsız Spot `BUY` ve `SELL` emirlerini destekler.
- Para çekme, marjin, borçlanma veya vadeli (futures) uç noktaları kodda mevcut değildir
  ve `security.assert_live_spot_request` tarafından koşulsuz engellenir.
- Her gerçek emirden önce:
  1. `GET /api/v3/time` ile sunucu saati senkronize edilir,
  2. `GET /api/v3/account` ile gerçek kullanılabilir USDC doğrulanır,
  3. `POST /api/v3/order/test` ile Binance ön-doğrulama testi yapılır,
  4. Yalnızca test başarılıysa `POST /api/v3/order` (`newOrderRespType=FULL`) gönderilir.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Mapping
from urllib.parse import urlencode

import httpx

from .binance_http import (
    BinanceTransportState,
    execute_signed_binance_request,
    sign_hmac_sha256,
)
from .config import AppConfig
from .domain import (
    Position,
    RiskSettings,
    SimulatedOrderResult,
    SymbolFilters,
    is_effective_dust_position,
    quantize_down,
)
from .market_data import _friendly_network_error
from .security import (
    STABLECOIN_OR_FIAT_ASSETS,
    LiveOrderBlockedError,
    assert_live_spot_request,
    redact_secrets,
)
from .storage import SQLiteStorage


class LiveExecutionError(RuntimeError):
    """Canlı Spot bağlantı, yetki veya emir doğrulama hatası."""

    def __init__(
        self,
        message: str,
        *,
        api_code: int | None = None,
        ambiguous: bool = False,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.api_code = api_code
        self.ambiguous = ambiguous
        self.status_code = status_code
        self.retryable = retryable


def _format_decimal_clean(val: Decimal) -> str:
    """Binance API parametreleri için bilimsel gösterim içermeyen düz ondalık metin üretir."""
    normalized = val.normalize()
    text = format(normalized, "f")
    return text if text else "0"


class BinanceLiveSpotExecutor:
    """HMAC-SHA256 imzalı, ön-testli (`order/test`) resmi Binance Spot v3 yürütücüsü."""

    def __init__(
        self,
        config: AppConfig,
        storage: SQLiteStorage,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self._external_client = http_client
        self._transport_state = BinanceTransportState()

    def _sign_params(self, params: dict[str, Any]) -> str:
        return sign_hmac_sha256(self.config.binance_api_secret, params)

    def _request_signed(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        ui_live_confirmed: bool,
    ) -> Any:
        if not self.config.has_binance_live_keys:
            raise LiveOrderBlockedError(
                "Canlı Spot işlem için .env dosyasında BINANCE_API_KEY ve BINANCE_API_SECRET tanımlı olmalıdır."
            )

        return execute_signed_binance_request(
            state=self._transport_state,
            base_url=self.config.binance_live_base_url,
            method=method,
            path=path,
            api_key=self.config.binance_api_key,
            api_secret=self.config.binance_api_secret,
            recv_window_ms=self.config.binance_recv_window_ms,
            timeout_sec=10.0,
            params=params,
            external_client=self._external_client,
            security_assert_fn=lambda m, u: assert_live_spot_request(
                m,
                u,
                config_live_enabled=self.config.enable_live_trading,
                ui_live_confirmed=ui_live_confirmed,
            ),
            error_cls=LiveExecutionError,
            error_prefix="Binance Canlı Spot API hatası",
            ambiguous_post_paths=frozenset({"/api/v3/order"}),
            body_snippet_limit=240,
        )

    def fetch_live_spot_balances(
        self, *, ui_live_confirmed: bool, quote_asset: str = "USDC"
    ) -> dict[str, Decimal]:
        """Gerçek Binance Spot hesabındaki serbest (`free`) varlık bakiyelerini okur ve SQLite'ta günceller."""
        with self.storage.execution_lock():
            return self._fetch_live_spot_balances(ui_live_confirmed=ui_live_confirmed, quote_asset=quote_asset)

    def fetch_live_spot_account_snapshot(
        self, *, ui_live_confirmed: bool, quote_asset: str = "USDC"
    ) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
        """Gerçek Binance Spot hesabındaki (`free`, `free + locked`) varlık bakiyelerini birlikte döner."""
        with self.storage.execution_lock():
            return self._fetch_live_spot_account_snapshot(
                ui_live_confirmed=ui_live_confirmed, quote_asset=quote_asset
            )

    def _fetch_live_spot_balances(
        self, *, ui_live_confirmed: bool, quote_asset: str,
    ) -> dict[str, Decimal]:
        free_balances, _ = self._fetch_live_spot_account_snapshot(
            ui_live_confirmed=ui_live_confirmed, quote_asset=quote_asset
        )
        return free_balances

    def _fetch_live_spot_account_snapshot(
        self, *, ui_live_confirmed: bool, quote_asset: str,
    ) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
        q_norm = quote_asset.strip().upper() if quote_asset.strip().upper() in {"USDT", "USDC"} else "USDC"
        self.storage.ensure_quote_asset("LIVE_SPOT", q_norm)
        earn_key = f"LD{q_norm}"

        data = self._request_signed("GET", "/api/v3/account", ui_live_confirmed=ui_live_confirmed)
        if not bool(data.get("canTrade", False)):
            raise LiveExecutionError(
                "Binance hesabınızda veya API anahtarınızda Spot alım-satım yetkisi ('canTrade') kapalı görünüyor."
            )

        balances_free: dict[str, Decimal] = {}
        balances_total: dict[str, Decimal] = {}
        for item in data.get("balances", []):
            asset = str(item.get("asset", "")).upper()
            free_amt = Decimal(str(item.get("free", "0")))
            locked_amt = Decimal(str(item.get("locked", "0")))
            tot_amt = free_amt + locked_amt
            if free_amt > 0 or asset in {"USDC", "USDT", "LDUSDC", "LDUSDT"}:
                balances_free[asset] = free_amt
            if tot_amt > 0 or asset in {"USDC", "USDT", "LDUSDC", "LDUSDT"}:
                balances_total[asset] = tot_amt

        quote_free = balances_free.get(q_norm, Decimal("0"))
        earn_free = balances_free.get(earn_key, Decimal("0"))
        self.storage.set_state("live_spot_cash_usdc", f"{quote_free:.8f}")
        self.storage.set_state("live_earn_ldusdc", f"{earn_free:.8f}")
        self.storage.set_state("live_balance_refresh_required", "0")

        perms = {str(p).upper() for p in data.get("permissions", [])}
        spot_perm = ("SPOT" in perms) or any(p.startswith("TRD_GRP_") for p in perms)
        self.storage.set_state("live_spot_permission_ok", "1" if spot_perm else "0")
        return balances_free, balances_total

    @staticmethod
    def _client_order_id(signal_key: str, side: str) -> str:
        # The same persisted signal must never turn into a second exchange order.
        digest = hashlib.sha256(f"LIVE_SPOT:{side}:{signal_key}".encode()).hexdigest()[:27]
        return f"BSP{side[0]}{digest}"

    def _new_intent(
        self, *, filters: SymbolFilters, quantity: Decimal, signal_key: str,
        side: str, strategy_source: str, **extra: Any,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        client_id = self._client_order_id(signal_key, side)
        return {
            "signal_key": signal_key, "client_order_id": client_id,
            "symbol": filters.symbol, "side": side, "status": "PREPARED",
            "base_asset": filters.base_asset, "quote_asset": filters.quote_asset,
            "strategy_source": strategy_source, "created_at_iso": now,
            "requested_qty": str(quantity), "applied_qty": "0", "applied_quote": "0",
            "applied_commissions": {},
            "order_params": {
                "symbol": filters.symbol, "side": side, "type": "MARKET",
                "quantity": _format_decimal_clean(quantity),
                "newClientOrderId": client_id, "newOrderRespType": "FULL",
            },
            **extra,
        }

    def reconcile_pending_orders(self, *, ui_live_confirmed: bool) -> list[SimulatedOrderResult]:
        """Recover persisted submissions without ever issuing another POST order.

        Call on startup and before an analysis cycle. An unresolved exchange response
        remains pending; new entries stay blocked until its outcome is established.
        """
        results: list[SimulatedOrderResult] = []
        with self.storage.execution_lock():
            for intent in self.storage.list_pending_live_intents():
                try:
                    result = self._reconcile_intent(intent, ui_live_confirmed=ui_live_confirmed)
                    if result is not None:
                        results.append(result)
                except (LiveExecutionError, LiveOrderBlockedError) as exc:
                    self.storage.update_live_intent(
                        intent["signal_key"], status="UNKNOWN", error=redact_secrets(str(exc)),
                    )
        return results

    def _reconcile_intent(
        self, intent: dict[str, Any], *, ui_live_confirmed: bool,
    ) -> SimulatedOrderResult | None:
        try:
            response = self._request_signed(
                "GET", "/api/v3/order",
                {"symbol": intent["symbol"], "origClientOrderId": intent["client_order_id"]},
                ui_live_confirmed=ui_live_confirmed,
            )
        except LiveExecutionError as exc:
            # PREPARED is persisted before test; SUBMITTED is committed before POST.
            # Only PREPARED can be proven never to have been sent. A temporary
            # -2013 after an uncertain POST is not permission to submit again.
            if exc.api_code == -2013 and intent["status"] == "PREPARED":
                self.storage.update_live_intent(intent["signal_key"], status="REJECTED", error=str(exc))
                return None
            raise
        return self._apply_response(intent, response, ui_live_confirmed=ui_live_confirmed)

    def _submit_intent(
        self, intent: dict[str, Any], *, ui_live_confirmed: bool,
        stop_requested: Callable[[], bool] | None = None,
    ) -> SimulatedOrderResult:
        key = intent["signal_key"]
        # Each operation commits independently: neither this nor the POST belongs
        # inside the inventory transaction. A crash can always be reconciled.
        self.storage.save_live_intent(key, intent)
        def check_stop() -> None:
            if stop_requested is not None and stop_requested():
                # This helper is used only before THIS fresh intent's first POST.
                # Never reject an old uncertain submission merely because stopped.
                self.storage.update_live_intent(key, status="REJECTED", error="Kullanıcı durdurdu; gerçek emir gönderilmedi.")
                raise LiveExecutionError("Durdurma isteği alındı; gerçek emir gönderilmedi.")

        check_stop()
        try:
            self._request_signed(
                "POST", "/api/v3/order/test", intent["order_params"],
                ui_live_confirmed=ui_live_confirmed,
            )
        except (LiveExecutionError, LiveOrderBlockedError) as exc:
            self.storage.update_live_intent(key, status="REJECTED", error=redact_secrets(str(exc)))
            raise
        self.storage.update_live_intent(key, status="SUBMITTED")
        check_stop()
        try:
            response = self._request_signed(
                "POST", "/api/v3/order", intent["order_params"],
                ui_live_confirmed=ui_live_confirmed,
            )
        except LiveExecutionError as exc:
            if not exc.ambiguous:
                self.storage.update_live_intent(key, status="REJECTED", error=redact_secrets(str(exc)))
                raise
            self.storage.update_live_intent(key, status="UNKNOWN", error=redact_secrets(str(exc)))
            current = self.storage.get_live_intent(key)
            try:
                result = self._reconcile_intent(current, ui_live_confirmed=ui_live_confirmed)
                if result is not None:
                    return result
            except (LiveExecutionError, LiveOrderBlockedError):
                pass
            raise LiveExecutionError(
                "Emir sonucu belirsiz. Kimliği kalıcı olarak kaydedildi; "
                "borsada doğrulanana kadar yeni alımlar durduruldu.", ambiguous=True,
            ) from exc
        return self._apply_response(intent, response, ui_live_confirmed=ui_live_confirmed)

    def _complete_fills(
        self, intent: dict[str, Any], response: dict[str, Any], *, ui_live_confirmed: bool,
    ) -> list[dict[str, Any]]:
        executed = Decimal(str(response.get("executedQty", "0")))
        if executed == 0:
            return []
        fills = response.get("fills")
        if fills is None:
            if not response.get("orderId"):
                raise LiveExecutionError("Gerçekleşen emrin kimliği yok; muhasebe uzlaştırma bekliyor.", ambiguous=True)
            fills = self._request_signed(
                "GET", "/api/v3/myTrades",
                {"symbol": intent["symbol"], "orderId": response["orderId"], "limit": 1000},
                ui_live_confirmed=ui_live_confirmed,
            )
        if not isinstance(fills, list) or sum((Decimal(str(f.get("qty", "0"))) for f in fills), Decimal(0)) != executed:
            raise LiveExecutionError(
                "Emir dolumları ve komisyonları henüz tam doğrulanamadı; yeni alımlar durduruldu.", ambiguous=True,
            )
        return fills

    def _apply_response(
        self, intent: dict[str, Any], response: dict[str, Any], *, ui_live_confirmed: bool,
    ) -> SimulatedOrderResult:
        key = intent["signal_key"]
        self.storage.update_live_intent(key, response=response)
        try:
            status = str(response.get("status", "")).upper()
            terminal = status in {"FILLED", "CANCELED", "REJECTED", "EXPIRED", "EXPIRED_IN_MATCH"}
            if not terminal and status not in {"NEW", "PARTIALLY_FILLED", "PENDING_NEW", "PENDING_CANCEL"}:
                raise ValueError("Bilinmeyen emir durumu")
            executed = Decimal(str(response.get("executedQty", "0")))
            quote = Decimal(str(response.get("cummulativeQuoteQty", "0")))
            requested = Decimal(intent["requested_qty"])
            if response.get("symbol") is not None and response["symbol"] != intent["symbol"]:
                raise ValueError("Yanıt sembolü kayıtlı emirle uyuşmuyor")
            if response.get("clientOrderId") is not None and response["clientOrderId"] != intent["client_order_id"]:
                raise ValueError("Yanıt müşteri emir kimliği kayıtlı emirle uyuşmuyor")
            if not executed.is_finite() or not quote.is_finite() or executed < 0 or quote < 0:
                raise ValueError("Geçersiz gerçekleşen miktar/tutar")
            if "quote_budget" in intent:
                if quote > Decimal(intent["quote_budget"]):
                    raise ValueError("Gerçekleşen tutar onaylı harcama tavanını aşıyor")
            elif executed > requested:
                raise ValueError("Gerçekleşen miktar istenen miktarı aşıyor")
            if executed > 0 and quote <= 0:
                raise ValueError("Gerçekleşen tutar eksik")
            fills = self._complete_fills(intent, response, ui_live_confirmed=ui_live_confirmed)
            commissions: dict[str, Decimal] = {}
            base_fee_value = Decimal(0)
            for fill in fills:
                asset = str(fill.get("commissionAsset", "")).upper()
                amount = Decimal(str(fill.get("commission", "0")))
                fill_price = Decimal(str(fill["price"]))
                fill_qty = Decimal(str(fill["qty"]))
                if not amount.is_finite() or amount < 0 or (amount > 0 and not asset):
                    raise ValueError("Geçersiz komisyon")
                if not fill_price.is_finite() or fill_price <= 0 or not fill_qty.is_finite() or fill_qty <= 0:
                    raise ValueError("Geçersiz dolum fiyatı veya miktarı")
                commissions[asset] = commissions.get(asset, Decimal(0)) + amount
                if asset == intent["base_asset"]:
                    base_fee_value += amount * fill_price
        except (ValueError, KeyError, TypeError, AttributeError, ArithmeticError, LiveExecutionError) as exc:
            self.storage.update_live_intent(key, status="UNKNOWN", error=redact_secrets(str(exc)))
            if isinstance(exc, LiveExecutionError):
                raise
            raise LiveExecutionError("Emir yanıtı doğrulanamadı; uzlaştırma gerekiyor.", ambiguous=True) from exc

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        date = self.storage.istanbul_date_str(now)
        with self.storage.atomic():
            current = self.storage.get_live_intent(key)
            previous_qty = Decimal(current.get("applied_qty", "0"))
            previous_quote = Decimal(current.get("applied_quote", "0"))
            previous_fees = {a: Decimal(v) for a, v in current.get("applied_commissions", {}).items()}
            delta_qty, delta_quote = executed - previous_qty, quote - previous_quote
            delta_fees = {
                a: commissions.get(a, Decimal(0)) - previous_fees.get(a, Decimal(0))
                for a in commissions.keys() | previous_fees.keys()
            }
            if delta_qty < 0 or delta_quote < 0 or any(v < 0 for v in delta_fees.values()):
                raise LiveExecutionError("Borsa dolumları geriye gitti; otomatik işlem durduruldu.", ambiguous=True)
            base_fee = delta_fees.get(intent["base_asset"], Decimal(0))
            quote_fee = delta_fees.get(intent["quote_asset"], Decimal(0))
            unknown = {a: str(v) for a, v in commissions.items() if a not in {intent["base_asset"], intent["quote_asset"]} and v > 0}
            if unknown:
                raw = self.storage.get_state("live_unvalued_commissions", "{}") or "{}"
                unvalued = json.loads(raw)
                unvalued[intent["client_order_id"]] = unknown
                self.storage.set_state("live_unvalued_commissions", json.dumps(unvalued, sort_keys=True))
            avg = delta_quote / delta_qty if delta_qty > 0 else Decimal(0)
            pnl = Decimal(0)
            net_qty = delta_qty
            if delta_qty > 0:
                if intent["side"] == "BUY":
                    net_qty = delta_qty - base_fee
                    if net_qty <= 0:
                        raise LiveExecutionError("Alış komisyonu net miktarı aşıyor; uzlaştırma gerekiyor.", ambiguous=True)
                    pos = self.storage.get_position(intent["position_id"], track="LIVE_SPOT")
                    if pos is None:
                        entry = quote / executed
                        self.storage.insert_position(
                            Position(
                                position_id=intent["position_id"], symbol=intent["symbol"],
                                quantity=net_qty, entry_price=entry, cost_basis_usdc=delta_quote,
                                # Base-asset fees already reduced inventory. Adding them here
                                # as a cash expense would count the very same fee twice.
                                entry_fee_usdc=quote_fee, opened_at_iso=intent["created_at_iso"],
                                signal_key=key, strategy_source=intent["strategy_source"],
                                stop_loss_price=entry * (1 - Decimal(intent["stop_loss_pct"])) if Decimal(intent["stop_loss_pct"]) > 0 else None,
                                take_profit_price=entry * (1 + Decimal(intent["take_profit_pct"])) if Decimal(intent["take_profit_pct"]) > 0 else None,
                            ), track="LIVE_SPOT",
                        )
                    elif pos.status == "OPEN":
                        entry = quote / executed
                        recycle_dust = bool(intent.get("recycle_dust"))
                        sl_pct = Decimal(str(intent.get("stop_loss_pct", "0")))
                        tp_pct = Decimal(str(intent.get("take_profit_pct", "0")))
                        self.storage.update_position_remaining(
                            pos.position_id, quantity=pos.quantity + net_qty,
                            cost_basis_usdc=pos.cost_basis_usdc + delta_quote,
                            entry_fee_usdc=pos.entry_fee_usdc + quote_fee, entry_price=entry,
                            stop_loss_price=entry * (1 - sl_pct) if (recycle_dust and sl_pct > 0) else None,
                            take_profit_price=entry * (1 + tp_pct) if (recycle_dust and tp_pct > 0) else None,
                            update_exit_targets=recycle_dust,
                        )
                    else:
                        raise LiveExecutionError("Devam eden alışın yerel pozisyonu kapalı; uzlaştırma gerekiyor.")
                else:
                    pos = self.storage.get_position(intent["position_id"], track="LIVE_SPOT")
                    debit_qty = delta_qty + base_fee
                    if pos is None or pos.status != "OPEN" or debit_qty > pos.quantity:
                        raise LiveExecutionError("Satış miktarı yerel açık pozisyonla uyuşmuyor; uzlaştırma gerekiyor.")
                    ratio = debit_qty / pos.quantity
                    sold_cost, sold_entry_fee = pos.cost_basis_usdc * ratio, pos.entry_fee_usdc * ratio
                    pnl = delta_quote - quote_fee - sold_cost - sold_entry_fee
                    if debit_qty == pos.quantity:
                        self.storage.update_position_closed(
                            pos.position_id, closed_at_iso=now_iso, closed_istanbul_date=date,
                            exit_price=avg, exit_fee_usdc=quote_fee, realized_pnl_usdc=pnl,
                            close_reason=intent["close_reason"],
                        )
                    else:
                        # Keep even a sub-step residual visible: dust still belongs to the account.
                        self.storage.update_position_remaining(
                            pos.position_id, quantity=pos.quantity - debit_qty,
                            cost_basis_usdc=pos.cost_basis_usdc - sold_cost,
                            entry_fee_usdc=pos.entry_fee_usdc - sold_entry_fee,
                        )
                        suffix = hashlib.sha256(f"{key}:{executed}".encode()).hexdigest()[:16]
                        self.storage.insert_position(
                            replace(pos, position_id=f"LIVE-CLOSED-{suffix}", quantity=debit_qty,
                                    cost_basis_usdc=sold_cost, entry_fee_usdc=sold_entry_fee),
                            track="LIVE_SPOT",
                        )
                        self.storage.update_position_closed(
                            f"LIVE-CLOSED-{suffix}", closed_at_iso=now_iso, closed_istanbul_date=date,
                            exit_price=avg, exit_fee_usdc=quote_fee, realized_pnl_usdc=pnl,
                            close_reason=intent["close_reason"],
                        )
                self.storage.mark_signal_processed(key, str(response.get("orderId", intent["client_order_id"])), now_iso, track="LIVE_SPOT")
                # Account reads may already include this fill (especially after a
                # restart). Applying a local cash delta would count it twice.
                self.storage.set_state("live_balance_refresh_required", "1")
            mapped = "PARTIALLY_FILLED" if executed > 0 and (not terminal or (status != "FILLED" and executed < requested)) else ("FILLED" if executed > 0 else "EXPIRED_UNFILLED")
            fee_value = quote_fee + base_fee_value - Decimal(current.get("applied_base_fee_value", "0"))
            warning = " Üçüncü varlık komisyonunun değeri bilinmiyor; K/Z geçicidir, yeni alımlar durduruldu." if unknown else ""
            result = SimulatedOrderResult(
                order_id=f"LIVE-{response.get('orderId', intent['client_order_id'])}-{_format_decimal_clean(executed)}",
                symbol=intent["symbol"], side=intent["side"], order_type="MARKET", status=mapped,
                requested_qty=requested, filled_qty=net_qty, avg_fill_price=avg,
                gross_quote_usdc=delta_quote, fee_usdc=fee_value, slippage_bps=Decimal(0),
                signal_key=key, strategy_source=intent["strategy_source"], created_at_iso=now_iso,
                realized_pnl_usdc=pnl,
                explanation=f"GERÇEK SPOT {intent['side']} ({status}): {net_qty} {intent['base_asset']} @ {avg:.8f} {intent['quote_asset']}." + warning,
            )
            if delta_qty > 0 or (terminal and not current.get("zero_result_recorded") and previous_qty == 0):
                self.storage.record_order(result, date, track="LIVE_SPOT")
            self.storage.update_live_intent(
                key, status="APPLIED" if terminal else "SUBMITTED", response=response,
                applied_qty=str(executed), applied_quote=str(quote),
                applied_commissions={a: str(v) for a, v in commissions.items()},
                applied_base_fee_value=str(base_fee_value), unknown_commissions=unknown,
                zero_result_recorded=terminal and executed == 0,
            )
        if delta_qty > 0:
            try:
                self.fetch_live_spot_balances(ui_live_confirmed=ui_live_confirmed, quote_asset=intent["quote_asset"])
            except Exception:
                # Inventory and the exchange result are already committed. Keep
                # the refresh-required flag; every new BUY must successfully read
                # the authoritative account before it can submit another order.
                result = replace(result, explanation=result.explanation + " Bakiye yenilenemedi; yeni alım öncesi hesap tekrar doğrulanacak.")
        return result

    def _check_signal_available(self, signal_key: str) -> None:
        if self.storage.get_live_intent(signal_key) is not None or self.storage.has_processed_signal(signal_key, track="LIVE_SPOT"):
            raise LiveExecutionError(f"Mükerrer canlı sinyal engellendi: {signal_key}")

    def execute_live_market_buy(
        self, *, filters: SymbolFilters, approved_qty: Decimal, signal_key: str,
        strategy_source: str, settings: RiskSettings, ui_live_confirmed: bool,
        approved_notional: Decimal | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> SimulatedOrderResult:
        """Persist one order identity before sending, then apply its fills atomically."""
        with self.storage.execution_lock():
            if settings.leverage != 1:
                raise LiveExecutionError("Canlı Spot yalnızca kaldıraçsız (1x) çalışır.")
            self.storage.ensure_quote_asset("LIVE_SPOT", filters.quote_asset)
            self.reconcile_pending_orders(ui_live_confirmed=ui_live_confirmed)
            self._check_signal_available(signal_key)
            if self.storage.list_pending_live_intents():
                raise LiveExecutionError("Sonucu doğrulanmamış canlı emir var; yeni alımlar durduruldu.")
            if self.storage.get_state("live_unvalued_commissions", "{}") not in {"", "{}", None}:
                raise LiveExecutionError("Üçüncü varlık komisyonları değerlenmeden yeni canlı alım yapılamaz.")
            est_price = (
                (approved_notional / approved_qty)
                if (approved_notional is not None and approved_notional > 0 and approved_qty > 0)
                else None
            )
            opened = self.storage.get_open_positions(track="LIVE_SPOT")
            active_opened = [
                p
                for p in opened
                if not is_effective_dust_position(
                    p,
                    filters if p.symbol == filters.symbol else None,
                    mark_price=est_price if p.symbol == filters.symbol else None,
                )
            ]
            dust_same_sym = next(
                (
                    p
                    for p in opened
                    if p.symbol == filters.symbol
                    and is_effective_dust_position(p, filters, mark_price=est_price)
                ),
                None,
            )
            if any(p.symbol == filters.symbol for p in active_opened) or len(active_opened) >= settings.max_open_positions:
                raise LiveExecutionError("Açık pozisyon veya aynı coin sınırı nedeniyle yeni alım engellendi.")
            qty = quantize_down(approved_qty, filters.market_step_size)
            if not qty.is_finite() or qty <= 0 or qty < max(filters.min_qty, filters.market_min_qty):
                raise LiveExecutionError("Canlı alış miktarı geçerli minimum miktarı karşılamıyor.")
            balances = self.fetch_live_spot_balances(ui_live_confirmed=ui_live_confirmed, quote_asset=filters.quote_asset)
            free_quote = balances.get(filters.quote_asset, Decimal(0))
            if free_quote <= 0:
                raise LiveExecutionError("Gerçek serbest Spot bakiyesi yetersiz; alım yapılmadı.")
            pos_id = (
                dust_same_sym.position_id
                if dust_same_sym is not None
                else f"LIVE-POS-{hashlib.sha256(signal_key.encode()).hexdigest()[:20]}"
            )
            intent = self._new_intent(
                filters=filters, quantity=qty, signal_key=signal_key, side="BUY",
                strategy_source=strategy_source,
                position_id=pos_id,
                recycle_dust=(dust_same_sym is not None),
                stop_loss_pct=str(settings.stop_loss_pct), take_profit_pct=str(settings.take_profit_pct),
            )
            cfg_min_buy = getattr(settings, "min_buy_notional_usdc", Decimal("0"))
            min_buy_floor = max(filters.min_notional, cfg_min_buy)
            if approved_notional is None and cfg_min_buy > Decimal("0"):
                raise LiveExecutionError(
                    f"Minimum {min_buy_floor:.2f} {filters.quote_asset} alt sınırı aktifken onaylı alım bütçesi zorunludur."
                )
            if approved_notional is not None:
                if not approved_notional.is_finite() or approved_notional <= 0:
                    raise LiveExecutionError("Onaylı canlı alış bütçesi geçersiz.")
                budget = quantize_down(
                    min(approved_notional, settings.max_position_size_usdc,
                        free_quote / (1 + settings.estimated_fee_rate)), Decimal("0.00000001"),
                )
                if budget < min_buy_floor:
                    raise LiveExecutionError(
                        f"Güncel bakiye ve komisyon payıyla minimum emir tutarı ({min_buy_floor:.2f} {filters.quote_asset}) karşılanamıyor."
                    )
                if cfg_min_buy > Decimal("0") and est_price is not None and est_price > Decimal("0"):
                    est_exec_qty = quantize_down(budget / est_price, filters.market_step_size)
                    est_exec_quote = est_exec_qty * est_price
                    if est_exec_quote < min_buy_floor:
                        raise LiveExecutionError(
                            f"Lot adımı yuvarlaması sonrası gerçekleşecek tutar (~{est_exec_quote:.2f} {filters.quote_asset}), "
                            f"minimum alım alt sınırının ({min_buy_floor:.2f} {filters.quote_asset}) altında kaldığı için alım yapılmadı."
                        )
                intent["quote_budget"] = str(budget)
                del intent["order_params"]["quantity"]
                intent["order_params"]["quoteOrderQty"] = _format_decimal_clean(budget)
            return self._submit_intent(intent, ui_live_confirmed=ui_live_confirmed, stop_requested=stop_requested)

    def execute_live_market_sell(
        self, *, position: Position, filters: SymbolFilters, signal_key: str,
        strategy_source: str, close_reason: str, ui_live_confirmed: bool,
        estimated_bid: Decimal | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> SimulatedOrderResult:
        """Sell only currently tracked inventory, retaining every unfilled residual."""
        with self.storage.execution_lock():
            self.storage.ensure_quote_asset("LIVE_SPOT", filters.quote_asset)
            self.reconcile_pending_orders(ui_live_confirmed=ui_live_confirmed)
            self._check_signal_available(signal_key)
            if any(i["symbol"] == filters.symbol for i in self.storage.list_pending_live_intents()):
                raise LiveExecutionError("Bu coin için sonucu belirsiz emir var; satış önce uzlaştırılmalı.")
            current = self.storage.get_position(position.position_id, track="LIVE_SPOT")
            if current is None or current.status != "OPEN" or current.symbol != filters.symbol:
                raise LiveExecutionError("Pozisyon artık açık değil; eski satış isteği engellendi.")
            if current.leverage != 1:
                raise LiveExecutionError("Kaldıraçlı pozisyon Spot yürütücüsünde satılamaz.")
            balances = self.fetch_live_spot_balances(ui_live_confirmed=ui_live_confirmed, quote_asset=filters.quote_asset)
            qty = quantize_down(min(current.quantity, balances.get(filters.base_asset, Decimal(0))), filters.market_step_size)
            if qty <= 0 or qty < max(filters.min_qty, filters.market_min_qty):
                raise LiveExecutionError(f"Satılabilir {filters.base_asset} minimumun altında; kalan miktar açık kayıtta korunuyor.")
            if estimated_bid is not None and estimated_bid > Decimal(0):
                est_notional = qty * estimated_bid
                if est_notional < filters.min_notional:
                    raise LiveExecutionError(
                        f"Satılabilir {qty} {filters.base_asset} tutarı (~{est_notional:.2f} {filters.quote_asset}) "
                        f"borsa minimum emir alt sınırının ({filters.min_notional:.2f} {filters.quote_asset}) altında; "
                        "kalan miktar açık kayıtta korunuyor."
                    )
            intent = self._new_intent(
                filters=filters, quantity=qty, signal_key=signal_key, side="SELL",
                strategy_source=strategy_source, position_id=current.position_id, close_reason=close_reason,
            )
            return self._submit_intent(intent, ui_live_confirmed=ui_live_confirmed, stop_requested=stop_requested)

    @staticmethod
    def _infer_base_asset(symbol: str, spot_filters_map: Mapping[str, SymbolFilters] | None = None) -> str:
        if spot_filters_map and symbol in spot_filters_map:
            return spot_filters_map[symbol].base_asset
        for quote_sfx in ("USDT", "USDC", "FDUSD"):
            if symbol.endswith(quote_sfx) and len(symbol) > len(quote_sfx):
                return symbol[: -len(quote_sfx)]
        return symbol

    def reconcile_external_spot_positions(
        self,
        *,
        spot_filters_map: Mapping[str, SymbolFilters] | None = None,
        bid_by_symbol: Mapping[str, Decimal] | None = None,
        ui_live_confirmed: bool,
        close_reason: str = "HARICI_MANUEL_SATIS",
        total_balances: Mapping[str, Decimal] | None = None,
        inspect_my_trades: bool = True,
        quote_asset: str = "USDT",
    ) -> list[dict[str, Any]]:
        """
        Binance uygulaması/web üzerinden manuel satılan veya BNB'ye dönüştürülen varlıkların
        SQLite `LIVE_SPOT` açık pozisyon kayıtlarını gerçek cüzdan bakiyesiyle (`free + locked`)
        uzlaştırır; hayalet açık pozisyonları ve çifte bakiye sayımını anında temizler.
        """
        with self.storage.execution_lock():
            self.reconcile_pending_orders(ui_live_confirmed=ui_live_confirmed)
            open_positions = self.storage.get_open_positions(track="LIVE_SPOT")
            if not open_positions:
                return []

            pending_syms = {
                str(i.get("symbol", "")).upper()
                for i in self.storage.list_pending_live_intents()
            }
            if total_balances is None:
                _, total_balances = self._fetch_live_spot_account_snapshot(
                    ui_live_confirmed=ui_live_confirmed, quote_asset=quote_asset
                )

            known_venue_ids: set[str] = set()
            if inspect_my_trades:
                for row in self.storage.get_orders(track="LIVE_SPOT", limit=500):
                    oid = str(row.get("order_id", ""))
                    if oid.startswith("LIVE-"):
                        parts = oid.split("-")
                        if len(parts) >= 2 and parts[1]:
                            known_venue_ids.add(parts[1])

            ordered_positions = sorted(open_positions, key=lambda p: str(p.opened_at_iso or ""))
            wallet_avail: dict[str, Decimal] = {}
            reconciled: list[dict[str, Any]] = []

            for pos in ordered_positions:
                if pos.symbol in pending_syms or pos.quantity <= Decimal("0"):
                    continue
                base_asset = self._infer_base_asset(pos.symbol, spot_filters_map)
                avail = wallet_avail.setdefault(
                    base_asset, Decimal(str(total_balances.get(base_asset, Decimal("0"))))
                )
                tol = max(Decimal("0.00000001"), pos.quantity * Decimal("0.005"))
                if avail >= pos.quantity - tol:
                    wallet_avail[base_asset] = max(Decimal("0"), avail - pos.quantity)
                    continue

                kept_qty = max(Decimal("0"), min(pos.quantity, avail))
                wallet_avail[base_asset] = max(Decimal("0"), avail - kept_qty)
                sold_qty = pos.quantity - kept_qty
                if sold_qty <= Decimal("0"):
                    continue

                exit_px = Decimal("0")
                ext_fee_quote = Decimal("0")
                matched_venue_trade = False

                if inspect_my_trades:
                    try:
                        opened_ms = int(datetime.fromisoformat(pos.opened_at_iso).timestamp() * 1000)
                    except Exception:
                        opened_ms = 0
                    try:
                        raw_trades = self._request_signed(
                            "GET",
                            "/api/v3/myTrades",
                            {"symbol": pos.symbol, "limit": 30},
                            ui_live_confirmed=ui_live_confirmed,
                        )
                    except Exception:
                        raw_trades = []
                    if isinstance(raw_trades, list):
                        qty_sum = Decimal("0")
                        quote_sum = Decimal("0")
                        qfee_sum = Decimal("0")
                        for tr in raw_trades:
                            if not isinstance(tr, dict) or bool(tr.get("isBuyer", True)):
                                continue
                            tr_oid = str(tr.get("orderId", ""))
                            if tr_oid and tr_oid in known_venue_ids:
                                continue
                            try:
                                tr_time = int(tr.get("time", 0))
                                tr_qty = Decimal(str(tr.get("qty", "0")))
                                tr_quote = Decimal(str(tr.get("quoteQty", "0")))
                                if tr_quote <= Decimal("0") and tr_qty > Decimal("0"):
                                    tr_quote = tr_qty * Decimal(str(tr.get("price", "0")))
                                tr_comm = Decimal(str(tr.get("commission", "0")))
                                tr_comm_asset = str(tr.get("commissionAsset", "")).upper()
                            except Exception:
                                continue
                            if tr_time < max(0, opened_ms - 5000) or tr_qty <= Decimal("0") or tr_quote <= Decimal("0"):
                                continue
                            qty_sum += tr_qty
                            quote_sum += tr_quote
                            if tr_comm > Decimal("0") and tr_comm_asset in {"USDT", "USDC"}:
                                qfee_sum += tr_comm
                        if qty_sum > Decimal("0") and quote_sum > Decimal("0"):
                            exit_px = quote_sum / qty_sum
                            scale = min(Decimal("1"), sold_qty / qty_sum)
                            ext_fee_quote = qfee_sum * scale
                            matched_venue_trade = True

                if exit_px <= Decimal("0"):
                    bid_px = Decimal(str((bid_by_symbol or {}).get(pos.symbol, Decimal("0"))))
                    exit_px = bid_px if bid_px > Decimal("0") else pos.entry_price
                    if close_reason == "BINANCE_TOZ_OPTIMIZASYONU" and bid_px > Decimal("0"):
                        ext_fee_quote = (sold_qty * exit_px) * Decimal("0.02")

                now = datetime.now(timezone.utc)
                now_iso = now.isoformat()
                ist_date = self.storage.istanbul_date_str(now)

                with self.storage.atomic():
                    cur_pos = self.storage.get_position(pos.position_id, track="LIVE_SPOT")
                    if cur_pos is None or cur_pos.status != "OPEN":
                        continue
                    fully_closed = kept_qty <= Decimal("0.00000001")
                    ratio = Decimal("1") if fully_closed else (sold_qty / cur_pos.quantity)
                    sold_cost = cur_pos.cost_basis_usdc if fully_closed else (cur_pos.cost_basis_usdc * ratio)
                    sold_entry_fee = cur_pos.entry_fee_usdc if fully_closed else (cur_pos.entry_fee_usdc * ratio)
                    gross_quote = sold_qty * exit_px
                    pnl = gross_quote - ext_fee_quote - sold_cost - sold_entry_fee

                    if fully_closed:
                        self.storage.update_position_closed(
                            cur_pos.position_id,
                            closed_at_iso=now_iso,
                            closed_istanbul_date=ist_date,
                            exit_price=exit_px,
                            exit_fee_usdc=ext_fee_quote,
                            realized_pnl_usdc=pnl,
                            close_reason=close_reason,
                        )
                    else:
                        self.storage.update_position_remaining(
                            cur_pos.position_id,
                            quantity=kept_qty,
                            cost_basis_usdc=max(Decimal("0"), cur_pos.cost_basis_usdc - sold_cost),
                            entry_fee_usdc=max(Decimal("0"), cur_pos.entry_fee_usdc - sold_entry_fee),
                        )
                        suffix = hashlib.sha256(
                            f"EXT:{cur_pos.position_id}:{now_iso}:{sold_qty}".encode()
                        ).hexdigest()[:16]
                        closed_id = f"LIVE-EXT-{suffix}"
                        self.storage.insert_position(
                            replace(
                                cur_pos,
                                position_id=closed_id,
                                quantity=sold_qty,
                                cost_basis_usdc=sold_cost,
                                entry_fee_usdc=sold_entry_fee,
                            ),
                            track="LIVE_SPOT",
                        )
                        self.storage.update_position_closed(
                            closed_id,
                            closed_at_iso=now_iso,
                            closed_istanbul_date=ist_date,
                            exit_price=exit_px,
                            exit_fee_usdc=ext_fee_quote,
                            realized_pnl_usdc=pnl,
                            close_reason=close_reason,
                        )

                    prefix = "DUST" if close_reason == "BINANCE_TOZ_OPTIMIZASYONU" else "EXT"
                    ord_hash = hashlib.sha256(
                        f"{close_reason}:{cur_pos.position_id}:{now_iso}:{sold_qty}".encode()
                    ).hexdigest()[:12]
                    label_tr = (
                        "BİNANCE TOZ→BNB DÖNÜŞÜMÜ"
                        if close_reason == "BINANCE_TOZ_OPTIMIZASYONU"
                        else (
                            "HARİCİ MANUEL SATIŞ (BORSA EŞLEŞTİ)"
                            if matched_venue_trade
                            else "HARİCİ BAKİYE UZLAŞTIRMASI"
                        )
                    )
                    self.storage.record_order(
                        SimulatedOrderResult(
                            order_id=f"{prefix}-{ord_hash}",
                            symbol=cur_pos.symbol,
                            side="SELL",
                            order_type="MARKET",
                            status="FILLED",
                            requested_qty=sold_qty,
                            filled_qty=sold_qty,
                            avg_fill_price=exit_px,
                            gross_quote_usdc=gross_quote,
                            fee_usdc=ext_fee_quote,
                            slippage_bps=Decimal("0"),
                            signal_key=f"{prefix}:{cur_pos.position_id}:{ord_hash}",
                            strategy_source=close_reason,
                            created_at_iso=now_iso,
                            realized_pnl_usdc=pnl,
                            explanation=f"{label_tr}: {_format_decimal_clean(sold_qty)} {base_asset} @ ~{exit_px:.6g} USDT.",
                        ),
                        ist_date,
                        track="LIVE_SPOT",
                    )
                    reconciled.append(
                        {
                            "symbol": cur_pos.symbol,
                            "asset": base_asset,
                            "sold_qty": _format_decimal_clean(sold_qty),
                            "kept_qty": _format_decimal_clean(Decimal("0") if fully_closed else kept_qty),
                            "exit_price": f"{exit_px:.8g}",
                            "gross_usdt": f"{gross_quote:.2f}",
                            "realized_pnl_usdc": f"{pnl:.4f}",
                            "close_reason": close_reason,
                        }
                    )

            return reconciled

    def _liquidate_untracked_spot_balance(
        self,
        *,
        filters: SymbolFilters,
        sell_qty: Decimal,
        ui_live_confirmed: bool,
        reason_label: str = "KUCUK_COIN_OPTIMIZASYONU",
    ) -> dict[str, Any]:
        """Yerel açık pozisyonu olmayan küçük Spot bakiyesini doğrudan USDT'ye satar."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        ist_date = self.storage.istanbul_date_str(now)
        sig_key = f"UNTRACKED_SELL:{filters.symbol}:{int(now.timestamp() * 1000)}"
        client_id = self._client_order_id(sig_key, "SELL")
        order_params = {
            "symbol": filters.symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": _format_decimal_clean(sell_qty),
            "newClientOrderId": client_id,
            "newOrderRespType": "FULL",
        }
        self._request_signed("POST", "/api/v3/order/test", order_params, ui_live_confirmed=ui_live_confirmed)
        resp = self._request_signed("POST", "/api/v3/order", order_params, ui_live_confirmed=ui_live_confirmed)
        executed = Decimal(str(resp.get("executedQty", "0")))
        quote = Decimal(str(resp.get("cummulativeQuoteQty", "0")))
        avg_px = (quote / executed) if executed > 0 else Decimal("0")
        q_fee = Decimal("0")
        for f in resp.get("fills") or []:
            if isinstance(f, dict) and str(f.get("commissionAsset", "")).upper() in {"USDT", "USDC"}:
                q_fee += Decimal(str(f.get("commission", "0")))
        if executed > 0:
            self.storage.record_order(
                SimulatedOrderResult(
                    order_id=f"OPT-{resp.get('orderId', client_id)}-{_format_decimal_clean(executed)}",
                    symbol=filters.symbol,
                    side="SELL",
                    order_type="MARKET",
                    status="FILLED",
                    requested_qty=sell_qty,
                    filled_qty=executed,
                    avg_fill_price=avg_px,
                    gross_quote_usdc=quote,
                    fee_usdc=q_fee,
                    slippage_bps=Decimal("0"),
                    signal_key=sig_key,
                    strategy_source=reason_label,
                    created_at_iso=now_iso,
                    realized_pnl_usdc=Decimal("0"),
                    explanation=f"KÜÇÜK COİN SATIŞI: {_format_decimal_clean(executed)} {filters.base_asset} -> {quote:.2f} {filters.quote_asset}.",
                ),
                ist_date,
                track="LIVE_SPOT",
            )
        return {
            "symbol": filters.symbol,
            "asset": filters.base_asset,
            "sold_qty": _format_decimal_clean(executed),
            "received_usdt": f"{quote:.2f}",
        }

    def optimize_small_spot_balances(
        self,
        *,
        spot_filters_map: Mapping[str, SymbolFilters],
        bid_by_symbol: Mapping[str, Decimal],
        ui_live_confirmed: bool,
        max_small_usdt: Decimal = Decimal("9.20"),
        keep_bnb_reserve_usdt: Decimal = Decimal("2.00"),
    ) -> dict[str, Any]:
        """
        1) Manuel satılmış hayalet açık pozisyonları gerçek Binance bakiyesiyle uzlaştırır.
        2) Satılabilir küçük Spot bakiyelerini (`< max_small_usdt` ve açık maliyeti `< max_small_usdt`)
           doğrudan USDT'ye (`MARKET SELL`) çevirir (`>= 10 USDT` aktif pozisyonlara ASLA dokunmaz!).
        3) Borsa alt sınırının altındaki toz (dust) bakiyeleri resmi Binance `/sapi/v1/asset/dust`
           uç noktasıyla BNB'ye süpürür (değeri yüksekten düşüğe sıralı).
        4) BNB komisyon rezervi (`~2 USDT`) üzerinde satılabilir fazlalık oluşursa USDT'ye çevirir ve
           dönüşen tüm toz kayıtları SQLite'ta kapatır.
        """
        with self.storage.execution_lock():
            free_bal, total_bal = self._fetch_live_spot_account_snapshot(
                ui_live_confirmed=ui_live_confirmed, quote_asset="USDT"
            )
            reconciled_manual = self.reconcile_external_spot_positions(
                spot_filters_map=spot_filters_map,
                bid_by_symbol=bid_by_symbol,
                ui_live_confirmed=ui_live_confirmed,
                close_reason="HARICI_MANUEL_SATIS",
                total_balances=total_bal,
                inspect_my_trades=True,
                quote_asset="USDT",
            )

            # Aktif >= max_small_usdt pozisyonları kesin koruma kümesine al
            open_after_reconcile = self.storage.get_open_positions(track="LIVE_SPOT")
            protected_assets: set[str] = set()
            open_by_symbol: dict[str, Position] = {}
            for p in open_after_reconcile:
                b_asset = self._infer_base_asset(p.symbol, spot_filters_map)
                bid_now = bid_by_symbol.get(p.symbol, Decimal("0"))
                cur_val = p.quantity * (bid_now if bid_now > 0 else p.entry_price)
                if p.cost_basis_usdc >= max_small_usdt or cur_val >= max_small_usdt:
                    protected_assets.add(b_asset)
                else:
                    open_by_symbol.setdefault(p.symbol, p)

            sold_to_usdt: list[dict[str, Any]] = []
            sell_errors: list[str] = []

            # Aşama 1: Doğrudan USDT'ye satılabilen küçük coinleri (< max_small_usdt) nakde çevir
            for asset, free_qty in sorted(free_bal.items()):
                if (
                    free_qty <= Decimal("0")
                    or asset in STABLECOIN_OR_FIAT_ASSETS
                    or asset == "BNB"
                    or asset.startswith("LD")
                    or asset in protected_assets
                ):
                    continue
                symbol = f"{asset}USDT"
                flt = spot_filters_map.get(symbol)
                bid_px = bid_by_symbol.get(symbol, Decimal("0"))
                tot_qty = total_bal.get(asset, free_qty)
                if flt is None or not flt.is_spot_trading_allowed or flt.status != "TRADING" or bid_px <= Decimal("0"):
                    continue
                if (tot_qty * bid_px) >= max_small_usdt:
                    protected_assets.add(asset)
                    continue
                sell_qty = quantize_down(free_qty, flt.market_step_size)
                if sell_qty < max(flt.min_qty, flt.market_min_qty) or (sell_qty * bid_px) < flt.min_notional:
                    continue

                tracked_pos = open_by_symbol.get(symbol)
                try:
                    if tracked_pos is not None:
                        pos_sell_qty = quantize_down(min(tracked_pos.quantity, free_qty), flt.market_step_size)
                        if pos_sell_qty >= max(flt.min_qty, flt.market_min_qty) and (pos_sell_qty * bid_px) >= flt.min_notional:
                            ord_res = self.execute_live_market_sell(
                                position=tracked_pos,
                                filters=flt,
                                signal_key=f"OPT_SMALL_SELL:{tracked_pos.position_id}:{int(time.time() * 1000)}",
                                strategy_source="KUCUK_COIN_OPTIMIZASYON",
                                close_reason="KUCUK_COIN_OPTIMIZASYONU",
                                ui_live_confirmed=ui_live_confirmed,
                                estimated_bid=bid_px,
                            )
                            sold_to_usdt.append(
                                {
                                    "symbol": symbol,
                                    "asset": asset,
                                    "sold_qty": _format_decimal_clean(ord_res.filled_qty),
                                    "received_usdt": f"{ord_res.gross_quote_usdc:.2f}",
                                }
                            )
                            rem_free = max(Decimal("0"), free_qty - ord_res.filled_qty)
                            rem_sell = quantize_down(rem_free, flt.market_step_size)
                            if rem_sell >= max(flt.min_qty, flt.market_min_qty) and (rem_sell * bid_px) >= flt.min_notional:
                                extra_res = self._liquidate_untracked_spot_balance(
                                    filters=flt, sell_qty=rem_sell, ui_live_confirmed=ui_live_confirmed
                                )
                                sold_to_usdt.append(extra_res)
                        else:
                            extra_res = self._liquidate_untracked_spot_balance(
                                filters=flt, sell_qty=sell_qty, ui_live_confirmed=ui_live_confirmed
                            )
                            sold_to_usdt.append(extra_res)
                    else:
                        extra_res = self._liquidate_untracked_spot_balance(
                            filters=flt, sell_qty=sell_qty, ui_live_confirmed=ui_live_confirmed
                        )
                        sold_to_usdt.append(extra_res)
                except Exception as exc:
                    sell_errors.append(f"{asset}: {redact_secrets(str(exc))[:140]}")

            # Aşama 2: Kalan satılamayan küçük/toz bakiyeleri Binance Dust API ile BNB'ye dönüştür
            converted_to_bnb: list[dict[str, Any]] = []
            total_bnb_received = Decimal("0")
            dust_note = ""
            try:
                dust_info = self._request_signed(
                    "POST",
                    "/sapi/v1/asset/dust-btc",
                    {"accountType": "SPOT"},
                    ui_live_confirmed=ui_live_confirmed,
                )
                details = dust_info.get("details", []) if isinstance(dust_info, dict) else []
                eligible_items: list[tuple[Decimal, str]] = []
                for item in details:
                    if not isinstance(item, dict):
                        continue
                    d_asset = str(item.get("asset", "")).strip().upper()
                    if (
                        not d_asset
                        or d_asset in STABLECOIN_OR_FIAT_ASSETS
                        or d_asset == "BNB"
                        or d_asset.startswith("LD")
                        or d_asset in protected_assets
                    ):
                        continue
                    try:
                        to_bnb = Decimal(str(item.get("toBNB", "0")))
                        amt_free = Decimal(str(item.get("amountFree", "0")))
                    except Exception:
                        continue
                    if to_bnb <= Decimal("0") or amt_free <= Decimal("0"):
                        continue
                    sym_check = f"{d_asset}USDT"
                    bid_chk = bid_by_symbol.get(sym_check, Decimal("0"))
                    if bid_chk > Decimal("0") and (amt_free * bid_chk) >= max_small_usdt:
                        continue
                    eligible_items.append((to_bnb, d_asset))

                eligible_items.sort(key=lambda x: x[0], reverse=True)
                dust_assets = [a for _, a in eligible_items]

                for start_idx in range(0, len(dust_assets), 20):
                    batch = dust_assets[start_idx : start_idx + 20]
                    if not batch:
                        break
                    try:
                        dust_resp = self._request_signed(
                            "POST",
                            "/sapi/v1/asset/dust",
                            {"asset": batch, "accountType": "SPOT"},
                            ui_live_confirmed=ui_live_confirmed,
                        )
                        if isinstance(dust_resp, dict):
                            for tr in dust_resp.get("transferResult") or []:
                                if not isinstance(tr, dict):
                                    continue
                                f_asset = str(tr.get("fromAsset", "")).upper()
                                f_amt = Decimal(str(tr.get("amount", "0")))
                                bnb_amt = Decimal(str(tr.get("transferedAmount", "0")))
                                total_bnb_received += bnb_amt
                                converted_to_bnb.append(
                                    {
                                        "asset": f_asset,
                                        "amount": _format_decimal_clean(f_amt),
                                        "bnb_received": _format_decimal_clean(bnb_amt),
                                    }
                                )
                    except Exception as batch_exc:
                        if start_idx == 0:
                            dust_note = redact_secrets(str(batch_exc))[:200]
                        break
            except Exception as exc:
                dust_note = redact_secrets(str(exc))[:200]

            # Aşama 3: Son cüzdan durumunu oku; yüksek BNB fazlası varsa USDT'ye çevir; dönüşen tozları kapat
            free_bal_after, total_bal_after = self._fetch_live_spot_account_snapshot(
                ui_live_confirmed=ui_live_confirmed, quote_asset="USDT"
            )
            bnb_flt = spot_filters_map.get("BNBUSDT")
            bnb_bid = bid_by_symbol.get("BNBUSDT", Decimal("0"))
            bnb_free_now = free_bal_after.get("BNB", Decimal("0"))
            if (
                bnb_flt is not None
                and bnb_flt.is_spot_trading_allowed
                and bnb_bid > Decimal("0")
                and "BNB" not in protected_assets
            ):
                bnb_usd_now = bnb_free_now * bnb_bid
                excess_bnb_usd = bnb_usd_now - keep_bnb_reserve_usdt
                if excess_bnb_usd >= max(bnb_flt.min_notional, Decimal("5.20")):
                    excess_bnb_qty = quantize_down(excess_bnb_usd / bnb_bid, bnb_flt.market_step_size)
                    if excess_bnb_qty >= max(bnb_flt.min_qty, bnb_flt.market_min_qty):
                        try:
                            bnb_sale = self._liquidate_untracked_spot_balance(
                                filters=bnb_flt,
                                sell_qty=excess_bnb_qty,
                                ui_live_confirmed=ui_live_confirmed,
                                reason_label="BNB_FAZLASI_OPTIMIZASYONU",
                            )
                            sold_to_usdt.append(bnb_sale)
                            free_bal_after, total_bal_after = self._fetch_live_spot_account_snapshot(
                                ui_live_confirmed=ui_live_confirmed, quote_asset="USDT"
                            )
                        except Exception as exc:
                            sell_errors.append(f"BNB: {redact_secrets(str(exc))[:140]}")

            reconciled_dust = self.reconcile_external_spot_positions(
                spot_filters_map=spot_filters_map,
                bid_by_symbol=bid_by_symbol,
                ui_live_confirmed=ui_live_confirmed,
                close_reason="BINANCE_TOZ_OPTIMIZASYONU",
                total_balances=total_bal_after,
                inspect_my_trades=False,
                quote_asset="USDT",
            )

            new_free_usdt = free_bal_after.get("USDT", Decimal("0"))
            new_free_bnb = free_bal_after.get("BNB", Decimal("0"))
            new_bnb_usdt = new_free_bnb * bnb_bid if bnb_bid > Decimal("0") else Decimal("0")

            parts_msg: list[str] = []
            if reconciled_manual:
                syms_m = ", ".join(sorted({r["asset"] for r in reconciled_manual}))
                parts_msg.append(f"{len(reconciled_manual)} manuel satış uzlaştırıldı ({syms_m})")
            if sold_to_usdt:
                syms_s = ", ".join(f"{r['asset']} ({r['received_usdt']} USDT)" for r in sold_to_usdt)
                parts_msg.append(f"{len(sold_to_usdt)} küçük coin doğrudan USDT'ye satıldı: {syms_s}")
            if converted_to_bnb:
                syms_d = ", ".join(r["asset"] for r in converted_to_bnb[:10])
                more_d = f" +{len(converted_to_bnb) - 10}" if len(converted_to_bnb) > 10 else ""
                parts_msg.append(
                    f"{len(converted_to_bnb)} toz bakiye BNB'ye dönüştürüldü (+{_format_decimal_clean(total_bnb_received)} BNB: {syms_d}{more_d})"
                )
            elif dust_note:
                if "-6005" in dust_note or "hour" in dust_note.lower() or "frequent" in dust_note.lower():
                    parts_msg.append("Binance toz→BNB dönüşümü saatlik bekleme süresinde")
                else:
                    parts_msg.append(f"Toz dönüşüm notu: {dust_note[:110]}")
            if not parts_msg:
                parts_msg.append("Optimize edilecek ek küçük bakiye veya uzlaştırılacak manuel satış bulunmadı")
            parts_msg.append(f"Güncel serbest nakit: {new_free_usdt:.2f} USDT (BNB rezervi: ~{new_bnb_usdt:.2f} USDT)")

            return {
                "ok": True,
                "reconciled_manual_count": len(reconciled_manual),
                "reconciled_manual": reconciled_manual,
                "sold_to_usdt_count": len(sold_to_usdt),
                "sold_to_usdt": sold_to_usdt,
                "converted_to_bnb_count": len(converted_to_bnb),
                "converted_to_bnb": converted_to_bnb,
                "reconciled_dust_count": len(reconciled_dust),
                "total_bnb_received": _format_decimal_clean(total_bnb_received),
                "free_usdt": f"{new_free_usdt:.2f}",
                "free_bnb": _format_decimal_clean(new_free_bnb),
                "free_bnb_usdt": f"{new_bnb_usdt:.2f}",
                "dust_note": dust_note,
                "sell_errors": sell_errors,
                "message": " • ".join(parts_msg),
            }
