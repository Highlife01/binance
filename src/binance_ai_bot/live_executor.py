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
from typing import Any, Callable
from urllib.parse import urlencode

import httpx

from .config import AppConfig
from .domain import (
    Position,
    RiskSettings,
    SimulatedOrderResult,
    SymbolFilters,
    is_substep_dust_position,
    quantize_down,
)
from .market_data import _friendly_network_error
from .security import LiveOrderBlockedError, assert_live_spot_request, redact_secrets
from .storage import SQLiteStorage


class LiveExecutionError(RuntimeError):
    """Canlı Spot bağlantı, yetki veya emir doğrulama hatası."""

    def __init__(self, message: str, *, api_code: int | None = None, ambiguous: bool = False) -> None:
        super().__init__(message)
        self.api_code = api_code
        self.ambiguous = ambiguous


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

    def _sign_params(self, params: dict[str, Any]) -> str:
        query_str = urlencode(params)
        signature = hmac.new(
            self.config.binance_api_secret.encode("utf-8"),
            query_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query_str}&signature={signature}"

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

        base_url = self.config.binance_live_base_url
        full_url = f"{base_url}{path}"
        assert_live_spot_request(
            method,
            full_url,
            config_live_enabled=self.config.enable_live_trading,
            ui_live_confirmed=ui_live_confirmed,
        )

        client = self._external_client or httpx.Client(timeout=10.0)
        close_after = self._external_client is None
        try:
            # Önce sunucu saatini alarak yerel saat kaymasından (-1021 Timestamp) koru
            time_url = f"{base_url}/api/v3/time"
            assert_live_spot_request(
                "GET",
                time_url,
                config_live_enabled=self.config.enable_live_trading,
                ui_live_confirmed=ui_live_confirmed,
            )
            try:
                t_resp = client.get(time_url)
                server_time_ms = int(t_resp.json()["serverTime"])
            except Exception as exc:
                raise LiveExecutionError(_friendly_network_error(exc, time_url)) from exc

            local_ms = int(time.time() * 1000)
            drift_ms = abs(server_time_ms - local_ms)
            if drift_ms > 30_000:
                raise LiveExecutionError(
                    f"Windows sistem saatiniz Binance sunucusundan {drift_ms // 1000} sn sapmış durumda. "
                    "Lütfen Windows Tarih ve Saat ayarlarından 'Şimdi eşitle' yapın."
                )

            payload = dict(params or {})
            payload["recvWindow"] = self.config.binance_recv_window_ms
            payload["timestamp"] = server_time_ms
            signed_query = self._sign_params(payload)
            headers = {"X-MBX-APIKEY": self.config.binance_api_key}

            req_url = f"{full_url}?{signed_query}"
            try:
                resp = client.request(method.upper(), req_url, headers=headers)
            except Exception as exc:
                raise LiveExecutionError(
                    _friendly_network_error(exc, full_url),
                    ambiguous=method.upper() == "POST" and path == "/api/v3/order",
                ) from exc

            if resp.status_code != 200:
                safe_body = redact_secrets(resp.text[:240])
                try:
                    api_code = int(resp.json().get("code", 0))
                except (ValueError, TypeError, AttributeError):
                    api_code = None
                raise LiveExecutionError(
                    f"Binance Canlı Spot API hatası (HTTP {resp.status_code}): {safe_body}",
                    api_code=api_code,
                    ambiguous=(
                        resp.status_code >= 500 or resp.status_code == 408
                        or api_code in {-1000, -1001, -1006, -1007}
                        or (api_code == -2010 and "duplicate" in resp.text.lower())
                    ),
                )
            try:
                return resp.json()
            except ValueError as exc:
                raise LiveExecutionError(
                    "Binance yanıtı çözümlenemedi; emir sonucu doğrulanmalıdır.",
                    ambiguous=method.upper() == "POST" and path == "/api/v3/order",
                ) from exc
        finally:
            if close_after:
                client.close()

    def fetch_live_spot_balances(
        self, *, ui_live_confirmed: bool, quote_asset: str = "USDC"
    ) -> dict[str, Decimal]:
        """Gerçek Binance Spot hesabındaki serbest (`free`) varlık bakiyelerini okur ve SQLite'ta günceller."""
        with self.storage.execution_lock():
            return self._fetch_live_spot_balances(ui_live_confirmed=ui_live_confirmed, quote_asset=quote_asset)

    def _fetch_live_spot_balances(
        self, *, ui_live_confirmed: bool, quote_asset: str,
    ) -> dict[str, Decimal]:
        q_norm = quote_asset.strip().upper() if quote_asset.strip().upper() in {"USDT", "USDC"} else "USDC"
        self.storage.ensure_quote_asset("LIVE_SPOT", q_norm)
        earn_key = f"LD{q_norm}"

        data = self._request_signed("GET", "/api/v3/account", ui_live_confirmed=ui_live_confirmed)
        if not bool(data.get("canTrade", False)):
            raise LiveExecutionError(
                "Binance hesabınızda veya API anahtarınızda Spot alım-satım yetkisi ('canTrade') kapalı görünüyor."
            )

        balances: dict[str, Decimal] = {}
        for item in data.get("balances", []):
            asset = str(item.get("asset", "")).upper()
            free_amt = Decimal(str(item.get("free", "0")))
            if free_amt > 0 or asset in {"USDC", "USDT", "LDUSDC", "LDUSDT"}:
                balances[asset] = free_amt

        quote_free = balances.get(q_norm, Decimal("0"))
        earn_free = balances.get(earn_key, Decimal("0"))
        self.storage.set_state("live_spot_cash_usdc", f"{quote_free:.8f}")
        self.storage.set_state("live_earn_ldusdc", f"{earn_free:.8f}")
        self.storage.set_state("live_balance_refresh_required", "0")

        perms = {str(p).upper() for p in data.get("permissions", [])}
        spot_perm = ("SPOT" in perms) or any(p.startswith("TRD_GRP_") for p in perms)
        self.storage.set_state("live_spot_permission_ok", "1" if spot_perm else "0")
        return balances

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
            opened = self.storage.get_open_positions(track="LIVE_SPOT")
            active_opened = [
                p
                for p in opened
                if not is_substep_dust_position(p, filters if p.symbol == filters.symbol else None)
            ]
            dust_same_sym = next(
                (p for p in opened if p.symbol == filters.symbol and is_substep_dust_position(p, filters)),
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
            if approved_notional is not None:
                if not approved_notional.is_finite() or approved_notional <= 0:
                    raise LiveExecutionError("Onaylı canlı alış bütçesi geçersiz.")
                budget = quantize_down(
                    min(approved_notional, settings.max_position_size_usdc,
                        free_quote / (1 + settings.estimated_fee_rate)), Decimal("0.00000001"),
                )
                if budget < filters.min_notional:
                    raise LiveExecutionError("Güncel bakiye ve komisyon payıyla minimum emir tutarı karşılanamıyor.")
                intent["quote_budget"] = str(budget)
                del intent["order_params"]["quantity"]
                intent["order_params"]["quoteOrderQty"] = _format_decimal_clean(budget)
            return self._submit_intent(intent, ui_live_confirmed=ui_live_confirmed, stop_requested=stop_requested)

    def execute_live_market_sell(
        self, *, position: Position, filters: SymbolFilters, signal_key: str,
        strategy_source: str, close_reason: str, ui_live_confirmed: bool,
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
            intent = self._new_intent(
                filters=filters, quantity=qty, signal_key=signal_key, side="SELL",
                strategy_source=strategy_source, position_id=current.position_id, close_reason=close_reason,
            )
            return self._submit_intent(intent, ui_live_confirmed=ui_live_confirmed, stop_requested=stop_requested)
