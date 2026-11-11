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
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import httpx

from .config import AppConfig
from .domain import (
    Position,
    RiskSettings,
    SimulatedOrderResult,
    SymbolFilters,
    quantize_down,
)
from .market_data import _friendly_network_error
from .security import LiveOrderBlockedError, assert_live_spot_request, redact_secrets
from .storage import SQLiteStorage


class LiveExecutionError(RuntimeError):
    """Canlı Spot bağlantı, yetki veya emir doğrulama hatası."""


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
                raise LiveExecutionError(_friendly_network_error(exc, full_url)) from exc

            if resp.status_code != 200:
                safe_body = redact_secrets(resp.text[:240])
                raise LiveExecutionError(
                    f"Binance Canlı Spot API hatası (HTTP {resp.status_code}): {safe_body}"
                )
            return resp.json()
        finally:
            if close_after:
                client.close()

    def fetch_live_spot_balances(
        self, *, ui_live_confirmed: bool, quote_asset: str = "USDC"
    ) -> dict[str, Decimal]:
        """Gerçek Binance Spot hesabındaki serbest (`free`) varlık bakiyelerini okur ve SQLite'ta günceller."""
        q_norm = quote_asset.strip().upper() if quote_asset.strip().upper() in {"USDT", "USDC"} else "USDC"
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

        perms = {str(p).upper() for p in data.get("permissions", [])}
        spot_perm = ("SPOT" in perms) or any(p.startswith("TRD_GRP_") for p in perms)
        self.storage.set_state("live_spot_permission_ok", "1" if spot_perm else "0")
        return balances

    def execute_live_market_buy(
        self,
        *,
        filters: SymbolFilters,
        approved_qty: Decimal,
        signal_key: str,
        strategy_source: str,
        settings: RiskSettings,
        ui_live_confirmed: bool,
    ) -> SimulatedOrderResult:
        """
        Önce `/api/v3/order/test` ile doğrular, ardından gerçek Spot MARKET BUY emrini gönderir.
        """
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        ist_date = self.storage.istanbul_date_str(now_utc)

        if self.storage.has_processed_signal(signal_key, track="LIVE_SPOT"):
            raise LiveExecutionError(f"Mükerrer canlı sinyal engellendi: {signal_key}")

        qty = quantize_down(approved_qty, filters.market_step_size)
        client_oid = f"BSPOT{uuid.uuid4().hex[:18].upper()}"
        order_params = {
            "symbol": filters.symbol,
            "side": "BUY",
            "type": "MARKET",
            "quantity": _format_decimal_clean(qty),
            "newClientOrderId": client_oid,
            "newOrderRespType": "FULL",
        }

        # 1. Ön-doğrulama testi (gerçek bakiye kilitlemeden filtre/imza doğrulaması)
        self._request_signed(
            "POST", "/api/v3/order/test", order_params, ui_live_confirmed=ui_live_confirmed
        )

        # 2. Gerçek Spot emir gönderimi
        resp = self._request_signed(
            "POST", "/api/v3/order", order_params, ui_live_confirmed=ui_live_confirmed
        )

        executed_qty = Decimal(str(resp.get("executedQty", "0")))
        cumm_quote = Decimal(str(resp.get("cummulativeQuoteQty", "0")))
        avg_price = (cumm_quote / executed_qty) if executed_qty > 0 else Decimal("0")

        # Fills dizisinden gerçek kesilen komisyonu hesapla
        q_ccy = filters.quote_asset
        fee_usdc = Decimal("0")
        for fill in resp.get("fills", []):
            comm = Decimal(str(fill.get("commission", "0")))
            comm_asset = str(fill.get("commissionAsset", "")).upper()
            if comm_asset == q_ccy:
                fee_usdc += comm
            elif comm_asset == filters.base_asset and avg_price > 0:
                fee_usdc += comm * avg_price
                # Komisyon baz varlıktan kesildiyse net eldeki miktarı düşür
                executed_qty = max(Decimal("0"), executed_qty - comm)

        eff_qty = quantize_down(executed_qty, filters.market_step_size)
        order_id = f"LIVE-{resp.get('orderId', client_oid)}"
        raw_status = str(resp.get("status", "FILLED")).upper()
        mapped_status = "PARTIALLY_FILLED" if raw_status == "PARTIALLY_FILLED" else "FILLED"

        if eff_qty > 0:
            stop_px = avg_price * (Decimal("1") - settings.stop_loss_pct) if settings.stop_loss_pct > 0 else None
            tp_px = avg_price * (Decimal("1") + settings.take_profit_pct) if settings.take_profit_pct > 0 else None
            pos = Position(
                position_id=f"LIVE-POS-{uuid.uuid4().hex[:10].upper()}",
                symbol=filters.symbol,
                quantity=eff_qty,
                entry_price=avg_price,
                cost_basis_usdc=cumm_quote,
                entry_fee_usdc=fee_usdc,
                opened_at_iso=now_iso,
                signal_key=signal_key,
                strategy_source=strategy_source,
                stop_loss_price=stop_px,
                take_profit_price=tp_px,
                status="OPEN",
            )
            self.storage.insert_position(pos, track="LIVE_SPOT")
            self.storage.mark_signal_processed(signal_key, order_id, now_iso, track="LIVE_SPOT")

        res = SimulatedOrderResult(
            order_id=order_id,
            symbol=filters.symbol,
            side="BUY",
            order_type="MARKET",
            status=mapped_status,  # type: ignore[arg-type]
            requested_qty=qty,
            filled_qty=eff_qty,
            avg_fill_price=avg_price,
            gross_quote_usdc=cumm_quote,
            fee_usdc=fee_usdc,
            slippage_bps=Decimal("0"),
            signal_key=signal_key,
            strategy_source=strategy_source,
            explanation=(
                f"GERÇEK SPOT ALIM ({mapped_status}): {eff_qty} {filters.base_asset} "
                f"@ {avg_price:.6f} {q_ccy} (Toplam: {cumm_quote:.2f} {q_ccy}, Komisyon: {fee_usdc:.4f} {q_ccy})."
            ),
            created_at_iso=now_iso,
        )
        self.storage.record_order(res, ist_date, track="LIVE_SPOT")
        return res

    def execute_live_market_sell(
        self,
        *,
        position: Position,
        filters: SymbolFilters,
        signal_key: str,
        strategy_source: str,
        close_reason: str,
        ui_live_confirmed: bool,
    ) -> SimulatedOrderResult:
        """Gerçek açık Spot pozisyonu Binance üzerinde MARKET SELL ile kapatır."""
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        ist_date = self.storage.istanbul_date_str(now_utc)
        q_ccy = filters.quote_asset

        # Hesaptaki gerçek serbest baz varlık miktarını doğrula (dust farkı nedeniyle -2010 hatasını önler)
        balances = self.fetch_live_spot_balances(
            ui_live_confirmed=ui_live_confirmed, quote_asset=q_ccy
        )
        free_base = balances.get(filters.base_asset, Decimal("0"))
        sell_qty = quantize_down(min(position.quantity, free_base), filters.market_step_size)
        if sell_qty < max(filters.min_qty, filters.market_min_qty) or sell_qty <= Decimal("0"):
            raise LiveExecutionError(
                f"Gerçek hesapta satılabilir yeterli {filters.base_asset} miktarı bulunamadı "
                f"(serbest: {free_base}, gereken min: {filters.min_qty})."
            )

        client_oid = f"SSELL{uuid.uuid4().hex[:18].upper()}"
        order_params = {
            "symbol": filters.symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": _format_decimal_clean(sell_qty),
            "newClientOrderId": client_oid,
            "newOrderRespType": "FULL",
        }

        self._request_signed(
            "POST", "/api/v3/order/test", order_params, ui_live_confirmed=ui_live_confirmed
        )
        resp = self._request_signed(
            "POST", "/api/v3/order", order_params, ui_live_confirmed=ui_live_confirmed
        )

        executed_qty = Decimal(str(resp.get("executedQty", "0")))
        cumm_quote = Decimal(str(resp.get("cummulativeQuoteQty", "0")))
        avg_price = (cumm_quote / executed_qty) if executed_qty > 0 else Decimal("0")

        exit_fee_usdc = Decimal("0")
        for fill in resp.get("fills", []):
            comm = Decimal(str(fill.get("commission", "0")))
            comm_asset = str(fill.get("commissionAsset", "")).upper()
            if comm_asset == q_ccy:
                exit_fee_usdc += comm

        net_proceeds = cumm_quote - exit_fee_usdc
        realized_pnl = net_proceeds - position.cost_basis_usdc - position.entry_fee_usdc
        order_id = f"LIVE-{resp.get('orderId', client_oid)}"

        self.storage.update_position_closed(
            position.position_id,
            closed_at_iso=now_iso,
            closed_istanbul_date=ist_date,
            exit_price=avg_price,
            exit_fee_usdc=exit_fee_usdc,
            realized_pnl_usdc=realized_pnl,
            close_reason=close_reason,
        )
        self.storage.mark_signal_processed(signal_key, order_id, now_iso, track="LIVE_SPOT")
        self.fetch_live_spot_balances(ui_live_confirmed=ui_live_confirmed, quote_asset=q_ccy)

        res = SimulatedOrderResult(
            order_id=order_id,
            symbol=filters.symbol,
            side="SELL",
            order_type="MARKET",
            status="FILLED",
            requested_qty=sell_qty,
            filled_qty=executed_qty,
            avg_fill_price=avg_price,
            gross_quote_usdc=cumm_quote,
            fee_usdc=exit_fee_usdc,
            slippage_bps=Decimal("0"),
            signal_key=signal_key,
            strategy_source=strategy_source,
            explanation=(
                f"GERÇEK SPOT SATIŞ ({close_reason}): {executed_qty} {filters.base_asset} "
                f"@ {avg_price:.6f} {q_ccy} (Net: {net_proceeds:.2f} {q_ccy}, K/Z: {realized_pnl:+.4f} {q_ccy})."
            ),
            created_at_iso=now_iso,
            realized_pnl_usdc=realized_pnl,
        )
        self.storage.record_order(res, ist_date, track="LIVE_SPOT")
        return res
