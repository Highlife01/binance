"""
Üç Kilitli Gerçek Binance Çapraz Marjin (Cross Margin) Kaldıraç Yürütücüsü (`BinanceLiveMarginExecutor`).

GÜVENLİK MİMARİSİ:
- Yalnızca `.env` `ENABLE_LIVE_TRADING=true` + `ui_live_confirmed=True` + `ui_margin_confirmed=True`
  üçlü kilidi açıkken çalışır.
- Sanal (`MAIN`) ve Canlı Spot (`LIVE_SPOT`) kayıtlarıyla ASLA karışmaz (`LIVE_MARGIN` izinde tutulur).
- Spot cüzdanınıza otomatik dokunmaz; yalnızca Marjin cüzdanına ayırdığınız teminatla işlem yapar.
- Alımda `sideEffectType="MARGIN_BUY"` (yalnızca gerektiğinde borç alır), satışta ise
  `sideEffectType="AUTO_REPAY"` (satış geliriyle borç + faizi anında otomatik kapatır, asla açığa satış/short
  borcu açmaz) kullanır.
- `marginLevel < 1.50` olduğunda likidasyon güvenliği için yeni borçlu alımı reddeder.
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
from .live_executor import LiveExecutionError, _format_decimal_clean
from .market_data import _friendly_network_error
from .risk import compute_liquidation_price
from .security import LiveOrderBlockedError, assert_live_margin_request, redact_secrets
from .storage import SQLiteStorage

MIN_SAFE_MARGIN_LEVEL = Decimal("1.50")


class BinanceLiveMarginExecutor:
    """HMAC-SHA256 imzalı, borç limit ve marjin seviyesi korumalı resmi Binance Cross Margin yürütücüsü."""

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
        ui_margin_confirmed: bool,
    ) -> Any:
        if not self.config.has_binance_live_keys:
            raise LiveOrderBlockedError(
                "Canlı Marjin işlem için .env dosyasında BINANCE_API_KEY ve BINANCE_API_SECRET tanımlı olmalıdır."
            )

        base_url = self.config.binance_live_base_url
        full_url = f"{base_url}{path}"
        assert_live_margin_request(
            method,
            full_url,
            config_live_enabled=self.config.enable_live_trading,
            ui_live_confirmed=ui_live_confirmed,
            ui_margin_confirmed=ui_margin_confirmed,
        )

        client = self._external_client or httpx.Client(timeout=10.0)
        close_after = self._external_client is None
        try:
            time_url = f"{base_url}/api/v3/time"
            assert_live_margin_request(
                "GET",
                time_url,
                config_live_enabled=self.config.enable_live_trading,
                ui_live_confirmed=ui_live_confirmed,
                ui_margin_confirmed=ui_margin_confirmed,
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
                safe_body = redact_secrets(resp.text[:260])
                if "-3003" in safe_body or "Margin account does not exist" in safe_body:
                    raise LiveExecutionError(
                        "Binance Çapraz Marjin (Cross Margin) cüzdanınız henüz açılmamış görünüyor (-3003). "
                        "Binance uygulamasından Cüzdanlar -> Marjin (Cross) ekranına girip kısa testi onaylayarak "
                        "Marjin hesabını açın ve Spot cüzdanınızdan Marjin cüzdanına USDT aktarın."
                    )
                raise LiveExecutionError(
                    f"Binance Canlı Marjin API hatası (HTTP {resp.status_code}): {safe_body}"
                )
            return resp.json()
        finally:
            if close_after:
                client.close()

    def fetch_live_margin_balances(
        self,
        *,
        ui_live_confirmed: bool,
        ui_margin_confirmed: bool,
        quote_asset: str = "USDT",
    ) -> dict[str, Any]:
        """
        Gerçek Çapraz Marjin (`GET /sapi/v1/margin/account`) bakiye, borç, faiz ve marjin seviyesini okur;
        yeni işlemler için kalan serbest teminatı `live_margin_collateral_usdc` olarak SQLite'a yazar.
        """
        q_norm = quote_asset.strip().upper() if quote_asset.strip().upper() in {"USDT", "USDC"} else "USDT"
        self.storage.ensure_quote_asset("LIVE_MARGIN", q_norm)
        data = self._request_signed(
            "GET",
            "/sapi/v1/margin/account",
            ui_live_confirmed=ui_live_confirmed,
            ui_margin_confirmed=ui_margin_confirmed,
        )

        trade_enabled = bool(data.get("tradeEnabled", False))
        borrow_enabled = bool(data.get("borrowEnabled", False))
        try:
            margin_level = Decimal(str(data.get("marginLevel", "999")))
        except Exception:
            margin_level = Decimal("999")

        free_by_asset: dict[str, Decimal] = {}
        borrowed_by_asset: dict[str, Decimal] = {}
        interest_by_asset: dict[str, Decimal] = {}
        net_by_asset: dict[str, Decimal] = {}

        for item in data.get("userAssets", []):
            asset = str(item.get("asset", "")).upper()
            f_amt = Decimal(str(item.get("free", "0")))
            b_amt = Decimal(str(item.get("borrowed", "0")))
            i_amt = Decimal(str(item.get("interest", "0")))
            n_amt = Decimal(str(item.get("netAsset", "0")))
            if f_amt != 0 or b_amt != 0 or n_amt != 0 or asset in {"USDT", "USDC"}:
                free_by_asset[asset] = f_amt
                borrowed_by_asset[asset] = b_amt
                interest_by_asset[asset] = i_amt
                net_by_asset[asset] = n_amt

        quote_free = free_by_asset.get(q_norm, Decimal("0"))
        quote_borrowed = borrowed_by_asset.get(q_norm, Decimal("0"))
        quote_interest = interest_by_asset.get(q_norm, Decimal("0"))
        quote_net = net_by_asset.get(q_norm, Decimal("0"))

        # Kalan taahhüt edilmemiş öz teminat = quote_net + açık pozisyonların sanal borç dilimi toplamı
        open_positions = self.storage.get_open_positions(track="LIVE_MARGIN")
        allocated_debt = sum((getattr(p, "borrowed_usdc", Decimal("0")) for p in open_positions), Decimal("0"))
        available_collateral = max(Decimal("0"), quote_net + allocated_debt)
        if not trade_enabled:
            available_collateral = Decimal("0")

        self.storage.set_cash_balance(available_collateral, track="LIVE_MARGIN")
        self.storage.set_state("live_margin_level", f"{margin_level:.4f}")
        self.storage.set_state("live_margin_trade_enabled", "1" if trade_enabled else "0")
        self.storage.set_state("live_margin_borrow_enabled", "1" if borrow_enabled else "0")
        self.storage.set_state("live_margin_quote_borrowed", f"{quote_borrowed:.8f}")

        if not trade_enabled:
            raise LiveExecutionError(
                "Binance Çapraz Marjin hesabınız henüz etkin değil ('tradeEnabled=false'). "
                "Lütfen Binance uygulamasından Cüzdanlar -> Marjin (Cross) ekranından Marjin hesabını etkinleştirip "
                f"Spot cüzdanınızdan Marjin cüzdanına {q_norm} aktarın."
            )

        return {
            "trade_enabled": trade_enabled,
            "borrow_enabled": borrow_enabled,
            "margin_level": margin_level,
            "quote_free": quote_free,
            "quote_borrowed": quote_borrowed,
            "quote_interest": quote_interest,
            "quote_net": quote_net,
            "available_collateral": available_collateral,
            "free_by_asset": free_by_asset,
        }

    def execute_live_margin_buy(
        self,
        *,
        filters: SymbolFilters,
        approved_qty: Decimal,
        signal_key: str,
        strategy_source: str,
        settings: RiskSettings,
        ui_live_confirmed: bool,
        ui_margin_confirmed: bool,
        approved_notional: Decimal | None = None,
    ) -> SimulatedOrderResult:
        """
        Önce `/api/v3/order/test` ve `/sapi/v1/margin/maxBorrowable` kontrollerini yapar,
        ardından Çapraz Marjin'de `sideEffectType=MARGIN_BUY` ile kaldıraçlı uzun (long) alım açar.
        """
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        ist_date = self.storage.istanbul_date_str(now_utc)
        q_ccy = filters.quote_asset
        lev = max(1, min(5, int(getattr(settings, "leverage", 2))))

        if self.storage.has_processed_signal(signal_key, track="LIVE_MARGIN"):
            raise LiveExecutionError(f"Mükerrer canlı marjin sinyali engellendi: {signal_key}")

        status_info = self.fetch_live_margin_balances(
            ui_live_confirmed=ui_live_confirmed,
            ui_margin_confirmed=ui_margin_confirmed,
            quote_asset=q_ccy,
        )
        margin_level: Decimal = status_info["margin_level"]
        if margin_level < MIN_SAFE_MARGIN_LEVEL:
            raise LiveExecutionError(
                f"Marjin güvenlik seviyesi ({margin_level:.2f}) minimum güvenli sınırın ({MIN_SAFE_MARGIN_LEVEL:.2f}) "
                "altında olduğu için likidasyon koruması yeni alımı durdurdu."
            )

        qty = quantize_down(approved_qty, filters.market_step_size)
        client_oid = f"BMARG{uuid.uuid4().hex[:18].upper()}"

        # 1. Sembol ve miktar adımı ön-doğrulaması (/api/v3/order/test)
        test_params = {
            "symbol": filters.symbol,
            "side": "BUY",
            "type": "MARKET",
            "quantity": _format_decimal_clean(qty),
        }
        self._request_signed(
            "POST",
            "/api/v3/order/test",
            test_params,
            ui_live_confirmed=ui_live_confirmed,
            ui_margin_confirmed=ui_margin_confirmed,
        )

        # 2. Maksimum borçlanılabilir limit kontrolü (/sapi/v1/margin/maxBorrowable)
        mb_resp = self._request_signed(
            "GET",
            "/sapi/v1/margin/maxBorrowable",
            {"asset": q_ccy},
            ui_live_confirmed=ui_live_confirmed,
            ui_margin_confirmed=ui_margin_confirmed,
        )
        max_borrow = Decimal(str(mb_resp.get("amount", "0")))
        total_buying_power = status_info["quote_free"] + max_borrow
        if total_buying_power < filters.min_notional:
            raise LiveExecutionError(
                f"Marjin cüzdanındaki serbest {q_ccy} ({status_info['quote_free']:.2f}) + borç limiti "
                f"({max_borrow:.2f}) minimum emir tutarını ({filters.min_notional:.2f} {q_ccy}) karşılamıyor."
            )
        if (
            approved_notional is not None
            and approved_notional > Decimal("0")
            and approved_notional > total_buying_power * Decimal("0.985")
        ):
            scale = (total_buying_power * Decimal("0.985")) / approved_notional
            scaled_qty = quantize_down(approved_qty * scale, filters.market_step_size)
            if scaled_qty >= max(filters.min_qty, filters.market_min_qty):
                qty = scaled_qty

        # 3. Gerçek Çapraz Marjin alım emri (MARGIN_BUY -> bakiye yetmezse otomatik borçlanır)
        order_params = {
            "symbol": filters.symbol,
            "isIsolated": "FALSE",
            "side": "BUY",
            "type": "MARKET",
            "quantity": _format_decimal_clean(qty),
            "sideEffectType": "MARGIN_BUY" if lev > 1 else "NO_SIDE_EFFECT",
            "newClientOrderId": client_oid,
            "newOrderRespType": "FULL",
        }
        resp = self._request_signed(
            "POST",
            "/sapi/v1/margin/order",
            order_params,
            ui_live_confirmed=ui_live_confirmed,
            ui_margin_confirmed=ui_margin_confirmed,
        )

        executed_qty = Decimal(str(resp.get("executedQty", "0")))
        gross_executed_qty = executed_qty
        cumm_quote = Decimal(str(resp.get("cummulativeQuoteQty", "0")))
        avg_price = (cumm_quote / executed_qty) if executed_qty > 0 else Decimal("0")

        fee_usdc = Decimal("0")
        for fill in resp.get("fills", []):
            comm = Decimal(str(fill.get("commission", "0")))
            comm_asset = str(fill.get("commissionAsset", "")).upper()
            if comm_asset == q_ccy:
                fee_usdc += comm
            elif comm_asset == filters.base_asset and avg_price > 0:
                fee_usdc += comm * avg_price
                executed_qty = max(Decimal("0"), executed_qty - comm)
            elif comm > 0 and cumm_quote > 0:
                fill_qty = Decimal(str(fill.get("qty", "0")))
                fill_px = Decimal(str(fill.get("price", "0")))
                fill_quote = fill_qty * fill_px if (fill_qty > 0 and fill_px > 0) else cumm_quote
                fee_usdc += fill_quote * getattr(settings, "estimated_fee_rate", Decimal("0.00075"))

        eff_qty = quantize_down(executed_qty, filters.market_step_size)
        order_id = f"MARG-{resp.get('orderId', client_oid)}"
        raw_status = str(resp.get("status", "FILLED")).upper()
        mapped_status = "PARTIALLY_FILLED" if raw_status == "PARTIALLY_FILLED" else "FILLED"

        # Baz varlık komisyonu / adım yuvarlaması yüzünden kayda alınan miktar azalırsa maliyet tabanı da
        # aynı oranda küçültülür; aksi halde artakalan küsurat sahte zarar (phantom loss) olarak yazılırdı.
        kept_ratio = (eff_qty / gross_executed_qty) if gross_executed_qty > 0 else Decimal("1")
        kept_ratio = min(Decimal("1"), max(Decimal("0"), kept_ratio))
        pos_cost = cumm_quote * kept_ratio
        collateral_part = pos_cost / Decimal(lev)
        borrowed_part = pos_cost - collateral_part

        if eff_qty > 0:
            stop_px = avg_price * (Decimal("1") - settings.stop_loss_pct) if settings.stop_loss_pct > 0 else None
            tp_px = avg_price * (Decimal("1") + settings.take_profit_pct) if settings.take_profit_pct > 0 else None
            liq_px = compute_liquidation_price(quantity=eff_qty, borrowed_usdc=borrowed_part)
            pos = Position(
                position_id=f"MARG-POS-{uuid.uuid4().hex[:10].upper()}",
                symbol=filters.symbol,
                quantity=eff_qty,
                entry_price=avg_price,
                cost_basis_usdc=pos_cost,
                entry_fee_usdc=fee_usdc,
                opened_at_iso=now_iso,
                signal_key=signal_key,
                strategy_source=strategy_source,
                stop_loss_price=stop_px,
                take_profit_price=tp_px,
                status="OPEN",
                leverage=lev,
                borrowed_usdc=borrowed_part,
                liquidation_price=liq_px,
            )
            self.storage.insert_position(pos, track="LIVE_MARGIN")
            self.storage.mark_signal_processed(signal_key, order_id, now_iso, track="LIVE_MARGIN")
            self.fetch_live_margin_balances(
                ui_live_confirmed=ui_live_confirmed,
                ui_margin_confirmed=ui_margin_confirmed,
                quote_asset=q_ccy,
            )

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
                f"GERÇEK MARJİN ALIM [{lev}x] ({mapped_status}): {eff_qty} {filters.base_asset} "
                f"@ {avg_price:.6f} {q_ccy} (Toplam: {cumm_quote:.2f} {q_ccy}, Teminat: {collateral_part:.2f} {q_ccy}, "
                f"Borç Dilimi: {borrowed_part:.2f} {q_ccy}, Komisyon: {fee_usdc:.4f} {q_ccy})."
            ),
            created_at_iso=now_iso,
        )
        self.storage.record_order(res, ist_date, track="LIVE_MARGIN")
        return res

    def execute_live_margin_sell(
        self,
        *,
        position: Position,
        filters: SymbolFilters,
        signal_key: str,
        strategy_source: str,
        close_reason: str,
        ui_live_confirmed: bool,
        ui_margin_confirmed: bool,
    ) -> SimulatedOrderResult:
        """
        Açık Çapraz Marjin pozisyonunu `sideEffectType=AUTO_REPAY` ile satarak kapatır;
        elde edilen USDT ile birikmiş faiz ve anapara borcunu otomatik öder (asla short borcu açmaz).
        """
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        ist_date = self.storage.istanbul_date_str(now_utc)
        q_ccy = filters.quote_asset

        status_info = self.fetch_live_margin_balances(
            ui_live_confirmed=ui_live_confirmed,
            ui_margin_confirmed=ui_margin_confirmed,
            quote_asset=q_ccy,
        )
        free_base = status_info["free_by_asset"].get(filters.base_asset, Decimal("0"))
        sell_qty = quantize_down(min(position.quantity, free_base), filters.market_step_size)
        if sell_qty < max(filters.min_qty, filters.market_min_qty) or sell_qty <= Decimal("0"):
            raise LiveExecutionError(
                f"Marjin cüzdanında satılabilir yeterli {filters.base_asset} miktarı bulunamadı "
                f"(serbest: {free_base}, gereken min: {filters.min_qty})."
            )

        client_oid = f"MSELL{uuid.uuid4().hex[:18].upper()}"
        order_params = {
            "symbol": filters.symbol,
            "isIsolated": "FALSE",
            "side": "SELL",
            "type": "MARKET",
            "quantity": _format_decimal_clean(sell_qty),
            "sideEffectType": "AUTO_REPAY",
            "newClientOrderId": client_oid,
            "newOrderRespType": "FULL",
        }
        resp = self._request_signed(
            "POST",
            "/sapi/v1/margin/order",
            order_params,
            ui_live_confirmed=ui_live_confirmed,
            ui_margin_confirmed=ui_margin_confirmed,
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
            elif comm_asset == filters.base_asset and avg_price > 0:
                exit_fee_usdc += comm * avg_price
            elif comm > 0 and cumm_quote > 0:
                fill_qty = Decimal(str(fill.get("qty", "0")))
                fill_px = Decimal(str(fill.get("price", "0")))
                fill_quote = fill_qty * fill_px if (fill_qty > 0 and fill_px > 0) else cumm_quote
                exit_fee_usdc += fill_quote * Decimal("0.00075")

        net_proceeds = cumm_quote - exit_fee_usdc
        realized_pnl = net_proceeds - position.cost_basis_usdc - position.entry_fee_usdc
        order_id = f"MARG-{resp.get('orderId', client_oid)}"

        self.storage.update_position_closed(
            position.position_id,
            closed_at_iso=now_iso,
            closed_istanbul_date=ist_date,
            exit_price=avg_price,
            exit_fee_usdc=exit_fee_usdc,
            realized_pnl_usdc=realized_pnl,
            close_reason=close_reason,
        )
        self.storage.mark_signal_processed(signal_key, order_id, now_iso, track="LIVE_MARGIN")
        self.fetch_live_margin_balances(
            ui_live_confirmed=ui_live_confirmed,
            ui_margin_confirmed=ui_margin_confirmed,
            quote_asset=q_ccy,
        )

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
                f"GERÇEK MARJİN SATIŞ + OTOMATİK BORÇ ÖDEME ({close_reason}): "
                f"{executed_qty} {filters.base_asset} @ {avg_price:.6f} {q_ccy} "
                f"(Net: {net_proceeds:.2f} {q_ccy}, K/Z: {realized_pnl:+.4f} {q_ccy})."
            ),
            created_at_iso=now_iso,
            realized_pnl_usdc=realized_pnl,
        )
        self.storage.record_order(res, ist_date, track="LIVE_MARGIN")
        return res
