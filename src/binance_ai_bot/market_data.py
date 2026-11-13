"""Binance Global herkese açık piyasa verisi istemcisi, filtre keşfi ve tazelik denetimi."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx

from .config import AppConfig
from .domain import (
    Candle,
    MarketBundle,
    OrderBookLevel,
    OrderBookSnapshot,
    SymbolFilters,
)
from .security import (
    STABLECOIN_OR_FIAT_ASSETS,
    assert_public_readonly_request,
)

INTERVAL_TO_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class MarketDataError(RuntimeError):
    """Piyasa verisi alınamadığında sade Türkçe açıklama taşıyan temel hata."""


class StaleOrIncompleteDataError(MarketDataError):
    """Eksik, boşluklu veya güncelliğini yitirmiş piyasa verisi tespit edildiğinde fırlatılır."""


class RateLimitCooldownError(MarketDataError):
    """Binance hız sınırı (429/418) nedeniyle kontrollü bekleme aktifken fırlatılır."""


def parse_symbol_filters(symbol_info: dict[str, Any]) -> SymbolFilters:
    """Binance `exchangeInfo` içindeki tek bir sembol nesnesinden dinamik emir filtrelerini çıkarır."""
    filters_by_type: dict[str, dict[str, Any]] = {
        str(f.get("filterType", "")): f for f in symbol_info.get("filters", [])
    }

    price_f = filters_by_type.get("PRICE_FILTER", {})
    lot_f = filters_by_type.get("LOT_SIZE", {})
    mlot_f = filters_by_type.get("MARKET_LOT_SIZE", {})
    notional_f = filters_by_type.get("NOTIONAL") or filters_by_type.get("MIN_NOTIONAL") or {}

    step_size = Decimal(str(lot_f.get("stepSize", "0.00000001")))
    min_qty = Decimal(str(lot_f.get("minQty", "0.00000001")))
    max_qty = Decimal(str(lot_f.get("maxQty", "900000000")))

    m_step_raw = Decimal(str(mlot_f.get("stepSize", "0")))
    m_min_raw = Decimal(str(mlot_f.get("minQty", "0")))
    m_max_raw = Decimal(str(mlot_f.get("maxQty", "0")))

    # Binance dokümantasyonuna göre MARKET_LOT_SIZE stepSize=0 ise LOT_SIZE stepSize geçerlidir.
    market_step_size = m_step_raw if m_step_raw > 0 else step_size
    market_min_qty = m_min_raw if m_min_raw > 0 else min_qty
    market_max_qty = m_max_raw if m_max_raw > 0 else max_qty

    min_notional = Decimal(str(notional_f.get("minNotional", "5.0")))
    max_notional = Decimal(str(notional_f.get("maxNotional", "9000000")))
    apply_min_to_market = bool(notional_f.get("applyMinToMarket", True))

    return SymbolFilters(
        symbol=str(symbol_info["symbol"]).upper(),
        base_asset=str(symbol_info["baseAsset"]).upper(),
        quote_asset=str(symbol_info["quoteAsset"]).upper(),
        status=str(symbol_info.get("status", "UNKNOWN")).upper(),
        is_spot_trading_allowed=bool(symbol_info.get("isSpotTradingAllowed", False)),
        min_price=Decimal(str(price_f.get("minPrice", "0"))),
        max_price=Decimal(str(price_f.get("maxPrice", "900000000"))),
        tick_size=Decimal(str(price_f.get("tickSize", "0.00000001"))),
        min_qty=min_qty,
        max_qty=max_qty,
        step_size=step_size,
        market_min_qty=market_min_qty,
        market_max_qty=market_max_qty,
        market_step_size=market_step_size,
        min_notional=min_notional,
        max_notional=max_notional,
        apply_min_to_market=apply_min_to_market,
    )


def extract_closed_candles(raw_klines: list[list[Any]], server_time_ms: int) -> list[Candle]:
    """
    Binance klines dizisinden YALNIZCA kapanmış mumları ayıklar.
    Henüz kapanmamış (close_time_ms >= server_time_ms) canlı mum kesinlikle dışarıda bırakılır.
    """
    closed: list[Candle] = []
    for row in raw_klines:
        if len(row) < 9:
            continue
        open_time_ms = int(row[0])
        close_time_ms = int(row[6])
        is_closed = close_time_ms < server_time_ms
        if not is_closed:
            continue
        closed.append(
            Candle(
                open_time_ms=open_time_ms,
                open=Decimal(str(row[1])),
                high=Decimal(str(row[2])),
                low=Decimal(str(row[3])),
                close=Decimal(str(row[4])),
                volume=Decimal(str(row[5])),
                close_time_ms=close_time_ms,
                quote_volume=Decimal(str(row[7])),
                trade_count=int(row[8]),
                is_closed=True,
            )
        )
    closed.sort(key=lambda c: c.open_time_ms)
    return closed


def validate_candles_and_book(
    *,
    closed_candles: list[Candle],
    order_book: OrderBookSnapshot,
    interval: str,
    server_time_ms: int,
    max_data_age_seconds: int,
    min_closed_candles: int = 35,
) -> None:
    """
    Eksik mum, zaman boşluğu, eski veri veya bozuk emir defteri durumunda işlem açılmasını engeller.
    """
    interval_ms = INTERVAL_TO_MS.get(interval)
    if interval_ms is None:
        raise StaleOrIncompleteDataError(f"Desteklenmeyen mum zaman dilimi: {interval}")

    if len(closed_candles) < min_closed_candles:
        raise StaleOrIncompleteDataError(
            f"Güvenilir sinyal hesabı için yeterli kapanmış mum yok "
            f"(mevcut: {len(closed_candles)}, gereken en az: {min_closed_candles})."
        )

    # Son kapanmış mumun güncelliği: En fazla 1 mum süresi + tolerans kadar geride olabilir.
    last_candle = closed_candles[-1]
    allowed_lag_ms = interval_ms + (max_data_age_seconds * 1000)
    actual_lag_ms = server_time_ms - last_candle.close_time_ms
    if actual_lag_ms > allowed_lag_ms:
        lag_sec = int(actual_lag_ms / 1000)
        raise StaleOrIncompleteDataError(
            f"Piyasa mum verisi güncel değil (son kapanmış mum {lag_sec} sn geride). "
            "Eski veriyle işlem üretilmez."
        )

    # Son 30 kapanmış mum arasında eksik mum / boşluk kontrolü
    recent = closed_candles[-30:]
    for prev_c, next_c in zip(recent, recent[1:]):
        diff = next_c.open_time_ms - prev_c.open_time_ms
        if diff != interval_ms:
            raise StaleOrIncompleteDataError(
                "Kapanmış mum serisinde eksik zaman dilimi (veri boşluğu) tespit edildi; işlem durduruldu."
            )

    # Emir defteri doğrulaması
    now_utc = datetime.now(timezone.utc)
    book_age_sec = (now_utc - order_book.fetched_at_utc).total_seconds()
    if book_age_sec > max_data_age_seconds:
        raise StaleOrIncompleteDataError(
            f"Emir defteri verisi eskimiş ({int(book_age_sec)} sn). Güncel veri olmadan işlem açılmaz."
        )

    if not order_book.bids or not order_book.asks:
        raise StaleOrIncompleteDataError("Emir defteri alış/satış kademeleri boş döndü.")

    if order_book.best_bid is None or order_book.best_ask is None:
        raise StaleOrIncompleteDataError("En iyi alış/satış kotasyonu okunamadı.")

    if order_book.best_bid <= 0 or order_book.best_ask <= 0 or order_book.best_bid >= order_book.best_ask:
        raise StaleOrIncompleteDataError(
            "Anormal alış/satış makası (bid >= ask) tespit edildi; veri güvenilir değil."
        )


def _friendly_network_error(exc: Exception, url: str) -> str:
    raw = str(exc)
    if "10054" in raw or "zorla kapatıldı" in raw.lower() or "connection reset" in raw.lower():
        return (
            "Binance Global veri sunucusuna bağlantı ağ/İSS düzeyinde sıfırlandı (WinError 10054). "
            "Türkiye'deki bazı internet servis sağlayıcılarında Binance Global alan adlarına kısıtlama "
            "uygulanabilmektedir. Güncel veri alınamadığı için güvenlik gereği yeni sanal işlem üretilmedi."
        )
    if isinstance(exc, httpx.TimeoutException):
        return f"Binance veri sunucusu zaman aşımına uğradı ({url}). İnternet bağlantınızı kontrol edin."
    return f"Binance herkese açık veri bağlantısı kurulamadı: {type(exc).__name__} ({raw[:160]})"


class BinancePublicDataClient:
    """
    Binance API anahtarı gerektirmeyen, yalnızca herkese açık verileri okuyan istemci.
    - Hız sınırı (429/418) geri çekilmesi,
    - Yedek resmi sunucu geçişi,
    - Windows uyku modundan uyanma (sleep/wake) tespiti içerir.
    """

    def __init__(
        self,
        config: AppConfig,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._external_client = http_client
        self._shared_client: httpx.Client | None = None
        self._base_urls = self._build_base_url_list()
        self._exchange_info_cache: dict[str, SymbolFilters] = {}
        self._exchange_info_fetched_at: float = 0.0
        self._cooldown_until_monotonic: float = 0.0
        self._consecutive_failures: int = 0
        self._last_wall_clock_ts: float | None = None
        self.last_used_weight_1m: int | None = None
        self.last_source_url: str = ""

    def _build_base_url_list(self) -> list[str]:
        candidates = [
            self.config.binance_public_base_url,
            "https://api.binance.me",
            self.config.binance_fallback_base_url,
            "https://data-api.binance.vision",
            "https://api.binance.com",
            "https://api-gcp.binance.com",
        ]
        unique: list[str] = []
        for url in candidates:
            cleaned = url.strip().rstrip("/")
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        return unique

    def detect_windows_wake_gap(self, expected_cycle_seconds: int = 60) -> bool:
        """
        Windows uyku / hazırda bekletme dönüşünü duvar saati sıçramasından tespit eder.
        Uyku dönüşü algılandığında eski önbelleği temizler.
        """
        now_wall = time.time()
        if self._last_wall_clock_ts is None:
            self._last_wall_clock_ts = now_wall
            return False

        elapsed = now_wall - self._last_wall_clock_ts
        self._last_wall_clock_ts = now_wall
        threshold = max(expected_cycle_seconds * 3, 180)
        if elapsed > threshold:
            # Uyku modundan uyanıldı veya uzun süre ara verildi; önbelleği sıfırla
            self._exchange_info_fetched_at = 0.0
            return True
        return False

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> tuple[Any, str]:
        now_mono = time.monotonic()
        if now_mono < self._cooldown_until_monotonic:
            wait_sec = int(self._cooldown_until_monotonic - now_mono) + 1
            raise RateLimitCooldownError(
                f"Hız sınırı veya bağlantı koruması aktif; {wait_sec} saniye sonra kontrollü yeniden denenecek."
            )

        last_error_msg = "Bilinmeyen veri hatası."
        client_to_use = self._get_client()

        try:
            for base_url in self._base_urls:
                full_url = f"{base_url}{path}"
                # Mimari güvenlik kilidi: Yalnızca beyaz listedeki salt-okunur GET uç noktaları!
                assert_public_readonly_request("GET", full_url, headers=None)
                try:
                    resp = client_to_use.get(full_url, params=params)
                except Exception as exc:
                    if self._external_client is not None:
                        last_error_msg = _friendly_network_error(exc, full_url)
                        continue
                    # Bayat keep-alive bağlantısı olabilir: istemciyi yenileyip bir kez daha dene
                    self._reset_client()
                    client_to_use = self._get_client()
                    try:
                        resp = client_to_use.get(full_url, params=params)
                    except Exception as exc2:
                        last_error_msg = _friendly_network_error(exc2, full_url)
                        continue

                weight_hdr = resp.headers.get("X-MBX-USED-WEIGHT-1M")
                if weight_hdr and weight_hdr.isdigit():
                    self.last_used_weight_1m = int(weight_hdr)

                if resp.status_code in (429, 418):
                    retry_after = int(resp.headers.get("Retry-After", "30") or "30")
                    backoff = min(max(retry_after, 15), 300)
                    self._cooldown_until_monotonic = time.monotonic() + backoff
                    raise RateLimitCooldownError(
                        f"Binance hız sınırı ({resp.status_code}) algılandı. "
                        f"IP koruması için {backoff} sn beklenecek."
                    )

                if resp.status_code >= 500:
                    last_error_msg = (
                        f"Binance sunucusu geçici hata döndürdü (HTTP {resp.status_code})."
                    )
                    continue

                if resp.status_code != 200:
                    last_error_msg = (
                        f"Binance veri isteği başarısız oldu (HTTP {resp.status_code}): {resp.text[:140]}"
                    )
                    continue

                self._consecutive_failures = 0
                self.last_source_url = base_url
                return resp.json(), base_url

            self._consecutive_failures += 1
            backoff_sec = min(2 ** min(self._consecutive_failures, 5), 30)
            self._cooldown_until_monotonic = time.monotonic() + backoff_sec
            raise MarketDataError(last_error_msg)
        finally:
            pass

    def _get_client(self) -> httpx.Client:
        """
        Kalıcı (keep-alive) HTTP istemcisi: her istekte yeni TCP bağlantısı açmak Cloud NAT port
        havuzunu tüketip "No route to host" hatalarına yol açıyordu; bağlantılar yeniden kullanılır.
        """
        if self._external_client is not None:
            return self._external_client
        if self._shared_client is None:
            self._shared_client = httpx.Client(
                timeout=8.0,
                limits=httpx.Limits(max_keepalive_connections=8, max_connections=16, keepalive_expiry=30.0),
            )
        return self._shared_client

    def _reset_client(self) -> None:
        client = self._shared_client
        self._shared_client = None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def discover_spot_pairs(
        self,
        *,
        quote_asset: str | None = None,
        allow_stablecoin_conversion: bool = False,
        force_refresh: bool = False,
    ) -> dict[str, SymbolFilters]:
        """
        Binance `exchangeInfo` üzerinden aktif, kaldıraçsız Spot (`USDT` ve `USDC`) çiftlerini ve
        emir filtrelerini dinamik olarak keşfeder (15 dakika önbellekli).
        """
        target_quotes = {quote_asset.strip().upper()} if quote_asset else {"USDC", "USDT"}
        now_mono = time.monotonic()
        if (
            not force_refresh
            and self._exchange_info_cache
            and (now_mono - self._exchange_info_fetched_at) < 900
        ):
            return {
                sym: flt
                for sym, flt in self._exchange_info_cache.items()
                if flt.quote_asset in target_quotes
                and (allow_stablecoin_conversion or flt.base_asset not in STABLECOIN_OR_FIAT_ASSETS)
            }

        data, _ = self._get_json("/api/v3/exchangeInfo")
        symbols_raw = data.get("symbols", [])
        discovered: dict[str, SymbolFilters] = {}
        for item in symbols_raw:
            q_asset = str(item.get("quoteAsset", "")).upper()
            if q_asset not in {"USDC", "USDT"}:
                continue
            parsed = parse_symbol_filters(item)
            if parsed.is_tradable_spot_for_quote(q_asset):
                discovered[parsed.symbol] = parsed

        if not discovered:
            raise MarketDataError("exchangeInfo içinde aktif Spot USDT/USDC işlem çifti bulunamadı.")

        self._exchange_info_cache = discovered
        self._exchange_info_fetched_at = now_mono

        return {
            sym: flt
            for sym, flt in discovered.items()
            if flt.quote_asset in target_quotes
            and (allow_stablecoin_conversion or flt.base_asset not in STABLECOIN_OR_FIAT_ASSETS)
        }

    def discover_usdc_spot_pairs(
        self,
        *,
        allow_stablecoin_conversion: bool = False,
        force_refresh: bool = False,
    ) -> dict[str, SymbolFilters]:
        """Geriye dönük uyumlu USDC Spot çift keşfi."""
        return self.discover_spot_pairs(
            quote_asset="USDC",
            allow_stablecoin_conversion=allow_stablecoin_conversion,
            force_refresh=force_refresh,
        )

    def fetch_top_usdc_pairs_by_volume(
        self, limit: int = 20, allow_stablecoin_conversion: bool = False
    ) -> list[dict[str, Any]]:
        """Keşfedilen USDC spot çiftlerini 24 saatlik USDC hacmine göre sıralar."""
        usdc_map = self.discover_usdc_spot_pairs(
            allow_stablecoin_conversion=allow_stablecoin_conversion
        )
        tickers, _ = self._get_json("/api/v3/ticker/24hr")
        ranked: list[dict[str, Any]] = []
        for t in tickers:
            sym = str(t.get("symbol", "")).upper()
            if sym not in usdc_map:
                continue
            flt = usdc_map[sym]
            ranked.append(
                {
                    "symbol": sym,
                    "base_asset": flt.base_asset,
                    "last_price": Decimal(str(t.get("lastPrice", "0"))),
                    "quote_volume_24h": Decimal(str(t.get("quoteVolume", "0"))),
                    "price_change_pct_24h": Decimal(str(t.get("priceChangePercent", "0"))),
                    "min_notional": flt.min_notional,
                    "step_size": flt.step_size,
                    "tick_size": flt.tick_size,
                }
            )
        ranked.sort(key=lambda x: x["quote_volume_24h"], reverse=True)
        return ranked[:limit]

    def fetch_closed_candles(self, symbol: str, interval: str, limit: int = 60) -> list[Candle]:
        """Üst zaman dilimi (ör. 15dk) trend onayı için yalnızca kapanmış mumları döndürür."""
        sym = symbol.strip().upper()
        raw_klines, _ = self._get_json(
            "/api/v3/klines", params={"symbol": sym, "interval": interval, "limit": limit}
        )
        return extract_closed_candles(raw_klines, server_time_ms=int(time.time() * 1000))

    def fetch_market_radar(
        self,
        universe_limit: int = 30,
        gainers_limit: int = 15,
        *,
        min_quote_volume: float = 5_000_000.0,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Tek ticker çağrısıyla hem dengeli Fırsat Evrenini hem Günün Yükselenleri Radarını üretir."""
        from .expert import rank_top_gainers, score_universe

        usdt_map = self.discover_spot_pairs(quote_asset="USDT", allow_stablecoin_conversion=False)
        tickers, _ = self._get_json("/api/v3/ticker/24hr")
        ranked = score_universe(
            tickers,
            tradable_usdt=usdt_map,
            min_quote_volume=min_quote_volume,
            limit=universe_limit,
        )
        gainers = rank_top_gainers(
            tickers,
            tradable_usdt=usdt_map,
            limit=gainers_limit,
        )
        return ranked, gainers

    def fetch_usdt_opportunities(
        self,
        limit: int = 30,
        *,
        min_quote_volume: float = 5_000_000.0,
    ) -> list[dict[str, Any]]:
        """24 saatlik ticker ile likit, ivmeli ve dar spreadli Spot USDT fırsat evrenini puanlar."""
        ranked, _ = self.fetch_market_radar(
            universe_limit=limit,
            gainers_limit=15,
            min_quote_volume=min_quote_volume,
        )
        return ranked

    def fetch_order_book(self, symbol: str, depth_limit: int = 20) -> OrderBookSnapshot:
        """Gerçekçi likidite ve kayma hesabı için emir defteri kademelerini çeker."""
        sym = symbol.strip().upper()
        data, src = self._get_json("/api/v3/depth", params={"symbol": sym, "limit": depth_limit})
        bids = [
            OrderBookLevel(price=Decimal(str(p)), quantity=Decimal(str(q)))
            for p, q in data.get("bids", [])
            if Decimal(str(q)) > 0
        ]
        asks = [
            OrderBookLevel(price=Decimal(str(p)), quantity=Decimal(str(q)))
            for p, q in data.get("asks", [])
            if Decimal(str(q)) > 0
        ]
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        return OrderBookSnapshot(
            symbol=sym,
            bids=bids,
            asks=asks,
            fetched_at_utc=datetime.now(timezone.utc),
            source_url=src,
        )

    def fetch_verified_market_bundle(
        self,
        symbol: str,
        interval: str,
        *,
        candle_limit: int = 120,
        allow_stablecoin_conversion: bool = False,
    ) -> MarketBundle:
        """
        Tek bir analiz adımı için:
        1. Uyku dönüşünü kontrol eder,
        2. Sunucu saatini alır,
        3. Sembolün aktif Spot (USDT veya USDC) kurallarını doğrular,
        4. Yalnızca kapanmış mumları ve canlı emir defterini çeker,
        5. Tazelik ve bütünlük kontrolünden geçirir.
        """
        sym = symbol.strip().upper()
        inferred_quote = "USDT" if sym.endswith("USDT") else "USDC"
        wake_gap = self.detect_windows_wake_gap()

        time_payload, src_url = self._get_json("/api/v3/time")
        server_time_ms = int(time_payload["serverTime"])

        spot_pairs = self.discover_spot_pairs(
            quote_asset=inferred_quote,
            allow_stablecoin_conversion=allow_stablecoin_conversion,
            force_refresh=wake_gap,
        )
        if sym not in spot_pairs:
            # Belki stabilcoin dönüşüm korumasından dolayı hariç tutuldu; nedenini açıkla
            all_cached = self._exchange_info_cache
            if sym in all_cached and all_cached[sym].base_asset in STABLECOIN_OR_FIAT_ASSETS:
                raise MarketDataError(
                    f"{sym} bir stabilcoin/itibari para dönüşüm çiftidir. "
                    f"{inferred_quote} izinsiz başka varlığa dönüştürülmez."
                )
            raise MarketDataError(
                f"{sym} çifti aktif Binance Spot {inferred_quote} listesinde bulunamadı veya işleme kapalı."
            )

        filters = spot_pairs[sym]
        raw_klines, kline_src = self._get_json(
            "/api/v3/klines",
            params={"symbol": sym, "interval": interval, "limit": candle_limit},
        )
        closed_candles = extract_closed_candles(raw_klines, server_time_ms=server_time_ms)
        order_book = self.fetch_order_book(sym, depth_limit=20)

        validate_candles_and_book(
            closed_candles=closed_candles,
            order_book=order_book,
            interval=interval,
            server_time_ms=server_time_ms,
            max_data_age_seconds=self.config.max_data_age_seconds,
        )

        return MarketBundle(
            symbol=sym,
            interval=interval,
            filters=filters,
            closed_candles=closed_candles,
            order_book=order_book,
            server_time_ms=server_time_ms,
            fetched_at_utc=datetime.now(timezone.utc),
            source_label=f"{kline_src} (Herkese Açık Spot Veri)",
            wake_gap_detected=wake_gap,
            used_weight_1m=self.last_used_weight_1m,
        )
