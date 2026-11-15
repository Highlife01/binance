"""
NautilusTrader Binance adaptörü ve RiskEngine desenlerinden uyarlanan ortak
imzalı HTTP taşıma, saat senkronizasyonu, hız sınırı (rate-limit) ve hata
sınıflandırma yardımcıları.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlencode

import httpx

from .market_data import _friendly_network_error
from .security import redact_secrets

# Binance HTTP ve API hata kodları (NautilusTrader binance/common/error.rs uyumlu)
BINANCE_RATE_LIMIT_STATUS = 429
BINANCE_IP_BAN_STATUS = 418
RETRYABLE_VENUE_CODES = frozenset({-1003, -1015, -1021})
AMBIGUOUS_VENUE_CODES = frozenset({-1000, -1001, -1006, -1007})
CLOCK_SYNC_TTL_SEC = 60.0
MAX_CLOCK_DRIFT_MS = 30_000


@dataclass
class BinanceTransportState:
    """Yürütücü başına sunucu saati sapması ve hız sınırı bekleme durumu."""

    clock_offset_ms: int = 0
    clock_synced_monotonic: float = 0.0
    cooldown_until_monotonic: float = 0.0
    last_used_weight_1m: int | None = None


def parse_binance_api_code(resp: httpx.Response) -> int | None:
    """Binance JSON hata gövdesinden tamsayı `code` alanını güvenli şekilde okur."""
    try:
        raw_code = resp.json().get("code")
        return int(raw_code) if raw_code is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def is_ambiguous_binance_failure(status_code: int, api_code: int | None, body_text: str) -> bool:
    """
    Emrin Binance tarafında işlenip işlenmediğinin belirsiz olduğu durumları
    NautilusTrader `classify_spot_http_failure` mantığıyla tespit eder.
    """
    return (
        status_code >= 500
        or status_code == 408
        or api_code in AMBIGUOUS_VENUE_CODES
        or (api_code == -2010 and "duplicate" in body_text.lower())
    )


def is_retryable_binance_failure(status_code: int, api_code: int | None) -> bool:
    """Geçici hız sınırı, IP bekleme veya sunucu hatası olup olmadığını döner."""
    return (
        status_code in (BINANCE_RATE_LIMIT_STATUS, BINANCE_IP_BAN_STATUS)
        or status_code >= 500
        or api_code in RETRYABLE_VENUE_CODES
    )


def parse_retry_after_sec(resp: httpx.Response, default_sec: int = 15) -> int:
    """`Retry-After` başlığını [5, 300] saniye aralığında güvenli şekilde çözer."""
    raw = resp.headers.get("Retry-After")
    try:
        parsed = int(str(raw).strip()) if raw is not None else default_sec
    except ValueError:
        parsed = default_sec
    return min(max(parsed, 5), 300)


def sign_hmac_sha256(secret: str, params: dict[str, Any]) -> str:
    """Binance HMAC-SHA256 imzalı sorgu dizgesi üretir (çoklu dizi parametreleri için `doseq=True`)."""
    query_str = urlencode(params, doseq=True)
    signature = hmac.new(
        secret.encode("utf-8"),
        query_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{query_str}&signature={signature}"


def execute_signed_binance_request(
    *,
    state: BinanceTransportState,
    base_url: str,
    method: str,
    path: str,
    api_key: str,
    api_secret: str,
    recv_window_ms: int,
    timeout_sec: float,
    params: dict[str, Any] | None,
    external_client: httpx.Client | None,
    security_assert_fn: Callable[..., None],
    error_cls: type[Exception],
    error_prefix: str,
    ambiguous_post_paths: frozenset[str],
    body_snippet_limit: int = 240,
    special_error_mapper: Callable[[int, int | None, str], Exception | None] | None = None,
) -> dict[str, Any] | list[Any]:
    """
    Ortak imzalı Binance Spot / Cross Margin HTTP isteği yürütücüsü.

    - Cooldown (429/418) aktifse IP ban riskini önlemek için isteği hemen reddeder.
    - Sunucu saat sapmasını 60 saniye önbellekler; `-1021` hatasında otomatik yeniler.
    - `X-MBX-USED-WEIGHT-1M` başlığını izler ve `429`/`418` yanıtlarında cooldown uygular.
    - Ağ veya sunucu hatalarında emir gönderim yolları için `ambiguous=True` işaretler.
    """
    now_mono = time.monotonic()
    if now_mono < state.cooldown_until_monotonic:
        wait_sec = max(1, int(state.cooldown_until_monotonic - now_mono) + 1)
        raise error_cls(
            f"Binance hız sınırı koruması aktif; {wait_sec} sn sonra tekrar deneyin.",
            retryable=True,
        )

    method_upper = method.upper()
    full_url = f"{base_url.rstrip('/')}{path}"
    time_url = f"{base_url.rstrip('/')}/api/v3/time"
    headers = {"X-MBX-APIKEY": api_key}

    security_assert_fn(method_upper, full_url)

    def _run_with_client(client: httpx.Client) -> dict[str, Any] | list[Any]:
        def _sync_timestamp(force_sync: bool = False) -> int:
            mono = time.monotonic()
            if (
                not force_sync
                and state.clock_synced_monotonic > 0.0
                and (mono - state.clock_synced_monotonic) < CLOCK_SYNC_TTL_SEC
            ):
                return int(time.time() * 1000) + state.clock_offset_ms

            security_assert_fn("GET", time_url)
            try:
                t_resp = client.get(time_url)
                server_time_ms = int(t_resp.json()["serverTime"])
            except Exception as exc:
                raise error_cls(_friendly_network_error(exc, time_url)) from exc

            local_ms = int(time.time() * 1000)
            drift_ms = abs(server_time_ms - local_ms)
            if drift_ms > MAX_CLOCK_DRIFT_MS:
                raise error_cls(
                    f"Windows sistem saatiniz Binance sunucusundan {drift_ms // 1000} sn sapmış durumda. "
                    "Lütfen Windows Tarih ve Saat ayarlarından 'Şimdi eşitle' yapın."
                )

            state.clock_offset_ms = server_time_ms - local_ms
            state.clock_synced_monotonic = time.monotonic()
            return server_time_ms

        is_idempotent_get = method_upper == "GET"
        is_ambiguous_post = method_upper == "POST" and path in ambiguous_post_paths
        attempts = 2 if is_idempotent_get else 1

        for attempt in range(attempts):
            server_time_ms = _sync_timestamp(force_sync=(attempt > 0))
            payload = dict(params or {})
            payload["recvWindow"] = recv_window_ms
            payload["timestamp"] = server_time_ms
            signed_query = sign_hmac_sha256(api_secret, payload)
            req_url = f"{full_url}?{signed_query}"

            try:
                resp = client.request(method_upper, req_url, headers=headers)
            except Exception as exc:
                raise error_cls(
                    _friendly_network_error(exc, full_url),
                    ambiguous=is_ambiguous_post,
                    retryable=not is_ambiguous_post,
                ) from exc

            weight_hdr = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            if weight_hdr and weight_hdr.isdigit():
                state.last_used_weight_1m = int(weight_hdr)

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise error_cls(
                        "Binance yanıtı çözümlenemedi; emir sonucu doğrulanmalıdır.",
                        ambiguous=is_ambiguous_post,
                    ) from exc

            api_code = parse_binance_api_code(resp)
            safe_body = redact_secrets(resp.text[:body_snippet_limit])

            if resp.status_code in (BINANCE_RATE_LIMIT_STATUS, BINANCE_IP_BAN_STATUS) or api_code in (-1003, -1015):
                default_wait = 60 if resp.status_code == BINANCE_IP_BAN_STATUS else 15
                backoff = parse_retry_after_sec(resp, default_sec=default_wait)
                state.cooldown_until_monotonic = time.monotonic() + backoff
                raise error_cls(
                    f"{error_prefix} hız sınırı (HTTP {resp.status_code}): {backoff} sn bekleme uygulandı. {safe_body}",
                    api_code=api_code,
                    status_code=resp.status_code,
                    retryable=True,
                )

            if api_code == -1021:
                state.clock_synced_monotonic = 0.0
                if attempt + 1 < attempts:
                    continue

            if special_error_mapper is not None:
                mapped = special_error_mapper(resp.status_code, api_code, safe_body)
                if mapped is not None:
                    raise mapped

            if api_code == -1013 and "NOTIONAL" in resp.text.upper():
                raise error_cls(
                    f"{error_prefix}: Emir tutarı Binance minimum işlem tutarı (NOTIONAL) filtresinin altında (-1013). "
                    "Küçük kalan bakiyeler borsa alt sınırından küçükse Binance uygulamasından BNB'ye dönüştürülebilir.",
                    api_code=api_code,
                    status_code=resp.status_code,
                )

            raise error_cls(
                f"{error_prefix} (HTTP {resp.status_code}): {safe_body}",
                api_code=api_code,
                status_code=resp.status_code,
                ambiguous=is_ambiguous_binance_failure(resp.status_code, api_code, resp.text),
                retryable=is_retryable_binance_failure(resp.status_code, api_code),
            )

        raise error_cls(f"{error_prefix}: İstek tamamlanamadı.")

    if external_client is not None:
        return _run_with_client(external_client)

    with httpx.Client(timeout=timeout_sec) as client:
        return _run_with_client(client)
