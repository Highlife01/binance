"""Güvenlik kilidi: Canlı emir engelleme, localhost doğrulaması ve AI veri hijyeni."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

# Bu sürümde canlı emir gönderme yeteneği mimari olarak kapalıdır.
LIVE_TRADING_CAPABLE: bool = False

ALLOWED_BINANCE_HOSTS: frozenset[str] = frozenset(
    {
        "data-api.binance.vision",
        "api.binance.com",
        "api-gcp.binance.com",
        "api1.binance.com",
        "api2.binance.com",
        "api3.binance.com",
        "api4.binance.com",
    }
)

# Yalnızca kimlik doğrulaması gerektirmeyen herkese açık salt-okunur GET uç noktaları
ALLOWED_PUBLIC_GET_PATHS: frozenset[str] = frozenset(
    {
        "/api/v3/ping",
        "/api/v3/time",
        "/api/v3/exchangeInfo",
        "/api/v3/klines",
        "/api/v3/ticker/bookTicker",
        "/api/v3/depth",
        "/api/v3/ticker/24hr",
    }
)

# Çift kilitli Canlı Spot modunda izin verilen TEK uç noktalar (Çekim, Marjin, Vadeli KESİNLİKLE YASAK)
ALLOWED_LIVE_SPOT_ENDPOINTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/v3/time"),
        ("GET", "/api/v3/account"),
        ("POST", "/api/v3/order/test"),
        ("POST", "/api/v3/order"),
        ("GET", "/api/v3/openOrders"),
        ("DELETE", "/api/v3/order"),
    }
)

# USDC'nin izinsiz başka stabilcoin veya itibari para çiftine dönüştürülmesini önleyen liste
STABLECOIN_OR_FIAT_ASSETS: frozenset[str] = frozenset(
    {"USDT", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "TRY", "BRL", "GBP", "AEUR"}
)

# Olası gizli anahtar / kimlik bilgisi sızıntı kalıpları
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|authorization)\s*[:=]\s*\S+"),
    re.compile(r"\b[A-Za-z0-9]{64}\b"),
)

# Harici güvenilmeyen metinlerdeki komut enjeksiyonu (prompt injection) kalıpları
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions?"),
    re.compile(r"(?i)(system\s*prompt|developer\s*message|execute\s+order|place\s+order)"),
    re.compile(r"(?i)(önceki\s+tüm\s+talimatları\s+unut|risk\s+sınırını\s+değiştir|emir\s+gönder)"),
)


class LiveOrderBlockedError(PermissionError):
    """Sanal modda gerçek emir veya kimlik doğrulamalı uç nokta çağrısı denendiğinde fırlatılır."""


class UnsafeHostBindingError(PermissionError):
    """Arayüz localhost (127.0.0.1 / localhost) dışına açılmak istendiğinde fırlatılır."""


def assert_public_readonly_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
) -> None:
    """
    Her HTTP isteğinden önce çalışarak gerçek emir gönderimini mimari olarak imkânsız kılar.
    - Yalnızca GET metoduna izin verir.
    - Yalnızca beyaz listedeki Binance herkese açık veri yollarına izin verir.
    - `X-MBX-APIKEY` veya `Authorization` başlığı taşıyan istekleri reddeder.
    """
    normalized_method = method.strip().upper()
    if normalized_method != "GET":
        raise LiveOrderBlockedError(
            f"Güvenlik Kilidi: Sanal deneme modunda '{normalized_method}' HTTP metoduna izin verilmez. "
            "Gerçek emir gönderme yeteneği kapalıdır."
        )

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise LiveOrderBlockedError(f"Güvenlik Kilidi: Yalnızca HTTPS veri kaynağı kullanılabilir ({url}).")

    if parsed.hostname not in ALLOWED_BINANCE_HOSTS:
        raise LiveOrderBlockedError(
            f"Güvenlik Kilidi: İzin verilmeyen sunucu adresi: '{parsed.hostname}'."
        )

    if parsed.path not in ALLOWED_PUBLIC_GET_PATHS:
        raise LiveOrderBlockedError(
            f"Güvenlik Kilidi: '{parsed.path}' uç noktası salt-okunur herkese açık veri listesinde değil! "
            "Canlı emir ve hesap uç noktaları kesinlikle engellenmiştir."
        )

    if headers:
        lowered = {k.lower() for k in headers}
        forbidden_headers = lowered.intersection({"x-mbx-apikey", "authorization"})
        if forbidden_headers:
            raise LiveOrderBlockedError(
                "Güvenlik Kilidi: Binance hesap kimlik bilgisi başlığı tespit edildi ve engellendi."
            )


def assert_live_spot_request(
    method: str,
    url: str,
    *,
    config_live_enabled: bool,
    ui_live_confirmed: bool,
) -> None:
    """
    Canlı Spot emir modülü için 5 katmanlı güvenlik kilidi:
    1. `.env` içinde `ENABLE_LIVE_TRADING=true` olmalı,
    2. Arayüzde kullanıcı açık onay kilidini (`ui_live_confirmed=True`) açmış olmalı,
    3. Yalnızca HTTPS ve resmi Binance sunucuları kullanılabilir,
    4. Yalnızca kaldıraçsız Spot uç noktalarına izin verilir; çekim/marjin/futures kesinlikle yasaktır.
    """
    if not config_live_enabled:
        raise LiveOrderBlockedError(
            "Güvenlik Kilidi: .env dosyasında ENABLE_LIVE_TRADING=true ayarlanmadan canlı işlem yapılamaz."
        )
    if not ui_live_confirmed:
        raise LiveOrderBlockedError(
            "Güvenlik Kilidi: Paneldeki 'Canlı Spot Emir Kilidi' kullanıcı tarafından onaylanmadan "
            "gerçek emir gönderilemez."
        )

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_BINANCE_HOSTS:
        raise LiveOrderBlockedError(
            f"Güvenlik Kilidi: Geçersiz canlı sunucu adresi: '{url}'."
        )

    norm_method = method.strip().upper()
    if (norm_method, parsed.path) not in ALLOWED_LIVE_SPOT_ENDPOINTS:
        raise LiveOrderBlockedError(
            f"Güvenlik Kilidi: ({norm_method} {parsed.path}) uç noktası yasaktır! "
            "Para çekme (withdraw), marjin ve vadeli (futures) işlemler mimari olarak engellenmiştir."
        )


def assert_localhost_only(host: str) -> None:
    """Web arayüzünün yalnızca yerel makineden erişilebilir olmasını zorunlu kılar."""
    if host.strip() not in {"127.0.0.1", "localhost", "::1"}:
        raise UnsafeHostBindingError(
            f"Güvenlik hatası: Arayüz yalnızca localhost (127.0.0.1) üzerinde çalışabilir, '{host}' reddedildi."
        )


def redact_secrets(text: str) -> str:
    """Metin içinde kazara bulunabilecek API anahtarı / sır kalıplarını maskeler."""
    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[GIZLI_BILGI_MASKELENDI]", cleaned)
    return cleaned


def sanitize_untrusted_external_text(raw_text: str, max_len: int = 280) -> str:
    """
    Harici haber veya piyasa notu metinlerini güvenilmeyen veri kabul eder;
    komut enjeksiyonu kalıplarını ve gizli dizgileri temizler.
    """
    if not raw_text:
        return ""
    cleaned = redact_secrets(raw_text)
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[GUVENILMEYEN_TALIMAT_FILTRELENDI]", cleaned)
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned).strip()
    return cleaned[:max_len]


def build_safe_ai_market_payload(
    *,
    symbol: str,
    interval: str,
    last_closed_candle_iso: str,
    close_price: str,
    spread_pct: str,
    ema_fast: str,
    ema_slow: str,
    rsi_14: str,
    macd_line: str,
    macd_signal: str,
    macd_hist: str,
    atr_14: str,
    volume_ratio: str,
    recent_returns_pct: list[str],
    has_open_position: bool,
    rule_baseline_action: str,
    external_notes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Modele gönderilecek veriyi yalnızca anonim piyasa istatistikleriyle sınırlar.
    Kişisel bilgi, bakiye detayı veya API anahtarı kesinlikle eklenmez.
    """
    payload: dict[str, Any] = {
        "parite": symbol,
        "zaman_dilimi": interval,
        "son_kapanmis_mum_zamani_utc": last_closed_candle_iso,
        "kapanis_fiyati_usdc": close_price,
        "anlik_spread_yuzde": spread_pct,
        "teknik_ozet": {
            "ema_9": ema_fast,
            "ema_21": ema_slow,
            "rsi_14": rsi_14,
            "macd": macd_line,
            "macd_sinyal": macd_signal,
            "macd_histogram": macd_hist,
            "atr_14": atr_14,
            "hacim_orani_20_ort": volume_ratio,
            "son_5_kapanmis_mum_degisim_yuzde": recent_returns_pct[:5],
            "kural_tabanli_referans_sinyal": rule_baseline_action,
        },
        "sanal_acik_pozisyon_var_mi": bool(has_open_position),
    }
    if external_notes:
        payload["guvenilmeyen_harici_veri_notlari"] = [
            sanitize_untrusted_external_text(item) for item in external_notes[:3] if item.strip()
        ]
    return payload
