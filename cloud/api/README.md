# TradeCebrail 7/24 Bulut API & Zamanlayıcı Motoru (`cloud/api`)

Bu servis Google Cloud Run (`europe-west1`, `--min-instances=0`, `--max-instances=1`) üzerinde çalışır ve iki görevi tek çatı altında güvenli biçimde birleştirir:

1. **7/24 Bulut Alım-Satım Döngüsü (`POST /api/cron/tick`):**
   - Cloud Scheduler (`tradecebrail-bot-tick`, `* * * * *`) her dakika `X-Cron-Secret` başlığıyla bu uç noktayı tetikler.
   - Bilgisayarınız kapalı olsa dahi 10 büyük USDT paritesini (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `ADAUSDT`, `AVAXUSDT`, `DOGEUSDT`, `LINKUSDT`, `SUIUSDT`) tarar, açık pozisyonların Zarar-Kes / Kâr-Al / Likidasyon seviyelerini korur ve uygun sinyallerde Gemini AI + risk süzgecinden geçirerek işlem yapar.
   - Her işlem ve ayar değişikliği sonunda `/tmp` içindeki SQLite veritabanı `sqlite3.Connection.backup` ile tek parça tutarlı dosyaya dönüştürülüp özel GCS bucket'ına (`gs://tradecebrail-bot-state-873885134851/sanal_sistem.db`) şifreli olarak yedeklenir.

2. **Sahip Doğrulamalı Web Kontrol Paneli API'si (`/api/dashboard`, `/api/settings`, `/api/tick`, `/api/close-positions`, `/api/reset-paper`):**
   - Firebase Google Sign-In ID Token doğrulaması zorunludur ve yalnızca `cebrailkara@gmail.com` (`email_verified=true`) kabul edilir.
   - Yalnızca `https://tradecebrail.web.app` ve `https://tradecebrail.firebaseapp.com` CORS kökenlerine izin verilir.
   - Tüm hata ve karar metinleri `redact_secrets` süzgecinden geçirilir; hiçbir API anahtarı veya gizli bilgi tarayıcıya ya da loglara sızmaz.
