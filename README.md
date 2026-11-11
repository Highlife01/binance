# Binance Global USDC Spot — Sanal Deneme & Çift Kilitli Canlı Karar Destek Sistemi

Windows 10/11 bilgisayarınızda yerel olarak (`127.0.0.1`) çalışan, Türkçe arayüzlü, **86 USDC** başlangıç sanal bakiyesiyle önce risksiz deneme yapan ve isteğe bağlı **5 Katmanlı Güvenlik Kilitli Canlı Spot** modülünü barındıran modüler Python işlem sistemidir.

> **ÖNEMLİ UYARI (Yatırım Tavsiyesi Değildir):**
> - Hiçbir algoritma veya yapay zekâ modeli **kâr garantisi ya da kesin kazanç sağlayamaz**.
> - Bilgisayarınız kapandığında, internet koptuğunda veya Windows uyku moduna geçtiğinde **sistem durur**; açık pozisyonları yerel yazılımdan takip edemez.
> - **Önce mutlaka Sanal Deneme modunda** sistemi, komisyon yükünü ve bağlantı kararlılığını gözlemleyin; sanal modda doğrulama yapmadan gerçek parayla işlem açmayın.

---

## 1. Hızlı Kurulum ve Çalıştırma (Windows 10 / 11)

Yönetici yetkisi gerekmez. Windows Güvenlik Duvarı veya antivirüs yazılımınızı **kapatmayın**.

1. Klasördeki **`kurulum.bat`** dosyasına çift tıklayın.
   - Otomatik olarak `.venv` sanal ortamını oluşturur, sabitlenmiş kütüphaneleri yükler, `.env` dosyasını hazırlar ve güvenlik/doğrulama testlerini çalıştırır.
2. Kurulum tamamlandıktan sonra **`baslat.bat`** dosyasına çift tıklayın.
   - Tarayıcınızda yalnızca sizin bilgisayarınızdan erişilebilen `http://127.0.0.1:8501` adresinde Türkçe kontrol paneli açılır.
3. Testleri istediğiniz zaman doğrulamak için **`testleri_calistir.bat`** dosyasına çift tıklayabilir veya terminalde şu komutu çalıştırabilirsiniz:
   ```powershell
   .\.venv\Scripts\pytest.exe
   ```

---

## 2. Çalışma Modları: Önce Sanal Deneme, Sonra Çift Kilitli Canlı Spot

### A) Varsayılan Mod: Sanal Deneme (`PAPER` — Önerilen Başlangıç)
- **Binance API anahtarı istemez**; yalnızca herkese açık verileri okur.
- Başlangıç sanal kasası **86.00 USDC** olarak başlar; gerçek Binance hesabınıza **kesinlikle dokunmaz**.
- Alışta satış defteri (`asks`), satışta alış defteri (`bids`) likiditesini yürüyerek gerçekçi spread, kayma ve komisyon (`%0.10` varsayılan, değiştirilebilir) uygular.
- Limit emirlerde fiyata yalnızca dokunulmasını %100 dolum saymaz; kuyruk önceliğini yansıtan `%40` kısmi dolum veya gerçekleşmeme (`EXPIRED_UNFILLED`) modelini kullanır.

### B) Gerçek Parayla İşlem: Çift Kilitli Canlı Spot Modu (`LIVE_SPOT`)
Gerçek **86 USDC** Spot bakiyenizle işlem yapmak istediğinizde sistem sizi kazara emir gönderimine karşı **5 güvenlik kilidiyle** korur:

1. **Önce Bağlantıyı Doğrulayın:** Sanal modda `Analiz Adımını Şimdi Çalıştır` düğmesine bastığınızda `WinError 10054` ağ engeli almadığınızdan emin olun (`SORUN_GIDERME.md` dosyasına bakın).
2. **Güvenli Binance API Anahtarı Oluşturun:**
   - Binance Global -> *API Management* ekranından yeni bir anahtar oluşturun.
   - **Enable Reading (Okuma):** Açık
   - **Enable Spot & Margin Trading:** Açık (Sistem yalnızca kaldıraçsız Spot kullanır; marjin uç noktaları kodda engellidir)
   - **Enable Withdrawals (Para Çekme):** **KESİNLİKLE KAPALI!**
   - **Enable Futures (Vadeli):** **KAPALI!**
   - Güvenlik için mümkünse kendi sabit IP adresinizi tanımlayın.
3. **Anahtarları Yalnızca Bilgisayarınızdaki `.env` Dosyasına Yazın (Sohbete Yazmayın!):**
   Klasördeki `.env` dosyasını Not Defteri ile açıp şu satırları düzenleyin:
   ```env
   ENABLE_LIVE_TRADING=true
   BINANCE_API_KEY=buraya_kendi_anahtariniz
   BINANCE_API_SECRET=buraya_kendi_gizli_anahtariniz
   ```
   *(Not: `.env` dosyası `.gitignore` içinde korumalıdır; Git deposuna gönderilmez.)*
4. **Paneldeki İki Aşamalı Onay Kilidini Açın:**
   Sol menüden **`Gerçek Para — Canlı Spot`** seçeneğini işaretleyin, risk onay kutusunu işaretleyin ve **`Canlı Spot Emir Kilidini Aç`** anahtarını etkinleştirin.
5. **Ön-Test Koruması (`/api/v3/order/test`):**
   Her gerçek emirden önce sistem sunucu saatini senkronize eder, gerçek serbest USDC bakiyenizi okur, **10 USDC** pozisyon sınırı ve **2 USDC** günlük zarar eşiğini denetler, önce Binance `order/test` uç noktasında doğrular ve yalnızca sorunsuzsa gerçek Spot emri iletir. Sanal ve canlı kayıtlar SQLite içinde ayrı izlerde (`MAIN` vs `LIVE_SPOT`) tutulur; birbirine karışmaz.

---

## 3. İsteğe Bağlı Yapay Zekâ (Gemini) Kurulumu

- `.env` dosyasında `GEMINI_API_KEY=` boş bırakılırsa sistem dürüstçe **`Yapay Zekâ Kapalı / Kural Tabanlı Deneme`** etiketiyle çalışır; matematiksel EMA/RSI/MACD kurallarını yapay zekâ gibi tanıtmaz.
- Etkinleştirmek için Google AI Studio üzerinden aldığınız anahtarı `.env` içindeki `GEMINI_API_KEY=` satırına yazmanız yeterlidir.
- Modele yalnızca anonim piyasa istatistikleri gönderilir; API anahtarlarınız veya kişisel bilgileriniz asla gönderilmez.
- Model doğrudan emir gönderemez, kod çalıştıramaz, risk sınırlarını değiştiremez; bozuk veya geciken yanıtlarda karar otomatik olarak **`BEKLE`** olur.
- Aynı kapanmış mum için önbellek (`SQLite Cache`), çağrı aralığı ve günlük çağrı kotası uygulanır.

---

## 4. Teknik Doğrulama Özeti ve Bilinen Sınırlamalar

- **Doğrulanan Binance Uç Noktaları:** Türkiye ağlarıyla doğrudan uyumlu resmî `https://api.binance.me` birincil uç noktası (ve yedek olarak `data-api.binance.vision` / `api.binance.com`) üzerinden `GET /api/v3/time`, `/api/v3/exchangeInfo`, `/api/v3/klines`, `/api/v3/depth`, `/api/v3/ticker/24hr` ile Canlı Spot modunda HMAC-SHA256 imzalı `GET /api/v3/account`, `POST /api/v3/order/test`, `POST /api/v3/order`.
- **Doğrulanan Gemini SDK ve Modeller:** Resmî `google-genai==3.2.0` paketi (`google-generativeai` eski paketi kullanılmamıştır). Desteklenen güncel model kodları: `gemini-3.5-flash-lite` (varsayılan), `gemini-3.8-flash`, `gemini-2.5-flash`.
- **Doğrulanamayan Dinamik Ücretler:** Kullanıcının kendi Binance VIP/promosyon komisyon oranı (ör. dönemsel USDC kampanyaları) herkese açık uç noktadan bilinemeyeceği için varsayılan Spot komisyonu `%0.10` alınmış ve panelden değiştirilebilir yapılmıştır. Gemini API anlık fatura tarifesi çalışma anında API'den çekilemediği için yapay zekâ maliyeti kesin fatura olarak değil, birim token varsayımına dayalı **tahmini gösterge** olarak sunulur.
- **Küçük Bakiye (86 USDC) Sınırlaması:** 10 USDC'lik işlemlerde gidiş-dönüş komisyon ve makas maliyeti yaklaşık `%0.26` başabaş eşiği yaratır. Çok kısa periyotlarda (1m) sık işlem yapmak bakiyeyi komisyonla eritir; `15m` veya `1h` gibi periyotlar tercih edilmelidir.
- **Yerel Çalışma Sınırlaması:** Bilgisayar kapandığında, yeniden başladığında veya Windows uyku moduna geçtiğinde bot çalışmaz. Risk sınırları ani fiyat boşluklarında (gap) veya likidite kurumalarında kesin kayıp garantisi veremez.


## Kesintisiz Arka Plan Botu (Sürekli Alım-Satım)

Panelin "Otomatik Döngü" özelliği yalnızca tarayıcı sekmesi açıkken çalışır. Tarayıcıdan bağımsız,
kesintisiz çalışma için:

| Dosya | Ne yapar |
|---|---|
| `bot_sanal.bat` | **Sanal Deneme (`2x` Kaldıraç)** — gerçek bakiyeye dokunmadan **Hızlı Aktif (`30 sn` döngü, `1m` mum, `2x` kaldıraç)** profili ile 10 coini tarar, aynı anda en fazla 5 farklı coinde pozisyon tutar ve anlık K/Z + likidasyon mesafesi gösterir |
| `bot_canli.bat` | **Gerçek Para (`1x` Spot)** — `.env` kilidi + konsolda `EVET` onayı ile 10 coini **30 sn döngüyle** tarar; aynı anda en fazla 5 farklı coinde (`5 × 10 USDT = maks 50 USDT`) borçsuz canlı Spot alım-satım yapar |
| `bot_marjin.bat` | **Gerçek Çapraz Marjin (`2x` Kaldıraç)** — 3 güvenlik kilidi + konsolda `EVET` onayı ile yalnızca Cross Margin cüzdanınıza ayırdığınız teminat üzerinden `MARGIN_BUY` / `AUTO_REPAY` ile `2x` kaldıraçlı alım-satım yapar (`marginLevel >= 1.50` korumalı) |

- Çoklu coin taraması hem **Sanal** hem **Canlı Spot** modunda ve web panelinde aktiftir: her 30 saniyede önce tüm açık pozisyonların zarar-kes / kâr-al / SAT çıkışı denetlenir, ardından boş slot varsa diğer coinler taranır. Aynı coine üst üste ekleme (Martingale) kesinlikle yapılmaz.

- İşlem profilleri (`src/binance_ai_bot/profiles.py`):
  - `HIZLI_30SN` *(varsayılan)*: 30 sn döngü, 1dk mum, EMA(5/13), gevşetilmiş hacim (0.15x), aynı anda en fazla 5 farklı coin, günde en fazla 60 pozisyon, %0.35 zarar-kes / %0.45 kâr-al + hızlı trend çıkışı.
  - `AKTIF`: 30 sn döngü, 5dk mum, aynı anda en fazla 3 farklı coin, günde en fazla 12 pozisyon, %0.8 zarar-kes / %1.2 kâr-al.
  - `TEMKINLI`: 60 sn döngü, 15dk mum, aynı anda 1 coin, günde en fazla 3 pozisyon, %2.5 zarar-kes / %4.5 kâr-al.
- Hiçbir profil işlem başına pozisyon sınırını (10 USDT/USDC) ve günlük toplam zarar eşiğini (2.00 USDT/USDC) gevşetmez.
- Aynı anda tek bot çalışabilir; bot çalışırken panelin otomatik döngüsü ve analiz düğmesi kilitlenir.
- Bot çalıştığı sürece Windows uykuya geçmez. Durdurmak için `Ctrl+C`, pencereyi kapatmak veya paneldeki **Arka Plan Botunu Durdur**.
- Kayıtlar: `logs/bot_YYYY-MM-DD.log`.
- Daha sık işlem daha fazla komisyon demektir; kâr garantisi yoktur.

## Emir Güvenliği ve Sonuçların Doğrulanması

- Panel, arka plan botu ve manuel kapatma aynı portföy emir kilidini paylaşır. Sanal işlemde nakit, pozisyon, emir ve sinyal kaydı birlikte tamamlanır; kayıt hatasında tamamı geri alınır.
- Canlı emir gönderilmeden önce sabit kimlikli emir niyeti kalıcı olarak kaydedilir. Yanıt kaybolursa veya program kapanırsa aynı emir yeniden gönderilmez: `GET /api/v3/order` ve gerektiğinde `GET /api/v3/myTrades` ile borsadaki durum doğrulanır. Belirsizlik çözülene kadar yeni alım bekletilir. Bu davranış Binance'in [zaman aşımı açıklamasıyla](https://developers.binance.com/en/docs/products/spot/rest-api) uyumludur.
- Kısmi satış yalnızca gerçekleşen miktarı kapatır. Kalan miktar ve minimum satış tutarının altındaki küçük bakiye kayıtta korunur. Baz varlıktan kesilen alış komisyonu iki kez maliyete eklenmez.
- Canlı alışlarda motor, onaylı para tutarını `quoteOrderQty` olarak gönderir. Emir sonrası kullanılabilir nakit yeniden hesaplanmak yerine borsadan okunur; yeniden başlatma aynı emri bakiyeden ikinci kez düşmez.
- BNB gibi üçüncü bir varlıkta kesilen komisyonun USDT/USDC değeri doğrulanamıyorsa ham varlık ve tutar saklanır, kâr/zararın geçici olduğu belirtilir ve yeni canlı alımlar bekletilir. `live_unvalued_commissions` kaydını silmek bir değerleme çözümü değildir; doğrulanmış değerleme olmadan kilit kaldırılmamalıdır. Mevcut pozisyonların koruyucu satışları devam edebilir.
- Günlük zarar ve yeni alım kontrollerinde bütün açık coinlerin güncel fiyatı kullanılır. Bir fiyat alınamazsa yeni alım yapılmaz. Mevcut coinin geçerli fiyatla zarar-kes/kâr-al denetimi devam eder. Günlük otomatik durdurma ertesi İstanbul gününde kaldırılır; kullanıcı durdurması ayrıca korunur.
- Her portföy izi tek para birimine bağlanır. USDT geçmişi USDC bakiyesiyle birleştirilemez; diğer para birimi için ayrı veritabanı kullanılır. Eski geçmişte iki para birimi zaten karışmışsa yeni alım engellenir. Modu bilinmeyen eski grafik/karar satırları yeni sanal veya canlı eğriye eklenmez.
- Durdurma isteği her coin öncesinde ve canlı emrin gönderiminden hemen önce kontrol edilir. Uzun hata beklemeleri 5 saniyelik aralıklarla kesilebilir; botun canlılık bilgisi bu sırada güncellenir. Başlamış bir ağ isteğinin sonucu ayrıca doğrulanır.

Geçmiş test artık seçilen profilin aynı strateji parametrelerini, günlük giriş/zarar sınırlarını ve gerçek veri seçildiğinde paritenin güncel borsa filtrelerini kullanır. Mum içindeki sıralama bilinmediğinde zarar-kes önce gerçekleşmiş sayılır; fiyat boşluğunda erişilemeyen hedef fiyattan dolum yazılmaz. Bu, tek parite için muhafazakâr bir OHLC modelidir; 30 saniyelik gerçek piyasa akışının veya eşzamanlı çoklu coin portföyünün birebir tekrarı değildir. Güncel borsa filtreleri geçmişteki kurallarla farklı olabilir. Sentetik seri yalnızca motor gösterimidir. Kaldıraçlı geçmiş test bu Spot doğrulama yolunda desteklenmez.

Test paketi gerçek Binance/Gemini hesabına bağlanmadan sahte yanıtlar ve geçici veritabanlarıyla çalışır:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Panel testleri ekran bileşenlerini doğrular; Windows test ortamındaki yerel socket kurulumundan bağımsız bir çizim adaptörü kullanır. Gerçek sunucu bağlantısını veya gerçek emir gönderimini doğruladıkları anlamına gelmez. Kod güncellemeleri çalışan eski bot sürecine kendiliğinden yüklenmez; güncel kodu kullanmak için süreç yeniden başlatılmalıdır.
