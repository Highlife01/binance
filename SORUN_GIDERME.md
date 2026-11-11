# Sorun Giderme Rehberi (Windows 10 / 11)

Bu rehber teknik bilgisi sınırlı kullanıcılar için en sık karşılaşılan durumları sade adımlarla açıklar. **Hiçbir adımda Windows Güvenlik Duvarını veya antivirüs programınızı kapatmanız gerekmez.**

---

## 1. `WinError 10054` / `Binance Global veri sunucusuna bağlantı sıfırlandı` Hatası
- **Nedeni:** Türkiye'deki bazı internet servis sağlayıcılarında `binance.com` ve `binance.vision` alan adlarına yönelik ağ düzeyinde bağlantı sıfırlaması uygulanabilmektedir.
- **Sistemin Güvenli Davranışı:** Program çökmez, hayalî işlem üretmez ve bakiyenizi korur.
- **Ne Yapabilirsiniz?**
  - Ev internetinizde standart DNS kaynaklı bir kısıtlama varsa, ağ/modem ayarlarınızda güvenilir genel DNS (ör. Cloudflare `1.1.1.1` veya Google `8.8.8.8`) kullanmayı deneyebilir veya mobil erişim noktanızla bağlantıyı test edebilirsiniz.
  - Bağlantı kesintisiz ve kararlı hale gelmeden **kesinlikle Canlı Spot modunu açmayın**.
  - Bağlantı yokken bile paneldeki **Kronolojik Doğrulama (Backtest)** sekmesinde *"Çevrimdışı Örnek Seri ile Doğrulama Motorunu Test Et"* düğmesiyle sistemin mantığını inceleyebilirsiniz.

---

## 2. `Minimum emir tutarı pozisyon üst sınırını aşıyor` Uyarısı
- **Nedeni:** Seçtiğiniz paritede Binance'in zorunlu kıldığı en düşük işlem tutarı (`minNotional`), sizin belirlediğiniz pozisyon üst sınırından (varsayılan `10.00 USDC`) yüksektir veya basamak yuvarlaması (`stepSize`) sonrası tutar sınırın altına düşmektedir.
- **Çözüm:** Sistem güvenliğiniz için sınırı kendiliğinden büyütmez. `BTCUSDC`, `ETHUSDC`, `SOLUSDC` gibi `minNotional` değeri `5 USDC` olan pariteleri seçebilir veya *Risk Ayarları* sekmesinden pozisyon üst sınırınızı bilinçli olarak (ör. `12 USDC`) güncelleyebilirsiniz.

---

## 3. `Windows sistem saatiniz Binance sunucusundan sapmış durumda (-1021)`
- **Nedeni:** Bilgisayarınızın saati birkaç saniye ileri veya geri kalmıştır.
- **Çözüm:**
  1. Windows görev çubuğundaki saate sağ tıklayıp **Tarih ve saati ayarla** menüsünü açın.
  2. **Saati otomatik olarak ayarla** seçeneğini açık konuma getirin.
  3. Alt kısımdaki **Şimdi eşitle (Sync now)** düğmesine basın.

---

## 4. Bilgisayar Ekranı Kapanınca Bot Duruyor
- **Nedeni:** Windows varsayılan olarak belirli bir süre sonra bilgisayarı **Uyku (Sleep)** moduna alır; uyku modunda tüm programlar ve internet bağlantısı duraklar.
- **Çözüm:**
  1. **Başlat -> Ayarlar -> Sistem -> Güç ve Pil (Power & Sleep)** bölümüne girin.
  2. Botu çalıştırdığınız süre boyunca *Ekran* kapanabilir ancak **Uyku (Sleep)** süresini **Hiçbir zaman (Never)** olarak ayarlayın.
  3. Sistem uyku dönüşünü otomatik algılar ve bayat veriyle emir açmamak için ilk döngüyü güvenli bekleme (`BEKLE`) olarak geçirir.

---

## 5. `kurulum.bat` Çalışırken `Python bulunamadı` Hatası
- **Çözüm:**
  1. Resmi [https://www.python.org/downloads/](https://www.python.org/downloads/) sayfasından **Python 3.12** Windows yükleyicisini indirin.
  2. Kurulumun ilk ekranında en alttaki **`Add python.exe to PATH`** kutucuğunu **mutlaka işaretleyin** ve *Install Now* deyin (yönetici izni istemeyen kullanıcı kurulumu varsayılandır).
  3. Ardından `kurulum.bat` dosyasına tekrar çift tıklayın.

---

## 6. Gemini Yapay Zekâ Yanıt Vermiyor veya Kota Doldu
- **Nedeni:** Ücretsiz katman dakikalık/günlük sınırına (`429 RESOURCE_EXHAUSTED`) ulaşılmış veya `.env` içindeki `GEMINI_API_KEY` hatalı girilmiş olabilir.
- **Sistemin Güvenli Davranışı:** Hata durumunda sistem otomatik olarak **`BEKLE`** kararına geçer; asla rastgele alım-satım yapmaz. Sol menüden günlük maksimum çağrı sayısını düşürebilir veya anahtarı boş bırakarak saf kural tabanlı modda devam edebilirsiniz.
