# İşlem sistemi ve kârlılık incelemesi

**Tarih: 15 Kasım 2026 — Europe/Istanbul**

Mevcut kanıtlar, gerçek parayla daha sık veya daha yüksek kaldıraçla işlem açmayı desteklemiyor. Yerel kapanmış işlem kayıtları zarar gösteriyor. Üç paritede dört profil için hazırlanan 12 geçmiş Spot temel testinin tamamı da maliyetlerden sonra negatif. Bu sonuçlar canlı sistemin bütün filtrelerini kapsamaz; herhangi bir profilin gelecekte mutlaka zarar edeceğini de kanıtlamaz. Kârlılığı doğrulanmış bir profil henüz yok.

İnceleme sırasında risk bütçesini aşabilen boyutlandırma, gevşetilebilen zarar-kes, eksik faiz hesabı, yenilenmeyen model önbelleği ve birbiriyle tutarsız araştırma metrikleri düzeltildi. Bu düzeltmeler yazılımın doğruluğunu iyileştirir; kazancı kanıtlamaz.

## Yerel işlem kayıtları

Kaynak: `data/sanal_sistem.db`, SQLite salt okunur bağlantı. Emir tarih aralığı 11–12 Kasım 2026. Buluttaki güncel hesap bakiyesi veya son işlemler okunmadı.

| Kayıt izi | Kapanmış işlem | Kazanan | Kayıtlı net sonuç | Kapanmış işlemlerin kayıtlı komisyonu |
|---|---:|---:|---:|---:|
| LIVE_SPOT — canlı Spot | 3 | 0 | −1,801281 USDT | 0,016931 USDT |
| MAIN — sanal portföy | 5 | 0 | −1,820520 USDT | 0,096619 USDT |
| RULE_SHADOW — kural karşılaştırması | 61 | 9 | −1,357548 USDT | 1,203537 USDT |

**Canlı sonuç geçicidir:** yerel kayıtta değerlenmemiş komisyonlar için iki kayıt bulunuyor. Üçüncü varlıkta kesilen ücretlerin değeri tamamlanmadan −1,801281 USDT nihai ve eksiksiz hesap sonucu sayılmamalı. Binance hesabıyla bu incelemede mutabakat yapılmadı.

Kural karşılaştırmasında kayıtlı ücretler net kaybın yaklaşık %88,7'sine eşit. Ücretler geri eklendiğinde dahi sonuç yaklaşık −0,154010 USDT; yalnızca komisyonu azaltmak bu örneği kendiliğinden kârlı yapmıyor. Kazanma oranı %14,8, işlem başına kayıtlı net sonuç yaklaşık −0,0223 USDT. Ayrıca beş açık karşılaştırma pozisyonu var; bunların güncel gerçekleşmemiş sonucu bu kapanmış işlem tablosuna dahil değil. Üç portföyün sonuçları tek bir yatırım getirisi gibi toplanmamalı.

## Gerçek geçmiş fiyatlarla sabit parametre testi

Kaynak: Binance herkese açık `data-api.binance.vision` mumları. Her BTCUSDT, ETHUSDT ve SOLUSDT paritesi için 1m, 5m, 15m dilimlerinde 1000 kapanmış mum kaydedildi. Son tarih **15 Kasım 2026 00:00 UTC / 03:00 İstanbul**. Parametreler profilden aynen alındı; test dilimindeki sonuca bakılarak ayarlanmadı. Kronolojik ayrım %60 eğitim, %20 doğrulama, %20 test.

Varsayımlar: ayrı ayrı 86 USDT başlangıç, 15 USDT pozisyon tavanı, 10 USDT minimum alım, 1x Spot, 2 USDT günlük zarar eşiği, yön başına %0,10 komisyon, yön başına 1,5 baz puan yarım makas + 2 baz puan ek kayma. Güncel miktar ve minimum emir filtreleri uygulandı. Profilin günlük giriş sınırı korundu. Başka pozisyona ayrılmayan para nakitte kaldı.

Tablodaki sonuçlar yalnızca son 200 mumluk test dilimine aittir. Her hücre ayrı bir 86 USDT senaryosudur; eşzamanlı portföy getirisi değildir.

| Profil | Mum | BTC net / işlem | ETH net / işlem | SOL net / işlem |
|---|---|---:|---:|---:|
| AL_YUKSELINCE_SAT | 1m | −1,5770 / 37 | −1,7693 / 36 | −1,4795 / 37 |
| HIZLI_30SN | 1m | −1,0859 / 24 | −0,9391 / 25 | −1,7117 / 40 |
| AKTIF | 5m | −0,3797 / 4 | −0,4178 / 4 | −0,0710 / 1 |
| TEMKINLI | 15m | −0,2083 / 1 | −0,1451 / 1 | −0,1685 / 1 |

Mum sayısı eşit olsa da takvim süreleri farklıdır: test dilimi 1m için 3 saat 20 dakika, 5m için 16 saat 40 dakika, 15m için 50 saattir. Bu yüzden satırlar arasında kazanç sıralamasıyla “en iyi profil” seçilemez. AKTIF ve TEMKINLI'de işlem sayısı yorum yapmak için özellikle yetersiz. Hızlı profillerde 15 işlem sınırının aşılması da kalıcı bir avantaj kanıtı oluşturmaz; gözlemler kısa ve birbirleriyle ilişkilidir.

**Model kapsamı:** Tek paritede OHLC temelli Spot testi. Sinyalden sonraki mum açılışında giriş/teknik çıkış; mum içinde zarar-kes ve hedef teması; belirsiz sırada zarar-kes önceliği. Açık miktar son fiyatta tahmini satış maliyetiyle değerlenir. Çoklu zaman dilimi, BTC rejimi ve quant giriş filtreleri, ATR boyutlandırması, kâr bekleme ve hareketli stop canlıdaki biçimiyle modellenmez. Güncel borsa filtreleri geçmişteki kurallardan farklı olabilir. Bu nedenle özellikle AL_YUKSELINCE_SAT satırı, tüm korumaları çalışan canlı profilin birebir tekrarı değildir.

Ham girdiler `reports/market_data/20261115T000000Z/`, ayrıntılı sonuçlar [profitability.json](reports/profitability.json) içinde. Ağ erişimi olmadan aynı girdilerle yeniden hesaplama:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_profitability.py
```

Yeni bir tarih için herkese açık veriyi indirmek:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_profitability.py --fetch --as-of 2026-11-15T00:00:00Z
```

Bu betik `.env` yüklemez, hesap anahtarı veya emir uç noktası kullanmaz, üretim veritabanına bağlanmaz.

## Kazancı zorlaştıran yapı

**Küçük hedef, yüksek başarı oranı gerektiriyor.** Sabit hedef/zarar-kes senaryosunda, yalnızca yön başına %0,10 ücret varsayımıyla:

| Profil | Fiyat zarar-kesi | Fiyat hedefi | Yaklaşık başabaş kazanma oranı |
|---|---:|---:|---:|
| AL_YUKSELINCE_SAT, 1x | %3,50 | %1,15 | %79,6 |
| HIZLI_30SN, 1x | %0,35 | %0,45 | %68,8 |

Bu hesap bütün işlemlerin yalnızca hedef veya stopta kapandığını varsayar; teknik çıkış ve hareketli stop dağılımını temsil etmez. Makas, kayma ve borç faizi dahil değildir. Gerçek başabaş gereksinimi maliyetlere ve çıkış dağılımına bağlıdır.

Temel testteki toplam gidiş-dönüş maliyeti yaklaşık %0,27'dir: 15 USDT için yaklaşık 0,0405 USDT. Fiyat yatay kalsa bile sık al-sat bu maliyeti tekrar tekrar öder. Gerçek ücret, hesaba ve pariteye göre değişebilir; %0,10 doğrulanmış hesap tarifesi değildir. [Binance komisyon açıklaması](https://developers.binance.me/en/docs/products/spot/faqs/commission_faq)

`min_net_profit_exit_pct` adı gerçekleşen net kâr oranını ifade etmiyor: motor bunu giriş fiyatının üzerindeki brüt fiyat artışı olarak kullanıyor. Örneğin %0,80 fiyat artışı, yön başına %0,10 ücretle yaklaşık %0,60 bırakır; kayma ve faiz bunu azaltır. Dinamik hedefin profil hedefinin 1,6 katıyla sınırlanması da her durumda 1,2 getiri/risk oranı sağlamaz; kod açıklaması buna göre düzeltildi.

AL_YUKSELINCE_SAT profilinde günlük ve eşzamanlı pozisyon değerleri 999'a kadar çıkabiliyor. Bulut yolu kaydedilmiş açık pozisyon sınırını ayrıca koruyor; yerel profil uygulaması daha yüksek profil sınırını kullanabiliyor. Çok sayıda ilişkili coine girmek, aynı piyasa düşüşündeki toplam riski büyütebilir. Bu incelemede profil limitleri veya canlı ayarlar değiştirilmedi.

## Uygulanan düzeltmeler

1. **Risk bütçesi korunuyor.** Hesaplanan tutar borsanın veya kullanıcının minimumuna yetmezse işlem atlanıyor. Minimuma ulaşmak için risk tavanı yükseltilmiyor; miktar adımı ve kayma tamponu tavanın içinde kalıyor. Özsermaye hesabı başarısız olduğunda başlangıçtaki büyük miktar korunmuyor.
2. **Mevcut stoplar gevşetilmiyor.** Bütçe, kaldıraç veya mod ayarı değişikliği açık işlemlerin ATR/hedef seviyelerini sıfırlamıyor. Profil değişikliği ve eski hedef geçişi mevcut stopu aşağı çekmiyor; kâra taşınmış hareketli stop korunuyor.
3. **Koruma kontrolü tarama bütçesiyle kesilmiyor.** Bulut taramasındaki 45 saniyelik yeni aday sınırı sonraki açık pozisyonların denetlenmesini engellemiyor. Ağ bağlantısı yine gecikebilir; bu değişiklik bir bağlantı veya kesin çıkış süresi garantisi değildir.
4. **Faiz günlük zarar hesabına doğru giriyor.** Yapılandırılmış saatlik faiz ve değerlendirme zamanı kullanılıyor. Gerçek marjin faiz tarifesi bu incelemede doğrulanmadı. [Binance borç/faiz API açıklaması](https://developers.binance.com/en/docs/catalog/core-trading-margin-trading/api/rest-api/borrow-repay)
5. **Stop yuvarlaması risk sınırını büyütmüyor.** Aşağı yuvarlama uygulanıyor; çok küçük pozitif stop sıfıra dönüşmüyor.
6. **Hızlı çıkış koruması gerçek sinyali tanıyor.** Stratejinin ürettiği gerekçe metniyle uyuşmayan denetim düzeltildi. Bu, amaçlanan davranışın onarımıdır; daha uzun beklemenin getiriyi artırdığı henüz ölçülmedi.
7. **Model kayan pencerede yenileniyor.** Sabit uzunluktaki mum listesinde değişmeyen sıra numarası yerine mum zamanı kullanılıyor. Farklı strateji parametreleri ayrı önbellek anahtarlarına sahip.
8. **Grafik ve istatistik aynı işlemlerden geliyor.** İkinci, farklı kurallı simülasyon kaldırıldı. Eğri, net getiri, ücretler ve Profit Factor ortak gerçekleşmelerden hesaplanıyor. Al-tut referansı aynı yürütülebilir başlangıç ve temel maliyetlerle değerleniyor. Kayıp işlem yoksa Profit Factor “Tanımsız” gösteriliyor.
9. **Test kapsamı görünür.** Araştırma ekranı 1x Spot temel testi olduğunu, seçili kaldıracın modellenmediğini, işlem sayısının yetersizliğini ve diğer sınırlamaları belirtiyor. Doğrudan araştırma çekirdeği desteklemediği kaldıraçlı girdiyi reddediyor. Geçmiş test kullanıcı minimum alım tutarını da uyguluyor.

## Sermayeyi büyütmeden önce gerekli kanıt

- Aynı takvim aralığında, farklı piyasa rejimlerini kapsayan daha uzun veriyle profilleri karşılaştırmak; örneğin 6–12 aylık veride yürüyen eğitim/doğrulama ve dokunulmamış son test bölümü kullanmak.
- Canlıdaki giriş filtreleri, minimum tutarlar, boyutlandırma, hareketli stop ve çoklu coin portföyünü içeren tekrar oynatma hazırlamak. Mevcut temel testi bu kapsamda kullanmamak.
- Ücretleri hesap tarifesiyle doğrulamak, değerlenmemiş komisyonları tamamlamak; normal ve daha yüksek kayma/makas senaryolarını ayrı ölçmek.
- Pozitif net beklenti, düşüş sınırı, maliyet dayanıklılığı ve birden fazla dönem/paritede tutarlılık aramak. Tamamlanmış işlem sayısını tek başına kabul koşulu yapmamak.
- Bu kanıt oluşana kadar yeni ayarları sanal modda gözlemek. Kaldıracı veya işlem sıklığını artırmak, pozitif beklenti oluşturmanın yerine geçmez.

İlk analiz aşamasında manuel gerçek emir gönderilmedi, açık işlemler değiştirilmedi ve bakiye sıfırlanmadı. Kullanıcının sonradan verdiği site üzerinden çalıştırma talebiyle aşağıdaki bulut dağıtımı tamamlandı. Önceden açık olan otomatik işlem ayarları korundu; normal zamanlayıcı taramaları dağıtımdan sonra devam ediyor.

## Doğrulama ve kalan çalışma

Tam çevrimdışı test paketi: **206 geçti** (`.\.venv\Scripts\python.exe -m pytest -q`). Başlangıçtaki 164 teste ek olarak boyutlandırma, faiz, stop koruması, zaman bütçesi, model yenilenmesi ve araştırma tutarlılığı için 42 regresyon senaryosu eklendi. JavaScript sözdizimi kontrolü de bu pakette yer alıyor. `git diff --check` temiz. Gerçek geçmiş veriyle 12 temel senaryo ayrıca çalıştırıldı; bu, testlerin gerçek emir göndermeyi doğruladığı anlamına gelmez.

Sonraki teknik öncelikler, uzun süren üst zaman dilimi isteklerinden sonra emir defterinin gönderim öncesinde tekrar tazelik denetimi, eksik üst zaman dilimi verisinde giriş davranışının netleştirilmesi ve paralel veri ön yüklemesinin zaman aşımında tüm işçileri bekleme davranışıdır. Bilgisayar/sunucu veya bağlantı durduğunda yerel stop denetiminin çalışmaması da mevcut mimarinin sınırıdır. Bu alanlar ve tam canlı tekrar oynatma tamamlanmadan uçtan uca güvenilirlik veya daha yüksek kazanç iddiası kurulamaz.

## Siteye dağıtım — 15 Kasım 2026

- Site: https://tradecebrail.web.app
- Firebase Hosting sürümü: `6ced7f05a2dae453`. Yayındaki HTML, test edilen kaynak dosyayla SHA-256 üzerinden birebir eşleşti.
- Cloud Run: `tradecebrail-bot-pfix-20261115`, trafik %100. İmaj özeti: `sha256:123b1916d6bd638a47514819abfc4582aba04ce697495244b36e31b8fa96320d`.
- Cloud Build: `a0d9c957-2a45-4389-9e3d-f8e3c571fe43`, başarılı. Yalnızca 25 uygulama/derleme dosyası yüklendi; yerel anahtarlar ve veritabanları pakete alınmadı.
- Mevcut 1x Canlı Spot modu, açık otomatik tarama, hesap sahibi doğrulaması ve ağ ayarları korundu. Her dakika çalışan zamanlayıcı, kısa geçiş duraklamasından sonra yeniden etkinleştirildi.
- Geçiş öncesi özel GCS yedeği: `backups/pre-profitfix-20261115-1728.db` (nesil `1794763812431245`). Eski sürüm `tradecebrail-bot-00028-gcf` silinmedi.
- Yeni sürümdeki ilk doğrulanan normal zamanlayıcı isteği: **15 Kasım 2026 20:32 İstanbul**, HTTP 200, yaklaşık 7,59 saniye. Manuel `/api/tick` çağrılmadı.
- Site üzerinden `/api/health` HTTP 200; oturumsuz `/api/dashboard` HTTP 401. Tarayıcı erişim hizmeti kullanılamadığı için oturum açılmış panelde görsel kontrol yapılamadı.

Bulut motoru yerel bilgisayarın açık olmasına bağlı değildir. Bu dağıtım kârlılık kanıtı oluşturmaz; yukarıdaki maliyet ve model sınırlamaları geçerlidir. İlk tablodaki kâr/zarar değerleri hâlâ 11–12 Kasım tarihli yerel geçmişe aittir; güncel bulut hesabının getirisi olarak yorumlanmamalıdır.
