# Claude için kalan görevler

Son güncelleme:14 Kasım2026. HANDOFF.md ayrıntıların ve mevcut doğrulamanın kaynağı.
`[x]` uygulanmış/doğrulanmış, `[ ]` açık. Bu liste bir tamamlanma iddiası değildir.

## Önce mevcut teslimi güvenceye al

- [x] Gerçek veri/model içeren yerel prototip.
- [x] TypeScript production build.
- [x] 13 Python testi.
- [x] Poyraxx hesabında private `mars-pathfinder-ai` deposu.
- [x] Kaynak, model, DEM,3 PDS kare takımı ve provenance için11 web commit.
- [x] Uzak depo ile yereldeki71 proje dosyası path/SHA-256 karşılaştırıldı (14 Kasım,
  commit `73011aa`): 71/71 path eşleşti, fazladan dosya yok; 70 dosya hash olarak aynı,
  boş `backend/__init__.py` raw uçtan 501 döndüğü için hash'lenemedi ama ağaçta var.
- [ ] Bu CLAUDE.md /HANDOFF.md /TASKS.md dosyalarını da private repoya ekle.
- [ ] Yerel Git geçmişini uzak11 commit ile güvenli şekilde eşitle.
  Yerelde henüz commit yok. Önce status ve remote'u incele; force-push yok.

## P0 — Canlı ücretsiz yayın

Canlı: **https://mars-pathfinder-ai.onrender.com** (servis `srv-dbs0p7brjlhs73criomg`,
Blueprint `exs-dbs0p0brjlhs73cri290`, Frankfurt, free). 14 Kasım 2026'da kuruldu.

- [x] Render dashboard kontrol edildi; hesap doğrulanmış.
- [x] Render GitHub App yalnızca `Poyraxx/mars-pathfinder-ai` için kuruldu (Only select repositories).
- [x] render.yaml Blueprint ile free servis oluşturuldu; auto-deploy main'den.
- [x] Docker Linux build ilk denemede başarılı (deploy başlangıçtan canlıya ~1 dk). Python bağımlılıkları hâlâ aralıkla
  pinli (ör. Render'da starlette 1.10 kuruldu); ileride sürüm kayması riski var.
- [x] Ücretli instance/DB/disk/GPU/kart seçilmedi.
- [x] `/healthz` (model_ready=true), `/api/catalog`, `/api/regions/gale`, `/` public URL'de 200.
- [x] Public URL'de dış istemciyle: görev → plan →3 rota (201.4 m; terrain/surface maks eğim
  7.85°) → SSE → 2 karede gerçek ONNX çıkarımı → `localization.rejected`, harita v0 →
  overlay PNG → duplicate 409 → JSON rapor. Tarayıcıda küre konsol hatasız yüklendi.
- [~] Render Free tek ölçüm: plan ~8.8 s duvar saati; inference_ms 401 (ilk) / 244 (ikinci).
  Free planda RAM metriği görünmüyor; olay zaman çizelgesinde OOM/restart yok. Soğuk açılış
  (uykudan uyanma) süresi ve çoklu kullanıcı yükü henüz ölçülmedi.
- [x] README'ye live URL, cold-start ve tek-çalıştırma ölçüm notu eklendi.

Kabul: dışarıdan açılabilen URL; hesaplama tarayıcıdaki kaydedilmiş JSON'dan değil,
Python sunucusundan geliyor. Model ve plan sonuçları gerçek. Private repo private kalıyor.

## P0 — Zorunlu bilimsel açık: gerçek kamera–harita bağı

- [ ] PDS etiketleri/PLACES kaynaklarını inceleyerek mevcut3 karenin bağımsız
  doğrulanabilir fiziksel landmark'larını bul. Adaylar uygun değilse uygun gerçek
  ürün takımı seç; veri/provenance pipeline'ını koru.
- [ ] Aynı fiziksel özellikleri hem kamera hem orbital görüntüde bağımsız tanımla.
  En az6 yayılmış kontrol, örtük tek objeye ait tekrarlı piksel listesi değil.
- [ ] `scripts/validate_registration.py` ile **fit edilmemiş mevcut dönüşümün**
  residual'larını ölç ve evidence dosyasını sakla. Kaynak hash'lerini kaydet.
- [ ] XY/XYE, görüntü boyutları, frame/site id, NED eksenleri, hata anlamı ve
  negatif/pozitif yükseklik yönlerini kontrol et. XYE için uygun metrik eleme ekle.
- [ ] Dönüşümü düzeltmen gerekirse fitting noktaları ile bağımsız validation
  noktalarını ayır. Aynı noktalarla fit+test yaparak başarı iddia etme.
- [ ] Kabul edilen gerçek kareyle grounded engel hücreleri → map_version artışı →
  gerçek route.replanned olayını ve eski/yeni rota farkını göster.
- [ ] Kayıt başarısız/eksikse harita değişmediğini koru; gate'i bypass etme.

Durum (14 Kasım 2026, yalnız yerelde; GitHub/Render'a henüz gönderilmedi):

- [x] Elle 6 kontrol noktası bu karelerde güvenilir değil: kareler 256×256 TRAV, menzil
  3–11 m, kayalar 10–40 cm (HiRISE 25 cm'de 1–2 piksel). Bunun yerine önceden sabitlenmiş
  `scripts/register_image_correlation.py`: kamera piksellerini XYZ + düzeltilmiş affine ile
  HiRISE ortho'ya yerleştir, DoG band-pass, ±8 m NCC kaydırma araması, 15 m+ uzaktaki
  2000 rastgele yerleştirmeyle ampirik null, 4 sabit filtre ayarı.
- [x] Sonuç: 3 karenin hepsinde tüm ayarlarda p ≤ 0.0025, en iyi kayma 0.35–0.75 m.
  Planlayıcı sınırı (max kayma + 0.5 m) 0.85–1.25 m. Kanıt:
  `data/prepared/pds/<frame>/registration-evidence.json` (girdi hash'leri dahil).
  DEM–stereo yükseklik eşleştirmesi bilgi vermedi (rölyef 0.4–0.6 m); kullanılmadı.
- [x] Gerçek füzyon: kabul edilen karelerle 47 hücre engel, harita v2, `route.replanned`.
  Kamera izdüşümünü kesen 50 m görevde surface rotası 50.00 → 52.49 m (eski rotanın
  9 hücresi bloklandı). Varsayılan 201 m rota izdüşümden geçmediği için değişmez.
- [x] Model `big_rock` pikselleri gözle incelendiğinde çoğu kum/çakıl/görüntü kenarıydı;
  gerçek bloklar yakalanmıyordu. Karar (kullanıcı tam yetki verdi): harita engelleri artık
  **stereo XYZ basamak yüksekliği** (`backend/perception.py::stereo_hazards`): yerel zemin =
  2 m pencerede 0.25 m hücre minimumlarının 20. persentili; basamak ≥ `profile.obstacle_step_m`
  (varsayılan 0.15 m); XYE normu ≤ 0.10 m; ≥6 piksellik bileşen. Model yalnız bağlam olarak
  gösteriliyor, haritaya yazmıyor (test ile korunuyor). Gözle: işaretlenenler kaya tepeleri;
  alçak/uzak kayalar kaçabiliyor. XYE filtresi olmadan uzak menzil gürültüsü sahte şerit üretiyor.
- [x] Kare 2 stereo engelleri: 15 hücre, harita v1; kesen 50 m görevde surface rotası
  50.00 → 51.66 m (eski rotanın 5 hücresi bloklu). Kare 3 başka konumdan aynı kayaları
  gördü ve yeni hücre eklemedi (`route.unchanged`) — kareler arası tutarlılık.
- [x] Arayüz: kayıt sınırı, stereo engel pikseli, "FRAME PROCESSED" (409 tekrarını önler).

Kabul: açık evidence + metre cinsinden bağımsız residual + gerçek geometrik harita
güncellemesi. Metadata tutarlılığı veya gözle güzel hizalama tek başına yeterli değil.
Bu tamamlanana kadar projenin ana füzyon hedefi tamamlandı deneme.

## P1 — Surface modelini iyileştir

- [ ] Önce training/validation sınıf dağılımını ölç. Held-out testte yalnız543
  big-rock pixel var ve recall0. Bu ciddi sınır, arayüz uyarısını kaldırma.
- [ ] Eğitim verisini big-rock içeriği ve gruplanmış bağımsızlık bakımından
  genişlet; sınıf dengeli crop/sampling gibi değişiklikleri validation'da değerlendir.
- [ ] RAD/EDR domain farkını incele. Yalnız test örneklerine göre eşik seçme.
- [ ] Uygun GPU bütçesiyle yerel eğitim;4 GB VRAM sınırını koru.
- [ ] Checkpoint, seed, split manifest, epoch history ve gerçek süreleri kaydet.
- [ ] ONNX export320 ve parity testini koru; BatchNorm eval hatasını geri getirme.
- [ ] Sınıf bazlı IoU/precision/recall/F1, confusion matrix ve ayrı final değerlendirme.
- [ ] Model kartı/UI metrikleri aynı kaynaktan gelsin; mevcut UI55.8%/0% metni
  statik yazılmış durumda, model değişirse güncellenmesi gerekiyor.

Kabul: gerçek ölçülmüş büyük kaya performansı ve sınırları. Başarı eşiği ve test
protokolü belirlenmeden modelin “güvenli tehlike detektörü” olduğu iddia edilemez.

## P1 — Uçtan uca ürün ve QA

- [ ] Mobil390×844'te rota terminal sonucu, drawer, intelligence, kamera, report
  ve sources akışını bitir. Desktop1440×900 ve tablet boyutunu da kontrol et.
- [ ] Üç bilim katmanının gerçek tile yüklemesini ve coverage/no-data durumunu test et.
- [ ] Start/goal/3 stop, invalid DEM, no-path, duplicate frame, wrong region,
  stale version, busy slot ve sunucu yeniden başlama durumlarını doğrula.
- [ ] Gözlem işlenmişse yeniden analiz düğmesinin davranışını iyileştir:
  şu an ikinci tıklama409 duplicate döndürür; net processed durumu gösterilebilir.
- [ ] Sources modalında focus trap, Escape ve focus return; tüm icon button
  etiketleri; çekmece klavye erişimi; yeterli kontrast ve okunabilir metin.
- [ ] API/TypeScript sözleşmelerini drift'e karşı kontrol et. İstenirse OpenAPI
  üzerinden üretim ekle; mevcut elle yazılmış tipleri generated diye adlandırma.
- [ ] Başarılı geometri bulunduğunda füzyon/map-version/replan concurrency
  testlerini gerçek küçük fixture'larla genişlet. Sentetik fiziksel eşleşme uydurma.
- [ ] Free deployment limitleri için SSE bağlantı sayısı, görev kapasitesi,
  tile cache ve memory ölçümlerini kontrol et.
- [ ] Navigasyon değerlendirmesini tekrar üret. Kabul edilmiş gözlem yokken C==B
  kalması doğrudur. Bağımsız collision truth yoksa oran/pazarlama iddiası üretme.

## P2 — Yarışma teslimi

- [ ] Güncel resmî challenge/Ankara teslim kurallarını yeniden doğrula.
- [ ] Gerçek ekranlardan30 saniyelik demo veya izin verilen7 slaytlık teslim hazırla.
  Şu an video/slayt dosyası oluşturulmadı; yalnız ekran görüntüleri var.
- [ ] README, model kartı, veri provenance, lisans/atıf ve AI kullanım açıklamasını
  son uygulama durumu ile tutarlı hale getir.
- [ ] Private repo kullanıcının açık tercihi. Yarışma public kod istiyorsa bunu
  kullanıcıyla çöz; kendi başına görünürlüğü değiştirme.
- [ ] “İki eğitilmiş AI” deme: mevcut Orbital deterministik. EVA güvenliği deme.
- [ ] Kamuya açık demo, indirilebilir rapor ve bilinen sınırlamalarla final kontrol.

## Şimdilik kapsam dışında

- Arbitrary kamera/fotoğrafı her yerde otomatik lokalize etmek.
- Mars'ın her noktasında1 m hassas planlama varmış gibi davranmak.
- İkinci öğrenilmiş orbital modeli sırf isim uğruna yetiştirmek.
- D*Lite/RL, suit consumables, uçuş veya insan güvenliği sertifikasyonu.
- Ücretli servis açmak ya da GitHub'a AI coauthor eklemek.
