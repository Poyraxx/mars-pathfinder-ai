# Claude için kalan görevler

Not: aşağıdaki eski devir maddelerinin bir kısmı tarihîdir. Güncel gerçek-footage
devamının sonucu dosyanın son bölümünde ve README'de bulunur.

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
- [x] GitHub'a yüklendi (commit `b3a5a14` dahil, tümü Poyraxx) ve Render'da canlı doğrulandı
  (14 Kasım): dış istemciyle kesen 50 m görev → kare 1: 0 engel px, `route.unchanged`;
  kare 2: 666 px, harita v1, `route.replanned`, surface 50.00 → 51.66 m; kare 3: 546 px,
  yeni hücre yok. Render Free: inference 799/346/324 ms; ilk gözlem 6.5 s duvar saati
  (scipy ilk yükleme dahil). Tek çalıştırma, benchmark değil.

Kabul: açık evidence + metre cinsinden bağımsız residual + gerçek geometrik harita
güncellemesi. Metadata tutarlılığı veya gözle güzel hizalama tek başına yeterli değil.
Bu tamamlanana kadar projenin ana füzyon hedefi tamamlandı deneme.

## 14 Kasım 2026 — ikinci tur (kullanıcı isteği: daha çok görsel, tüm Mars, uydu görüntü işleme, demo)

- [x] Yerel git uzak geçmişe bağlandı (`main` → `origin/main`), commit kimliği yalnız bu repo için
  `Poyraxx <156310194+Poyraxx@users.noreply.github.com>`. Co-authored-by yok.
- [x] 27 yeni gerçek Sol 57 Navcam RAD/XYZ/XYE karesi (`scripts/prepare_navcam_set.py`): sürüş dizisi
  6 kare + Rocknest panoraması 24 kare. Geometri 256×256 (blok-merkez örnek), tam çözünürlük görüntü ayrı.
- [x] Aynı önceden sabitlenmiş kayıt testi 30 karede: 13 kabul (sürüş 4, panorama NCAM00445 9; kayma
  0–0.75 m, p ≤ 0.005), 2 ret (kayma 2.5–8 m, p 0.29–0.80), 15 ölçülemedi (bölge içinde yoğun zemin
  yok). Ölçülemeyen/reddedilen kareler gösterilir ama haritaya yazmaz, rota çizilmez.
- [x] Rota kamera görüntüsünde: `backend/camera_route.py` — her rota örneği en yakın ölçülmüş stereo
  zemin pikseline (tolerans ≤ hata sınırı/2, min 0.35 m); kapanma/görüş dışı = boşluk. Eski rota kesikli.
  "Plan in this view": başlangıç = PLACES rover konumu, hedef = görüntü merkezindeki ölçülmüş zemin noktası.
  Kare 2'de: rota 8.24 m (kayaların içinden) → kamera güncellemesinden sonra 9.41 m (kayaların yanından).
- [x] Video benzeri oynatma: sürüş/panorama dizilerinde kare kare oynat/duraklat.
- [x] Tüm Mars kaba planlama: `backend/global_dem.py` — MOLA/HRSC 200 m LERC karoları canlı çekilir,
  ≤384² ızgara (≥163 m hücre). Jezero 5.9 km, Olympus Mons 89.6 km, Valles Marineris 229 km test edildi.
  Kaya/engel bilgisi yok; |enlem| ≤ 80°, mesafe sınırı yok (uzun rotada hücre büyür), 180° meridyeni destekli.
- [x] 2055 IAU adı (USGS Gazetteer) küre etiketleri + arama + tür/çap kartı.
- [x] Küresel planlama varsayılan (Gale seçmeden kürede başlangıç/hedef). Rota üzerinde/yakınında adlı
  yer şekilleri listesi; "Avoid" ile seçilen yer şekli IAU çapı kadar daire olarak terrain/surface
  rotalarına kapatılır (`backend/features.py`). Tharsis testi: 454 km / 8.8 km tırmanış (Pavonis Mons
  üstünden) → kaçınma ile 577 km / 3.9 km.
- [x] Öğrenilmiş Orbital model: MobileNetV3-Small, HiRISE landmark v3.2 (Zenodo 4002935, CC BY 4.0),
  resmî kaynak-görüntü ayrımı. Test (1793 orijinal): doğruluk 92.9%, macro-F1 69.5%; spider recall 0
  (42 örnek), impact ejecta 57% (7 örnek). CTX'e aktarım kontrolü: 131 IAU kraterinin 65.6%'sı "crater",
  3 çap kaydırılmış pencerelerde 8.4%. Rota koridoru boyunca CTX (~5 m) pencerelerinde canlı çalışır;
  bağlam bilgisi, tehlike haritası değil, rota maliyetine girmez.
- [x] AI4Mars v2 seti: train 2000 (820 büyük-kaya içeren, yalnız train'de tercihli), val 400, test 400
  (yansız). Yeniden eğitim 12 epoch (en iyi val epoch 6). Aynı 400 test görüntüsünde v1→v2: mIoU
  59.6→64.7%, big-rock IoU 0.06→0.20, recall 14→36%, precision 10→32% (val recall 84%). v2 yayında;
  büyük kaya hâlâ güvenilmez ve haritaya yazmaz. v1 model yedeği: `%LOCALAPPDATA%/mars-pathfinder/training/v1-backup/`.

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

## Devam noktası (14 Kasım 2026, kullanım sınırında durdu)

Kullanıcı isteği: senaryoların varsayılanı GERÇEK rover görüntüsü olacak, rota her karede canlı çizilecek
(GTA tarzı yeniden rota), sağda küresel harita; 3D render yalnız yardımcı simülasyon.
- [x] Mesafe sınırı kaldırıldı; 180° meridyeni destekli; bozuk DEM karosunda bir üst seviyeye düşme.
- [x] Replay motoru (`backend/replay.py`): gerçek PLACES pozları, 2 m sapmada yeniden rota, U dönüşü (>120°),
  stereo engel ile yeniden rota, özel hedef. Curiosity sol 55–57 (39 kare). Navcam TRAV kareleri yana bakıyor
  → rota görüntüde görünmüyor (doğru davranış). Yeni arayüz: Explore / Plan / Drive replay modları.
- [x] Engel dedektörü: 3 m genişlik ve 2 m² alan üstü bileşenler elenir (stereo kusuru/arazi şekli).
- [ ] SIRADAKİ: Perseverance ham kareleri (mars.nasa.gov raw_images API). Her karede CAHVORE kamera modeli
  (rover çerçevesi), `attitude` kuaterniyonu, `xyz` (site çerçevesinde rover), `site`/`drive` var.
  Site başlangıcı: https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json (drive 0 satırı).
  API sol filtresi: `&sol=14`. Sol 14 (ilk sürüş) yalnız 2 FLF karesi → sürüş sırasında çok karesi olan sol seç.
- [ ] CAHVORE 3D→2D uygula; `mars2020_hazcam_ops_stereo` XYZ ürünleriyle piksel hatasını ölç (doğrulama).
- [ ] Rota: küresel/yerel DEM'de planla, her gerçek kareye CAHVORE ile izdüşür; varsayılan görünüm gerçek kare.
- [ ] Demo videosu (MP4): gerçek kare + rota (sol) | harita + rover (sağ).

## Gerçek-footage devamı — Codex, 14 Kasım 2026

- [x] Bu klasöre ait5 Claude Code oturumunun kullanıcı/assistant konuşmaları ve
  Paint referansları incelendi; son kullanıcı düzeltmesi esas alındı.
- [x] Perseverance araştırmasına bağımlı kalmadan, aynı planlama bölgesindeki
  gerçek ileri bakan Curiosity Front Hazcam RAD/XYZ/XYE kaynakları bulundu.
- [x] 12 Hazcam ürünü: Sol50,52,53,55,56,57. Native1024 görüntü ve256 stereo grid.
  Mevcut katalog77 kare:65 Navcam+12 Hazcam,21 orbital image-registration kabulü.
- [x] NASA-AMMOS VICAR esaslı CAHVORE portu ve PDS rover/site dönüşümü. Fit
  yapılmadan her gerçek stereo noktasının kaynak pikseline reprojeksiyonu:
  p95 0.16–0.30 px. Bu mutlak orbital konum doğrulaması değildir.
- [x] Varsayılan senaryo gerçek4 çekim: Sol50→52→53→55. Yaklaşık60 m kayıtlı ilerleme,
  kamera ve haritada aynı kalan rota; 4.05 m sapmada gerçek yeniden planlama.
- [x] Aynı gerçek footage ile kaçırılmış waypoint senaryosu: önerilen U dönüşü,
  hedef görüntü arkasında ise yanıltıcı çizgi yerine yönlendirme ve haritada rota.
- [x] Geçilen rota kısmının kırpılması, son karede de replan, profile-aware cache.
- [x] Fotoğraf yüklenmeden yeni overlay gösterilmiyor; playback görüntüyü bekler.
- [x] İlk açılışta ortho yüklenmemesi (catalog bağımlılığı), rota kadrajının
  kesilmesi ve mobil başlık taşması düzeltildi. Whole planet / fit-route kontrolleri.
- [x] Gerçek kamera varsayılan; 3D simulation ayrı sekme. Oynat/duraklat/scrub.
- [x] İki30 s,1600×900,24 fps H.264 MP4; gerçek fotoğraf+HiRISE map, API'den indirme.
  Bunlar4 gerçek kareden hızlandırılmış timelapse; kesintisiz veya canlı video değil.
- [x] 29 Python testi ve production TypeScript/Vite build geçti.
- [x] Private main dalı Render'a yayımlandı. Public API'de iki senaryo, gerçek sapma/U dönüşü,
  MP4 Range indirme ve dosya hashleri doğrulandı. Linux'ta gerçek registration kanıtı
  kabul edildi; doğrulanmamış Hazcam kareleri engel haritasını değiştirmedi.

Korunan sınırlar: yeni4 featured Hazcam pozu bağımsız orbital eşleşmeyi geçmedi;
arşiv pozuyla görselleştirilir, engel füzyonuna girmez. Uzak görüntü çizgisi stereo
ile düşey hizalanmış DEM yüksekliğini kullanır. ML güvenilirliği/safety sertifikası
iddiası yok. Model eğitimleri bu turda yeniden yapılmadı.
