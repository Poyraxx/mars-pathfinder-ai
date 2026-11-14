# Mars Pathfinder AI — Codex → Claude devir notu

Tarih: 14 Kasım 2026, Europe/Istanbul. Bu belge bir tamamlanma raporu değildir.
Kullanıcı Codex çalışmasını durdurup Claude'a devretmemizi istedi. Yeni özellik,
eğitim veya yayın işi bu noktada durduruldu. Yerel inceleme sunucuları açık bırakıldı.

## 1. Proje ve kabul edilen kapsam

NASA Space Apps / Martian Map için gerçek Mars verisi kullanan araştırma prototipi.
Ana deneyim: Mars'ı 3D keşfet → Gale/Rocknest alanını seç → gerçek DEM üzerinde
rota karşılaştır → gerçek Navcam karesinde öğrenilmiş segmentasyon çalıştır →
doğrulanabilen gözlemi ortak haritaya aktar → yeniden planla → rapor dışa aktar.

İlk dört adım ve raporlama uygulanmış durumda. **Kamera verisinin bağımsız
orbital doğrulaması ve buna dayalı gerçek rota değişimi henüz gösterilemedi.**
Bu eksikliği saklayan animasyon veya yapay engel kullanılmadı.

Kullanıcı birkaç doğrulanmış arşiv karesini yeterli buldu. Serbest fotoğraf
lokalizasyonu MVP'de değil. Orbital modülün ikinci bir öğrenilmiş model yerine
deterministik DEM analizi yapması kabul edildi. İnsan EVA araştırma profili var;
oksijen/enerji/radyasyon ve insan güvenliği doğrulaması yok. Ücretsiz hizmetler,
soğuk açılış ve saniyeler/onlarca saniyelik çıkarım kabul edildi. Harcama yok.

Orijinal uzun brief:
`C:\Users\osura\.codex\attachments\b2891bbe-a921-4ba2-b3ed-73fb710955d8\Yapıştırılan metin.txt`

Briefteki eski “PLAN MODE / kod yazma” bölümü daha sonra kullanıcının “Please
implement the plan” talimatıyla geçersiz oldu. Şimdi uygulama aşamasındayız.
Yarışma tarih/teslim kurallarını gerekiyorsa resmî sayfadan tekrar doğrula;
bu belge yeni bir resmî şartname yerine geçmez.

## 2. Çalışan parçalar

| Alan | Durum |
|---|---|
| React + TypeScript + Cesium küresi | Gerçek Viking görüntüsü, gerçek MOLA/HRSC yükseklikleri |
| Bilim katmanları | Viking, MOLA renkli hillshade, THEMIS gündüz IR |
| Yerel planlama | Gale/Rocknest 512×512, native 1 m DEM, 25 cm ortho |
| Navigasyon | 8 komşulu A*, köşe kesme yok, mesafe/terrain/surface karşılaştırması |
| Metrikler | Mesafe, yükselme, maksimum eğim, yükseklik profili, örnek hızdan süre |
| Kamera | 3 gerçek eşleşen PDS RAD/XYZ/XYE ürün takımı |
| Surface modeli | Gerçek AI4Mars üzerinde fine-tune edilmiş LR-ASPP/MobileNetV3 |
| Inference | ONNX Runtime CPU, API üzerinden gerçek çıkarım, mask overlay |
| Koordinasyon | Asenkron işler, tek hesaplama slotu, sürümlü harita, SSE olay günlüğü |
| Füzyon | Gerçek geometri kodu var; mevcut karelerde bağımsız kayıt kapısı kapalı |
| Rapor | JSON dışa aktarımı, veri kaynakları, model bilgisi, olaylar |
| Yayın paketi | Dockerfile + Render Free Blueprint hazır |
| GitHub | Private repo oluşturuldu, kod/veri/model tarayıcıyla yüklendi |
| Render canlı URL | Henüz yok; e-posta doğrulaması kullanıcı tarafından yapıldı denildi |

## 3. Kaynak kod haritası

### Arayüz

- `src/App.tsx`: görev akışı, paneller, kamera, kaynak modalı, rapor indirme,
  SSE olaylarının terminal durumlarında mission state yenileme.
- `src/Globe.tsx`: tek Cesium sahnesi, Mars küresi, LERC DEM örnekleme,
  yerel DEM/ortho, rota ve başlangıç/hedef seçimi, katmanlar.
- `src/store.ts`: Zustand oturum/görünüm/seçim durumu, API yardımcı fonksiyonu.
- `src/types.ts`: elle yazılmış TypeScript sözleşmeleri; OpenAPI'den üretilmiş
  olduklarını iddia etme. Şu an üretim hattı yok.
- `src/styles.css`: desktop kontrol paneli, mobil çekmeceler, reduced-motion.
- `vite.config.ts`: Cesium Workers/Assets kopyası ve `/api` → port8000 proxy.

### Python API

- `backend/main.py`: FastAPI, katalog, görev/plan/gözlem/state/report/SSE,
  görüntü/overlay/DEM uçları, sabit NASA/Esri kaynaklarına sınırlı tile proxy.
- `backend/navigation.py`: DEM eğim/pürüzlülük, geçerli hücreler, maliyet gridleri,
  A*, Dijkstra referans seçeneği, waypoint legleri, metrikler, engel şişirme.
- `backend/coordinates.py`: Mars body499, planetocentric, east-positive,
  `[-180,180)` boylam; küresel yerel projeksiyon ve NED/ENU yardımcıları.
- `backend/perception.py`: ONNX session, ImageNet normalizasyonu, 4 sınıflı
  segmentasyon, native maske ve XYZ + PLACES affine dönüşümü, sıkı kayıt kapısı.
- `backend/data.py`: prepared-data konumu, bounded DEM mmap, katalog/model kartı.
- `backend/events.py`: geçici SQLite dosyasında artan sıra numaralı olaylar.
- `backend/schemas.py`: Pydantic istek sınırları, en fazla3 ara durak,
  varsayılan20° eğim/0.5 m/s hız/1 m açıklık araştırma profili.

Önemli API'ler: `/healthz`, `/api/catalog`, `/api/regions/gale`,
`POST /api/missions`, `POST /api/missions/{id}/plan`,
`POST /api/missions/{id}/observations`, `.../state`, `.../report`, `.../events`,
`/api/frames/{id}/image`, `.../overlays/{frame_id}`.

Plan ve inference işleri202 döner; sonuç SSE/state ile gelir. İşlerin tek slotu
background task başlamadan handler içinde rezerve ediliyor; yarış durumu düzeltildi.
Görev sayısı64 ile sınırlı. Disk/oturum kalıcılığı iddia edilmiyor.

## 4. Gerçek veri ve koordinat ayrıntıları

Hazır veri `data/prepared/` içinde, yaklaşık25 MB. Ham indirmeler/eğitim verileri
OneDrive dışında `C:\Users\osura\AppData\Local\mars-pathfinder\` altında.

### Gale DEM / görüntü

- Native1 m PLACES/HiRISE DEM crop, 512×512, %100 finite. Sentetik dolgu yok.
- `gale/elevation.npy`, `gale/metadata.json` planlamada kullanılıyor.
- DEM kaynağı `urn:nasa:pds:msl_places:data_maps:msl_orbital_dem::42.0`.
- NW yaklaşık E8146952.49279 / N-271815.07519 m.
- Areoid yükseklik aralığı yaklaşık -4525.687 .. -4500.194 m.
- `pds/region-ortho.png`: 2048×2048 @25 cm/pixel, yaklaşık512 m genişlik.
- `pds/orbital-crop.png`: kamera çevresinin64 m'lik256×256 ortho crop'u.
- Her crop'un JSON'u kaynak URL, hash, pixel/projeksiyon/bounds bilgilerini taşır.
- `scripts/prepare_region.py` alternatif büyük HiRISE DTM'yi denedi; Rocknest
  merkezinde NODATA nedeniyle doğru şekilde reddetti. Aktif DEM bu dosyadan gelmiyor.

### Üç PDS kare takımı

Sol57 / 2012-10-03, SITE4, in-drive interpolated localization:

1. `NLA_402556045XYZ_D0043480TRAV00022M1`
2. `NLA_402556100XYZ_D0043486TRAV00022M1`
3. `NLA_402556155XYZ_D0043492TRAV00022M1`

Her klasörde `reference.png`, `xyz.npy`, `xye.npy`, `valid.npy` var.
Reference ürünü **RAD_D**; ILT diye adlandırma. Görüntü/XYZ ilişkisi resmî
PDS etiketlerinden doğrulandı. Katalogda source URL, SHA, acquisition ve frame var.

Sadece statik Site4 orijini kullanmak yaklaşık7.3 m doğu kaymasına neden oluyordu.
`places.archive_site_to_global_ned_rotation` ve `...translation_m`, eski telemetry
ankrajını localized rover ankrajıyla değiştiren düzeltilmiş affine'dir. Bu dönüşüm
kullanılmalı; ilk Site origin'ine geri dönme. Test, affine'in rover anchor'ını
1e-6 m toleransla yeniden ürettiğini kontrol ediyor. Bu **bağımsız doğruluk kanıtı değil**.

PLACES SIS'teki localized_pos için beklenen0.5 m ifadesi bu interpolated in-drive
karelerin bağımsız hata sınırı olarak kullanılamaz. İlk karede stereo/DEM düşey
farkı yaklaşık1.1 m. Kaynaklar arası mutlak yükseklik uyumu iddia edilmedi.
XYE, eksen bazlı stereo tahmini hata bileşenleri; localization covariance veya
95% güven aralığı değildir. Füzyon kodu henüz XYE ile ek eleme yapmıyor.

**Tüm karelerde `registration.verified=false`.** Bağımsız fiziksel özellik
eşleştirmesi ölçülene kadar böyle kalmalı.

`scripts/validate_registration.py`, aynı fiziksel noktaya ait en az6 bağımsız
kamera/ortho eşini alır, mevcut dönüşümü fit etmeden residual ölçer, ampirik max
residual <=2 m ise evidence yazar ve kareyi açar. Bu eşikler araştırma toleransı;
istatistiksel güven sınırı değildir. Kontrolleri dönüşümün kendisinden üretip
aynı dönüşümü doğrulamak bilimsel olarak geçersiz olur. Hazır gerçek kontrol çifti yok.

### Küresel veri / render sınırları

- Mars sphere R3396190 m. Bilimsel areoid yükseklik ile render referans yüzeyi ayrı.
- Esri MOLA/HRSC200 m LERC, token gerektirmeyen gerçek global context.
- Sabit kaynak: `https://astro.arcgis.com/arcgis/rest/services/OnMars/MDEM200M/ImageServer/tile/{z}/{y}/{x}`.
- NASA Trek WMTS: Viking ve MOLA `maximumLevel=7`, THEMIS `maximumLevel=9`.
  Bunlar capability XML'lerinden kontrol edilip hem istemci hem API sınırı düzeltildi.
- THEMIS IR parlaklığı canlı sıcaklık değildir. Global kaba DEM yerel engel kanıtı değil.
- İsteğe bağlı scoped Cesium ion token ile Mars asset3644333 yolu var;
  token yok, o yol denenmedi. Varsayılan açık veri yolu çalışıyor.

## 5. Eğitim ve gerçek ölçümler

`scripts/fetch_training.py --count400` orijinal yayıncı ZIP'inden HTTP Range ile
400 train /80 validation /80 test Navcam görsel/etiketi seçti. Tam ZIP yaklaşık16 GB;
tamamını indirmek gerekmiyor. `scripts/remote_zip.py`206/range doğrular, bounded cache.
Gruplar3-sol SCLK bloklarından deterministik hash ile ayrıldı, grup başına12 kare
sınırı var, demo çevresi±3 sol dışlandı. Test crowd label altkümesi; expert gold değil.

Model LR-ASPP + MobileNetV3 / ImageNet backbone. Eğitim384 input, batch2,
gradient accumulation4, weighted CE + Dice, AdamW; ilk2 epoch backbone freeze.
Gerçek10 epoch RTX3050 Laptop4 GB üzerinde yaklaşık223.6 saniye sürdü.

`data/prepared/model-card.json` gerçek sonuçların kaynağı:

| Ölçüm | Değer |
|---|---:|
| ONNX320 test mIoU | 0.5580087769787435 |
| soil IoU | 0.8802 |
| bedrock IoU | 0.6450 |
| sand IoU | 0.7068 |
| big_rock IoU / recall | **0 / 0** |
| Testte big_rock truth pixel sayısı | 543 |
| PyTorch384 test mIoU | 0.593916 |
| ONNX/PyTorch pixel agreement,3 örnek | yaklaşık0.99999 /1 /0.99999 |
| Yerel CPU tek thread warm p50 /p95 | 21.66 /25.08 ms |

Bu model bir başlangıç deneyi. Büyük kaya tespiti doğrulanmış değil; held-out
başarısızlık arayüzde açıkça gösteriliyor. RAD demo kareleri ile EDR eğitim
görüntülerinin fotometrik farkı da var. Model skorları occupancy probability değil.
Etiket sözlüğü:0 soil,1 bedrock,2 sand,3 big_rock,255 ignore. Small-stone sınıfı yok.

İlk export'ta wrapper train mode / BatchNorm parity problemi çıktı. Düzeltildi:
`Export(model).eval()`, export sonrası `model.eval()`. `--export-only` ile gerçek
checkpoint'ten export tekrarlandı; eğitim tekrarı yapılmış gibi gösterilmedi.
İlk eğitimde tüm epoch history dosyası kaydedilmeden export hata verdi; tam eski
history mevcut değil. `training/run.json` gerçek10 epoch/223.6 s bilgisini tutar.
Script artık sonraki eğitimlerde history'yi her epoch yazar. Eksik eski history üretme.

Model: `data/prepared/surface.onnx` (~12.9 MB).
Checkpoint/manifest/cache: `%LOCALAPPDATA%\mars-pathfinder\training\`.
Test splitini model seçimi için kullanma; iyileştirmeleri validation'da seçip yeni,
ayrı final değerlendirme politikası belirle.

`navigation-evaluation.json`:10 gerçek start/goal çiftinde30 A/B/C sonucu.
Şu anda C==B; kabul edilmiş grounded gözlem sayısı0. Bağımsız collision truth yok;
kanıtlanmamış collision azalması veya Surface katkısı yüzdesi iddia edilmedi.

## 6. Yerel ortam ve komutlar

Windows, PowerShell,32 GB RAM, NVIDIA RTX3050 Laptop **4 GB VRAM**.
Sistem PATH'inde gerçek node/git/python yoktu; Codex runtime kullanıldı.

```powershell
$taskRoot = 'C:\Users\osura\OneDrive\Masaüstü\nasa'
$taskRuntime = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies'
$taskPython = 'C:\Users\osura\AppData\Local\mars-pathfinder\venv\Scripts\python.exe'
$taskGit = Join-Path $taskRuntime 'native\git\cmd\git.exe'
$taskPnpm = Join-Path $taskRuntime 'bin\fallback\pnpm.cmd'
$env:PATH = (Join-Path $taskRuntime 'node\bin') + ';' + $env:PATH
Set-Location -LiteralPath $taskRoot

# Backend + derlenmiş frontend (port8000; zaten açıksa tekrar çalıştırma)
.\scripts\start.ps1

# Ayrı terminal: geliştirme arayüzü (port5173)
& $taskPnpm dev

# Doğrulama
& $taskPnpm run build
& $taskPython -m pytest tests -q

# Veri/eğitim sadece ihtiyaç varsa; hazır veriyi yeniden indirmek gerekmiyor
& $taskPython scripts/prepare_pds.py --frames 3 --ortho-crop --dem-crop --region-ortho
& $taskPython -m scripts.assemble_data
$env:MARS_DLL_DIR = Join-Path $taskRuntime 'node\bin'
& $taskPython scripts/train_surface.py --epochs 10
```

Yüklü ortam: Python3.12.14, Node24.22.1, pnpm11.25; Python venv tüm çalışma
bağımlılıklarını ve torch/torchvision CUDA128 içeriyor. Frontend bağımlılıklarının
tam sürümleri pnpm-lock.yaml'de. Production requirements PyTorch/GDAL içermez.

Windows ONNX/PyTorch için runtime Node bin'deki VC DLL'leri gerekiyor.
`.local-run.json` python/dll_dir içerir, gitignore'da. Backend bunu okur.
`os.add_dll_directory()` handle'ının canlı tutulması gerekir; kodda tutuluyor.
Global Visual C++ yüklemesi yapılmadı. Bu makinede DLL sorunu olursa önce bu yolu kontrol et.

Devirde açık süreçler (PID'ler sonra değişebilir):

- Uvicorn launcher9196, gerçek Python21532, `127.0.0.1:8000`, reload kapalı.
- Vite node14660, `127.0.0.1:5173`.
- Süren eğitim veya indirme işi yok. Biten komutları yeniden başlatma.

## 7. Doğrulanmış davranış ve QA açıkları

- Son `pnpm run build`: başarılı. Cesium bundle yaklaşık4.26 MB /1.15 MB gzip;
  büyük chunk uyarısı var, build hatası yok.
- Son pytest: **13 passed**, Starlette/httpx deprecation uyarısı var.
- Testler: A* / Dijkstra, köşe kesme, koordinatlar, gerçek DEM/API validation,
  stale map, yanlış tile seviyeleri, dönüşüm anchor tutarlılığı, kayıt kapısı,
  background task başlamadan compute slot reservation.
- Desktop gerçek tarayıcı/WebGL testi yapıldı: Mars küresi, Gale, gerçek route,
 201 m /7 min /7.8° örnek sonuç, gerçek57 ms kamera çıkarımı,6 olay,
 `localization.rejected`, ortak harita v0. Bunlar o oturumun gerçek sonuçları.
- Mobil390×844 gerçek Chrome viewport'unda drawer/start/plan kontrolleri tıklandı.
  Mobil route işleminin terminal sonucunu ve tüm overflow/katmanları henüz
  bitmiş QA diye sunma. Codex kullanıcı isteğiyle bu noktada durdu.
- IAB viewport capability bu oturumda390'a geçmeyi uygulamadı (DOM1440 kaldı).
  Chrome viewport override390×844 ile gerçekten çalıştı. Bundan yanlış mobil test
  başarısı çıkarma. Override'lar devir kapanışında reset edilecek.
- Tüm katmanlar için final görsel QA, mobil intelligence/export, klavye/modal
  focus trap/Escape davranışı tamamlanmadı. Sidebar gizliyken visibility düzeltildi.
- Gerçek Docker Linux build henüz çalıştırılmadı. Render deployment testi yok.

`artifacts/mission-desktop.jpg`, `github-private.jpg`, `render-authorization.jpg`
yerel kanıt görselleri; artifacts gitignore'da. Mask görseli de mevcut.

## 8. GitHub'ın tam durumu

Repo: **https://github.com/Poyraxx/mars-pathfinder-ai — PRIVATE**.
Kullanıcı private depo açılmasını açıkça istedi. Public'e çevirme.
Kod, model, DEM,3 kamera ve provenance toplam71 yerel tracked dosya olarak
11 web commit ile yüklendi. Uzak arşivi indirip tüm hash'leri yerelle karşılaştırma
adımı henüz yapılmadı; bunu tamamlanmış sayma.

Son görünen remote commit:
`73011aad0c8c920518ce9a4f0d475081604d8a27`
mesajı `Add third matched Navcam image and stereo geometry`.
Repo başlığında Private, Contributors(1) altında yalnızca @Poyraxx görüldü.

Yerel `.git` initialize edildi, `origin` bu repoya ayarlı, dosyalar staged,
**yerel commit yok**. Uzak11 commit, tarayıcıyla kullanıcı hesabı adına üretildi.
Yerelin uzak geçmişle eşitlendiğini varsayma. Zorla push/reset/overwrite yapma.
Git Credential Manager mevcut (2.9.1), fakat `github list` boş; gh CLI bulunamadı.
GitHub MCP bu private repoya404 döndü; connector erişimi yok. Buna erişim varmış
gibi güvenme; tarayıcıdaki kullanıcı oturumu ile yükleme yapıldı.

Chrome ChatGPT eklentisinde dosya URL erişimi kullanıcı tarafından açıldı.
Kullanıcı açık bırakılmasını istedi. Erişim tokenı/parola dosyalara yazılmadı.
`node_modules`, `.local-run.json`, ham/eğitim verileri, secrets ve artifacts yüklenmedi.
Bu3 yeni devir belgesi henüz uzak depoda yok; yalnızca yerelde oluşturuluyor.

## 9. Render — güncel (14 Kasım 2026, Claude)

Canlı: https://mars-pathfinder-ai.onrender.com. Render GitHub App yalnızca bu repoya
kuruldu; Blueprint (render.yaml, free, Frankfurt) ile servis oluşturuldu, ilk build başarılı.
Public URL'de görev/plan/SSE/ONNX/rapor akışı dış istemciyle doğrulandı; ölçümler TASKS.md'de.
Aşağıdaki alt bölüm Codex devrindeki tarihî durumdur.

## 9a. Render'da kalınan nokta (Codex devri, tarihî)

Dockerfile ve render.yaml, Frankfurt'ta **free** tek Python/ONNX servisini hedefler.
Docker multi-stage Node24 → Python3.12 slim, libgomp1, non-root user, port env.
Frontend build ve prepared data aynı container'da. Tek Uvicorn worker.

Render'a giriş başlatıldı. Kullanıcı GitHub identity ve read-email OAuth iznini
onayladı. Render e-posta doğrulaması istedi; son kullanıcı mesajı doğruladığını
söylüyor. Bundan sonra dashboard tekrar incelenmedi. **Henüz servis oluşturulmadı,
repo bağlantısı/Render GitHub App kurulumu tamamlanmadı, gerçek public URL yok.**

Chrome'da mevcut sekmeler:
- GitHub repo (Code menüsü açık olabilir).
- Render giriş/doğrulama sayfası: dashboard.render.com.
- Yerel uygulama Chrome ve in-app browser sekmeleri.

Giriş sonrası sadece bu repo için gereken erişimi seç. Başka private repolara
geniş yetki verme. Hesap/Terms veya yeni GitHub App yetkileri çıkarsa kendi
çalışma ortamının onay kurallarını uygula; kullanıcının genel ayar devri her
hassas işlemin per-action onayı yerine geçmez. Ücretli plan/kart ekleme.

Önce free deployment build log'unu izle. Runtime ve gerçekONNX çıkarımı + SSE'yi
public URL'de denemeden “yayında çalışıyor” deme. Free tier uyuyabilir, soğuk açılır,
ephemeral disktir; model warm local21 ms değeri Render performans ölçümü değildir.
Free RAM/CPU altında OOM/timeout kontrolü gerekiyor.

## 10. Tasarım araçları ve belgeler

Kullanıcının talebiyle kuruldu:

- `C:\Users\osura\.codex\skills\impeccable\SKILL.md`
  kaynak: `pbakaus/impeccable`, `.agents/skills/impeccable`.
- `C:\Users\osura\.codex\skills\design-taste-frontend\SKILL.md`
  kaynak: `Leonxlnx/taste-skill`, `skills/taste-skill`.

Skill içerikleri okunup mevcut bilimsel dashboard yönüne uygulandı. Impeccable
context launcher exit1 verdi; kullanıcıya söylendi ve proje doğrudan okunarak
devam edildi. Taste headless browser QA launcher'ı bu Codex oturumunun CUA kuralları
nedeniyle çalıştırılmadı. “Taste gate geçti” veya “skill testleri geçti” deme.
Mevcut DOM ekranına göre el ile/CUA görsel kontrol yapıldı.

## 11. Bir sonraki çalışma

Öncelik ve kabul ölçütleri TASKS.md'de. En kritik iki açık: canlı public deployment
ve bağımsız kamera/harita eşleştirmesi. Modelin big-rock başarısızlığını ayrıca
iyileştirmeden başarılı tehlike tespiti/Surface faydası iddia edilemez.
Kullanıcı şu anda devir belgesi istedi; önce bu belgelerden projeyi devral,
tamamlanmış işleri yeniden üretme ve eksikleri tamamlanmış gibi anlatma.
