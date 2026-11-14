# Current continuation — 15 November 2026

Latest follow-up: [REPLAY_VISUAL_FIXES.md](REPLAY_VISUAL_FIXES.md). Full-approach
paint taper, an explicitly projected destination flag, and persistent replay
entities/DEM heights replace the abrupt taper and zoom/stationary-frame flicker.
Only replay annotation/render behavior changed; panel layout and models stay intact.

The interrupted corridor/continuous-route work after `d99922b` is documented in
[CONTINUATION_REVIEW.md](CONTINUATION_REVIEW.md). Read it before the older notes.
Connected measured side exits replace the unfinished free-pivot shortcut; near
and map routes use one tapered geometry shared by browser/video. Map-only context
is labelled unverified. Rock memory can use current stereo reconfirmation and
expires by elapsed frames. Model v11 and the global map remain unchanged.

# Previous work: local Hazcam obstacles (14 November 2026) — PROMOTED (v11)

Done by Claude after Codex ran out of tokens. User sensitivity: match the red reference rocks,
ignore only very small gravel, include large raised bedrock blocks, never mistake rover equipment
for rocks. UI, global 3D map, recorded motion and other modules preserved. Details: MODEL_REFINEMENT.md.

- `data/prepared/navigation.onnx` (SegFormer-B0 local head v11, 512 px, sha 6dcf4882…) and
  `navigation-model-card.json` are installed. Global v3 `surface.onnx` is unchanged.
- Selected on validation only with the pre-declared objective (v11 0.552 > v10 0.548 > v10-normalized
  0.534); ONNX parity 100 %; peak 314 MiB. Frozen before the 3 fresh test scenes were opened once.
- Validation (5 scenes): 27/36 rocks found, rover hardware as obstacle 0.05 %.
  Fresh test (3 scenes): 7/20 rocks found (35 %), hardware as obstacle 0.002 %.
  Previous runtime (v3 big_rock) on the same scenes: 0/36 and 0/20, hardware-as-rock 9 % / 4 %.
- Appearance only proposes; stereo relief vs nearby measured ground decides occupancy.
- The 3 test scenes are now consumed. Dataset v6 (+10 reviewed scenes, superpixel tool) trained a
  challenger v13: better on discrete rocks/hardware but failed the pre-declared IoU gate on the new
  test_v6 (block terrain), so v11 stays. v11 on independent test_v6: 20/32 rocks (62 %). test_v6 is
  consumed too; next = more raised-block scenes + a new holdout.
- Speed: default replays are computed during the Docker build (sealed by input hash, labelled as
  build-time); A* and route sampling were sped up with identical outputs. Render: ~1 s instead of minutes.
- 15 Nov: camera route continues past the measured corridor as a thin line along the projected map
  plan to the destination (dotted where hidden, ends in a dot, no arrow head). The local planner aims
  at the map route 6 m ahead so the two connect. Stereo-confirmed rocks (raised steps + confirmed rocks)
  are remembered in global coordinates for 12 frames / 10 m and fill only cells the current view
  cannot measure (orange). Distant 4-12 m stereo rock candidates are drawn as orange rings.
- Dataset v7 (+8 scenes incl. raised blocks, new test_v7). Challenger v14: validation 0.614 vs 0.552,
  but failed the IoU gate on test_v7 (0.38 vs 0.42, only 7 objects), so v11 stays. test_v7 consumed;
  next = a larger holdout (>= 30 objects) before another promotion.
- NVIDIA-derived weights: non-commercial research/evaluation only (LICENSES/NVIDIA-SegFormer.txt).

---

# Claude için başlangıç

## Güncel kullanıcı isteği — önce bunu esas al

**En son (Claude, 14 Kasım gece):** Kullanıcının 5 isteği tamamlandı (ayrıntı TASKS.md son bölüm):
rota roverin ayağından bağlı çıkar; hedef arkadaysa U salınımı veya yerinde dönüş oku; oynatma
önbellekli ve kesintisiz; Explore/Plan'da ×12 abartılmış MOLA kabartma + DEM yüksekliğinde rota;
model v3 (test mIoU 0.665, big_rock IoU 0.213). Stereo sözde-etiket ince ayarı denendi, kaya
skorunu düşürdüğü için kullanılmadı (negatif sonuç README'de). v2 yedeği:
`%LOCALAPPDATA%/mars-pathfinder/training/backup-v2`.

**En son (Claude, 14 Kasım akşam):** Kullanıcının "GTA minimap çizgisi değil, gerçek engelden
kaçan, zemine oturan AR rota" isteği `backend/local_navigation.py` ile tamamlandı. Stereo XYZ
(hata ≤0.10 m, ~4 m'ye kadar güvenilir) → 10 cm zemin ızgarası; engel = ≥0.25 m basamak, 0.2 m
düzleştirilmiş eğim >20°, stereo ile doğrulanan model kayası. Gövde modeli: 2.8 m genişlik koridoru
(merkez çizgisi ölçülmüş engellerden ≥1.62 m), rota ölçülmüş zeminden geçer, koridorun ≥%50'si
ölçülmüş olmalı; rover kendi gövdesi altında yerinde dönebilir. A* + çarpışma kontrollü kısaltma +
yeniden kontrol edilen Chaikin yumuşatma; PDS CAHVORE ile 0.5 m zemin şeridi, rover genişliği kenarları
ve kırmızı engel işaretleri. Hedef arkadaysa U dönüşü. 142 karenin 63'ünde rota (Jake Matijevic kayası
soldan dolaşılıyor); diğerlerinde neden yazılır, sahte çizgi yok. 45 test geçti; 3 MP4 yenilendi.

Son düzeltme: kullanıcı çizginin her zaman roverin alt-ortasından çıkmasını ve
"sağa yöneldi" gibi gerçek hareket bilgisinin ayrı bir algoritmayla gösterilmesini
istedi. Varsayılan **Centre guide**, [0.5,0.94] rover işaretine bağlı yönlendirme
HUD'udur; zemin ölçümü gibi sunulmaz. Gerçek kalibre izdüşüm **Ground projection**
seçeneğinde korunur. `backend/guidance.py` yönlendirmeyi kalan rota/baş açısından,
Recorded motion bilgisini ise ardışık gerçek PLACES poz farkından hesaplar.
Ekran görüntüsündeki Sol54 13:36:36 karesi birebir kontrol edildi: sol yönlendirme
çizgisi artık merkezden çıkar; kayıtlı konum o anda değişmediği için hareket etiketi
Position unchanged'dir.32 test geçti. Üç MP4 de yeni merkez kılavuzuyla yenilendi.

Son proje sohbetleri ve kullanıcının Paint çizimi okundu. İstenen varsayılan,
gerçek rover görüntüsü solda ve aynı rota/konumun Mars haritasında sağda olması;
rover kaydı ilerledikçe geçen yolun geride kalması, sapınca yeniden rota ve gerekirse
U dönüşü önerisi. 3D render yalnız yardımcı görünüm. Bir görüntüye göz kararı çizgi
koymak veya 3D simülasyonu gerçek footage yerine göstermek isteği karşılamıyor.

Güncel uygulama: kullanıcının "4 aşırı az, başka videolarda ara" düzeltmesinden
sonra142 gerçek Front Hazcam ürünü (Sol40–150),207 toplam kare hazırlandı.
Varsayılan142 çekimlik interaktif kayıt yaklaşık459 m kaydedilmiş hareket,
19 replan ve2 dakikalık MP4 içerir. İki kısa4-kare tanılama senaryosu korunur.
NASA'nın192 gerçek kareli Perseverance Sol200 filmi ve5 dakika49 saniyelik
Curiosity filmi uygulamadan doğrudan oynatılır; bunlara uydurma planner eşleşmesi
eklenmez. Başlangıçta uzun gerçek kamera/harita kaydı açılır. Pozlar
arşiv PLACES kaynaklıdır; bu4 Hazcam görüntüsü bağımsız orbital eşleşme testini
geçmedi ve haritaya engel yazmaz.142 kamera kalibrasyonunda p95 0.14–0.37 piksel hatası,
mutlak harita doğruluğu anlamına gelmez. Uzak çizgi yüksekliği native DEM'den
gelir. İnteraktif kayıt ve2 dakikalık film142 gerçek çekimden oluşan timelapse'tir;
çekimler arasında konum veya görüntü üretilmez.30 test ve production build geçti.

Önceki devir ve tarihî görev maddeleri aşağıda korunmuştur; güncel uygulama için
README'nin "Real-footage navigation" bölümü ve TASKS.md'nin son bölümü esas alınır.

Bu proje Codex'ten Claude'a devrediliyor. Kullanıcı 14 Kasım 2026'da Codex'in
uygulama/yayın çalışmalarını durdurmasını ve devam edebilmen için devir belgeleri
hazırlamasını istedi. Projeyi yeniden kurarak başlama; çalışan uygulama, gerçek
veriler ve eğitilmiş model zaten var.

Önce sırayla oku:

1. [HANDOFF.md](HANDOFF.md): mevcut durum, mimari, ortam, GitHub/Render, riskler.
2. [TASKS.md](TASKS.md): kalan işlerin önceliği ve kabul ölçütleri.
3. [README.md](README.md), [PRODUCT.md](PRODUCT.md), [DESIGN.md](DESIGN.md).
4. Bilimsel ayrıntı gerektiğinde `data/prepared/frames.json`,
   `data/prepared/model-card.json`, `data/prepared/pds/pds-candidate.json`.

## Kullanıcı tercihleri

- Kullanıcıyla Türkçe iletişim kur. Gereksiz tekrar onayı isteme; yetkilendirilmiş
  ve geri alınabilir işleri kendin tamamla. Harcama yetkisi verilmedi.
- GitHub deposu **private** kalacak: https://github.com/Poyraxx/mars-pathfinder-ai
- Kendini/Claude'u/Codex'i commit yazarı, coauthor veya katkıcı olarak ekleme.
  `Co-authored-by` ekleme. Yarışma için AI araç kullanımını dürüstçe açıklamak
  ayrı bir gerekliliktir; mevcut README açıklamasını bu tercihle karıştırma.
- Ücretsiz hizmetlerle başla. Ücretli GPU, Render planı veya abonelik açma.
- Tasarım için kullanıcı `impeccable` ve `taste-skill` kurdurdu. İkisi de yerel
  Codex skills klasöründe; konumlar HANDOFF.md'de. Mevcut NASA Eyes esinli bilimsel
  kontrol panelini genel bir pazarlama sayfasına dönüştürme.
- İkinci öğrenilmiş orbital model zorunluluğu, uygulanabilir uçtan uca sistem
  lehine gevşetildi. Orbital modülün deterministik olduğunu açık tut.
- Birkaç gerçek arşiv karesiyle **gerçek kamera–harita eşleştirme ve buna dayalı
  yeniden planlama hâlâ zorunlu hedef**. Şu an bu hedef tamamlanmış değil.
- Demo, sunucuda canlı hesaplama yapmalı. Önceden kaydedilmiş sonuçları canlı
  çıkarım gibi sunma. Ücretsiz sunucu uyuması/soğuk açılışı kabul edildi.

## Bilimsel doğruluk sınırları

Sahte veri, uydurulmuş eşleşme, test metriği veya engel üretme. Mevcut kamera
karelerinin bağımsız harita doğrulaması yok; güven kapısını yalnızca demo güzel
görünsün diye açma. Modelin held-out testinde büyük kaya recall değeri **0%**.
İnsan EVA güvenliği, uçuş yeterliliği veya güvenli yol garantisi iddia etme.

## Hızlı yerel başlangıç

Çalışma klasörü: `C:\Users\osura\OneDrive\Masaüstü\nasa`.

Uygulama devir anında zaten `http://127.0.0.1:8000` ve geliştirme arayüzü
`http://127.0.0.1:5173` üzerinde açıktı. Önce kontrol et; ikinci sunucu başlatma.

```powershell
Set-Location -LiteralPath 'C:\Users\osura\OneDrive\Masaüstü\nasa'
.\scripts\start.ps1
```

Test/derleme komutları ve PATH dışındaki Node/Python/Git konumları HANDOFF.md'de.
Yerel `.git` henüz commit içermiyor; dosyalar staged. Uzak depo ise tarayıcıdan
11 gerçek commit ile dolduruldu. Körlemesine force-push/reset yapma.

## Güncel durum (14 Kasım 2026, Claude oturumu)

Render free servisi canlı: https://mars-pathfinder-ai.onrender.com — public URL'de uçtan
uca akış doğrulandı (ayrıntı TASKS.md P0). Uzak repo 71/71 dosya yerelle eşleşiyor.
Yerel git hâlâ commitsiz: bu ortamda GitHub kimlik bilgisi yok, `git fetch` çalışmıyor.
Kamera–harita eşleştirmesi görüntü korelasyonu ile ölçüldü (3 kare, 0.35–0.75 m, p ≤ 0.0025);
harita engelleri stereo basamak yüksekliğinden geliyor, model haritaya yazmıyor. Kullanıcı
proje bitene kadar karar yetkisini Claude'a verdi; GitHub'a Claude'u yazar/katkıcı/coauthor
olarak ekleme (kullanıcı ayrıca vurguladı). Sıradaki: model iyileştirme (P1), mobil QA, teslim.
İkinci tur (aynı gün): 30 Navcam karesi (13 kayıtlı), kamera görüntüsünde rota, tüm Mars kaba planlama,
2055 IAU adı, öğrenilmiş HiRISE/CTX orbital model, AI4Mars v2 yeniden eğitim. Ayrıntı TASKS.md.
Yerel git artık çalışıyor (kullanıcı git kurdu); commit kimliği repo-yerel Poyraxx noreply.

## İlk devam noktası (Codex devri; tarihî)

Kullanıcı Render e-postasını doğruladığını söyledi. Dashboard durumu henüz tekrar
kontrol edilmedi; kamuya açık servis oluşturulmadı. Kod ve veri GitHub'a yüklendi.
Önce uzak dosyaları yerelle karşılaştır, ardından ücretsiz Render yayınını tamamla.
Handoff dosyaları yalnızca yerelde oluşturuldu; kendileri henüz GitHub'a yüklenmedi.
