# Claude için başlangıç

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
Sıradaki zorunlu hedef: gerçek kamera–harita eşleştirmesi (TASKS.md P0 bilimsel açık).

## İlk devam noktası (Codex devri; tarihî)

Kullanıcı Render e-postasını doğruladığını söyledi. Dashboard durumu henüz tekrar
kontrol edilmedi; kamuya açık servis oluşturulmadı. Kod ve veri GitHub'a yüklendi.
Önce uzak dosyaları yerelle karşılaştır, ardından ücretsiz Render yayınını tamamla.
Handoff dosyaları yalnızca yerelde oluşturuldu; kendileri henüz GitHub'a yüklenmedi.
