# Sera Tarımında Pekiştirmeli Öğrenme: PPO ile Domates Serası Kontrolü

Şanlıurfa benzeri sıcak ve kurak bir iklimde, plastik bir domates serasının **ısıtma, havalandırma, sulama ve gübre** kararlarını her gün birlikte veren bir PPO ajanı. Sera simülatörü ve PPO **sıfırdan** yazılmıştır (Gymnasium + PyTorch); tüm grafik ve tablolar ham CSV loglarından üretilir.

**Burak Kılcı** · 230202111 · Kocaeli Üniversitesi Bilgisayar Mühendisliği
BLM446 Pekiştirmeli Öğrenme dönem projesi, genişletilmiş analiz (Ekim 2026)

---

## Sonuç özeti

20 test senaryosu (eğitimde ve hiçbir seçimde kullanılmadı), deterministik politika:

| Politika | Ödül | Verim (t/ha) |
|---|---:|---:|
| Hiçbir şey yapmama | −8.57 ± 0.10 | 0.02 |
| Her şeyi açma | 10.88 ± 2.35 | 1.07 |
| Orijinal kural (elle seçilmiş eşikler) | 23.81 ± 1.41 | 1.45 |
| Ayarlı kural (validation'da 900 kombinasyon) | 126.35 ± 3.31 | 6.11 |
| **PPO, 5 seed** | **129.18 ± 3.57** | **6.23** |

- Orijinal kurala göre **4.30×** verim. Ancak bu kural toprağı 120 günün 120'sinde su stresinde tutuyor.
- Eşikleri adilce ayarlanmış kurala göre **1.02×** verim; eşleştirilmiş fark **2.83 ± 0.31** ödül, beş seed'in her biri **20 senaryonun 20'sinde** önde.
- Kalan farkın kaynağı ölçüldü: daha iyi su ve sıcaklık yönetimi ve **gübrenin sezona göre zamanlanması** (PPO gübresinin %11'ini sezonun son üçte birinde veriyor, ayarlı kural %26).
- Dürüst sonuç: bu ortamın en iyi politikası sabit eşiklere yakın; PPO doğru çalışma noktasını kural bilgisi olmadan buluyor, ama PÖ'nün avantajı küçük.

Ayrıntılar: [`rapor/230202111_sera_rapor.pdf`](rapor/230202111_sera_rapor.pdf) (IEEE formatı, 9 sayfa).

---

## Proje yapısı

```
├── config.yaml               # TÜM sayısal parametreler (ortam, ödül, PPO, deney, arama)
├── config.py                 # config.yaml'ı okur
├── greenhouse_env.py         # Gymnasium sera ortamı: iklim, toprak–su–azot, bitki
├── baselines.py              # Do-nothing, max-out, hand-crafted, ayarlı hand-crafted
├── ppo_agent.py              # Sıfırdan PPO; eğitim, CSV loglama, değerlendirme
├── test_env.py               # Duman testi ve hand-crafted trajektorisi
├── hyperparameter_sweep.py   # lr × clip_eps taraması (5 seed, 150k adım)
├── tune_handcrafted.py       # Hand-crafted eşiklerinin validation'da grid search'ü
├── final_evaluation.py       # 20 test senaryosunda final değerlendirme
├── plot_results.py           # Ham loglardan 5 grafik ve özet tablolar
├── policy_analysis.py        # Trajektori, davranış özeti, ödül ayrıştırması, sondaj
├── dashboard.py              # Etkileşimli Tkinter arayüzü
├── visualize_episode.py      # Episode animasyonu (MP4)
├── run_all.sh / run_all.ps1  # Tüm hattı sırayla çalıştırır
├── requirements.txt, seeds.txt, DEGISIKLIK_LISTESI.md
├── runs/ppo_seed{0,1,2,3,42}/  # Final checkpoint'ler + episodes/updates/evals CSV
├── sonuclar/                 # Final sonuçlar, özet tablolar; loglar/ = tüm ham CSV loglar
├── grafikler/                # 5 zorunlu grafik, trajektori, sondaj, episode videosu
└── rapor/                    # LaTeX kaynağı ve PDF
```

## Kurulum

Python 3.11, NVIDIA GPU önerilir (CPU'da da çalışır, daha yavaş).

```bash
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Çalıştırma

Tüm hat: `bash run_all.sh` (Windows: `.\run_all.ps1`). Adım adım:

| Komut | Ne yapar? | Süre (RTX 2060) |
|---|---|---|
| `python test_env.py` | Ortam ve baseline duman testi | ~20 sn |
| `python ppo_agent.py --all` | 5 seed × 500k adım eğitim | ~2.5 saat |
| `python ppo_agent.py --seed 0 --steps 20480 --out runs_test` | Kısa deneme eğitimi | ~1 dk |
| `python hyperparameter_sweep.py` | 25 kısa eğitim (lr, clip_eps) | ~3.2 saat |
| `python tune_handcrafted.py` | 900 eşik kombinasyonu, validation | ~1–2 dk |
| `python final_evaluation.py` | Final test, eşleştirilmiş karşılaştırma | ~1–2 dk |
| `python plot_results.py` | 5 grafik, `sonuclar.csv`, `grafik_ozet.csv` | birkaç sn |
| `python policy_analysis.py` | Davranış analizi ve politika sondajı | ~1–2 dk |
| `python dashboard.py` | Etkileşimli arayüz | — |

Repodaki checkpoint'ler ve loglarla eğitim yapmadan `final_evaluation.py`, `plot_results.py` ve `policy_analysis.py` doğrudan çalıştırılabilir. Hiperparametre taramasının 25 ara checkpoint'i repoya konmadı; ham logu `sonuclar/loglar/sweep_log.csv`'de.

## Ortam ve MDP

- **Durum (13):** iç sıcaklık, iç nem, CO₂, toprak suyu, toprak azotu, toprak sıcaklığı, LAI, biyokütle, gelişme aşaması (DVS), dış sıcaklık, dış nem, ışınım, normalize gün
- **Eylem (4, sürekli):** ısıtma (0–200 W/m²), havalandırma (%0–100), sulama (0–10 mm/gün), gübre (0–5 kg N/ha/gün)
- **Ödül:** günlük büyüme − su, enerji ve gübre maliyetleri − stres ve aşırı nem cezaları; sezon sonunda verime bağlı hasat bonusu (`greenhouse_env.py` → `_compute_reward`)
- **Episode:** 120 gün ya da DVS ≥ 2.0; γ = 0.99
- **Stokastiklik:** sezonluk hava, günlük iklim geçişi ve biyolojik büyüme gürültüsü; yedi parametrenin hepsi `config.yaml → environment.stochasticity`

## Tekrar üretilebilirlik

- Rastgelelik `numpy.random.default_rng(seed)` ve `torch.manual_seed(seed)` ile; seed grupları `seeds.txt`'te.
- Senaryolar ayrıktır: eğitim (0, 1, 2, 3, 42), validation (3000+), test (2000–2019). Test senaryoları hiçbir seçimde kullanılmaz.
- `plot_results.py` grafikleri yalnızca `sonuclar/loglar/` altındaki ham CSV'lerden çizer.
- Aynı makinede aynı seed aynı öğrenme eğrisini verir; GPU'daki paralel toplamlar nedeniyle farklı donanımlarda küçük sayısal farklar olabilir.

## Sınırlamalar

Hava verisi sentetik, sera modeli sadeleştirilmiş ve gerçek ölçümle kalibre edilmemiştir. Rakip olarak yalnızca oransal (P) kontrolcü kullanılmıştır; PI ve MPC ile karşılaştırma gelecek çalışmadır.

## Kaynaklar

- J. Schulman vd., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347, 2017.
- R. S. Sutton ve A. G. Barto, *Reinforcement Learning: An Introduction*, 2. baskı, MIT Press, 2018.
- B. H. E. Vanthoor vd., *A methodology for model-based greenhouse design*, Biosystems Engineering 110(4), 2011.

Tam kaynakça raporda.
