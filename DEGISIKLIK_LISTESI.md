# Değişiklik Listesi

**Burak Kılcı (230202111)** · BLM446 Pekiştirmeli Öğrenme
Dönem sonu teslimine (Mayıs 2026) göre genişletilmiş analizde (Ekim 2026) yapılan değişiklikler.

## 1. Kod

| Dosya | Değişiklik |
|---|---|
| `config.yaml` | **Yeni.** Tüm sayısal parametreler tek dosyada: 69 ortam parametresi, ödül ağırlıkları, PPO hiperparametreleri, seed grupları, tarama ve eşik arama ayarları. |
| `config.py` | Artık sabit değer içermiyor; `config.yaml`'ı okuyor. Değişken adları korundu. |
| `greenhouse_env.py` | Koda gömülü tüm ayarlanabilir sayılar (gürültü seviyeleri, hava trendi, başlangıç durumu, nem eşiği vb.) `config.yaml`'a taşındı. Davranış değişmedi (bkz. Bölüm 4). |
| `baselines.py` | **Yeni.** Do-nothing, max-out ve hand-crafted politikalar tek yerde; eşikler `config.yaml`'da. Ayarlı hand-crafted desteği eklendi. |
| `ppo_agent.py` | Algoritma değişmedi. Eklenenler: `episodes.csv`, `updates.csv` (policy/value loss, entropi, yaklaşık KL, clip oranı) ve `evals.csv` (her 10 güncellemede deterministik test) ham logları; komut satırı argümanları (`--seed`, `--all`, `--steps`, `--eval`). |
| `test_env.py` | Politikalar `baselines.py`'dan alınıyor; seedler ve grafik çizgileri `config.yaml`'dan. Docstring'deki "0.25 hedef" ifadesi koddaki gerçek değerle (0.28) uyumsuzdu, düzeltildi. |
| `hyperparameter_sweep.py` | Değerler `config.yaml → sweep`'ten okunuyor. |
| `tune_handcrafted.py` | **Yeni.** Hand-crafted eşiklerinin validation senaryolarında 900 kombinasyonluk grid search'ü. |
| `final_evaluation.py` | Sadeleştirildi; ayarlı kural ve eşleştirilmiş karşılaştırma eklendi; özetini UTF-8 dosyaya yazıyor. |
| `plot_results.py` | **Yeni.** Beş zorunlu grafiği yalnızca ham loglardan çizer; `sonuclar.csv`, `hiperparametre_ozet.csv`, `grafik_ozet.csv` üretir. |
| `policy_analysis.py` | **Yeni.** Trajektori karşılaştırması, davranış özeti, ödül ayrıştırması ve politika sondajı. |
| `run_all.sh`, `run_all.ps1`, `.gitignore` | Güncellendi / yeni. |
| `compare_seeds.py` | **Silindi**; yerini `plot_results.py` (Grafik 1) aldı. |

## 2. Deneyler

- Beş ana eğitim (seed 0, 1, 2, 3, 42) **aynı kod sürümüyle** yeniden koşuldu. İlk teslimde seed 0, 1, 42 Nisan'da; seed 2 ve 3 Mayıs'ta eğitilmişti.
- Hiperparametre duyarlılığı: lr ∈ {0.0001, 0.0003, 0.001} × clip_eps ∈ {0.1, 0.2, 0.3}, her biri 5 seed × 150k adım.
- Ayarlı hand-crafted kontrolcü: validation senaryolarında (3000–3009) 900 kombinasyon; test senaryoları aramada kullanılmadı.
- Eşleştirilmiş senaryo karşılaştırması, ödül ayrıştırması, davranış analizi ve politika sondajı.

## 3. Sonuçlar

| | İlk teslim | Genişletilmiş analiz |
|---|---:|---:|
| PPO test ödülü (20 senaryo) | 129.41 ± 3.61 | 129.18 ± 3.57 |
| PPO test verimi | 6.24 t/ha | 6.23 t/ha |
| Orijinal kurala göre verim | 4.31× | 4.30× |
| Ayarlı kurala göre verim | — | 1.02× |

Yeniden üretilen sonuçlar ilk teslimle virgülden sonra ikinci basamakta örtüşmektedir.

## 4. Doğrulama testleri

- Parametreler `config.yaml`'a taşınırken ortam, önceki sürümle 24 episode ve 2 568 adım boyunca **bit düzeyinde** karşılaştırıldı: gözlemler, ödüller ve ödül bileşenleri birebir aynı.
- `baselines.py`'daki hand-crafted politika, önceki sürümle 25 senaryoda eylem eylem aynı; duman testi sonuçları değişmedi (24.34 / 1.47 t/ha).

## 5. Düzeltilen iddialar

- **"PPO proaktif sular":** desteklenmedi. Sulama–LAI korelasyonu PPO'da 0.79, orijinal kuralda 0.98. Çıkarıldı.
- **Dashboard'dan aktarılan havalandırma değerleri** ("yüksek LAI'de havalandırma %12" vb.): politika sondajıyla doğrulanmadı (PPO bu koşullarda havalandırmayı ≈ %87–100 açıyor). Çıkarıldı.
- **Isıtıcı üst sınırı:** eski sunumda 100 W/m² yazıyordu; koddaki değer 200 W/m². Düzeltildi.
- **"4.3× üstünlük":** korunuyor, ancak yalnızca orijinal (kötü ayarlı) kurala karşı geçerli olduğu ve ayarlı kurala karşı farkın %2 olduğu açıkça belirtiliyor.

## 6. Rapor ve sunum

Rapor (IEEE, 9 sayfa) ve sunum (28 slayt) yeni sonuçlarla baştan yazıldı: zorunlu beş grafik ve 4 cümlelik yorumları, "Neden PÖ?" bölümü, ödül formülünün kod satırlarıyla eşleştirilmesi, adil karşılaştırma bölümü ve PPO'nun zayıf noktaları eklendi.
