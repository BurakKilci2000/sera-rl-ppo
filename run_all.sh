#!/usr/bin/env bash
# ============================================================
# run_all.sh — Tüm deney hattını sırayla çalıştırır
# Burak Kılcı (230202111)
#
# Kullanım (Linux/macOS ya da Windows'ta Git Bash):
#   bash run_all.sh
# Windows PowerShell için: run_all.ps1
#
# Süreler (RTX 2060): 5 seed ana eğitim ~2.5 saat,
# hiperparametre taraması (25 × 150k adım) ~3.2 saat.
# Eğitimi tamamlanmış adımlar tekrar çalıştırılınca atlanmaz;
# ana eğitimi yeniden koşturmak istemiyorsan 2. adımı yorum satırı yap.
# ============================================================
set -e

echo ">>> [1/8] Duman testi (ortam ve baseline'lar)"
python test_env.py

echo ">>> [2/8] PPO ana eğitimi: 5 seed × 500k adım"
python ppo_agent.py --all

echo ">>> [3/8] Hiperparametre taraması (lr, clip_eps)"
python hyperparameter_sweep.py

echo ">>> [4/8] Hand-crafted eşiklerinin validation senaryolarında ayarlanması"
python tune_handcrafted.py

echo ">>> [5/8] Final değerlendirme: 20 test senaryosu"
python final_evaluation.py

echo ">>> [6/8] Beş zorunlu grafik ve özet tablolar"
python plot_results.py

echo ">>> [7/8] Politika davranış analizi (trajektori, ödül ayrıştırması, sondaj)"
python policy_analysis.py

echo ">>> [8/8] Episode animasyonu (isteğe bağlı)"
python visualize_episode.py

echo "TAMAMLANDI: grafikler/ ve sonuclar/ klasörlerine bakın."
