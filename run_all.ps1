# ============================================================
# run_all.ps1 — Tüm deney hattı (Windows PowerShell)
# Burak Kılcı (230202111)
#
# Kullanım:  .\run_all.ps1
# (İlk kez çalıştırırken gerekirse:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
# ============================================================
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$steps = @(
    @("Duman testi (ortam ve baseline'lar)", "test_env.py", @()),
    @("PPO ana eğitimi: 5 seed x 500k adım", "ppo_agent.py", @("--all")),
    @("Hiperparametre taraması (lr, clip_eps)", "hyperparameter_sweep.py", @()),
    @("Hand-crafted eşiklerinin ayarlanması", "tune_handcrafted.py", @()),
    @("Final değerlendirme: 20 test senaryosu", "final_evaluation.py", @()),
    @("Beş zorunlu grafik ve özet tablolar", "plot_results.py", @()),
    @("Politika davranış analizi", "policy_analysis.py", @()),
    @("Episode animasyonu (isteğe bağlı)", "visualize_episode.py", @())
)

$i = 1
foreach ($s in $steps) {
    Write-Host ">>> [$i/$($steps.Count)] $($s[0])"
    python $s[1] @($s[2])
    if ($LASTEXITCODE -ne 0) { throw "Hata: $($s[1])" }
    $i++
}
Write-Host "TAMAMLANDI: grafikler\ ve sonuclar\ klasörlerine bakın."
