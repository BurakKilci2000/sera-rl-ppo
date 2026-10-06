"""
config.py
=========
config.yaml dosyasını okuyup diğer modüllere sunar.

Tüm sayısal değerler (ödül ağırlıkları, PPO hiperparametreleri, seedler)
config.yaml'da tutulur; bu dosyada ve diğer modüllerde sabit sayı yoktur.
Bir değeri değiştirmek için yalnızca config.yaml düzenlenir.

Değişken adları eski config.py ile aynı bırakıldı; diğer dosyaların
import satırları değişmeden çalışır.
"""
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

REWARD_WEIGHTS = CFG["reward_weights"]           # ödül ağırlıkları
PPO_HP         = CFG["ppo"]                      # PPO hiperparametreleri
EVAL_CFG       = CFG["evaluation"]               # periyodik deterministik eval
BASELINE_REF   = CFG["baseline_reference"]       # grafik referans çizgileri
PLOT_CFG       = CFG["plotting"]                 # grafik ayarları
HANDCRAFTED    = CFG["handcrafted"]              # hand-crafted baseline eşikleri
HC_SEARCH      = CFG["handcrafted_search"]       # ayarlı hand-crafted araması
SWEEP_CFG      = CFG["sweep"]                    # hiperparametre taraması
ANALYSIS_CFG   = CFG["analysis"]                 # politika davranış analizi
EVAL_SEEDS     = CFG["experiment"]["eval_seeds"]  # eğitim seedleri
TEST_SEEDS     = CFG["experiment"]["test_seeds"]  # final test senaryoları
SMOKE_SEEDS    = CFG["experiment"]["smoke_test_seeds"]     # duman testi senaryoları
TRAJ_SEED      = CFG["experiment"]["trajectory_seed"]      # trajektori senaryosu
TRAJ_PPO_SEED  = CFG["experiment"]["trajectory_ppo_seed"]  # trajektorideki PPO ajanı

# Ortam parametreleri: config.yaml'da okunabilirlik için alt bölümlere
# ayrılmıştır; GreenhouseParams için tek düz sözlüğe açılır.
ENV_PARAMS = {}
for _section, _values in CFG["environment"].items():
    for _key, _val in _values.items():
        if _key in ENV_PARAMS:
            raise KeyError(f"config.yaml: '{_key}' anahtarı iki kez tanımlı")
        ENV_PARAMS[_key] = _val
