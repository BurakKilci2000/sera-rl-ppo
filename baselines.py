"""
baselines.py
============
PPO ile karşılaştırılan basit politikalar. Hepsi tek yerde tanımlıdır;
test_env.py, final_evaluation.py, tune_handcrafted.py ve policy_analysis.py
buradan import eder.

  - Do-nothing           : hiçbir kaynak kullanmaz (tüm eylemler minimum)
  - Max-out              : her şeyi sürekli tam açar (tüm eylemler maksimum)
  - Hand-crafted         : oransal (P) kontrolcü; eşikler config.yaml → handcrafted
  - Hand-crafted (ayarlı): aynı kontrolcü, eşikleri validation senaryolarında
                           grid search ile seçilmiş (tune_handcrafted.py)

Tüm politikalar gözlemi alır, [-1, 1] aralığında 4 boyutlu eylem döner.
"""
import os
import numpy as np
import yaml

from config import HANDCRAFTED as HC
from greenhouse_env import GreenhouseEnv

IDX = GreenhouseEnv.IDX
TUNED_PATH = "./sonuclar/handcrafted_ayarli.yaml"


def policy_donothing(obs):
    return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)


def policy_maxout(obs):
    return np.array([+1.0, +1.0, +1.0, +1.0], dtype=np.float32)


def make_handcrafted(hc):
    """Verilen eşiklerle bir oransal kontrolcü üretir.

    Her eylem kendi eşiğine bakar; değişkenler arası etkileşimi görmez
    (LAI'yi, gelişme aşamasını ve sezonu hiç kullanmaz)."""
    def policy(obs):
        T_in   = obs[IDX["T_in"]]
        H_in   = obs[IDX["H_in"]]
        W_soil = obs[IDX["W_soil"]]
        N_soil = obs[IDX["N_soil"]]

        # Sıcaklık: hedefin üstündeyse havalandır, altındaysa ısıt
        err_T = T_in - hc["T_target"]
        if err_T > 0:
            a_vent = min(1.0, err_T / hc["T_band"])
            a_heat = 0.0
        else:
            a_vent = 0.0
            a_heat = min(1.0, -err_T / hc["T_band"])

        # Aşırı nem → ek havalandırma
        if H_in > hc["H_vent_threshold"]:
            a_vent = max(a_vent, hc["H_vent_min"])

        # Sulama: toprak suyu hedefin altındaysa
        if W_soil < hc["W_target"]:
            a_irr = min(1.0, (hc["W_target"] - W_soil) / hc["W_band"])
        else:
            a_irr = 0.0

        # Gübre: azot hedefin altındaysa
        if N_soil < hc["N_target"]:
            a_fert = min(1.0, (hc["N_target"] - N_soil) / hc["N_band"])
        else:
            a_fert = 0.0

        # [0, 1] → [-1, 1]
        return 2.0 * np.array([a_heat, a_vent, a_irr, a_fert], dtype=np.float32) - 1.0
    return policy


# Orijinal hand-crafted: config.yaml'daki eşiklerle
policy_handcrafted = make_handcrafted(HC)


def load_tuned_handcrafted():
    """Ayarlı eşikler varsa (tune_handcrafted.py çıktısı) politikayı döner, yoksa None."""
    if not os.path.exists(TUNED_PATH):
        return None
    with open(TUNED_PATH, encoding="utf-8") as f:
        best = yaml.safe_load(f)["en_iyi_esikler"]
    return make_handcrafted({**HC, **best})


BASELINES = {
    "Do-nothing":   policy_donothing,
    "Max-out":      policy_maxout,
    "Hand-crafted": policy_handcrafted,
}
_tuned = load_tuned_handcrafted()
if _tuned is not None:
    BASELINES["Hand-crafted (ayarlı)"] = _tuned
