"""
tune_handcrafted.py
===================
Hand-crafted kontrolcünün eşiklerini grid search ile ayarlar; böylece PPO
kötü ayarlanmış bir kurala karşı değil, elle ayarlanabilecek makul bir
kurala karşı da kıyaslanır.

Adillik kuralları (config.yaml → handcrafted_search):
  - Seçim yalnızca validation senaryolarında yapılır (3000+).
    Test senaryoları (2000+) bu scriptte hiç kullanılmaz.
  - Seçim ölçütü PPO'nun optimize ettiği şeyle aynıdır: ortalama episode ödülü.

Çıktılar:
  sonuclar/handcrafted_arama.csv   ← denenen her kombinasyon (ödüle göre sıralı)
  sonuclar/handcrafted_ayarli.yaml ← en iyi eşikler (baselines.py bunu okur)

Sonrasında: python final_evaluation.py  (ayarlı kural test senaryolarında değerlendirilir)

Kullanım:
  python tune_handcrafted.py
"""
import os
import csv
import itertools
import time
import numpy as np
import yaml

from greenhouse_env import GreenhouseEnv, GreenhouseParams
from baselines import make_handcrafted, TUNED_PATH
from config import HANDCRAFTED, HC_SEARCH

RES_DIR = "./sonuclar"


def evaluate(policy, seeds):
    env = GreenhouseEnv(GreenhouseParams())
    R, Y = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            total += r
            done = term or trunc
        R.append(total)
        Y.append(info["yield_ton_ha"])
    return np.array(R), np.array(Y)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    s0 = HC_SEARCH["search_seed_start"]
    seeds = list(range(s0, s0 + HC_SEARCH["n_search_seeds"]))
    grid = HC_SEARCH["grid"]
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"{len(combos)} kombinasyon × {len(seeds)} validation senaryosu "
          f"(seed {seeds[0]}–{seeds[-1]})")

    t0, rows = time.time(), []
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        R, Y = evaluate(make_handcrafted({**HANDCRAFTED, **params}), seeds)
        rows.append({**params,
                     "val_odul_ort": round(float(R.mean()), 3),
                     "val_odul_std": round(float(R.std()), 3),
                     "val_verim_ort": round(float(Y.mean()), 4)})
        if i % 20 == 0:
            print(f"  {i}/{len(combos)}  ({time.time() - t0:.0f} s)")

    rows.sort(key=lambda r: r["val_odul_ort"], reverse=True)
    with open(os.path.join(RES_DIR, "handcrafted_arama.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Orijinal eşiklerin aynı validation senaryolarındaki performansı (referans)
    R0, Y0 = evaluate(make_handcrafted(HANDCRAFTED), seeds)

    best = rows[0]
    out = {
        "aciklama": "tune_handcrafted.py çıktısı; seçim validation senaryolarında, ölçüt ödül",
        "validation_seedleri": seeds,
        "en_iyi_esikler": {k: float(best[k]) for k in keys},
        "val_odul_ort": best["val_odul_ort"],
        "val_verim_ort": best["val_verim_ort"],
        "orijinal_val_odul_ort": round(float(R0.mean()), 3),
        "orijinal_val_verim_ort": round(float(Y0.mean()), 4),
    }
    with open(TUNED_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)

    print(f"\nOrijinal eşikler   : ödül {R0.mean():7.2f} | verim {Y0.mean():.2f} t/ha")
    print("En iyi 5 kombinasyon (validation):")
    for r in rows[:5]:
        desc = ", ".join(f"{k}={r[k]:g}" for k in keys)
        print(f"  {desc:58s} ödül {r['val_odul_ort']:7.2f} | verim {r['val_verim_ort']:.2f}")
    print(f"\nSeçilen eşikler → {TUNED_PATH}")
    print("Sıradaki adım: python final_evaluation.py")


if __name__ == "__main__":
    main()
