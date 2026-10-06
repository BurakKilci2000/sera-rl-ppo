"""
hyperparameter_sweep.py
=======================
Hiperparametre duyarlılık taraması (Grafik 4).

İki parametre × üç değer × beş seed (değerler config.yaml → sweep):
  - lr        : öğrenme oranı
  - clip_eps  : PPO kırpma yarıçapı
Ortak nokta (lr = baseline_lr, clip = baseline_clip) bir kez eğitilir ve
her iki taramada da kullanılır.

Tarama, hesaplama bütçesi için kısa eğitimle (sweep_steps) yapılır; final
değerlendirme tam eğitimle (ppo.total_timesteps) yapılmıştır.

Çıktılar:
  runs_sweep/<config>_seed<N>/ppo_checkpoint.pt
  runs_sweep/sweep_log.csv      ← tüm eğitimlerin episode bazlı ham logu
  runs_sweep/sweep_summary.csv  ← config × seed final özeti

Eğitimi tamamlanmış konfigürasyonlar atlanır; yarıda kalırsa kaldığı
yerden devam eder.

Kullanım:
  python hyperparameter_sweep.py
"""
import os
import csv
import time
import numpy as np
import torch

from ppo_agent import train
from config import SWEEP_CFG, PLOT_CFG

OUT_DIR = "./runs_sweep"
W_FIN   = PLOT_CFG["final_window_episodes"]


def cfg_dir(tag, seed):
    return os.path.join(OUT_DIR, f"{tag}_seed{seed}")


def build_configs():
    """(etiket, override) listesi; ortak baseline noktası tekrar edilmez."""
    configs = []
    for lr in SWEEP_CFG["lr_values"]:
        configs.append((f"lr_{lr:g}", {"lr": lr, "clip_eps": SWEEP_CFG["baseline_clip"]}))
    for ce in SWEEP_CFG["clip_eps_values"]:
        if ce == SWEEP_CFG["baseline_clip"]:
            continue
        configs.append((f"clip_{ce:g}", {"lr": SWEEP_CFG["baseline_lr"], "clip_eps": ce}))
    return configs


def run_one(tag, seed, **overrides):
    save_dir = cfg_dir(tag, seed)
    if os.path.exists(os.path.join(save_dir, "ppo_checkpoint.pt")):
        print(f"  [atla] {tag} seed={seed} zaten var.")
        return
    print(f"  [eğit] {tag} seed={seed}  {overrides}")
    train(save_dir=save_dir, total_timesteps=SWEEP_CFG["sweep_steps"], seed=seed, **overrides)


def collect_logs(configs):
    """Checkpoint'lerdeki episode geçmişinden iki CSV üretir."""
    log_rows, summary_rows = [], []
    for tag, ov in configs:
        for seed in SWEEP_CFG["sweep_seeds"]:
            path = os.path.join(cfg_dir(tag, seed), "ppo_checkpoint.pt")
            if not os.path.exists(path):
                print(f"  [uyarı] eksik: {path}")
                continue
            hist = torch.load(path, map_location="cpu", weights_only=False)["history"]
            ep_r, ep_y = hist["ep_r"], hist["ep_y"]
            for i, (r, y) in enumerate(zip(ep_r, ep_y)):
                log_rows.append({"config": tag, "lr": ov["lr"], "clip_eps": ov["clip_eps"],
                                 "seed": seed, "episode": i + 1, "reward": r,
                                 "yield_ton_ha": y})
            n = min(W_FIN, len(ep_r))
            summary_rows.append({"config": tag, "lr": ov["lr"], "clip_eps": ov["clip_eps"],
                                 "seed": seed,
                                 "final_reward_mean": float(np.mean(ep_r[-n:])),
                                 "final_reward_std": float(np.std(ep_r[-n:])),
                                 "final_yield_mean": float(np.mean(ep_y[-n:])),
                                 "final_yield_std": float(np.std(ep_y[-n:])),
                                 "n_episodes": len(ep_r)})

    for name, rows in [("sweep_log.csv", log_rows), ("sweep_summary.csv", summary_rows)]:
        with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  {len(rows)} satır → {name}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    configs = build_configs()
    seeds = SWEEP_CFG["sweep_seeds"]
    total = len(configs) * len(seeds)
    print(f"Toplam {total} eğitim ({len(configs)} config × {len(seeds)} seed), "
          f"her biri {SWEEP_CFG['sweep_steps']:,} adım.\n")

    t0, done = time.time(), 0
    for tag, ov in configs:
        print(f"== {tag}  {ov} ==")
        for seed in seeds:
            run_one(tag, seed, **ov)
            done += 1
            print(f"  ilerleme: {done}/{total}  ({(time.time() - t0) / 60:.1f} dk)\n")

    print("Loglar toplanıyor...")
    collect_logs(configs)


if __name__ == "__main__":
    main()
