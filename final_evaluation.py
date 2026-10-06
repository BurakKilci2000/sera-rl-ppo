"""
final_evaluation.py
===================
Eğitilmiş 5 PPO ajanını ve baseline politikaları (ayarlı hand-crafted
varsa o da dahil), eğitimde hiç görülmemiş
20 hava senaryosunda (config.yaml → experiment.test_seeds) deterministik
olarak test eder. PPO için eylem a = mu_theta(s): keşif gürültüsü yok.

Çıktı:
  sonuclar/final_eval_results.csv — politika × senaryo ham sonuçları
  sonuclar/final_eval_ozet.txt     — konsol özetinin UTF-8 kopyası
  (Grafik 5 ve sonuclar.csv, plot_results.py tarafından bu dosyadan üretilir)

Kullanım:
  python final_evaluation.py
"""
from __future__ import annotations
import os
import csv
import numpy as np
import torch

from greenhouse_env import GreenhouseEnv, GreenhouseParams
from ppo_agent import ActorCritic, RunningNorm, get_device
from baselines import BASELINES
from config import EVAL_SEEDS, TEST_SEEDS

RUNS_DIR = "./runs"
OUT_DIR  = "./sonuclar"
OUT_CSV  = os.path.join(OUT_DIR, "final_eval_results.csv")
OUT_TXT  = os.path.join(OUT_DIR, "final_eval_ozet.txt")


class Tee:
    """Konsola basılanı aynı anda UTF-8 bir dosyaya da yazar."""
    def __init__(self, path, stream):
        self.file = open(path, "w", encoding="utf-8")
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def load_ppo_policy(seed: int):
    """Checkpoint'ten ajanı ve eğitimdeki normalizasyonu yükler."""
    device = get_device()
    ckpt_path = os.path.join(RUNS_DIR, f"ppo_seed{seed}", "ppo_checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    env = GreenhouseEnv(GreenhouseParams())
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = ActorCritic(obs_dim, act_dim, ckpt["hp"]["hidden_size"]).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    onorm = RunningNorm(obs_dim)
    onorm.mean, onorm.M2, onorm.count = (
        ckpt["obs_norm_mean"], ckpt["obs_norm_M2"], ckpt["obs_norm_count"])

    def policy(obs):
        o = torch.tensor(onorm.normalize(obs), dtype=torch.float32, device=device)
        with torch.no_grad():
            return agent.actor(o).cpu().numpy()   # deterministik: a = mu(s)
    return policy


def run_episode(env, policy, scenario_seed: int):
    obs, _ = env.reset(seed=scenario_seed)
    total_r, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total_r += r
        done = term or trunc
    return total_r, info["yield_ton_ha"], info["biomass_g_m2"]


def evaluate(name, policy, env, rows):
    R, Y = [], []
    for s in TEST_SEEDS:
        r, y, b = run_episode(env, policy, s)
        R.append(r); Y.append(y)
        rows.append({"policy": name, "scenario_seed": s, "reward": r,
                     "yield_ton_ha": y, "biomass_g_m2": b})
    R, Y = np.asarray(R), np.asarray(Y)
    print(f"  {name:16s}  R = {R.mean():7.2f} ± {R.std():5.2f}  | "
          f"Y = {Y.mean():.2f} ± {Y.std():.2f} t/ha")
    return R, Y


def main():
    import sys
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout = Tee(OUT_TXT, sys.stdout)
    env = GreenhouseEnv(GreenhouseParams())
    rows = []

    print("=" * 70)
    print(f"  Final değerlendirme: {len(TEST_SEEDS)} test senaryosu (deterministik)")
    print("=" * 70)

    print("\nBaseline politikalar:")
    base = {name: evaluate(name, pol, env, rows) for name, pol in BASELINES.items()}

    print("\nPPO ajanları:")
    ppo_R, ppo_Y = [], []
    for seed in EVAL_SEEDS:
        R, Y = evaluate(f"PPO seed={seed}", load_ppo_policy(seed), env, rows)
        ppo_R.append(R); ppo_Y.append(Y)

    # Seed düzeyinde özet: her seed'in 20 senaryo ortalaması → seedler arası ort. ± std
    seed_means_R = np.array([r.mean() for r in ppo_R])
    seed_means_Y = np.array([y.mean() for y in ppo_Y])
    all_R, all_Y = np.concatenate(ppo_R), np.concatenate(ppo_Y)
    print(f"\nPPO, {len(EVAL_SEEDS)} seed:")
    print(f"  seedler arası  R = {seed_means_R.mean():.2f} ± {seed_means_R.std():.2f}  | "
          f"Y = {seed_means_Y.mean():.2f} ± {seed_means_Y.std():.2f} t/ha")
    print(f"  tüm episodelar R = {all_R.mean():.2f} ± {all_R.std():.2f}  | "
          f"Y = {all_Y.mean():.2f} ± {all_Y.std():.2f} t/ha")

    for ref in ["Hand-crafted", "Hand-crafted (ayarlı)"]:
        if ref in base:
            r_R, r_Y = base[ref][0].mean(), base[ref][1].mean()
            print(f"{ref:22s}'a göre: ödül {all_R.mean() / r_R:.2f}×, "
                  f"verim {all_Y.mean() / r_Y:.2f}×")
    if "Hand-crafted (ayarlı)" not in base:
        print("  (Ayarlı hand-crafted yok; önce: python tune_handcrafted.py)")
    else:
        # Eşleştirilmiş karşılaştırma: aynı senaryoda PPO (5 seed ort.) − ayarlı kural
        ppo_per_scen = np.mean(np.stack(ppo_R), axis=0)          # (20,)
        diff = ppo_per_scen - base["Hand-crafted (ayarlı)"][0]
        wins = int((diff > 0).sum())
        print(f"\nEşleştirilmiş fark (PPO − ayarlı), {len(TEST_SEEDS)} senaryo: "
              f"{diff.mean():.2f} ± {diff.std():.2f} ödül | PPO {wins}/{len(diff)} senaryoda önde")
        for s, R in zip(EVAL_SEEDS, ppo_R):
            d = R - base["Hand-crafted (ayarlı)"][0]
            print(f"  PPO seed={s:<3d}  fark {d.mean():6.2f}  |  {int((d > 0).sum())}/{len(d)} senaryoda önde")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "scenario_seed", "reward",
                                          "yield_ton_ha", "biomass_g_m2"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nHam sonuçlar: {OUT_CSV}")
    print(f"Bu özet: {OUT_TXT}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stream


if __name__ == "__main__":
    main()
