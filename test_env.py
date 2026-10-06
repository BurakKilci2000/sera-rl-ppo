"""
test_env.py
===========
GreenhouseEnv için duman testi (smoke test).

Üç baseline politikayı (baselines.py) duman testi senaryolarında
(config.yaml → experiment.smoke_test_seeds) koşturur:
  1) Do-nothing   (tüm eylemler minimumda)
  2) Max-out      (tüm eylemler maksimumda — savurgan)
  3) Hand-crafted (oransal kural tabanlı kontrolcü; eşikler config.yaml → handcrafted)

Beklenti: hand-crafted > do-nothing ve hand-crafted > max-out.
Bu sağlanmıyorsa ortam anlamlı bir öğrenme sinyali üretmiyor demektir.

Ayrıca bir hand-crafted episode'unun trajektorisini çizer:
  grafikler/trajektori_handcrafted.png
(PPO ile yan yana karşılaştırma için: policy_analysis.py)

Kullanım:
  python test_env.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greenhouse_env import GreenhouseEnv, GreenhouseParams
from config import SMOKE_SEEDS, TRAJ_SEED
# Eski isimler korunur: başka dosyalar "from test_env import policy_handcrafted" yapabilir
from baselines import (policy_donothing as policy_do_nothing,
                       policy_maxout as policy_max_out,
                       policy_handcrafted)

FIG_DIR = "./grafikler"
IDX = GreenhouseEnv.IDX


# ---------------------------------------------------------------------------
# Episode koşturucu
# ---------------------------------------------------------------------------
def run_episode(env: GreenhouseEnv, policy_fn, seed: int):
    obs, info = env.reset(seed=seed)
    total_r, done, last_info = 0.0, False, info
    while not done:
        obs, r, term, trunc, info = env.step(policy_fn(obs))
        total_r += r
        last_info = info
        done = term or trunc
    return total_r, last_info


def evaluate_policy(env: GreenhouseEnv, policy_fn, name: str, seeds=SMOKE_SEEDS):
    rewards, yields, biomass = [], [], []
    for s in seeds:
        r, info = run_episode(env, policy_fn, seed=s)
        rewards.append(r)
        yields.append(info["yield_ton_ha"])
        biomass.append(info["biomass_g_m2"])
    print(f"  {name:18s}  "
          f"reward = {np.mean(rewards):8.2f} ± {np.std(rewards):5.2f}  | "
          f"yield = {np.mean(yields):5.2f} ± {np.std(yields):4.2f} ton/ha  | "
          f"B = {np.mean(biomass):6.1f} g/m²")
    return {"name": name, "rewards": rewards, "yields": yields, "biomass": biomass}


# ---------------------------------------------------------------------------
# Tek bir episode'un trajektorisi
# ---------------------------------------------------------------------------
def record_trajectory(env: GreenhouseEnv, policy_fn, seed: int = TRAJ_SEED):
    obs, info = env.reset(seed=seed)
    keys = ["T_in", "T_out", "H_in", "W_soil", "N_soil", "LAI", "B", "DVS", "I_rad"]
    traj = {k: [] for k in keys + ["a_heat", "a_vent", "a_irr", "a_fert", "reward"]}
    done = False
    while not done:
        new_obs, r, term, trunc, info = env.step(policy_fn(obs))
        for k in keys:
            traj[k].append(obs[IDX[k]])
        sa = info["action_scaled"]
        traj["a_heat"].append(sa["heat"]); traj["a_vent"].append(sa["vent"])
        traj["a_irr"].append(sa["irr"]);   traj["a_fert"].append(sa["fert"])
        traj["reward"].append(r)
        obs = new_obs
        done = term or trunc
    return {k: np.array(v) for k, v in traj.items()}, info


def plot_trajectory(traj: dict, info: dict, params: GreenhouseParams,
                    path: str, title: str = "Hand-crafted politika trajektorisi"):
    p = params
    days = np.arange(len(traj["T_in"]))
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)

    ax = axes[0, 0]
    ax.plot(days, traj["T_out"], label="T_out", color="#888", lw=1)
    ax.plot(days, traj["T_in"],  label="T_in",  color="#c0392b", lw=1.5)
    ax.axhspan(p.T_opt_low, p.T_opt_high, color="#2ecc71", alpha=0.10, label="optimum")
    ax.set_ylabel("Sıcaklık (°C)"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("İklim")

    ax = axes[0, 1]
    ax.plot(days, traj["W_soil"], label="W_soil", color="#2980b9")
    ax.axhline(p.wilting_point, ls="--", color="#888", lw=0.8, label="solma noktası")
    ax.axhline(p.field_capacity, ls="--", color="#555", lw=0.8, label="tarla kapasitesi")
    ax.set_ylabel("Toprak suyu (hac.)"); ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(days, traj["N_soil"], color="#27ae60", lw=1.2)
    ax2.set_ylabel("Toprak N (kg/ha)", color="#27ae60")
    ax.set_title("Toprak")

    ax = axes[1, 0]
    ax.plot(days, traj["B"], label="B (g/m²)", color="#16a085", lw=1.6)
    ax.set_ylabel("Biyokütle (g/m²)"); ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(days, traj["LAI"], color="#8e44ad", lw=1.2)
    ax2.set_ylabel("LAI", color="#8e44ad")
    ax.set_title("Bitki")

    ax = axes[1, 1]
    ax.plot(days, traj["DVS"], color="#d35400", lw=1.5)
    ax.set_ylabel("DVS"); ax.set_ylim(0, p.DVS_maturity * 1.025)
    ax.set_title("Gelişme aşaması")

    ax = axes[2, 0]
    ax.plot(days, traj["a_heat"], label="ısıtma", color="#e67e22")
    ax.plot(days, traj["a_vent"], label="havalandırma", color="#3498db")
    ax.set_ylabel("İklim eylemi [0, 1]"); ax.legend(fontsize=8); ax.set_xlabel("Gün")

    ax = axes[2, 1]
    ax.plot(days, traj["a_irr"],  label="sulama", color="#1abc9c")
    ax.plot(days, traj["a_fert"], label="gübre",  color="#9b59b6")
    ax.set_ylabel("Toprak eylemi [0, 1]"); ax.legend(fontsize=8); ax.set_xlabel("Gün")

    fig.suptitle(f"{title}  |  final B = {info['biomass_g_m2']:.0f} g/m²  |  "
                 f"verim ≈ {info['yield_ton_ha']:.2f} ton/ha", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Trajektori kaydedildi: {path}")


# ---------------------------------------------------------------------------
# Ana
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print(f"  GreenhouseEnv duman testi  ({len(SMOKE_SEEDS)} episode × 3 politika)")
    print("=" * 72)

    params = GreenhouseParams()
    env = GreenhouseEnv(params)

    # API kontrolü
    obs, info = env.reset(seed=SMOKE_SEEDS[0])
    assert env.observation_space.contains(obs), "obs space ihlali!"
    obs2, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs2), "step sonrası obs space ihlali!"
    print("  ✓ API kontrolü geçti (obs/action space tutarlı)\n")

    res = [evaluate_policy(env, pol, name) for name, pol in
           [("Do-nothing", policy_do_nothing),
            ("Max-out", policy_max_out),
            ("Hand-crafted", policy_handcrafted)]]

    print()
    dn_r, mx_r, hc_r = (np.mean(x["rewards"]) for x in res)
    if hc_r > dn_r and hc_r > mx_r:
        print("  ✓ Hand-crafted, do-nothing ve max-out'tan daha iyi.")
        print("    Ortam anlamlı bir öğrenme sinyali üretiyor.")
    else:
        print("  ✗ DİKKAT: hand-crafted en iyi değil. Ödül/dinamik ayarı gerekli.")

    print()
    traj, info = record_trajectory(env, policy_handcrafted, seed=TRAJ_SEED)
    plot_trajectory(traj, info, params, os.path.join(FIG_DIR, "trajektori_handcrafted.png"))
    print(f"  ✓ Hand-crafted final biyokütle = {info['biomass_g_m2']:.1f} g/m²")
    print(f"  ✓ Hand-crafted verim           = {info['yield_ton_ha']:.2f} ton/ha")
    print(f"  ✓ Hand-crafted final DVS       = {info['DVS']:.2f}")
