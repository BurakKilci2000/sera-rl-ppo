"""
policy_analysis.py
==================
PPO'nun hand-crafted'tan NASIL farklı davrandığını ölçer. Rapordaki ve
sunumdaki davranış iddiaları yalnızca bu scriptin ürettiği loglara dayanır.

  1) Trajektori karşılaştırması
     Aynı hava senaryosunda (experiment.trajectory_seed) hand-crafted ve PPO
     (experiment.trajectory_ppo_seed) 120 gün boyunca yan yana.
       → grafikler/trajektori_karsilastirma.png
       → sonuclar/loglar/trajektori_handcrafted.csv, trajektori_ppo.csv

  2) Davranış özeti (20 test senaryosu)
     Su stresli gün sayısı, toplam sulama/drenaj/gübre, sıcaklığın optimumda
     kaldığı gün sayısı vb. Hand-crafted: 20 senaryo; PPO: 5 seed × 20 senaryo.
       → sonuclar/politika_davranis_ozet.csv

  3) Politika sondajı
     PPO trajektorisinin analysis.probe_day günündeki gerçek durum alınır;
     yalnızca iç sıcaklık ve LAI (biyokütleyle tutarlı olarak) değiştirilir,
     5 PPO ajanının önerdiği eylemler kaydedilir.
       → grafikler/politika_sondaji.png
       → sonuclar/loglar/politika_sondaji.csv

Ön koşul: ppo_agent.py --all tamamlanmış olmalı.

Kullanım:
  python policy_analysis.py
"""
from __future__ import annotations
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greenhouse_env import GreenhouseEnv, GreenhouseParams
from baselines import policy_handcrafted, load_tuned_handcrafted
from config import (EVAL_SEEDS, TEST_SEEDS, TRAJ_SEED, TRAJ_PPO_SEED, ANALYSIS_CFG)

RES_DIR = "./sonuclar"
LOG_DIR = os.path.join(RES_DIR, "loglar")
FIG_DIR = "./grafikler"
IDX = GreenhouseEnv.IDX

HC_COLOR, PPO_COLOR, TUNED_COLOR = "#7f7f7f", "#1f77b4", "#7b1fa2"


# ======================================================================
# Episode kaydı
# ======================================================================
def record(policy, scenario_seed):
    """Bir episode'u koşturur. Döner: (günlük kayıtlar, toplam ödül, verim,
    her günün eylem öncesi tam gözlem vektörleri)."""
    env = GreenhouseEnv(GreenhouseParams())
    p = env.p
    obs, _ = env.reset(seed=scenario_seed)
    rows, obs_list, done, total_r = [], [], False, 0.0
    while not done:
        obs_list.append(obs.copy())
        a = policy(obs)
        new_obs, r, term, trunc, info = env.step(a)
        sa = info["action_scaled"]
        total_r += r
        rows.append({
            "day": len(rows),
            "T_in": float(obs[IDX["T_in"]]), "T_out": float(obs[IDX["T_out"]]),
            "H_in": float(obs[IDX["H_in"]]), "W_soil": float(obs[IDX["W_soil"]]),
            "N_soil": float(obs[IDX["N_soil"]]), "LAI": float(obs[IDX["LAI"]]),
            "B": float(obs[IDX["B"]]), "DVS": float(obs[IDX["DVS"]]),
            "heat": float(sa["heat"]), "vent": float(sa["vent"]),
            "irr_mm": float(sa["irr"]) * p.irrigation_max_mm,
            "fert_kg_ha": float(sa["fert"]) * p.fertilizer_max_kg_ha,
            "f_T": info["f_T"], "g_W": info["g_W"], "h_N": info["h_N"],
            "drainage_mm": info["drainage_mm"], "ET_mm": info["ET_mm"],
            "reward": float(r),
            **{f"odul_{k}": float(v) for k, v in info["reward_parts"].items()},
        })
        obs = new_obs
        done = term or trunc
    return rows, total_r, info["yield_ton_ha"], obs_list


def save_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def col(rows, key):
    return np.array([r[key] for r in rows])


# ======================================================================
# 1) Trajektori karşılaştırması
# ======================================================================
def trajectory_comparison(ppo_policy):
    hc, hc_R, hc_Y, _ = record(policy_handcrafted, TRAJ_SEED)
    pp, pp_R, pp_Y, pp_obs = record(ppo_policy, TRAJ_SEED)
    save_rows(hc, os.path.join(LOG_DIR, "trajektori_handcrafted.csv"))
    save_rows(pp, os.path.join(LOG_DIR, "trajektori_ppo.csv"))
    tuned_policy = load_tuned_handcrafted()
    tu = None
    if tuned_policy is not None:
        tu, tu_R, tu_Y, _ = record(tuned_policy, TRAJ_SEED)
        save_rows(tu, os.path.join(LOG_DIR, "trajektori_handcrafted_ayarli.csv"))
    p = GreenhouseParams()

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    lab_hc = f"Hand-crafted ({hc_Y:.2f} t/ha)"
    lab_pp = f"PPO seed={TRAJ_PPO_SEED} ({pp_Y:.2f} t/ha)"
    lab_tu = f"Hand-crafted ayarlı ({tu_Y:.2f} t/ha)" if tu is not None else ""

    ax = axes[0, 0]
    ax.axhspan(p.T_opt_low, p.T_opt_high, color="#2ecc71", alpha=0.12,
               label=f"optimum {p.T_opt_low:g}–{p.T_opt_high:g} °C")
    ax.plot(col(hc, "day"), col(hc, "T_out"), color="#bbbbbb", lw=1, label="dış sıcaklık")
    ax.plot(col(hc, "day"), col(hc, "T_in"), color=HC_COLOR, lw=1.4, label=lab_hc)
    ax.plot(col(pp, "day"), col(pp, "T_in"), color=PPO_COLOR, lw=1.6, label=lab_pp)
    if tu is not None:
        ax.plot(col(tu, "day"), col(tu, "T_in"), color=TUNED_COLOR, lw=1.3, ls="--", label=lab_tu)
    ax.set_ylabel("İç sıcaklık (°C)")
    ax.set_title("İklim")

    ax = axes[0, 1]
    ax.axhline(p.wilting_point, ls="--", color="#999", lw=0.8, label="solma noktası")
    ax.axhline(p.field_capacity, ls="--", color="#555", lw=0.8, label="tarla kapasitesi")
    ax.plot(col(hc, "day"), col(hc, "W_soil"), color=HC_COLOR, lw=1.4, label="Hand-crafted")
    ax.plot(col(pp, "day"), col(pp, "W_soil"), color=PPO_COLOR, lw=1.6, label="PPO")
    if tu is not None:
        ax.plot(col(tu, "day"), col(tu, "W_soil"), color=TUNED_COLOR, lw=1.3, ls="--", label="Hand-crafted (ayarlı)")
    ax.set_ylabel("Toprak suyu (hacimsel)")
    ax.set_title("Toprak suyu")

    ax = axes[1, 0]
    ax.plot(col(hc, "day"), col(hc, "B"), color=HC_COLOR, lw=1.4, label="Hand-crafted")
    ax.plot(col(pp, "day"), col(pp, "B"), color=PPO_COLOR, lw=1.6, label="PPO")
    if tu is not None:
        ax.plot(col(tu, "day"), col(tu, "B"), color=TUNED_COLOR, lw=1.3, ls="--", label="Hand-crafted (ayarlı)")
    ax.set_ylabel("Biyokütle (g/m²)")
    ax.set_title("Bitki büyümesi")

    ax = axes[1, 1]
    ax.plot(col(hc, "day"), col(hc, "N_soil"), color=HC_COLOR, lw=1.4, label="Hand-crafted")
    ax.plot(col(pp, "day"), col(pp, "N_soil"), color=PPO_COLOR, lw=1.6, label="PPO")
    if tu is not None:
        ax.plot(col(tu, "day"), col(tu, "N_soil"), color=TUNED_COLOR, lw=1.3, ls="--", label="Hand-crafted (ayarlı)")
    ax.set_ylabel("Toprak azotu (kg N/ha)")
    ax.set_title("Toprak azotu")

    ax = axes[2, 0]
    ax.plot(col(hc, "day"), col(hc, "irr_mm"), color=HC_COLOR, lw=1.2, label="Hand-crafted")
    ax.plot(col(pp, "day"), col(pp, "irr_mm"), color=PPO_COLOR, lw=1.4, label="PPO")
    if tu is not None:
        ax.plot(col(tu, "day"), col(tu, "irr_mm"), color=TUNED_COLOR, lw=1.3, ls="--", label="Hand-crafted (ayarlı)")
    ax.set_ylabel("Sulama (mm/gün)")
    ax.set_title("Sulama eylemi")

    ax = axes[2, 1]
    ax.plot(col(hc, "day"), col(hc, "vent"), color=HC_COLOR, lw=1.2, label="Hand-crafted havalandırma")
    ax.plot(col(pp, "day"), col(pp, "vent"), color=PPO_COLOR, lw=1.4, label="PPO havalandırma")
    if tu is not None:
        ax.plot(col(tu, "day"), col(tu, "vent"), color=TUNED_COLOR, lw=1.3, ls="--", label="Ayarlı havalandırma")
    ax.plot(col(hc, "day"), col(hc, "heat"), color=HC_COLOR, lw=1.0, ls=":", label="Hand-crafted ısıtma")
    ax.plot(col(pp, "day"), col(pp, "heat"), color=PPO_COLOR, lw=1.0, ls=":", label="PPO ısıtma")
    ax.set_ylabel("Eylem oranı [0, 1]")
    ax.set_title("İklim eylemleri")

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    for ax in axes[2]:
        ax.set_xlabel("Gün")
    extra = f", ayarlı {tu_R:.1f}" if tu is not None else ""
    fig.suptitle(f"Trajektori karşılaştırması  |  aynı hava senaryosu (seed {TRAJ_SEED}, eğitimde "
                 f"görülmedi)  |  ödül: hand-crafted {hc_R:.1f}{extra}, PPO {pp_R:.1f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "trajektori_karsilastirma.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ trajektori: hand-crafted {hc_Y:.2f} t/ha, PPO {pp_Y:.2f} t/ha")
    return pp, pp_obs


# ======================================================================
# 2) Davranış özeti (20 test senaryosu)
# ======================================================================
def episode_metrics(rows, yield_t):
    thr = ANALYSIS_CFG["water_stress_threshold"]
    irr, lai = col(rows, "irr_mm"), col(rows, "LAI")
    corr = float(np.corrcoef(irr, lai)[0, 1]) if irr.std() > 0 and lai.std() > 0 else float("nan")
    return {
        "verim_t_ha": yield_t,
        "sezon_uzunlugu_gun": len(rows),
        "sicaklik_optimum_gun": int((col(rows, "f_T") >= 1.0).sum()),
        "su_stresli_gun": int((col(rows, "g_W") < thr).sum()),
        "ort_toprak_suyu": float(col(rows, "W_soil").mean()),
        "toplam_sulama_mm": float(irr.sum()),
        "toplam_drenaj_mm": float(col(rows, "drainage_mm").sum()),
        "toplam_gubre_kg_ha": float(col(rows, "fert_kg_ha").sum()),
        "ort_isitma_orani": float(col(rows, "heat").mean()),
        "ort_havalandirma_orani": float(col(rows, "vent").mean()),
        "ort_azot_stres_faktoru_hN": float(col(rows, "h_N").mean()),
        "sulama_LAI_korelasyonu": corr,
        # Gübrenin ne kadarı sezonun son üçte birinde verildi?
        "gec_sezon_gubre_orani": float(col(rows, "fert_kg_ha")[2 * len(rows) // 3:].sum()
                                       / max(1e-9, col(rows, "fert_kg_ha").sum())),
        # Ödül bileşenlerinin sezon toplamları (fark hangi kalemden geliyor?)
        **{f"toplam_{k}": float(col(rows, k).sum())
           for k in rows[0] if k.startswith("odul_")},
        "toplam_odul": float(col(rows, "reward").sum()),
    }


def behaviour_summary(load_ppo_policy):
    def over_scenarios(policy):
        ms = []
        for s in TEST_SEEDS:
            rows, _, y, _ = record(policy, s)
            ms.append(episode_metrics(rows, y))
        return {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}

    hc = over_scenarios(policy_handcrafted)
    tuned_policy = load_tuned_handcrafted()
    tu = over_scenarios(tuned_policy) if tuned_policy is not None else None
    per_seed = [over_scenarios(load_ppo_policy(s)) for s in EVAL_SEEDS]

    out = []
    for k in hc:
        vals = np.array([m[k] for m in per_seed])
        row = {"metrik": k, "hand_crafted": round(hc[k], 3)}
        if tu is not None:
            row["hand_crafted_ayarli"] = round(tu[k], 3)
        row["ppo_ort"] = round(float(np.nanmean(vals)), 3)
        row["ppo_seed_std"] = round(float(np.nanstd(vals)), 3)
        out.append(row)
    path = os.path.join(RES_DIR, "politika_davranis_ozet.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"  ✓ {path}")
    return out


# ======================================================================
# 3) Politika sondajı
# ======================================================================
def policy_probe(ppo_rows, ppo_obs, load_ppo_policy):
    p = GreenhouseParams()
    day = min(ANALYSIS_CFG["probe_day"], len(ppo_obs) - 1)
    base = ppo_rows[day]
    obs = ppo_obs[day]   # PPO trajektorisinin o günkü gerçek, tam gözlem vektörü

    rows = []
    for seed in EVAL_SEEDS:
        policy = load_ppo_policy(seed)
        for T in ANALYSIS_CFG["probe_T_values"]:
            for lai in ANALYSIS_CFG["probe_LAI_values"]:
                o = obs.copy()
                o[IDX["T_in"]] = T
                o[IDX["LAI"]] = lai
                o[IDX["B"]] = lai / (p.SLA * p.leaf_fraction)   # LAI ile tutarlı biyokütle
                a = GreenhouseEnv._scale_action(policy(o))
                rows.append({"ppo_seed": seed, "T_in": T, "LAI": lai,
                             "B": round(float(o[IDX["B"]]), 1),
                             "isitma": float(a[0]), "havalandirma": float(a[1]),
                             "sulama": float(a[2]), "gubre": float(a[3])})
    save_rows(rows, os.path.join(LOG_DIR, "politika_sondaji.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for ax, act, title in zip(axes.flat, ["isitma", "havalandirma", "sulama", "gubre"],
                              ["Isıtma", "Havalandırma", "Sulama", "Gübre"]):
        for c, T in zip(colors, ANALYSIS_CFG["probe_T_values"]):
            lais = np.array(ANALYSIS_CFG["probe_LAI_values"])
            vals = np.array([[r[act] for r in rows if r["T_in"] == T and r["LAI"] == l
                              and r["ppo_seed"] == s] for s in EVAL_SEEDS for l in lais]
                            ).reshape(len(EVAL_SEEDS), len(lais))
            m, sd = vals.mean(axis=0), vals.std(axis=0)
            ax.fill_between(lais, m - sd, m + sd, color=c, alpha=0.15)
            ax.plot(lais, m, "o-", color=c, lw=2, ms=4, label=f"iç sıcaklık {T:g} °C")
        ax.set_title(title)
        ax.set_ylabel("Önerilen eylem oranı [0, 1]")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Yaprak alan indeksi (LAI)")
    fig.suptitle(f"Politika sondajı  |  {len(EVAL_SEEDS)} PPO ajanı ort. ± std  |  "
                 f"taban durum: seed {TRAJ_SEED} senaryosunun {day}. günü, "
                 f"diğer durum değişkenleri sabit", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "politika_sondaji.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ sondaj: {len(rows)} ölçüm (gün {day}, taban T_in = {base['T_in']:.1f} °C, "
          f"LAI = {base['LAI']:.2f})")
    return rows


def main():
    from final_evaluation import load_ppo_policy   # torch gerektirir
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("1) Trajektori karşılaştırması...")
    ppo_rows, ppo_obs = trajectory_comparison(load_ppo_policy(TRAJ_PPO_SEED))
    print("2) Davranış özeti (20 test senaryosu)...")
    summary = behaviour_summary(load_ppo_policy)
    print("3) Politika sondajı...")
    policy_probe(ppo_rows, ppo_obs, load_ppo_policy)

    has_tu = "hand_crafted_ayarli" in summary[0]
    print("\n" + "=" * 82)
    print(f"  {'metrik':30s} {'hand-crafted':>12s} {'ayarlı' if has_tu else '':>10s} {'PPO (5 seed)':>18s}")
    print("=" * 82)
    for r in summary:
        tu_txt = f"{r['hand_crafted_ayarli']:10.3f}" if has_tu else " " * 10
        print(f"  {r['metrik']:30s} {r['hand_crafted']:12.3f} {tu_txt} "
              f"{r['ppo_ort']:10.3f} ± {r['ppo_seed_std']:.3f}")


if __name__ == "__main__":
    main()
