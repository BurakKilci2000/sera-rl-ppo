"""
plot_results.py
===============
Ham logları toplar, zorunlu 5 grafiği ve seed × metrik özetini üretir.

Akış:
  1) runs/ppo_seed*/{episodes,evals,updates}.csv ve runs_sweep/sweep_log.csv
     → sonuclar/loglar/ altına kopyalanır (teslim edilen ham loglar).
  2) Grafikler YALNIZCA sonuclar/loglar/ ve sonuclar/final_eval_results.csv
     dosyalarından çizilir; aynı loglarla herkes aynı grafiği üretir.
  3) sonuclar/sonuclar.csv, hiperparametre_ozet.csv ve grafik_ozet.csv yazılır.
     grafik_ozet.csv: grafik yorumlarında kullanılan her sayı (kilometre taşları,
     eval–eğitim farkı, loss değerleri) gözle okunmaz, loglardan hesaplanır.

Grafikler (grafikler/ klasörüne):
  grafik1_ogrenme_egrisi.png   eğitim episode getirisi, 5 seed ort. ± std
  grafik2_eval_egrisi.png      deterministik test eğrisi, 5 seed ort. ± std
  grafik3_loss.png             policy/value loss, entropi, KL ve clip oranı
  grafik4_hiperparametre.png   lr ve clip_eps taraması, 5 seed ort. ± std
  grafik5_baseline.png         20 test senaryosunda baseline karşılaştırması

Ön koşul: ppo_agent.py --all, hyperparameter_sweep.py ve final_evaluation.py
çalıştırılmış olmalı.

Kullanım:
  python plot_results.py
"""
from __future__ import annotations
import os
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EVAL_SEEDS, BASELINE_REF, PLOT_CFG, SWEEP_CFG, EVAL_CFG, TEST_SEEDS

RUNS_DIR   = "./runs"
SWEEP_DIR  = "./runs_sweep"
RES_DIR    = "./sonuclar"
LOG_DIR    = os.path.join(RES_DIR, "loglar")
FIG_DIR    = "./grafikler"
FINAL_CSV  = os.path.join(RES_DIR, "final_eval_results.csv")
TUNED      = "Hand-crafted (ayarlı)"

W      = PLOT_CFG["smoothing_window"]
W_LOSS = PLOT_CFG["loss_smoothing_window"]
W_FIN  = PLOT_CFG["final_window_episodes"]
N_SEED = len(EVAL_SEEDS)

SEED_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728",
               "#8c564b", "#e377c2", "#17becf"]
MEAN_COLOR = "#111111"


# ======================================================================
# Yardımcılar
# ======================================================================
def read_csv(path):
    """CSV → {sütun: np.array}. Sayıya çevrilebilen sütunlar float olur."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0].keys():
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) for v in vals])
        except ValueError:
            out[k] = np.array(vals)
    return out


def smooth(x, w):
    """Hareketli ortalama. Çıktı len(x) - w + 1 uzunluğunda (ilk tam pencereden itibaren)."""
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def mean_std(curves):
    """Farklı uzunluktaki eğrileri en kısaya hizalar; seedler arası ort. ve std."""
    L = min(len(c) for c in curves)
    arr = np.stack([c[:L] for c in curves])
    return arr, arr.mean(axis=0), arr.std(axis=0), L


def band_plot(ax, x, curves, mean, std, label_mean, color=MEAN_COLOR, show_seeds=True):
    if show_seeds:
        for i, (c, s) in enumerate(zip(curves, EVAL_SEEDS)):
            ax.plot(x, c, color=SEED_COLORS[i % len(SEED_COLORS)], lw=0.8, alpha=0.35,
                    label=f"seed={s}")
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, label="ort. ± std")
    ax.plot(x, mean, color=color, lw=2.4, label=label_mean)


def finish(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ======================================================================
# 1) Ham logları topla
# ======================================================================
def collect_logs():
    os.makedirs(LOG_DIR, exist_ok=True)
    for s in EVAL_SEEDS:
        for name in ["episodes", "evals", "updates"]:
            src = os.path.join(RUNS_DIR, f"ppo_seed{s}", f"{name}.csv")
            if not os.path.exists(src):
                raise FileNotFoundError(f"Eksik log: {src} (önce: python ppo_agent.py --all)")
            shutil.copy2(src, os.path.join(LOG_DIR, f"seed{s}_{name}.csv"))
    sweep_src = os.path.join(SWEEP_DIR, "sweep_log.csv")
    if os.path.exists(sweep_src):
        shutil.copy2(sweep_src, os.path.join(LOG_DIR, "sweep_log.csv"))
    elif os.path.exists(os.path.join(LOG_DIR, "sweep_log.csv")):
        print(f"  ✓ {sweep_src} yok; sonuclar/loglar içindeki mevcut kopya kullanılıyor")
    else:
        print(f"  ⚠ {sweep_src} yok; Grafik 4 atlanacak")
    print(f"  ✓ ham loglar → {LOG_DIR}")


def baseline_refs():
    """Grafik 1–2 referans çizgileri: final test (20 senaryo) ortalamaları.
    Böylece grafikler ve sonuç tablosu aynı sayıyı gösterir."""
    if not os.path.exists(FINAL_CSV):
        print("  ⚠ final_eval_results.csv yok; config'teki duman testi değerleri kullanılıyor")
        return dict(BASELINE_REF), "duman testi"
    d = read_csv(FINAL_CSV)
    m = lambda p, c: float(d[c][d["policy"] == p].mean())
    refs = {"hand_crafted_reward": round(m("Hand-crafted", "reward"), 2),
            "max_out_reward":      round(m("Max-out", "reward"), 2),
            "do_nothing_reward":   round(m("Do-nothing", "reward"), 2),
            "hand_crafted_yield":  round(m("Hand-crafted", "yield_ton_ha"), 2)}
    if (d["policy"] == TUNED).any():
        refs["tuned_reward"] = round(m(TUNED, "reward"), 2)
        refs["tuned_yield"] = round(m(TUNED, "yield_ton_ha"), 2)
    return refs, "test ort."


def load_seed_logs(name):
    return {s: read_csv(os.path.join(LOG_DIR, f"seed{s}_{name}.csv")) for s in EVAL_SEEDS}


# ======================================================================
# Grafik 1 — Öğrenme eğrisi (eğitim, stokastik politika)
# ======================================================================
def graph1(eps, refs, src, notes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, ylabel in [(axes[0], "reward", "Episode getirisi (toplam ödül)"),
                            (axes[1], "yield_ton_ha", "Hasat verimi (ton/ha)")]:
        curves = [smooth(eps[s][col], W) for s in EVAL_SEEDS]
        arr, mean, std, L = mean_std(curves)
        x = np.arange(W, W + L)
        band_plot(ax, x, list(arr), mean, std, f"{N_SEED} seed ortalaması")
        if col == "reward":
            lines = [("hand_crafted_reward", "hand-crafted", "green"),
                     ("max_out_reward", "max-out", "orange"),
                     ("do_nothing_reward", "do-nothing", "red")]
            if "tuned_reward" in refs:
                lines.insert(0, ("tuned_reward", "hand-crafted (ayarlı)", "purple"))
            for key, name, c in lines:
                ax.axhline(refs[key], ls="--", color=c, lw=1.1,
                           label=f"{name} ({refs[key]}, {src})")
                if mean[0] >= refs[key]:
                    notes.append(("G1", f"{name} seviyesi", "eğitimin başından beri üstünde"))
                else:
                    hit = np.nonzero(mean >= refs[key])[0]
                    notes.append(("G1", f"{name} seviyesinin ilk aşıldığı episode",
                                  int(x[hit[0]]) if hit.size else "aşılmadı"))
            final = float(mean[-100:].mean())
            notes.append(("G1", "son 100 noktanın ortalaması (plato)", round(final, 2)))
            p95 = np.nonzero(mean >= 0.95 * final)[0]
            notes.append(("G1", "platonun %95'ine ulaşılan episode", int(x[p95[0]])))
            notes.append(("G1", "seedler arası std, son nokta", round(float(std[-1]), 2)))
        else:
            ax.axhline(refs["hand_crafted_yield"], ls="--", color="green", lw=1.2,
                       label=f"hand-crafted ({refs['hand_crafted_yield']}, {src})")
            if "tuned_yield" in refs:
                ax.axhline(refs["tuned_yield"], ls="--", color="purple", lw=1.2,
                           label=f"hand-crafted ayarlı ({refs['tuned_yield']}, {src})")
        ax.set_xlabel("Episode (1 episode = 1 yetiştirme sezonu)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", ncol=2)
    axes[0].set_title("Episode getirisi")
    axes[1].set_title("Hasat verimi")
    fig.suptitle(f"Grafik 1 — Öğrenme eğrisi (eğitim)  |  {N_SEED} seed, ort. ± std, "
                 f"hareketli ortalama w = {W} episode", fontsize=12, fontweight="bold")
    finish(fig, "grafik1_ogrenme_egrisi.png")


# ======================================================================
# Grafik 2 — Test (eval) eğrisi: deterministik politika
# ======================================================================
def graph2(eps, evs, refs, src, notes):
    n_val = EVAL_CFG["n_validation_episodes"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ecol, tcol, ylabel in [
            (axes[0], "eval_reward_mean", "reward", "Episode getirisi (toplam ödül)"),
            (axes[1], "eval_yield_mean", "yield_ton_ha", "Hasat verimi (ton/ha)")]:
        # Deterministik eval: her nokta n_val validation senaryosunun ortalaması
        steps = evs[EVAL_SEEDS[0]]["global_step"]
        e_arr, e_mean, e_std, L = mean_std([evs[s][ecol] for s in EVAL_SEEDS])
        x = steps[:L]
        ax.fill_between(x, e_mean - e_std, e_mean + e_std, color="#1f77b4", alpha=0.18)
        ax.plot(x, e_mean, "o-", color="#1f77b4", lw=2.2, ms=4,
                label=f"Deterministik test (a = μθ(s)), {N_SEED} seed ort. ± std")

        # Karşılaştırma: aynı adımlarda eğitim (stokastik politika) eğrisi
        t_curves = []
        for s in EVAL_SEEDS:
            sm = smooth(eps[s][tcol], W)
            gs = eps[s]["global_step"][W - 1:]
            t_curves.append(np.interp(x, gs, sm))
        t_arr, t_mean, t_std, _ = mean_std(t_curves)
        ax.fill_between(x, t_mean - t_std, t_mean + t_std, color="#7f7f7f", alpha=0.15)
        ax.plot(x, t_mean, "--", color="#555555", lw=1.8,
                label=f"Eğitim (stokastik politika), hareketli ort. w = {W}")

        ref = refs["hand_crafted_reward"] if ecol == "eval_reward_mean" \
            else refs["hand_crafted_yield"]
        ax.axhline(ref, ls=":", color="green", lw=1.2, label=f"hand-crafted ({ref}, {src})")
        tkey = "tuned_reward" if ecol == "eval_reward_mean" else "tuned_yield"
        if tkey in refs:
            ax.axhline(refs[tkey], ls=":", color="purple", lw=1.4,
                       label=f"hand-crafted ayarlı ({refs[tkey]}, {src})")
        if ecol == "eval_reward_mean":
            for i, when in [(0, "ilk"), (-1, "son")]:
                notes.append(("G2", f"{when} eval adımı", int(x[i])))
                notes.append(("G2", f"{when} eval: deterministik ödül", round(float(e_mean[i]), 2)))
                notes.append(("G2", f"{when} eval: eğitim (stokastik) ödül", round(float(t_mean[i]), 2)))
                notes.append(("G2", f"{when} eval: fark (det. − stok.)",
                              round(float(e_mean[i] - t_mean[i]), 2)))
        ax.set_xlabel("Kümülatif ortam adımı (gün)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center right")
    axes[0].set_title("Episode getirisi")
    axes[1].set_title("Hasat verimi")
    fig.suptitle(f"Grafik 2 — Test (eval) eğrisi  |  her {EVAL_CFG['eval_interval_updates']} "
                 f"güncellemede {n_val} validation senaryosu, {N_SEED} seed ort. ± std",
                 fontsize=12, fontweight="bold")
    finish(fig, "grafik2_eval_egrisi.png")


# ======================================================================
# Grafik 3 — Loss eğrileri
# ======================================================================
def graph3(ups, notes):
    panels = [("pg_loss", "Policy (clipped surrogate) kaybı", "Kayıp", False),
              ("v_loss", "Değer fonksiyonu kaybı", "0.5 · MSE", False),
              ("entropy", "Politika entropisi", "Entropi (nat)", False),
              ("approx_kl", "Yaklaşık KL ıraksaması", "KL (eski → yeni politika)", False)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, title, ylabel, logy) in zip(axes.flat, panels):
        curves = [smooth(ups[s][col], W_LOSS) for s in EVAL_SEEDS]
        arr, mean, std, L = mean_std(curves)
        x = ups[EVAL_SEEDS[0]]["global_step"][W_LOSS - 1:][:L]
        band_plot(ax, x, list(arr), mean, std, f"{N_SEED} seed ortalaması")
        if col == "approx_kl":
            ax2 = ax.twinx()
            c_arr, c_mean, c_std, _ = mean_std([smooth(ups[s]["clip_frac"], W_LOSS)
                                                for s in EVAL_SEEDS])
            ax2.fill_between(x, c_mean - c_std, c_mean + c_std, color="#d62728", alpha=0.12)
            ax2.plot(x, c_mean, color="#d62728", lw=1.8, label="clip oranı (sağ eksen)")
            ax2.set_ylabel("Kırpılan örnek oranı", color="#d62728")
            notes.append(("G3", "clip_frac: ilk değer", round(float(c_mean[0]), 4)))
            notes.append(("G3", "clip_frac: son değer", round(float(c_mean[-1]), 4)))
            ax2.legend(fontsize=8, loc="upper right")
        if logy:
            ax.set_yscale("log")
        notes.append(("G3", f"{col}: ilk değer", round(float(mean[0]), 4)))
        notes.append(("G3", f"{col}: son değer", round(float(mean[-1]), 4)))
        notes.append(("G3", f"{col}: en yüksek değer", round(float(mean.max()), 4)))
        notes.append(("G3", f"{col}: en yüksek değerin adımı", int(x[int(mean.argmax())])))
        ax.set_title(title)
        ax.set_xlabel("Kümülatif ortam adımı (gün)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle(f"Grafik 3 — PPO eğitim kayıpları  |  {N_SEED} seed, ort. ± std, "
                 f"hareketli ortalama w = {W_LOSS} güncelleme",
                 fontsize=12, fontweight="bold")
    finish(fig, "grafik3_loss.png")


# ======================================================================
# Grafik 4 — Hiperparametre duyarlılığı
# ======================================================================
def graph4():
    path = os.path.join(LOG_DIR, "sweep_log.csv")
    if not os.path.exists(path):
        return []
    d = read_csv(path)
    seeds = SWEEP_CFG["sweep_seeds"]

    def curves_for(lr, clip):
        out = []
        for s in seeds:
            m = np.isclose(d["lr"], lr) & np.isclose(d["clip_eps"], clip) & (d["seed"] == s)
            if m.sum() == 0:
                continue
            order = np.argsort(d["episode"][m])
            out.append(d["reward"][m][order])
        return out

    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sweeps = [(axes[0], "lr", SWEEP_CFG["lr_values"],
               lambda v: (v, SWEEP_CFG["baseline_clip"]),
               f"(a) Öğrenme oranı (clip_eps = {SWEEP_CFG['baseline_clip']})"),
              (axes[1], "clip_eps", SWEEP_CFG["clip_eps_values"],
               lambda v: (SWEEP_CFG["baseline_lr"], v),
               f"(b) PPO clip_eps (lr = {SWEEP_CFG['baseline_lr']:g})")]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for ax, pname, values, cfg_of, title in sweeps:
        for c, v in zip(colors, values):
            raw = curves_for(*cfg_of(v))
            if not raw:
                print(f"  ⚠ {pname}={v} verisi yok")
                continue
            arr, mean, std, L = mean_std([smooth(r, W) for r in raw])
            x = np.arange(W, W + L)
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
            ax.plot(x, mean, color=c, lw=2.2, label=f"{pname} = {v:g}  ({len(raw)} seed)")
            finals = np.array([r[-W_FIN:].mean() for r in raw])
            summary.append({"parametre": pname, "deger": v, "n_seed": len(raw),
                            "final_odul_ort": round(float(finals.mean()), 2),
                            "final_odul_std": round(float(finals.std()), 2)})
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode getirisi (toplam ödül)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right", title="ort. ± std")
    fig.suptitle(f"Grafik 4 — Hiperparametre duyarlılığı  |  {SWEEP_CFG['sweep_steps']:,} adım, "
                 f"{len(seeds)} seed, hareketli ortalama w = {W} episode",
                 fontsize=12, fontweight="bold")
    finish(fig, "grafik4_hiperparametre.png")
    return summary


# ======================================================================
# Grafik 5 — Baseline karşılaştırması (final test)
# ======================================================================
def graph5():
    d = read_csv(FINAL_CSV)
    base_names = ["Do-nothing", "Max-out", "Hand-crafted"]
    if (d["policy"] == TUNED).any():
        base_names.append(TUNED)
    order = base_names + [f"PPO seed={s}" for s in EVAL_SEEDS]
    groups = {p: (d["reward"][d["policy"] == p], d["yield_ton_ha"][d["policy"] == p])
              for p in order}
    ppo_R = np.concatenate([groups[f"PPO seed={s}"][0] for s in EVAL_SEEDS])
    ppo_Y = np.concatenate([groups[f"PPO seed={s}"][1] for s in EVAL_SEEDS])
    labels = [o.replace(" (ayarlı)", "\n(ayarlı)") for o in order] + [f"PPO\n({N_SEED} seed)"]
    data = {0: [groups[p][0] for p in order] + [ppo_R],
            1: [groups[p][1] for p in order] + [ppo_Y]}
    colors = ["#d9d9d9", "#bdbdbd", "#969696", "#b39ddb"][:len(base_names)] \
        + SEED_COLORS[:N_SEED] + ["#111111"]

    hc_R, hc_Y = groups["Hand-crafted"][0].mean(), groups["Hand-crafted"][1].mean()
    ratios = {0: ppo_R.mean() / hc_R, 1: ppo_Y.mean() / hc_Y}
    tuned_ratios = None
    if TUNED in groups:
        tuned_ratios = {0: ppo_R.mean() / groups[TUNED][0].mean(),
                        1: ppo_Y.mean() / groups[TUNED][1].mean()}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for i, (ax, ylabel, title) in enumerate([
            (axes[0], "Episode getirisi (toplam ödül)", "Episode getirisi"),
            (axes[1], "Hasat verimi (ton/ha)", "Hasat verimi")]):
        bp = ax.boxplot(data[i], labels=labels, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=5),
                        medianprops=dict(color="#222222", lw=1.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        sub = f"PPO / hand-crafted = {ratios[i]:.2f}×"
        if tuned_ratios:
            sub += f",  PPO / ayarlı = {tuned_ratios[i]:.2f}×"
        ax.set_title(f"{title}  —  {sub}", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle(f"Grafik 5 — Baseline karşılaştırması  |  {len(TEST_SEEDS)} test senaryosu "
                 f"(eğitimde görülmedi), deterministik politika, kutu: çeyrekler, ◇: ortalama",
                 fontsize=12, fontweight="bold")
    finish(fig, "grafik5_baseline.png")
    return groups


# ======================================================================
# sonuclar.csv — seed × metrik
# ======================================================================
def write_summary(eps, evs, groups, sweep_summary):
    rows = []
    for s in EVAL_SEEDS:
        tR, tY = groups[f"PPO seed={s}"]
        rows.append({
            "seed": s,
            "n_egitim_episode": len(eps[s]["reward"]),
            f"egitim_odul_son{W_FIN}": round(float(eps[s]["reward"][-W_FIN:].mean()), 2),
            f"egitim_verim_son{W_FIN}": round(float(eps[s]["yield_ton_ha"][-W_FIN:].mean()), 3),
            "val_odul_final": round(float(evs[s]["eval_reward_mean"][-1]), 2),
            "val_verim_final": round(float(evs[s]["eval_yield_mean"][-1]), 3),
            "test_odul_ort": round(float(tR.mean()), 2),
            "test_odul_std": round(float(tR.std()), 2),
            "test_verim_ort": round(float(tY.mean()), 3),
            "test_verim_std": round(float(tY.std()), 3),
        })
    keys = [k for k in rows[0] if k != "seed"]
    for label, fn in [("ortalama", np.mean), ("std", np.std)]:
        row = {"seed": label}
        for k in keys:
            row[k] = round(float(fn([r[k] for r in rows[:N_SEED]])), 3)
        rows.append(row)

    path = os.path.join(RES_DIR, "sonuclar.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ {path}")

    if sweep_summary:
        path = os.path.join(RES_DIR, "hiperparametre_ozet.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_summary[0].keys()))
            w.writeheader()
            w.writerows(sweep_summary)
        print(f"  ✓ {path}")
    return rows


def main():
    print("1) Ham loglar toplanıyor...")
    collect_logs()
    eps = load_seed_logs("episodes")
    evs = load_seed_logs("evals")
    ups = load_seed_logs("updates")

    refs, src = baseline_refs()
    notes = []
    print("2) Grafikler çiziliyor...")
    graph1(eps, refs, src, notes)
    graph2(eps, evs, refs, src, notes)
    graph3(ups, notes)
    sweep_summary = graph4()
    groups = graph5()

    print("3) Özet tablolar yazılıyor...")
    rows = write_summary(eps, evs, groups, sweep_summary)
    note_path = os.path.join(RES_DIR, "grafik_ozet.csv")
    with open(note_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grafik", "metrik", "deger"])
        w.writerows(notes)
    print(f"  ✓ {note_path}")

    print("\n" + "=" * 72)
    print("  sonuclar.csv (yorum yazarken bu sayıları kullan)")
    print("=" * 72)
    for r in rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in r.items()))
    if sweep_summary:
        print("\n  Hiperparametre taraması (son "
              f"{W_FIN} episode ortalaması, seedler arası ort. ± std):")
        for r in sweep_summary:
            print(f"    {r['parametre']:9s} = {r['deger']:<7g} → "
                  f"{r['final_odul_ort']:7.2f} ± {r['final_odul_std']:.2f}")


if __name__ == "__main__":
    main()
