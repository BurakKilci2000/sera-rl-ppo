"""
ppo_agent.py — Sıfırdan yazılmış PPO (PyTorch): eğitim ve değerlendirme.

Kullanım:
    python ppo_agent.py --seed 0                   # tek seed eğit
    python ppo_agent.py --all                      # config.yaml'daki 5 seed sırayla
    python ppo_agent.py --seed 0 --steps 20480     # kısa deneme eğitimi
    python ppo_agent.py --eval runs/ppo_seed42/ppo_checkpoint.pt

Her eğitim save_dir altına üç ham log yazar (grafikler bunlardan üretilir):
    episodes.csv : her eğitim episode'u   → Grafik 1 (öğrenme eğrisi)
    evals.csv    : periyodik deterministik test → Grafik 2 (eval eğrisi)
    updates.csv  : her PPO güncellemesi   → Grafik 3 (loss eğrisi)
Satırlar yazıldıkça diske işlenir; eğitim yarıda kesilse bile log kaybolmaz.
"""
from __future__ import annotations
import os, csv, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

from greenhouse_env import GreenhouseEnv, GreenhouseParams
from config import PPO_HP, EVAL_CFG, BASELINE_REF, PLOT_CFG, EVAL_SEEDS, TEST_SEEDS


# ======================================================================
# Hiperparametreler (config.yaml → config.py üzerinden)
# ======================================================================
HP = PPO_HP.copy()


# ======================================================================
# Aktör-Kritik ağı (ayrı gövdeler, durumdan bağımsız log_std)
# ======================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def get_action_and_value(self, obs, action=None):
        mean = self.actor(obs)
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


# ======================================================================
# Çalışan ortalama / varyans (Welford) — gözlem normalizasyonu
# ======================================================================
class RunningNorm:
    def __init__(self, dim: int):
        self.mean  = np.zeros(dim, dtype=np.float64)
        self.M2    = np.zeros(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        self.count += 1
        delta  = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)

    def normalize(self, x):
        std = np.sqrt(self.M2 / max(1.0, self.count - 1)) + 1e-8
        return (x - self.mean) / std


# ======================================================================
# Yardımcılar: cihaz, CSV log, deterministik koşu
# ======================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CsvLog:
    """Satır satır yazan CSV logger. flush() sayesinde eğitim kesilse de
    o ana kadarki satırlar diskte kalır."""
    def __init__(self, path: str, fields: list[str]):
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def write(self, **row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


@torch.no_grad()
def run_deterministic(agent, obs_norm, seeds, device):
    """Politikayı gürültüsüz çalıştırır: a = mu_theta(s), örnekleme yok.

    - Ayrı bir ortam nesnesi kullanılır → eğitim ortamının rastgelelik
      zinciri etkilenmez.
    - obs_norm yalnızca okunur (update çağrılmaz) → eğitimdeki
      normalizasyon istatistikleri değişmez.
    """
    env = GreenhouseEnv(GreenhouseParams())
    rewards, yields = [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        total_r, done = 0.0, False
        while not done:
            o = torch.tensor(obs_norm.normalize(obs), dtype=torch.float32, device=device)
            a = agent.actor(o).cpu().numpy()
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            done = term or trunc
        rewards.append(total_r)
        yields.append(info["yield_ton_ha"])
    return np.array(rewards), np.array(yields)


# ======================================================================
# Eğitim
# ======================================================================
def train(save_dir="./runs/ppo_run", **hp_overrides):
    hp = {**HP, **hp_overrides}
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    print(f"Hiperparametreler: {hp}")

    torch.manual_seed(hp["seed"])
    rng = np.random.default_rng(hp["seed"])     # np.random.seed() kullanılmaz

    env     = GreenhouseEnv(GreenhouseParams())
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent     = ActorCritic(obs_dim, act_dim, hp["hidden_size"]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=hp["lr"], eps=1e-5)
    obs_norm  = RunningNorm(obs_dim)

    # Validation seedleri: final test seedlerinden ayrı (config.yaml)
    v0 = EVAL_CFG["validation_seed_start"]
    val_seeds = list(range(v0, v0 + EVAL_CFG["n_validation_episodes"]))

    # Ham loglar (Grafik 1, 2, 3 bu dosyalardan çizilir)
    ep_log  = CsvLog(os.path.join(save_dir, "episodes.csv"),
                     ["episode", "global_step", "reward", "yield_ton_ha", "biomass_g_m2"])
    upd_log = CsvLog(os.path.join(save_dir, "updates.csv"),
                     ["update", "global_step", "pg_loss", "v_loss", "entropy",
                      "total_loss", "approx_kl", "clip_frac"])
    ev_log  = CsvLog(os.path.join(save_dir, "evals.csv"),
                     ["update", "global_step", "eval_reward_mean", "eval_reward_std",
                      "eval_yield_mean", "eval_yield_std"])

    # Rollout buffer'ı
    R = hp["rollout_steps"]
    b_obs  = torch.zeros((R, obs_dim), device=device)
    b_act  = torch.zeros((R, act_dim), device=device)
    b_logp = torch.zeros(R, device=device)
    b_rew  = torch.zeros(R, device=device)
    b_val  = torch.zeros(R, device=device)
    b_done = torch.zeros(R, device=device)

    obs, _ = env.reset(seed=hp["seed"])
    obs_norm.update(obs)
    obs_t = torch.tensor(obs_norm.normalize(obs), dtype=torch.float32, device=device)

    n_updates = hp["total_timesteps"] // R
    history = {"ep_r": [], "ep_y": [], "ep_b": []}
    ep_r, global_step, ep_count = 0.0, 0, 0
    t_start = time.time()

    for update in range(1, n_updates + 1):
        # ---------------- 1) Rollout topla ----------------
        for step in range(R):
            with torch.no_grad():
                a, lp, _, v = agent.get_action_and_value(obs_t)
            a_np = a.cpu().numpy()
            next_obs, r, term, trunc, info = env.step(a_np)
            done = term or trunc

            b_obs[step]  = obs_t
            b_act[step]  = a
            b_logp[step] = lp
            b_rew[step]  = float(r)
            b_val[step]  = v
            b_done[step] = float(done)

            ep_r += r
            global_step += 1
            if done:
                ep_count += 1
                history["ep_r"].append(ep_r)
                history["ep_y"].append(info["yield_ton_ha"])
                history["ep_b"].append(info["biomass_g_m2"])
                ep_log.write(episode=ep_count, global_step=global_step, reward=ep_r,
                             yield_ton_ha=info["yield_ton_ha"],
                             biomass_g_m2=info["biomass_g_m2"])
                ep_r = 0.0
                next_obs, _ = env.reset()

            obs_norm.update(next_obs)
            obs_t = torch.tensor(obs_norm.normalize(next_obs), dtype=torch.float32, device=device)

        # ---------------- 2) GAE hesapla ----------------
        with torch.no_grad():
            _, _, _, next_v = agent.get_action_and_value(obs_t)
        adv = torch.zeros(R, device=device)
        last_gae = 0.0
        for t in reversed(range(R)):
            nt = 1.0 - b_done[t]
            nv = next_v if t == R - 1 else b_val[t + 1]
            delta  = b_rew[t] + hp["gamma"] * nv * nt - b_val[t]
            adv[t] = last_gae = delta + hp["gamma"] * hp["gae_lambda"] * nt * last_gae
        ret = adv + b_val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------------- 3) PPO clipped güncellemesi ----------------
        stats = {k: [] for k in ["pg_loss", "v_loss", "entropy",
                                 "total_loss", "approx_kl", "clip_frac"]}
        idx = np.arange(R)
        mb = hp["minibatch_size"]
        for _epoch in range(hp["n_epochs"]):
            rng.shuffle(idx)
            for s in range(0, R, mb):
                mi = idx[s:s + mb]
                _, nlp, ent, nv = agent.get_action_and_value(b_obs[mi], b_act[mi])
                log_ratio = nlp - b_logp[mi]
                ratio = log_ratio.exp()
                pg1 = -adv[mi] * ratio
                pg2 = -adv[mi] * torch.clamp(ratio, 1 - hp["clip_eps"], 1 + hp["clip_eps"])
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss  = 0.5 * (nv - ret[mi]).pow(2).mean()
                e_loss  = ent.mean()
                loss = pg_loss + hp["vf_coef"] * v_loss - hp["ent_coef"] * e_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), hp["max_grad_norm"])
                optimizer.step()

                # Teşhis ölçütleri (öğrenmeyi etkilemez, sadece kaydedilir)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()   # eski→yeni politika uzaklığı
                    clip_frac = ((ratio - 1).abs() > hp["clip_eps"]).float().mean()
                stats["pg_loss"].append(pg_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["entropy"].append(e_loss.item())
                stats["total_loss"].append(loss.item())
                stats["approx_kl"].append(approx_kl.item())
                stats["clip_frac"].append(clip_frac.item())

        # Güncelleme başına minibatch ortalamaları → updates.csv
        upd_log.write(update=update, global_step=global_step,
                      **{k: float(np.mean(v)) for k, v in stats.items()})

        # ---------------- 4) Periyodik deterministik değerlendirme ----------------
        eval_msg = ""
        if update % EVAL_CFG["eval_interval_updates"] == 0 or update == n_updates:
            Rs, Ys = run_deterministic(agent, obs_norm, val_seeds, device)
            ev_log.write(update=update, global_step=global_step,
                         eval_reward_mean=float(Rs.mean()), eval_reward_std=float(Rs.std()),
                         eval_yield_mean=float(Ys.mean()), eval_yield_std=float(Ys.std()))
            eval_msg = f" | EVAL R {Rs.mean():7.2f}"

        # ---------------- 5) Konsol çıktısı ----------------
        if len(history["ep_r"]) > 0:
            n = min(10, len(history["ep_r"]))
            elapsed = time.time() - t_start
            print(f"Update {update:3d}/{n_updates} | step {global_step:>7d} | "
                  f"eps {ep_count:>4d} | "
                  f"R(last{n}) {np.mean(history['ep_r'][-n:]):7.2f} | "
                  f"Y(last{n}) {np.mean(history['ep_y'][-n:]):.2f} t/ha | "
                  f"({elapsed:.0f}s){eval_msg}")

    for lg in (ep_log, upd_log, ev_log):
        lg.close()

    ckpt = {
        "agent": agent.state_dict(),
        "obs_norm_mean": obs_norm.mean,
        "obs_norm_M2":   obs_norm.M2,
        "obs_norm_count": obs_norm.count,
        "history": history,
        "hp": hp,
    }
    torch.save(ckpt, os.path.join(save_dir, "ppo_checkpoint.pt"))
    print(f"\nKaydedildi: {save_dir}/ppo_checkpoint.pt")
    plot_curves(history, save_dir)
    return agent, history, obs_norm


# ======================================================================
# Tek seed için hızlı kontrol grafiği (asıl 5-seed grafikleri ayrı script)
# ======================================================================
def plot_curves(history, save_dir):
    if len(history["ep_r"]) < 2:
        print("Çok az episode, grafik atlanıyor.")
        return
    w = PLOT_CFG["smoothing_window"]
    eps = np.arange(1, len(history["ep_r"]) + 1)

    def smooth(x):
        x = np.asarray(x)
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(eps, history["ep_r"], alpha=0.25, color="steelblue", lw=0.7)
    sm = smooth(history["ep_r"])
    axes[0].plot(np.arange(len(sm)) + w, sm, color="steelblue", lw=2, label=f"PPO (hareketli ort. w={w})")
    ref = BASELINE_REF
    axes[0].axhline(ref["hand_crafted_reward"], ls="--", color="green",  lw=1,
                    label=f"hand-crafted ({ref['hand_crafted_reward']})")
    axes[0].axhline(ref["max_out_reward"], ls="--", color="orange", lw=1,
                    label=f"max-out ({ref['max_out_reward']})")
    axes[0].axhline(ref["do_nothing_reward"], ls="--", color="red", lw=1,
                    label=f"do-nothing ({ref['do_nothing_reward']})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode getirisi (toplam ödül)")
    axes[0].set_title("Öğrenme eğrisi")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(eps, history["ep_y"], alpha=0.25, color="darkgreen", lw=0.7)
    sm = smooth(history["ep_y"])
    axes[1].plot(np.arange(len(sm)) + w, sm, color="darkgreen", lw=2, label=f"PPO (hareketli ort. w={w})")
    axes[1].axhline(ref["hand_crafted_yield"], ls="--", color="green", lw=1,
                    label=f"hand-crafted ({ref['hand_crafted_yield']})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Verim (ton/ha)")
    axes[1].set_title("Verim gelişimi")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(save_dir, "learning_curves.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Eğri kaydedildi: {out}")


# ======================================================================
# Eğitilmiş ajanı final test senaryolarında değerlendir
# ======================================================================
def evaluate(ckpt_path: str):
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    env = GreenhouseEnv(GreenhouseParams())
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = ActorCritic(obs_dim, act_dim, ckpt["hp"]["hidden_size"]).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    onorm = RunningNorm(obs_dim)
    onorm.mean, onorm.M2, onorm.count = ckpt["obs_norm_mean"], ckpt["obs_norm_M2"], ckpt["obs_norm_count"]

    Rs, Ys = run_deterministic(agent, onorm, TEST_SEEDS, device)
    print(f"Değerlendirme ({len(TEST_SEEDS)} test senaryosu): "
          f"R = {Rs.mean():.2f} ± {Rs.std():.2f} | "
          f"Y = {Ys.mean():.2f} ± {Ys.std():.2f} ton/ha")
    return Rs, Ys


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Sera PPO eğitimi / değerlendirmesi")
    ap.add_argument("--seed", type=int, help="tek bir seed ile eğit")
    ap.add_argument("--all", action="store_true", help="config.yaml'daki tüm eğitim seedleri")
    ap.add_argument("--steps", type=int, default=None, help="toplam adım (varsayılan: config.yaml)")
    ap.add_argument("--out", default="./runs", help="çıktı kök klasörü")
    ap.add_argument("--eval", metavar="CKPT", help="checkpoint'i test senaryolarında değerlendir")
    args = ap.parse_args()

    if args.eval:
        evaluate(args.eval)
        return

    if args.all:
        seeds = EVAL_SEEDS
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        ap.error("--seed N, --all veya --eval CKPT vermelisin")

    overrides = {} if args.steps is None else {"total_timesteps": args.steps}
    for s in seeds:
        train(save_dir=os.path.join(args.out, f"ppo_seed{s}"), seed=s, **overrides)


if __name__ == "__main__":
    main()
