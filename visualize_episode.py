"""
visualize_episode.py
====================
Eğitilmiş PPO ajanının bir episode'unu sera kesiti olarak görselleştirir
ve MP4 video kaydeder.

Kullanım:
    python visualize_episode.py

Senaryo ve ajan config.yaml'dan okunur (experiment.trajectory_seed ve
experiment.trajectory_ppo_seed); böylece video, policy_analysis.py'daki
trajektori grafiğiyle aynı senaryoyu gösterir.

Çıktı: grafikler/episode_ppo.mp4 (FFmpeg yoksa .gif)

Görsel öğeler:
  - Üstte sera kesiti: zemin (toprak rengi su seviyesine göre değişir),
    bitki (gün gün büyüyen gövde + yapraklar), iç hava (sıcaklık rengi),
    güneş, ısıtıcı/havalandırma/sulama/gübre eylemleri
  - Altta canlı zaman serisi: sıcaklık, biyokütle, toprak suyu, eylemler
"""
from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse

from greenhouse_env import GreenhouseEnv, GreenhouseParams
from ppo_agent import ActorCritic, RunningNorm
from config import TRAJ_SEED, TRAJ_PPO_SEED

# ============================================================
# AYARLAR (senaryo ve ajan config.yaml'dan)
# ============================================================
CKPT_PATH = f"./runs/ppo_seed{TRAJ_PPO_SEED}/ppo_checkpoint.pt"
OUT_PATH  = "grafikler/episode_ppo.mp4"
SEED      = TRAJ_SEED            # trajektori grafikleriyle aynı senaryo
FPS       = 8                    # 8 fps × 120 frame = 15 sn video
DPI       = 110

# ============================================================
# Eğitilmiş ajanı yükle
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
env = GreenhouseEnv(GreenhouseParams())
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

agent = ActorCritic(obs_dim, act_dim, ckpt["hp"]["hidden_size"]).to(device)
agent.load_state_dict(ckpt["agent"]); agent.eval()
onorm = RunningNorm(obs_dim)
onorm.mean, onorm.M2, onorm.count = (
    ckpt["obs_norm_mean"], ckpt["obs_norm_M2"], ckpt["obs_norm_count"]
)

# ============================================================
# Episode'u koştur, tüm trajektoriyi topla
# ============================================================
print("Episode koşturuluyor...")
obs, _ = env.reset(seed=SEED)
traj = {k: [] for k in ["T_in", "T_out", "H_in", "W_soil", "N_soil",
                         "LAI", "B", "DVS", "I_rad",
                         "a_heat", "a_vent", "a_irr", "a_fert", "reward"]}
total_r = 0.0; done = False
while not done:
    o = torch.tensor(onorm.normalize(obs), dtype=torch.float32, device=device)
    with torch.no_grad():
        a_t = agent.actor(o)               # deterministik: a = mu(s)
    a = a_t.cpu().numpy()
    next_obs, r, term, trunc, info = env.step(a)
    traj["T_in"].append(obs[0]); traj["T_out"].append(obs[9])
    traj["H_in"].append(obs[1]); traj["W_soil"].append(obs[3])
    traj["N_soil"].append(obs[4]); traj["LAI"].append(obs[6])
    traj["B"].append(obs[7]);    traj["DVS"].append(obs[8])
    traj["I_rad"].append(obs[11])
    sa = info["action_scaled"]
    traj["a_heat"].append(sa["heat"]); traj["a_vent"].append(sa["vent"])
    traj["a_irr"].append(sa["irr"]);   traj["a_fert"].append(sa["fert"])
    traj["reward"].append(r)
    total_r += r
    obs = next_obs; done = term or trunc

T = len(traj["T_in"])
final_yield = info["yield_ton_ha"]
final_B     = info["biomass_g_m2"]
print(f"Episode bitti: {T} gün | toplam ödül = {total_r:.2f} | "
      f"verim = {final_yield:.2f} t/ha | biyokütle = {final_B:.0f} g/m²")
for k in traj: traj[k] = np.asarray(traj[k])


# ============================================================
# Renk yardımcıları (yalnızca görsel)
# ============================================================
def temp_to_color(T_c):
    """Sıcaklığa göre arka plan rengi: soğuk-mavi → ılık-yeşilimsi → sıcak-kırmızı."""
    t = np.clip(T_c / 40.0, 0, 1)
    if t < 0.5:
        f = t / 0.5
        return (0.4 + 0.45 * f, 0.6 + 0.35 * f, 0.95 - 0.20 * f)
    f = (t - 0.5) / 0.5
    return (0.85 + 0.10 * f, 0.95 - 0.55 * f, 0.75 - 0.55 * f)


def soil_color(W):
    """Toprak nemine göre kahverengi tonu: kuru-açık → ıslak-koyu."""
    p = env.p
    f = np.clip((W - p.wilting_point) / (p.field_capacity - p.wilting_point), 0, 1)
    return (0.75 - 0.40 * f, 0.55 - 0.30 * f, 0.30 - 0.20 * f)


# ============================================================
# Şekil ve eksen kurulumu
# ============================================================
fig = plt.figure(figsize=(12, 7), facecolor="#f7f7f7")
gs = fig.add_gridspec(3, 3, height_ratios=[3.2, 1, 1], hspace=0.35, wspace=0.35,
                      left=0.06, right=0.97, top=0.93, bottom=0.08)

ax_main  = fig.add_subplot(gs[0, :])      # büyük sera kesiti
ax_temp  = fig.add_subplot(gs[1, 0])
ax_bio   = fig.add_subplot(gs[1, 1])
ax_water = fig.add_subplot(gs[1, 2])
ax_act1  = fig.add_subplot(gs[2, :2])
ax_act2  = fig.add_subplot(gs[2, 2])

# --- Sera kesiti ---
ax_main.set_xlim(0, 10); ax_main.set_ylim(0, 5.5)
ax_main.set_aspect("equal"); ax_main.axis("off")
ax_main.add_patch(Rectangle((1, 1), 8, 3.5, fill=False, edgecolor="#222", linewidth=2.5))
ax_main.add_patch(Polygon([[1, 4.5], [5, 5.2], [9, 4.5]], closed=True,
                          facecolor="#dde8ee", edgecolor="#222", linewidth=2))

inside_air = Rectangle((1.05, 1.05), 7.9, 3.4, facecolor=(0.85, 0.95, 0.75),
                       edgecolor="none", zorder=1)
ax_main.add_patch(inside_air)
soil_rect = Rectangle((1.05, 1.05), 7.9, 0.55, facecolor=(0.55, 0.40, 0.20),
                      edgecolor="none", zorder=2)
ax_main.add_patch(soil_rect)

plant_artists = []

sun = Circle((9.5, 5.0), 0.35, facecolor="#ffd23f", edgecolor="#e6a800",
             linewidth=1.5, zorder=5)
ax_main.add_patch(sun)

heater_glow = Circle((1.5, 1.9), 0.25, facecolor="#ff6b35",
                     edgecolor="none", alpha=0.0, zorder=4)
ax_main.add_patch(heater_glow)
ax_main.text(1.5, 1.5, "ISI", ha="center", va="center", fontsize=8,
             color="#7d2c0f", fontweight="bold", zorder=5)

vent_line, = ax_main.plot([7.5, 8.5], [4.5, 4.5], color="#3498db",
                          linewidth=4, zorder=6, alpha=0.0)
ax_main.text(8.0, 4.85, "VENT", ha="center", va="center", fontsize=8,
             color="#1a5276", fontweight="bold", zorder=5)

rain_drops = []
for i in range(8):
    d, = ax_main.plot([], [], "o", color="#3498db", markersize=4, alpha=0.0, zorder=3)
    rain_drops.append(d)

fert_dots = []
for i in range(6):
    d, = ax_main.plot([], [], "s", color="#9b59b6", markersize=5, alpha=0.0, zorder=3)
    fert_dots.append(d)

title_txt = ax_main.text(0.3, 5.30, "", fontsize=11, fontweight="bold", color="#1a3a5c")
weather_txt = ax_main.text(0.3, 0.45, "", fontsize=9, ha="left", va="center",
                           color="#444",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                     edgecolor="#bbb", alpha=0.85))
T_label = ax_main.text(5.0, 4.10, "", ha="center", va="center",
                       fontsize=11, color="#222", fontweight="bold", zorder=10)

# --- Alttaki zaman serileri ---
days_axis = np.arange(T)

ax_temp.set_title("İç sıcaklık (°C)", fontsize=9)
ax_temp.set_xlim(0, T); ax_temp.set_ylim(5, 45)
ax_temp.axhspan(env.p.T_opt_low, env.p.T_opt_high, color="#2ecc71", alpha=0.15)
ax_temp.grid(alpha=0.3)
T_in_line,  = ax_temp.plot([], [], color="#c0392b", lw=1.5, label="iç")
T_out_line, = ax_temp.plot([], [], color="#888",    lw=1.0, label="dış")
ax_temp.legend(fontsize=7, loc="upper left")

ax_bio.set_title("Biyokütle (g/m²)", fontsize=9)
ax_bio.set_xlim(0, T); ax_bio.set_ylim(0, max(traj["B"].max() * 1.1, 50))
ax_bio.grid(alpha=0.3)
B_line, = ax_bio.plot([], [], color="#16a085", lw=2)

ax_water.set_title("Toprak suyu (hacimsel)", fontsize=9)
ax_water.set_xlim(0, T); ax_water.set_ylim(0.05, 0.40)
ax_water.axhline(env.p.wilting_point, ls="--", color="#888", lw=0.8)
ax_water.axhline(env.p.field_capacity, ls="--", color="#888", lw=0.8)
ax_water.grid(alpha=0.3)
W_line, = ax_water.plot([], [], color="#2980b9", lw=1.5)

ax_act1.set_title("İklim eylemleri", fontsize=9)
ax_act1.set_xlim(0, T); ax_act1.set_ylim(-0.05, 1.05)
ax_act1.grid(alpha=0.3)
heat_line, = ax_act1.plot([], [], color="#e67e22", lw=1.3, label="ısıtma")
vent_line2, = ax_act1.plot([], [], color="#3498db", lw=1.3, label="havalandırma")
ax_act1.legend(fontsize=7, loc="upper right")

ax_act2.set_title("Toprak eylemleri", fontsize=9)
ax_act2.set_xlim(0, T); ax_act2.set_ylim(-0.05, 1.05)
ax_act2.grid(alpha=0.3)
irr_line, = ax_act2.plot([], [], color="#1abc9c", lw=1.3, label="sulama")
fert_line, = ax_act2.plot([], [], color="#9b59b6", lw=1.3, label="gübre")
ax_act2.legend(fontsize=7, loc="upper right")

fig.suptitle(f"PPO ajanı (seed {TRAJ_PPO_SEED}) — test senaryosu {SEED}, deterministik politika",
             fontsize=13, fontweight="bold")


# ============================================================
# Bitki çizici
# ============================================================
def draw_plant(ax, t):
    """Frame t'de bitkiyi çiz (gövde + yapraklar + meyve). Önceki çizimleri sil."""
    for art in plant_artists:
        art.remove()
    plant_artists.clear()

    LAI = traj["LAI"][t]
    DVS = traj["DVS"][t]
    h = 0.2 + 1.8 * min(DVS, 1.0)          # yükseklik gelişme aşamasıyla artar

    for cx in (3.0, 5.0, 7.0):
        stem = Rectangle((cx - 0.05, 1.55), 0.10, h, facecolor="#2d6a3e",
                         edgecolor="#1f4a2c", linewidth=0.8, zorder=3)
        ax.add_patch(stem); plant_artists.append(stem)

        n_leaves = int(np.clip(LAI, 0, 5))  # yaprak sayısı LAI ile artar
        for i in range(n_leaves):
            y_leaf = 1.55 + (i + 1) * h / (n_leaves + 1)
            sign = -1 if i % 2 == 0 else +1
            leaf_w = 0.30 + 0.10 * min(LAI / 3.0, 1.0)
            leaf = Ellipse((cx + sign * (0.12 + leaf_w / 2), y_leaf), leaf_w, 0.18,
                           angle=sign * 18, facecolor="#3a8a4a", edgecolor="#205030",
                           linewidth=0.6, zorder=4)
            ax.add_patch(leaf); plant_artists.append(leaf)

        if DVS > 1.0:                        # gelişme ilerleyince meyve
            for i in range(int(np.clip((DVS - 1.0) * 4, 0, 3))):
                fr = Circle((cx + (0.10 if i % 2 else -0.10), 1.65 + h * 0.4 + i * 0.20),
                            0.08, facecolor="#d63031", edgecolor="#7d1414",
                            linewidth=0.6, zorder=5)
                ax.add_patch(fr); plant_artists.append(fr)


# ============================================================
# Frame güncelleme
# ============================================================
def update(t):
    inside_air.set_facecolor(temp_to_color(traj["T_in"][t]))
    soil_rect.set_facecolor(soil_color(traj["W_soil"][t]))
    draw_plant(ax_main, t)

    heater_glow.set_alpha(0.7 * traj["a_heat"][t])
    vent_line.set_alpha(traj["a_vent"][t])

    irr = traj["a_irr"][t]
    for i, d in enumerate(rain_drops):
        if irr > 0.05:
            phase = (t * 7 + i * 13) % 10
            d.set_data([2.0 + i * 0.95], [4.0 - phase * 0.25])
            d.set_alpha(np.clip(irr, 0, 1))
        else:
            d.set_alpha(0.0)

    fert = traj["a_fert"][t]
    for i, d in enumerate(fert_dots):
        if fert > 0.05:
            d.set_data([1.7 + i * 1.15], [1.20 + (i % 2) * 0.15])
            d.set_alpha(np.clip(fert, 0, 1))
        else:
            d.set_alpha(0.0)

    bright = np.clip(traj["I_rad"][t] / 30.0, 0.3, 1.0)
    sun.set_facecolor((1.0, 0.82 * bright, 0.25 * bright))
    sun.set_radius(0.25 + 0.20 * bright)

    title_txt.set_text(f"Gün {t + 1:>3d}/{T}   |   DVS = {traj['DVS'][t]:.2f}   |   "
                       f"Biyokütle = {traj['B'][t]:6.1f} g/m²")
    weather_txt.set_text(f"Dış: {traj['T_out'][t]:.1f}°C, ışınım {traj['I_rad'][t]:.1f} MJ/m²")
    T_label.set_text(f"İç: {traj['T_in'][t]:.1f}°C")

    x = days_axis[:t + 1]
    T_in_line.set_data(x, traj["T_in"][:t + 1])
    T_out_line.set_data(x, traj["T_out"][:t + 1])
    B_line.set_data(x, traj["B"][:t + 1])
    W_line.set_data(x, traj["W_soil"][:t + 1])
    heat_line.set_data(x, traj["a_heat"][:t + 1])
    vent_line2.set_data(x, traj["a_vent"][:t + 1])
    irr_line.set_data(x, traj["a_irr"][:t + 1])
    fert_line.set_data(x, traj["a_fert"][:t + 1])
    return []


# ============================================================
# Animasyonu kaydet
# ============================================================
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
print(f"Animasyon oluşturuluyor: {OUT_PATH}  ({T} frame, {FPS} fps)")
anim = FuncAnimation(fig, update, frames=T, interval=1000 // FPS, blit=False, repeat=False)

try:
    writer = FFMpegWriter(fps=FPS, bitrate=2200,
                          metadata={"artist": "Sera-RL", "title": "PPO episode"})
    anim.save(OUT_PATH, writer=writer, dpi=DPI)
    print(f"✓ Kaydedildi: {OUT_PATH}")
except Exception as e:
    print(f"⚠ MP4 başarısız ({e}). GIF olarak kaydediliyor...")
    gif_path = OUT_PATH.replace(".mp4", ".gif")
    anim.save(gif_path, writer="pillow", fps=FPS, dpi=80)
    print(f"✓ Kaydedildi: {gif_path}")

plt.close(fig)
print(f"\nFinal sonuçlar: ödül = {total_r:.2f}, verim = {final_yield:.2f} t/ha, "
      f"biyokütle = {final_B:.0f} g/m²")
