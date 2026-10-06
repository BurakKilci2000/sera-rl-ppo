"""
dashboard.py
============
Eğitilmiş PPO ajanının etkileşimli politika keşif arayüzü.

Kullanım:  python dashboard.py

Sol tarafta 6 slider (iç sıcaklık, toprak suyu, toprak azotu, LAI, DVS, gün)
ile bir durum kurulur. Sağ tarafta:
  - Sera kesiti: anlık durumun görselleştirmesi
  - Aynı durumda üç politikanın önerdiği eylemler:
      PPO  |  Hand-crafted (ayarlı)  |  Hand-crafted (orijinal)

Not: Slider'larla kurulan durum sentetiktir. Slider'da olmayan değişkenler
(dış hava, iç nem, CO2, biyokütle) config.yaml'daki sezon trendinden ve
basit varsayımlardan türetilir; bu nedenle eğitimde hiç görülmemiş durum
kombinasyonları da üretilebilir. Politikanın gerçek senaryolardaki davranışı
için: policy_analysis.py (trajektori ve politika sondajı).
"""
from __future__ import annotations
import numpy as np
import torch
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse

from greenhouse_env import GreenhouseEnv, GreenhouseParams
from ppo_agent import ActorCritic, RunningNorm
from baselines import policy_handcrafted, load_tuned_handcrafted
from config import TRAJ_PPO_SEED

# ============================================================
# Eğitilmiş ajanı yükle (gösterilen ajan: config.yaml → trajectory_ppo_seed)
# ============================================================
CKPT_PATH = f"./runs/ppo_seed{TRAJ_PPO_SEED}/ppo_checkpoint.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
env = GreenhouseEnv(GreenhouseParams())
P = env.p
IDX = GreenhouseEnv.IDX
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

agent = ActorCritic(obs_dim, act_dim, ckpt["hp"]["hidden_size"]).to(device)
agent.load_state_dict(ckpt["agent"])
agent.eval()
onorm = RunningNorm(obs_dim)
onorm.mean, onorm.M2, onorm.count = (
    ckpt["obs_norm_mean"], ckpt["obs_norm_M2"], ckpt["obs_norm_count"]
)

# Ayarlı kural: tune_handcrafted.py çıktısı yoksa None
policy_tuned = load_tuned_handcrafted()


# ============================================================
# Politikalar: hepsi [0, 1] aralığında eylem döner
# ============================================================
def get_ppo_action(state):
    o = torch.tensor(onorm.normalize(state), dtype=torch.float32, device=device)
    with torch.no_grad():
        mean = agent.actor(o).cpu().numpy()          # deterministik: a = mu(s)
    return GreenhouseEnv._scale_action(mean)


def get_rule_action(policy, state):
    if policy is None:
        return None
    return GreenhouseEnv._scale_action(policy(state))


# ============================================================
# Slider değerlerinden 13 boyutlu durum
# ============================================================
def build_state(T_in, W_soil, N_soil, LAI, DVS, day):
    """Slider'da olmayan değişkenler: dış hava config'teki sezon trendinden
    (gürültüsüz ortalama), iç nem ve CO2 basit varsayımlardan, biyokütle
    LAI ile tutarlı olacak şekilde."""
    season = day / max(1, P.episode_days - 1)
    T_out = P.T_out_start + P.T_out_rise * season
    I_rad = P.I_rad_start + P.I_rad_rise * season
    H_out = P.H_out_start - P.H_out_drop * season
    H_in = float(np.clip(H_out + P.init_H_in_offset, 0.0, 1.0))
    CO2 = P.CO2_ambient - P.CO2_plant_draw_max * min(LAI, P.LAI_saturation) / P.LAI_saturation
    B = max(P.init_B, LAI / (P.SLA * P.leaf_fraction))
    s = np.zeros(13, dtype=np.float32)
    s[IDX["T_in"]], s[IDX["H_in"]], s[IDX["CO2_in"]] = T_in, H_in, CO2
    s[IDX["W_soil"]], s[IDX["N_soil"]], s[IDX["T_soil"]] = W_soil, N_soil, T_in
    s[IDX["LAI"]], s[IDX["B"]], s[IDX["DVS"]] = LAI, B, DVS
    s[IDX["T_out"]], s[IDX["H_out"]], s[IDX["I_rad"]] = T_out, H_out, I_rad
    s[IDX["day_norm"]] = day / P.episode_days
    return s


# ============================================================
# Renk yardımcıları (yalnızca görsel)
# ============================================================
def temp_to_color(T_c):
    t = np.clip(T_c / 40.0, 0, 1)
    if t < 0.5:
        f = t / 0.5
        return (0.4 + 0.45 * f, 0.6 + 0.35 * f, 0.95 - 0.20 * f)
    f = (t - 0.5) / 0.5
    return (0.85 + 0.10 * f, 0.95 - 0.55 * f, 0.75 - 0.55 * f)


def soil_color(W):
    f = np.clip((W - P.wilting_point) / (P.field_capacity - P.wilting_point), 0, 1)
    return (0.75 - 0.40 * f, 0.55 - 0.30 * f, 0.30 - 0.20 * f)


ACTION_NAMES = ["Isıtma", "Havalan.", "Sulama", "Gübre"]
ACTION_COLORS = ["#e67e22", "#3498db", "#1abc9c", "#9b59b6"]
DEFAULTS = dict(T_in=22.0, W_soil=0.30, N_soil=60.0, LAI=2.0, DVS=0.8, day=60)


# ============================================================
# Tkinter penceresi
# ============================================================
class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("PPO Sera Kontrol — Etkileşimli Politika Keşif Aracı")
        root.geometry("1400x760")
        root.configure(bg="#f7f7f7")

        left = tk.Frame(root, bg="#eaecef", width=320, padx=15, pady=15)
        right = tk.Frame(root, bg="#f7f7f7", padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(left, text="Durum Ayarları", font=("Arial", 14, "bold"),
                 bg="#eaecef", fg="#1a3a5c").pack(pady=(0, 12))

        # Slider aralıkları politikaların gerçek çalışma bölgesini kapsar:
        # toprak suyu tarla kapasitesine kadar, azot ayarlı kuralın hedefinin üstüne kadar
        self.sliders = {}
        specs = [
            ("T_in",   "İç sıcaklık (°C)",        5.0, 45.0, 0.1),
            ("W_soil", "Toprak suyu (m³/m³)",     0.05, P.field_capacity, 0.005),
            ("N_soil", "Toprak azotu (kg N/ha)",  0.0, 400.0, 1.0),
            ("LAI",    "Yaprak alan indeksi",     0.05, P.LAI_max, 0.05),
            ("DVS",    "Gelişme aşaması (DVS)",   0.0, P.DVS_maturity, 0.02),
            ("day",    f"Gün (0–{P.episode_days})", 0, P.episode_days, 1),
        ]
        for key, label, lo, hi, step in specs:
            frame = tk.Frame(left, bg="#eaecef")
            frame.pack(fill=tk.X, pady=4)
            tk.Label(frame, text=label, font=("Arial", 10), bg="#eaecef",
                     anchor="w").pack(fill=tk.X)
            value_label = tk.Label(frame, text="", font=("Arial", 11, "bold"),
                                   bg="#eaecef", fg="#2d5a8a")
            value_label.pack(anchor="e")
            scale = tk.Scale(frame, from_=lo, to=hi, resolution=step,
                             orient=tk.HORIZONTAL, length=280, showvalue=0,
                             bg="#eaecef", troughcolor="#cfd6dc", highlightthickness=0,
                             command=lambda v, k=key, lbl=value_label:
                                 self._on_slider(k, float(v), lbl))
            scale.set(DEFAULTS[key])
            scale.pack(fill=tk.X)
            self.sliders[key] = scale

        tk.Button(left, text="Varsayılana sıfırla", command=self.reset_sliders,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"),
                  activebackground="#2980b9", relief=tk.FLAT, pady=6).pack(fill=tk.X, pady=(15, 5))

        self.info_txt = tk.Label(left, text="", font=("Arial", 9), bg="#eaecef",
                                 fg="#333", wraplength=290, justify="left")
        self.info_txt.pack(fill=tk.X, pady=(10, 0))

        tk.Label(left, text=("Not: Slider'larla kurulan durum sentetiktir; dış hava, "
                             "nem ve CO₂ sezon trendinden türetilir. Gerçek senaryolardaki "
                             "davranış için policy_analysis.py çıktılarına bakın."),
                 font=("Arial", 8, "italic"), bg="#eaecef", fg="#666",
                 wraplength=290, justify="left").pack(fill=tk.X, side=tk.BOTTOM)

        self.fig = Figure(figsize=(10.5, 6.8), dpi=95, facecolor="#f7f7f7")
        gs = self.fig.add_gridspec(2, 3, height_ratios=[2.4, 1], hspace=0.42, wspace=0.28,
                                   left=0.05, right=0.98, top=0.93, bottom=0.10)
        self.ax_main = self.fig.add_subplot(gs[0, :])
        self.ax_bars = [self.fig.add_subplot(gs[1, i]) for i in range(3)]

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._setup_main_axes()
        self._setup_action_axes()
        self.update_view()

    # ------------------------------------------------------------------
    def _setup_main_axes(self):
        ax = self.ax_main
        ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Sera kesiti", fontsize=11, fontweight="bold")
        ax.add_patch(Rectangle((1, 1), 8, 3.5, fill=False, edgecolor="#222", linewidth=2.5))
        ax.add_patch(Polygon([[1, 4.5], [5, 5.2], [9, 4.5]], closed=True,
                             facecolor="#dde8ee", edgecolor="#222", linewidth=2))
        self.inside_air = Rectangle((1.05, 1.05), 7.9, 3.4, facecolor=(0.85, 0.95, 0.75),
                                    edgecolor="none", zorder=1)
        ax.add_patch(self.inside_air)
        self.soil = Rectangle((1.05, 1.05), 7.9, 0.55, facecolor=(0.55, 0.40, 0.20),
                              edgecolor="none", zorder=2)
        ax.add_patch(self.soil)
        self.sun = Circle((9.5, 5.0), 0.30, facecolor="#ffd23f", edgecolor="#e6a800",
                          linewidth=1.2, zorder=5)
        ax.add_patch(self.sun)
        self.title_txt = ax.text(0.3, 5.30, "", fontsize=10, fontweight="bold", color="#1a3a5c")
        self.weather_txt = ax.text(0.3, 0.55, "", fontsize=9, ha="left", color="#444",
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                             edgecolor="#bbb", alpha=0.85))
        self.T_label = ax.text(5.0, 4.15, "", ha="center", va="center", fontsize=11,
                               color="#222", fontweight="bold", zorder=10)
        self.plant_artists = []

    def _setup_action_axes(self):
        titles = [("PPO ajanı", "#2d5a8a"),
                  ("Hand-crafted (ayarlı)", "#6a1b9a"),
                  ("Hand-crafted (orijinal)", "#7d7d7d")]
        self.bars, self.bar_txts = [], []
        for ax, (title, color), alpha in zip(self.ax_bars, titles, [1.0, 0.75, 0.5]):
            ax.set_xlim(-0.5, 3.5); ax.set_ylim(0, 1.12)
            ax.set_xticks(range(4)); ax.set_xticklabels(ACTION_NAMES, fontsize=8)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_title(title, fontsize=10, fontweight="bold", color=color)
            ax.grid(alpha=0.3, axis="y")
            self.bars.append(ax.bar(range(4), [0] * 4, color=ACTION_COLORS,
                                    edgecolor="#333", linewidth=1, alpha=alpha))
            self.bar_txts.append([ax.text(i, 0.05, "", ha="center", fontsize=8,
                                          fontweight="bold") for i in range(4)])
        self.ax_bars[0].set_ylabel("Eylem oranı [0, 1]", fontsize=9)
        if policy_tuned is None:
            self.ax_bars[1].text(1.5, 0.55, "Ayarlı eşik yok:\npython tune_handcrafted.py",
                                 ha="center", va="center", fontsize=8, color="#6a1b9a")

    # ------------------------------------------------------------------
    def _draw_plant(self, LAI, DVS):
        for art in self.plant_artists:
            art.remove()
        self.plant_artists.clear()
        ax = self.ax_main
        h = 0.2 + 1.8 * min(DVS, 1.0)
        for cx in (3.0, 5.0, 7.0):
            stem = Rectangle((cx - 0.05, 1.55), 0.10, h, facecolor="#2d6a3e",
                             edgecolor="#1f4a2c", linewidth=0.8, zorder=3)
            ax.add_patch(stem); self.plant_artists.append(stem)
            n_leaves = int(np.clip(LAI, 0, 5))
            for i in range(n_leaves):
                y_leaf = 1.55 + (i + 1) * h / (n_leaves + 1)
                sign = -1 if i % 2 == 0 else +1
                leaf_w = 0.30 + 0.10 * min(LAI / 3.0, 1.0)
                leaf = Ellipse((cx + sign * (0.12 + leaf_w / 2), y_leaf), leaf_w, 0.18,
                               angle=sign * 18, facecolor="#3a8a4a", edgecolor="#205030",
                               linewidth=0.6, zorder=4)
                ax.add_patch(leaf); self.plant_artists.append(leaf)
            if DVS > 1.0:
                for i in range(int(np.clip((DVS - 1.0) * 4, 0, 3))):
                    fr = Circle((cx + (0.10 if i % 2 else -0.10), 1.65 + h * 0.4 + i * 0.20),
                                0.08, facecolor="#d63031", edgecolor="#7d1414",
                                linewidth=0.6, zorder=5)
                    ax.add_patch(fr); self.plant_artists.append(fr)

    # ------------------------------------------------------------------
    def _on_slider(self, key, value, label):
        fmt = {"N_soil": "{:.0f}", "day": "{:.0f}", "T_in": "{:.1f}",
               "LAI": "{:.2f}", "DVS": "{:.2f}"}.get(key, "{:.3f}")
        label.config(text=fmt.format(value))
        if hasattr(self, "bars"):
            self.update_view()

    def reset_sliders(self):
        for k, v in DEFAULTS.items():
            self.sliders[k].set(v)

    # ------------------------------------------------------------------
    def update_view(self):
        v = {k: float(s.get()) for k, s in self.sliders.items()}
        day = int(v["day"])
        s = build_state(v["T_in"], v["W_soil"], v["N_soil"], v["LAI"], v["DVS"], day)

        actions = [get_ppo_action(s),
                   get_rule_action(policy_tuned, s),
                   get_rule_action(policy_handcrafted, s)]

        self.inside_air.set_facecolor(temp_to_color(v["T_in"]))
        self.soil.set_facecolor(soil_color(v["W_soil"]))
        self._draw_plant(v["LAI"], v["DVS"])
        bright = np.clip(s[IDX["I_rad"]] / 30.0, 0.3, 1.0)
        self.sun.set_facecolor((1.0, 0.82 * bright, 0.25 * bright))
        self.sun.set_radius(0.22 + 0.18 * bright)
        self.title_txt.set_text(f"Gün {day}/{P.episode_days}   |   DVS = {v['DVS']:.2f}   |   "
                                f"LAI = {v['LAI']:.2f}   |   N = {v['N_soil']:.0f} kg/ha")
        self.weather_txt.set_text(f"Dış (sezon ortalaması): {s[IDX['T_out']]:.1f}°C, "
                                  f"ışınım {s[IDX['I_rad']]:.1f} MJ/m²")
        self.T_label.set_text(f"İç: {v['T_in']:.1f}°C")

        for bars, txts, a in zip(self.bars, self.bar_txts, actions):
            vals = a if a is not None else np.zeros(4)
            for i, b in enumerate(bars):
                b.set_height(vals[i])
                txts[i].set_position((i, max(vals[i] + 0.04, 0.06)))
                txts[i].set_text(f"{vals[i]:.2f}" if a is not None else "")

        names = ["ısıtma", "havalandırma", "sulama", "gübre"]
        lines = []
        for ref_name, ref in [("ayarlı kural", actions[1]), ("orijinal kural", actions[2])]:
            if ref is None:
                continue
            d = actions[0] - ref
            i = int(np.argmax(np.abs(d)))
            lines.append(f"PPO − {ref_name}: en büyük fark {names[i]} "
                         f"({actions[0][i]:.2f} / {ref[i]:.2f}, {d[i]:+.2f})")
        self.info_txt.config(text="\n".join(lines))
        self.canvas.draw_idle()


# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()
