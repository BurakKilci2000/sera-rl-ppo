"""
greenhouse_env.py
=================
Gymnasium API'sine uyumlu, sıfırdan yazılmış sera ortamı.

Domates yetiştirilen orta ölçekli plastik sera için RL ajanı,
ısıtma / havalandırma / sulama / gübre kontrolünü öğrenir.

Üç alt model:
  1) İklim:  iç sıcaklık, iç nem, iç CO2 (enerji-kütle dengesi)
  2) Toprak: hacimsel su (bucket), azot (alım + yıkanma)
  3) Bitki:  RUE-tabanlı biyokütle, LAI, DVS (gelişme aşaması)

Stokastiklik (parametreleri config.yaml → environment.stochasticity):
  - Dış hava (sıcaklık, radyasyon, nem) her episode'da yeniden örneklenir.
  - İklim geçişinde günlük gürültü (iç sıcaklık, nem, CO2).
  - Biyolojik büyüme oranı varyansı (~%5).

Tüm ayarlanabilir sayılar config.yaml'dadır. Kodda yalnızca birim
dönüşümleri, Hargreaves formül sabitleri ve gözlem uzayı sınırları kalır.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import REWARD_WEIGHTS, ENV_PARAMS

E = ENV_PARAMS   # kısaltma: aşağıdaki varsayılan değerler config.yaml'dan gelir


# ===========================================================================
# Parametreler (değerler config.yaml → environment bölümünde)
# ===========================================================================
@dataclass(frozen=True)
class GreenhouseParams:
    # --- Episode / zaman ---
    episode_days: int = E["episode_days"]
    DVS_maturity: float = E["DVS_maturity"]

    # --- Sera geometrisi ---
    floor_area_m2: float = E["floor_area_m2"]
    cover_area_m2: float = E["cover_area_m2"]

    # --- Termal ---
    cover_transmissivity: float = E["cover_transmissivity"]
    U_loss: float = E["U_loss"]
    U_vent_max: float = E["U_vent_max"]
    P_heat_max: float = E["P_heat_max"]
    thermal_relax: float = E["thermal_relax"]

    # --- İç iklim (nem, CO2) ---
    CO2_ambient: float = E["CO2_ambient"]
    CO2_min: float = E["CO2_min"]
    CO2_max: float = E["CO2_max"]
    CO2_plant_draw_max: float = E["CO2_plant_draw_max"]
    transp_base: float = E["transp_base"]
    transp_gain: float = E["transp_gain"]
    transp_to_humidity: float = E["transp_to_humidity"]
    LAI_saturation: float = E["LAI_saturation"]

    # --- Toprak (kök bölgesi) ---
    field_capacity: float = E["field_capacity"]
    wilting_point: float = E["wilting_point"]
    soil_depth_m: float = E["soil_depth_m"]
    Kc_base: float = E["Kc_base"]
    Kc_slope: float = E["Kc_slope"]
    Kc_max: float = E["Kc_max"]
    T_range_offset: float = E["T_range_offset"]
    T_range_min: float = E["T_range_min"]
    leach_max_frac: float = E["leach_max_frac"]
    soil_temp_memory: float = E["soil_temp_memory"]
    soil_temp_air_weight: float = E["soil_temp_air_weight"]

    # --- Bitki (domates) ---
    RUE: float = E["RUE"]
    PAR_fraction: float = E["PAR_fraction"]
    k_extinction: float = E["k_extinction"]
    SLA: float = E["SLA"]
    leaf_fraction: float = E["leaf_fraction"]
    LAI_max: float = E["LAI_max"]
    T_min_growth: float = E["T_min_growth"]
    T_opt_low: float = E["T_opt_low"]
    T_opt_high: float = E["T_opt_high"]
    T_max_growth: float = E["T_max_growth"]
    T_base_GDD: float = E["T_base_GDD"]
    DVS_thermal_demand: float = E["DVS_thermal_demand"]
    N_demand_per_g: float = E["N_demand_per_g"]
    N_half_saturation: float = E["N_half_saturation"]
    harvest_index: float = E["harvest_index"]

    # --- Aksiyon ölçekleri ---
    irrigation_max_mm: float = E["irrigation_max_mm"]
    fertilizer_max_kg_ha: float = E["fertilizer_max_kg_ha"]

    # --- Sentetik hava trendi ---
    T_out_start: float = E["T_out_start"]
    T_out_rise: float = E["T_out_rise"]
    I_rad_start: float = E["I_rad_start"]
    I_rad_rise: float = E["I_rad_rise"]
    I_rad_min: float = E["I_rad_min"]
    I_rad_max: float = E["I_rad_max"]
    H_out_start: float = E["H_out_start"]
    H_out_drop: float = E["H_out_drop"]
    H_out_min: float = E["H_out_min"]
    H_out_max: float = E["H_out_max"]

    # --- Stokastiklik ---
    weather_noise_T: float = E["weather_noise_T"]
    weather_noise_I: float = E["weather_noise_I"]
    weather_noise_H: float = E["weather_noise_H"]
    climate_noise_T: float = E["climate_noise_T"]
    climate_noise_H: float = E["climate_noise_H"]
    climate_noise_CO2: float = E["climate_noise_CO2"]
    biological_noise: float = E["biological_noise"]

    # --- Ödül şekli ---
    humidity_threshold: float = E["humidity_threshold"]

    # --- Başlangıç durumu ---
    init_T_in_offset: float = E["init_T_in_offset"]
    init_H_in_offset: float = E["init_H_in_offset"]
    init_W_soil: float = E["init_W_soil"]
    init_N_soil: float = E["init_N_soil"]
    init_LAI: float = E["init_LAI"]
    init_B: float = E["init_B"]

    # --- Ödül ağırlıkları (config.yaml → reward_weights) ---
    w_yield_terminal: float = REWARD_WEIGHTS["w_yield_terminal"]
    w_growth_daily:   float = REWARD_WEIGHTS["w_growth_daily"]
    w_water:          float = REWARD_WEIGHTS["w_water"]
    w_energy:         float = REWARD_WEIGHTS["w_energy"]
    w_fert:           float = REWARD_WEIGHTS["w_fert"]
    w_stress:         float = REWARD_WEIGHTS["w_stress"]
    w_humidity:       float = REWARD_WEIGHTS["w_humidity"]


# ===========================================================================
# Şanlıurfa benzeri sentetik hava üretici
# ===========================================================================
class WeatherGenerator:
    """
    Her episode başında bir yetiştirme sezonu (Mart-Temmuz) için günlük dış
    sıcaklık, radyasyon ve nem trajektörisi üretir:
    mevsimsel doğrusal trend + Gauss gürültüsü.
    """
    def __init__(self, params: GreenhouseParams, np_random: np.random.Generator):
        self.p = params
        self.np_random = np_random

    def generate_season(self) -> Dict[str, np.ndarray]:
        p = self.p
        D = p.episode_days
        days = np.arange(D)

        # Mevsimsel rampa: Mart başı → Temmuz sonu (0 → 1)
        season = days / max(1, D - 1)

        # Sıcaklık trendi
        T_base = p.T_out_start + p.T_out_rise * season
        T_noise = self.np_random.normal(0, p.weather_noise_T, size=D)
        T_out = T_base + T_noise

        # Radyasyon trendi (MJ/m²/gün)
        I_base = p.I_rad_start + p.I_rad_rise * season
        I_noise = self.np_random.normal(0, p.weather_noise_I, size=D)
        I_rad = np.clip(I_base + I_noise, p.I_rad_min, p.I_rad_max)

        # Nem trendi (yaza doğru kuruyor)
        H_base = p.H_out_start - p.H_out_drop * season
        H_noise = self.np_random.normal(0, p.weather_noise_H, size=D)
        H_out = np.clip(H_base + H_noise, p.H_out_min, p.H_out_max)

        return {"T_out": T_out, "I_rad": I_rad, "H_out": H_out}


# ===========================================================================
# Sera Ortamı
# ===========================================================================
class GreenhouseEnv(gym.Env):
    """
    Gymnasium ortamı.

    Observation (13D, sürekli):
        [T_in, H_in, CO2_in,
         W_soil, N_soil, T_soil,
         LAI, B, DVS,
         T_out, H_out, I_rad,
         day_norm]   (day_norm = t / episode_days)

    Action (4D, [-1, 1]):
        a[0] = ısıtma yoğunluğu      → [0, 1]
        a[1] = havalandırma açıklığı → [0, 1]
        a[2] = sulama (mm/gün)       → [0, 1] × irrigation_max
        a[3] = gübre (kg N/ha/gün)   → [0, 1] × fertilizer_max
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    IDX = {
        "T_in": 0, "H_in": 1, "CO2_in": 2,
        "W_soil": 3, "N_soil": 4, "T_soil": 5,
        "LAI": 6, "B": 7, "DVS": 8,
        "T_out": 9, "H_out": 10, "I_rad": 11,
        "day_norm": 12,
    }

    def __init__(self, params: Optional[GreenhouseParams] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.p = params if params is not None else GreenhouseParams()
        self.render_mode = render_mode

        # Aksiyonlar [-1, 1] (Gauss politika ile uyumlu); step içinde [0,1]'e çevrilir
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Gözlem uzayı sınırları (arayüz tanımı; bilinçli olarak geniş tutuldu)
        obs_low = np.array([
            -10.0, 0.0, 200.0,        # T_in, H_in, CO2
            0.0, 0.0, -10.0,          # W_soil, N_soil, T_soil
            0.0, 0.0, 0.0,            # LAI, B, DVS
            -10.0, 0.0, 0.0,          # T_out, H_out, I_rad
            0.0,                      # day_norm
        ], dtype=np.float32)
        obs_high = np.array([
            55.0, 1.0, 1500.0,
            0.6, 500.0, 50.0,
            10.0, 5000.0, 2.0,
            55.0, 1.0, 40.0,
            1.0,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.t: int = 0
        self.weather: Dict[str, np.ndarray] = {}
        self.state: np.ndarray = np.zeros(13, dtype=np.float32)
        self._last_dB: float = 0.0  # önceki adımdaki büyüme (azot alımı için)

    # ----------------------------------------------------------------------
    # Yardımcılar
    # ----------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return self.state.copy()

    def _get_info(self) -> Dict[str, Any]:
        idx = self.IDX
        return {
            "day": self.t,
            "biomass_g_m2": float(self.state[idx["B"]]),
            # g/m² × 10 → kg/ha; × harvest_index; /1000 → ton/ha
            "yield_ton_ha": float(self.state[idx["B"]] * 10 * self.p.harvest_index / 1000.0),
            "DVS": float(self.state[idx["DVS"]]),
        }

    @staticmethod
    def _scale_action(action: np.ndarray) -> np.ndarray:
        """[-1, 1] → [0, 1]."""
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        return (a + 1.0) / 2.0

    # ----------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        p = self.p

        wgen = WeatherGenerator(p, self.np_random)
        self.weather = wgen.generate_season()

        self.t = 0
        self._last_dB = 0.0

        idx = self.IDX
        s = np.zeros(13, dtype=np.float32)

        T_out0 = self.weather["T_out"][0]
        H_out0 = self.weather["H_out"][0]
        I_rad0 = self.weather["I_rad"][0]

        # Başlangıç: küçük fide, iç koşullar dışa yakın
        s[idx["T_in"]]   = T_out0 + p.init_T_in_offset
        s[idx["H_in"]]   = float(np.clip(H_out0 + p.init_H_in_offset, 0.0, 1.0))
        s[idx["CO2_in"]] = p.CO2_ambient
        s[idx["W_soil"]] = p.init_W_soil
        s[idx["N_soil"]] = p.init_N_soil
        s[idx["T_soil"]] = T_out0
        s[idx["LAI"]]    = p.init_LAI
        s[idx["B"]]      = p.init_B
        s[idx["DVS"]]    = 0.0
        s[idx["T_out"]]  = T_out0
        s[idx["H_out"]]  = H_out0
        s[idx["I_rad"]]  = I_rad0
        s[idx["day_norm"]] = 0.0

        self.state = s
        return self._get_obs(), self._get_info()

    # ----------------------------------------------------------------------
    # Alt modeller
    # ----------------------------------------------------------------------
    def _climate_step(self, a_heat: float, a_vent: float
                      ) -> Tuple[float, float, float]:
        """İç sıcaklık, nem ve CO2'yi günlük denge yaklaşımıyla günceller."""
        idx = self.IDX
        p = self.p
        T_in_old = float(self.state[idx["T_in"]])
        T_out    = float(self.state[idx["T_out"]])
        H_out    = float(self.state[idx["H_out"]])
        I_rad    = float(self.state[idx["I_rad"]])    # MJ/m²/gün
        LAI      = float(self.state[idx["LAI"]])

        # MJ/m²/gün → W/m² (birim dönüşümü)
        I_W = I_rad * 1e6 / 86400.0

        # Enerji dengesi (W/m²)
        Q_solar_in = p.cover_transmissivity * I_W
        Q_heat_in  = a_heat * p.P_heat_max
        U_total    = p.U_loss + a_vent * p.U_vent_max  # W/(m²·K)

        # Denge sıcaklığı
        denom = max(U_total, 1e-3)
        T_in_eq = T_out + (Q_solar_in + Q_heat_in) / denom

        # Termal kütle nedeniyle gevşeme + gürültü (stokastik geçiş)
        T_in = T_in_old + p.thermal_relax * (T_in_eq - T_in_old)
        T_in += self.np_random.normal(0.0, p.climate_noise_T)

        # Nem: transpirasyon ekler, havalandırma dış değere çeker
        transpiration_input = p.transp_base + p.transp_gain * min(LAI, p.LAI_saturation) / p.LAI_saturation
        H_in_old = float(self.state[idx["H_in"]])
        H_target = (1 - a_vent) * (H_in_old + transpiration_input * p.transp_to_humidity) + a_vent * H_out
        H_in = float(np.clip(H_target + self.np_random.normal(0, p.climate_noise_H), 0.0, 1.0))

        # CO2: bitki tüketimi (LAI ile orantılı), havalandırma dış değere çeker
        CO2_in_old = float(self.state[idx["CO2_in"]])
        plant_draw = p.CO2_plant_draw_max * min(LAI, p.LAI_saturation) / p.LAI_saturation
        CO2_target = (1 - a_vent) * (CO2_in_old - plant_draw) + a_vent * p.CO2_ambient
        CO2_in = float(np.clip(CO2_target + self.np_random.normal(0, p.climate_noise_CO2),
                               p.CO2_min, p.CO2_max))

        return T_in, H_in, CO2_in

    def _soil_step(self, a_irr_norm: float, a_fert_norm: float, T_in: float
                   ) -> Tuple[float, float, float, float, float]:
        """Toprak su (W) ve azot (N). Geri döner: (W_new, N_new, T_soil, ET, drainage_mm)"""
        idx = self.IDX
        p = self.p
        W_old   = float(self.state[idx["W_soil"]])
        N_old   = float(self.state[idx["N_soil"]])
        I_rad   = float(self.state[idx["I_rad"]])
        T_out   = float(self.state[idx["T_out"]])
        LAI     = float(self.state[idx["LAI"]])

        irrigation_mm = a_irr_norm * p.irrigation_max_mm
        fert_kg_ha    = a_fert_norm * p.fertilizer_max_kg_ha
        soil_depth_mm = p.soil_depth_m * 1000.0           # m → mm

        # Hargreaves ET0 (mm/gün); 0.0023, 17.8, 2.45 formülün kendi sabitleri
        T_range = max(p.T_range_min, abs(T_in - T_out) + p.T_range_offset)
        ET0 = 0.0023 * max(0.0, T_in + 17.8) * np.sqrt(T_range) * I_rad / 2.45
        Kc = min(p.Kc_max, p.Kc_base + p.Kc_slope * min(LAI, p.LAI_saturation))
        ET_potential = ET0 * Kc

        # Su mevcudiyeti
        avail_frac = (W_old - p.wilting_point) / max(1e-6, p.field_capacity - p.wilting_point)
        avail_frac = float(np.clip(avail_frac, 0.0, 1.0))
        ET_actual = ET_potential * avail_frac

        # Su dengesi (hacimsel su içeriği)
        W_new = W_old + (irrigation_mm - ET_actual) / soil_depth_mm
        # Drenaj (tarla kapasitesinin üstü kaybedilir)
        if W_new > p.field_capacity:
            drainage_mm = (W_new - p.field_capacity) * soil_depth_mm
            W_new = p.field_capacity
        else:
            drainage_mm = 0.0
        W_new = float(max(0.0, W_new))

        # Azot dengesi — alım: önceki günkü dB'ye orantılı (g/m² × 10 → kg/ha)
        N_uptake_kg_ha = self._last_dB * p.N_demand_per_g * 10.0
        # Yıkanma: drenaj suyu oranında kayıp
        leach_frac = drainage_mm / max(1.0, soil_depth_mm * p.field_capacity)
        leach_frac = float(np.clip(leach_frac, 0.0, p.leach_max_frac))
        N_new = N_old + fert_kg_ha - N_uptake_kg_ha - leach_frac * N_old
        N_new = float(max(0.0, N_new))

        # Toprak sıcaklığı (havayı gecikmeyle takip eder)
        T_soil_old = float(self.state[idx["T_soil"]])
        T_soil = p.soil_temp_memory * T_soil_old + p.soil_temp_air_weight * T_in

        return W_new, N_new, T_soil, ET_actual, drainage_mm

    def _plant_step(self, T_in: float, W_new: float, N_new: float
                    ) -> Tuple[float, float, float, float, float, float, float]:
        """Bitki büyümesi. Geri döner: (B_new, LAI_new, DVS_new, dB, f_T, g_W, h_N)."""
        idx = self.IDX
        p = self.p
        B_old   = float(self.state[idx["B"]])
        LAI_old = float(self.state[idx["LAI"]])
        DVS_old = float(self.state[idx["DVS"]])
        I_rad   = float(self.state[idx["I_rad"]])

        # Sıcaklık stresi (trapez)
        if T_in <= p.T_min_growth or T_in >= p.T_max_growth:
            f_T = 0.0
        elif T_in < p.T_opt_low:
            f_T = (T_in - p.T_min_growth) / (p.T_opt_low - p.T_min_growth)
        elif T_in <= p.T_opt_high:
            f_T = 1.0
        else:
            f_T = (p.T_max_growth - T_in) / (p.T_max_growth - p.T_opt_high)

        # Su stresi (lineer)
        g_W = (W_new - p.wilting_point) / max(1e-6, p.field_capacity - p.wilting_point)
        g_W = float(np.clip(g_W, 0.0, 1.0))

        # Azot stresi (Michaelis-Menten benzeri)
        h_N = N_new / (N_new + p.N_half_saturation)

        # Işık yakalama (Beer yasası)
        light_intercept = 1.0 - np.exp(-p.k_extinction * LAI_old)

        # Fotosentetik aktif ışınım
        I_PAR = p.PAR_fraction * I_rad   # MJ/m²/gün

        # Günlük biyokütle artışı (g/m²) + biyolojik varyans (stokastik)
        dB = p.RUE * I_PAR * light_intercept * f_T * g_W * h_N
        dB *= (1.0 + self.np_random.normal(0.0, p.biological_noise))
        dB = float(max(0.0, dB))

        B_new = B_old + dB
        LAI_new = float(min(p.SLA * p.leaf_fraction * B_new, p.LAI_max))

        # DVS — büyüme derece günleri ile
        GDD = max(0.0, T_in - p.T_base_GDD)
        DVS_new = float(min(DVS_old + GDD / p.DVS_thermal_demand, p.DVS_maturity))

        return B_new, LAI_new, DVS_new, dB, f_T, g_W, h_N

    def _compute_reward(self, a_heat: float, a_vent: float,
                        a_irr_norm: float, a_fert_norm: float,
                        dB: float, f_T: float, g_W: float, h_N: float,
                        H_in: float, terminated: bool) -> Tuple[float, Dict[str, float]]:
        p = self.p
        # Anlık ödül parçaları
        r_growth  =  p.w_growth_daily * dB
        c_water   =  p.w_water  * a_irr_norm
        c_energy  =  p.w_energy * a_heat
        c_fert    =  p.w_fert   * a_fert_norm
        # Stres = (1 - f_T) + (1 - g_W) + (1 - h_N): 0 (stres yok) ile 3 (max) arası
        c_stress  =  p.w_stress * (3.0 - f_T - g_W - h_N)
        # Aşırı nem hastalık riski
        c_humid   =  p.w_humidity * max(0.0, H_in - p.humidity_threshold)

        r = r_growth - c_water - c_energy - c_fert - c_stress - c_humid

        # Terminal verim bonusu (ton/ha)
        r_term = 0.0
        if terminated:
            B = float(self.state[self.IDX["B"]])
            yield_kg_ha = B * 10.0 * p.harvest_index   # g/m² × 10 → kg/ha; HI ile çarp
            yield_ton_ha = yield_kg_ha / 1000.0
            r_term = p.w_yield_terminal * yield_ton_ha
            r += r_term

        parts = {
            "r_growth": r_growth, "c_water": c_water, "c_energy": c_energy,
            "c_fert": c_fert, "c_stress": c_stress, "c_humid": c_humid,
            "r_terminal": r_term,
        }
        return float(r), parts

    # ----------------------------------------------------------------------
    # Step
    # ----------------------------------------------------------------------
    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        idx = self.IDX
        a = self._scale_action(action)
        a_heat, a_vent, a_irr, a_fert = a[0], a[1], a[2], a[3]

        # 1) İklim güncellemesi
        T_in, H_in, CO2_in = self._climate_step(a_heat, a_vent)

        # 2) Toprak (yeni T_in'i kullanır)
        W_new, N_new, T_soil, ET, drainage = self._soil_step(a_irr, a_fert, T_in)

        # 3) Bitki (yeni T_in, W, N)
        B_new, LAI_new, DVS_new, dB, f_T, g_W, h_N = self._plant_step(T_in, W_new, N_new)

        # State güncelle
        self.state[idx["T_in"]]   = T_in
        self.state[idx["H_in"]]   = H_in
        self.state[idx["CO2_in"]] = CO2_in
        self.state[idx["W_soil"]] = W_new
        self.state[idx["N_soil"]] = N_new
        self.state[idx["T_soil"]] = T_soil
        self.state[idx["LAI"]]    = LAI_new
        self.state[idx["B"]]      = B_new
        self.state[idx["DVS"]]    = DVS_new

        # Sonraki günün dış koşulları
        self.t += 1
        terminated = (self.t >= self.p.episode_days) or (DVS_new >= self.p.DVS_maturity)
        truncated = False

        if not terminated:
            self.state[idx["T_out"]]    = float(self.weather["T_out"][self.t])
            self.state[idx["H_out"]]    = float(self.weather["H_out"][self.t])
            self.state[idx["I_rad"]]    = float(self.weather["I_rad"][self.t])
            self.state[idx["day_norm"]] = self.t / self.p.episode_days

        self._last_dB = dB

        reward, parts = self._compute_reward(
            a_heat, a_vent, a_irr, a_fert,
            dB, f_T, g_W, h_N, H_in, terminated
        )

        info = self._get_info()
        info.update({
            "ET_mm": ET, "drainage_mm": drainage,
            "f_T": f_T, "g_W": g_W, "h_N": h_N,
            "dB": dB, "reward_parts": parts,
            "action_scaled": {"heat": a_heat, "vent": a_vent,
                              "irr": a_irr, "fert": a_fert},
        })
        return self._get_obs(), reward, terminated, truncated, info

    # ----------------------------------------------------------------------
    # Render (basit konsol)
    # ----------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        s = self.state
        idx = self.IDX
        print(f"Day {self.t:3d} | T_in {s[idx['T_in']]:5.1f}°C | "
              f"W {s[idx['W_soil']]:.2f} | N {s[idx['N_soil']]:5.1f} | "
              f"LAI {s[idx['LAI']]:.2f} | B {s[idx['B']]:6.1f} g/m² | "
              f"DVS {s[idx['DVS']]:.2f}")
