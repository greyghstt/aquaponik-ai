from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import SEED


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    start_hour: int
    duration_hours: int
    intensity: float


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    hour = result["timestamp"].dt.hour.to_numpy()
    day = result["timestamp"].dt.dayofyear.to_numpy()
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_sin"] = np.sin(2 * np.pi * day / 365)
    result["day_cos"] = np.cos(2 * np.pi * day / 365)
    return result


def _smooth_random_walk(rng: np.random.Generator, n: int, scale: float, pull: float) -> np.ndarray:
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = values[i - 1] * (1 - pull) + rng.normal(0, scale)
    return values


def _pulse(n: int, start: int, duration: int, peak: float) -> np.ndarray:
    values = np.zeros(n)
    if start >= n:
        return values
    end = min(n, start + duration)
    x = np.linspace(0, 1, end - start)
    shape = np.sin(np.pi * x)
    values[start:end] = peak * np.maximum(shape, 0)
    return values


def _make_scenarios(n: int, rng: np.random.Generator, training: bool) -> list[SimulationScenario]:
    if not training:
        # Poster window intentionally contains mild, explainable incidents.
        return [
            SimulationScenario("feeding_ammonia", 24, 20, 0.90),
            SimulationScenario("biofilter_nitrite", 64, 26, 0.95),
            SimulationScenario("hot_greenhouse", 104, 30, 1.10),
            SimulationScenario("oxygen_stress", 136, 22, 1.00),
            SimulationScenario("fish_stress_combo", 166, 28, 1.15),
            SimulationScenario("low_nutrient", 204, 44, 1.10),
            SimulationScenario("ph_drift", 250, 34, 1.20),
            SimulationScenario("low_water_level", 292, 36, 1.30),
        ]

    candidates = [
        "feeding_ammonia",
        "biofilter_nitrite",
        "hot_greenhouse",
        "low_nutrient",
        "ph_drift",
        "low_water_level",
        "oxygen_stress",
        "fish_stress_combo",
    ]
    scenarios: list[SimulationScenario] = []
    for idx, start in enumerate(range(36, n - 72, 42)):
        name = candidates[idx % len(candidates)]
        scenarios.append(
            SimulationScenario(
                name=name,
                start_hour=start + int(rng.integers(-8, 10)),
                duration_hours=int(rng.integers(22, 58)),
                intensity=float(rng.uniform(0.80, 1.25)),
            )
        )
    return scenarios


def generate_sensor_data(
    days: int,
    end_timestamp: str = "2026-05-23 12:00:00",
    seed: int = SEED,
    training: bool = False,
) -> pd.DataFrame:
    """Generate aquaponic sensor data with coupled dynamics, not pure random noise."""

    rng = np.random.default_rng(seed + (17 if training else 0))
    periods = days * 24
    timestamps = pd.date_range(end=end_timestamp, periods=periods, freq="h")
    hour = timestamps.hour.to_numpy()
    day_index = np.arange(periods) / 24

    solar_shape = np.maximum(0, np.sin(np.pi * (hour - 6) / 12)) ** 1.55
    cloud_daily = np.repeat(rng.uniform(0.72, 1.03, days), 24)[:periods]
    slow_weather = _smooth_random_walk(rng, periods, scale=0.08, pull=0.018)
    light_noise = rng.normal(0, 450, periods)
    intensitas_cahaya_lux = np.clip(38000 * solar_shape * cloud_daily + light_noise, 0, 43000)
    intensitas_cahaya_lux[solar_shape == 0] = np.clip(rng.normal(45, 25, (solar_shape == 0).sum()), 0, 130)

    suhu_udara_c = (
        23.6
        + 6.0 * solar_shape
        + 1.2 * np.sin(2 * np.pi * day_index / 9)
        + slow_weather
        + rng.normal(0, 0.32, periods)
    )
    kelembapan_pct = (
        83
        - 0.95 * (suhu_udara_c - 24)
        - 0.00012 * intensitas_cahaya_lux
        + 5.0 * (solar_shape == 0)
        + rng.normal(0, 1.3, periods)
    )

    suhu_air_c = np.empty(periods)
    suhu_air_c[0] = 26.2 + rng.normal(0, 0.15)
    for i in range(1, periods):
        target = 25.6 + 0.30 * (suhu_udara_c[i] - 25.5) + 0.55 * solar_shape[i]
        suhu_air_c[i] = suhu_air_c[i - 1] + 0.075 * (target - suhu_air_c[i - 1]) + rng.normal(0, 0.045)

    feed_hours = ((hour == 7) | (hour == 17)).astype(float)
    feed_response = np.zeros(periods)
    for i in range(periods):
        if feed_hours[i]:
            decay_len = min(9, periods - i)
            feed_response[i : i + decay_len] += 0.13 * np.exp(-np.arange(decay_len) / 3.8)

    amonia_mg_l = 0.10 + feed_response + _smooth_random_walk(rng, periods, 0.008, 0.12)
    nitrit_mg_l = 0.055 + np.convolve(amonia_mg_l - 0.10, np.ones(7) / 7, mode="same") * 0.24
    nitrat_mg_l = np.empty(periods)
    nitrat_mg_l[0] = 42 + rng.normal(0, 1.0)
    ec_ms_cm = np.empty(periods)
    ec_ms_cm[0] = 1.42 + rng.normal(0, 0.03)
    level_air_pct = np.empty(periods)
    level_air_pct[0] = 93.5 + rng.normal(0, 0.25)
    ph_air = np.empty(periods)
    ph_air[0] = 6.95 + rng.normal(0, 0.03)

    scenarios = _make_scenarios(periods, rng, training=training)
    scenario_flags = np.array(["normal"] * periods, dtype=object)
    ammonia_event = np.zeros(periods)
    nitrite_event = np.zeros(periods)
    heat_event = np.zeros(periods)
    nutrient_event = np.zeros(periods)
    level_event = np.zeros(periods)
    ph_event = np.zeros(periods)
    oxygen_event = np.zeros(periods)

    for scenario in scenarios:
        p = _pulse(periods, scenario.start_hour, scenario.duration_hours, scenario.intensity)
        active = p > 0
        scenario_flags[active] = scenario.name
        if scenario.name == "feeding_ammonia":
            ammonia_event += 0.78 * p
        elif scenario.name == "biofilter_nitrite":
            nitrite_event += 0.72 * p
            ammonia_event += 0.20 * p
        elif scenario.name == "hot_greenhouse":
            heat_event += 5.10 * p
            oxygen_event += 0.15 * p
        elif scenario.name == "low_nutrient":
            nutrient_event += 1.15 * p
        elif scenario.name == "ph_drift":
            ph_event += 0.95 * p * rng.choice([-1, 1])
        elif scenario.name == "low_water_level":
            level_event += 38.0 * p
        elif scenario.name == "oxygen_stress":
            oxygen_event += 2.70 * p
        elif scenario.name == "fish_stress_combo":
            heat_event += 4.80 * p
            oxygen_event += 2.50 * p
            ammonia_event += 0.38 * p
            nitrite_event += 0.28 * p

    suhu_air_c = suhu_air_c + heat_event
    suhu_udara_c = suhu_udara_c + heat_event * 1.35
    kelembapan_pct = kelembapan_pct - heat_event * 4.3
    amonia_mg_l = amonia_mg_l + ammonia_event
    nitrit_mg_l = nitrit_mg_l + nitrite_event + np.roll(ammonia_event, 8) * 0.28

    for i in range(1, periods):
        plant_uptake = 0.040 + 0.025 * solar_shape[i] + 0.010 * max(ec_ms_cm[i - 1] - 1.15, 0)
        nitrification = 0.026 + 0.18 * nitrit_mg_l[i - 1]
        nitrat_mg_l[i] = nitrat_mg_l[i - 1] + nitrification - plant_uptake + rng.normal(0, 0.16)
        ec_target = 0.95 + nitrat_mg_l[i] / 55
        ec_ms_cm[i] = ec_ms_cm[i - 1] + 0.050 * (ec_target - ec_ms_cm[i - 1]) + rng.normal(0, 0.010)
        evap = 0.026 + 0.030 * solar_shape[i] + 0.012 * max(suhu_udara_c[i] - 28, 0)
        refill = 0.0
        if hour[i] == 6 and (i // 24) % 3 == 0:
            refill = 6.5
        if hour[i] == 6 and level_air_pct[i - 1] < 72:
            refill = max(refill, 12.0)
        level_air_pct[i] = level_air_pct[i - 1] - evap + refill + rng.normal(0, 0.035)
        ph_target = 6.95 - 0.010 * (nitrat_mg_l[i] - 40) + 0.012 * np.sin(2 * np.pi * i / (24 * 5))
        ph_air[i] = ph_air[i - 1] + 0.035 * (ph_target - ph_air[i - 1]) + rng.normal(0, 0.010)

    nitrat_mg_l = nitrat_mg_l - 32 * nutrient_event
    ec_ms_cm = ec_ms_cm - 0.72 * nutrient_event
    level_air_pct = level_air_pct - level_event
    ph_air = ph_air + ph_event

    do_baseline = (
        7.65
        - 0.18 * (suhu_air_c - 26)
        - 0.000010 * intensitas_cahaya_lux
        - 0.42 * (amonia_mg_l > 0.45)
        - 0.16 * (nitrit_mg_l > 0.25)
        - oxygen_event
        + rng.normal(0, 0.08, periods)
    )
    do_mg_l = np.clip(do_baseline, 2.6, 8.4)

    # Use a 500-scale conversion so EC 1.0-1.8 mS/cm maps to 500-900 ppm.
    tds_ppm = ec_ms_cm * 500 + rng.normal(0, 8, periods)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "scenario": scenario_flags,
            "ph_air": np.clip(ph_air, 5.85, 8.18),
            "suhu_air_c": np.clip(suhu_air_c, 23.0, 33.8),
            "do_mg_l": do_mg_l,
            "amonia_mg_l": np.clip(amonia_mg_l + rng.normal(0, 0.010, periods), 0.02, 1.45),
            "nitrit_mg_l": np.clip(nitrit_mg_l + rng.normal(0, 0.008, periods), 0.01, 1.05),
            "nitrat_mg_l": np.clip(nitrat_mg_l + rng.normal(0, 0.32, periods), 5, 85),
            "ec_ms_cm": np.clip(ec_ms_cm + rng.normal(0, 0.012, periods), 0.62, 2.25),
            "tds_ppm": np.clip(tds_ppm, 390, 1450),
            "level_air_pct": np.clip(level_air_pct, 47, 99),
            "suhu_udara_c": np.clip(suhu_udara_c, 20.5, 39.0),
            "kelembapan_pct": np.clip(kelembapan_pct, 42, 96),
            "intensitas_cahaya_lux": np.round(intensitas_cahaya_lux, 0),
        }
    )
    return build_time_features(df).round(
        {
            "ph_air": 2,
            "suhu_air_c": 2,
            "do_mg_l": 2,
            "amonia_mg_l": 3,
            "nitrit_mg_l": 3,
            "nitrat_mg_l": 2,
            "ec_ms_cm": 3,
            "tds_ppm": 1,
            "level_air_pct": 2,
            "suhu_udara_c": 2,
            "kelembapan_pct": 2,
        }
    )
