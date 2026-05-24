from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import SEED, SIMULATION_INTERVAL_MINUTES


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    start_step: int
    duration_steps: int
    intensity: float


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    hour = (result["timestamp"].dt.hour + result["timestamp"].dt.minute / 60).to_numpy()
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


def _hours_to_steps(hours: float, steps_per_hour: int) -> int:
    return int(round(hours * steps_per_hour))


def _step_pull(hourly_pull: float, steps_per_hour: int) -> float:
    return 1 - (1 - hourly_pull) ** (1 / steps_per_hour)


def _make_scenarios(total_hours: int, steps_per_hour: int, rng: np.random.Generator, training: bool) -> list[SimulationScenario]:
    if not training:
        # Dashboard window intentionally contains mild, explainable incidents.
        return [
            SimulationScenario("feeding_ammonia", _hours_to_steps(24, steps_per_hour), _hours_to_steps(20, steps_per_hour), 0.90),
            SimulationScenario("biofilter_nitrite", _hours_to_steps(64, steps_per_hour), _hours_to_steps(26, steps_per_hour), 0.95),
            SimulationScenario("hot_greenhouse", _hours_to_steps(104, steps_per_hour), _hours_to_steps(30, steps_per_hour), 1.10),
            SimulationScenario("oxygen_stress", _hours_to_steps(136, steps_per_hour), _hours_to_steps(22, steps_per_hour), 1.00),
            SimulationScenario("fish_stress_combo", _hours_to_steps(166, steps_per_hour), _hours_to_steps(28, steps_per_hour), 1.15),
            SimulationScenario("low_nutrient", _hours_to_steps(230, steps_per_hour), _hours_to_steps(48, steps_per_hour), 1.10),
            SimulationScenario("ph_drift", _hours_to_steps(300, steps_per_hour), _hours_to_steps(36, steps_per_hour), 1.20),
            SimulationScenario("low_water_level", _hours_to_steps(392, steps_per_hour), _hours_to_steps(42, steps_per_hour), 1.30),
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
    for idx, start_hour in enumerate(range(36, total_hours - 72, 42)):
        name = candidates[idx % len(candidates)]
        start_hour = start_hour + int(rng.integers(-8, 10))
        duration_hours = int(rng.integers(22, 58))
        scenarios.append(
            SimulationScenario(
                name=name,
                start_step=_hours_to_steps(start_hour, steps_per_hour),
                duration_steps=_hours_to_steps(duration_hours, steps_per_hour),
                intensity=float(rng.uniform(0.80, 1.25)),
            )
        )
    return scenarios


def generate_sensor_data(
    days: int,
    end_timestamp: str = "2026-05-23 12:00:00",
    seed: int = SEED,
    training: bool = False,
    interval_minutes: int = SIMULATION_INTERVAL_MINUTES,
) -> pd.DataFrame:
    """Generate aquaponic sensor data with coupled dynamics, not pure random noise."""

    rng = np.random.default_rng(seed + (17 if training else 0))
    steps_per_hour = 60 // interval_minutes
    total_hours = days * 24
    periods = total_hours * steps_per_hour
    timestamps = pd.date_range(end=end_timestamp, periods=periods, freq=f"{interval_minutes}min")
    hour = timestamps.hour.to_numpy()
    minute = timestamps.minute.to_numpy()
    hour_float = hour + minute / 60
    day_index = np.arange(periods) / (24 * steps_per_hour)
    dt_hours = interval_minutes / 60

    solar_shape = np.maximum(0, np.sin(np.pi * (hour_float - 6) / 12)) ** 1.55
    cloud_daily = np.repeat(rng.uniform(0.72, 1.03, days), 24 * steps_per_hour)[:periods]
    slow_weather = _smooth_random_walk(
        rng,
        periods,
        scale=0.08 * np.sqrt(dt_hours),
        pull=_step_pull(0.018, steps_per_hour),
    )
    light_noise = rng.normal(0, 280, periods)
    intensitas_cahaya_lux = np.clip(38000 * solar_shape * cloud_daily + light_noise, 0, 43000)
    intensitas_cahaya_lux[solar_shape == 0] = np.clip(rng.normal(45, 25, (solar_shape == 0).sum()), 0, 130)

    suhu_udara_c = (
        23.6
        + 6.0 * solar_shape
        + 1.2 * np.sin(2 * np.pi * day_index / 9)
        + slow_weather
        + rng.normal(0, 0.16, periods)
    )
    kelembapan_pct = (
        83
        - 0.95 * (suhu_udara_c - 24)
        - 0.00012 * intensitas_cahaya_lux
        + 5.0 * (solar_shape == 0)
        + rng.normal(0, 0.65, periods)
    )

    suhu_air_c = np.empty(periods)
    suhu_air_c[0] = 26.2 + rng.normal(0, 0.15)
    water_temp_pull = _step_pull(0.075, steps_per_hour)
    for i in range(1, periods):
        target = 25.6 + 0.30 * (suhu_udara_c[i] - 25.5) + 0.55 * solar_shape[i]
        suhu_air_c[i] = suhu_air_c[i - 1] + water_temp_pull * (target - suhu_air_c[i - 1]) + rng.normal(0, 0.014)

    feed_hours = (((hour == 7) | (hour == 17)) & (minute == 0)).astype(float)
    feed_response = np.zeros(periods)
    for i in range(periods):
        if feed_hours[i]:
            decay_len = min(_hours_to_steps(9, steps_per_hour), periods - i)
            response_hours = np.arange(decay_len) / steps_per_hour
            feed_response[i : i + decay_len] += 0.13 * np.exp(-response_hours / 3.8)

    amonia_mg_l = 0.10 + feed_response + _smooth_random_walk(
        rng,
        periods,
        0.008 * np.sqrt(dt_hours),
        _step_pull(0.12, steps_per_hour),
    )
    nitrite_window = _hours_to_steps(7, steps_per_hour)
    nitrit_mg_l = 0.055 + np.convolve(amonia_mg_l - 0.10, np.ones(nitrite_window) / nitrite_window, mode="same") * 0.24
    nitrat_mg_l = np.empty(periods)
    nitrat_mg_l[0] = 42 + rng.normal(0, 1.0)
    ec_ms_cm = np.empty(periods)
    ec_ms_cm[0] = 1.42 + rng.normal(0, 0.03)
    level_air_pct = np.empty(periods)
    level_air_pct[0] = 93.5 + rng.normal(0, 0.25)
    ph_air = np.empty(periods)
    ph_air[0] = 6.95 + rng.normal(0, 0.03)

    scenarios = _make_scenarios(total_hours, steps_per_hour, rng, training=training)
    scenario_flags = np.array(["normal"] * periods, dtype=object)
    ammonia_event = np.zeros(periods)
    nitrite_event = np.zeros(periods)
    heat_event = np.zeros(periods)
    nutrient_event = np.zeros(periods)
    level_event = np.zeros(periods)
    ph_event = np.zeros(periods)
    oxygen_event = np.zeros(periods)

    for scenario in scenarios:
        p = _pulse(periods, scenario.start_step, scenario.duration_steps, scenario.intensity)
        active = p > 0
        scenario_flags[active] = scenario.name
        if scenario.name == "feeding_ammonia":
            ammonia_event += 0.78 * p
        elif scenario.name == "biofilter_nitrite":
            nitrite_event += 0.92 * p
            ammonia_event += 0.08 * p
        elif scenario.name == "hot_greenhouse":
            heat_event += 5.90 * p
            oxygen_event += 0.15 * p
        elif scenario.name == "low_nutrient":
            nutrient_event += 1.15 * p
        elif scenario.name == "ph_drift":
            ph_event += 0.95 * p * rng.choice([-1, 1])
        elif scenario.name == "low_water_level":
            level_event += 50.0 * p
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
    nitrit_mg_l = nitrit_mg_l + nitrite_event + np.roll(ammonia_event, _hours_to_steps(8, steps_per_hour)) * 0.28

    ec_pull = _step_pull(0.050, steps_per_hour)
    ph_pull = _step_pull(0.035, steps_per_hour)
    for i in range(1, periods):
        plant_uptake = (0.040 + 0.025 * solar_shape[i] + 0.010 * max(ec_ms_cm[i - 1] - 1.15, 0)) * dt_hours
        nitrification = (0.026 + 0.18 * nitrit_mg_l[i - 1]) * dt_hours
        nitrat_mg_l[i] = nitrat_mg_l[i - 1] + nitrification - plant_uptake + rng.normal(0, 0.045)
        ec_target = 0.95 + nitrat_mg_l[i] / 55
        ec_ms_cm[i] = ec_ms_cm[i - 1] + ec_pull * (ec_target - ec_ms_cm[i - 1]) + rng.normal(0, 0.0035)
        evap = (0.026 + 0.030 * solar_shape[i] + 0.012 * max(suhu_udara_c[i] - 28, 0)) * dt_hours
        refill = 0.0
        if hour[i] == 6 and minute[i] == 0 and (i // (24 * steps_per_hour)) % 3 == 0:
            refill = 6.5
        if hour[i] == 6 and minute[i] == 0 and level_air_pct[i - 1] < 72:
            refill = max(refill, 12.0)
        level_air_pct[i] = level_air_pct[i - 1] - evap + refill + rng.normal(0, 0.010)
        ph_target = 6.95 - 0.010 * (nitrat_mg_l[i] - 40) + 0.012 * np.sin(2 * np.pi * i / (24 * steps_per_hour * 5))
        ph_air[i] = ph_air[i - 1] + ph_pull * (ph_target - ph_air[i - 1]) + rng.normal(0, 0.0035)

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
        + rng.normal(0, 0.035, periods)
    )
    do_mg_l = np.clip(do_baseline, 2.6, 8.4)

    # Use a 500-scale conversion so EC 1.0-1.8 mS/cm maps to 500-900 ppm.
    tds_ppm = ec_ms_cm * 500 + rng.normal(0, 4, periods)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "scenario": scenario_flags,
            "ph_air": np.clip(ph_air, 5.85, 8.18),
            "suhu_air_c": np.clip(suhu_air_c, 23.0, 33.8),
            "do_mg_l": do_mg_l,
            "amonia_mg_l": np.clip(amonia_mg_l + rng.normal(0, 0.004, periods), 0.02, 1.45),
            "nitrit_mg_l": np.clip(nitrit_mg_l + rng.normal(0, 0.0035, periods), 0.01, 1.05),
            "nitrat_mg_l": np.clip(nitrat_mg_l + rng.normal(0, 0.10, periods), 5, 85),
            "ec_ms_cm": np.clip(ec_ms_cm + rng.normal(0, 0.004, periods), 0.62, 2.25),
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
