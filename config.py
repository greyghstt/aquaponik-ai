from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODEL_PATH = OUTPUT_DIR / "rf_model.joblib"

SEED = 42
TRAINING_DAYS = 90
POSTER_DAYS = 14

SENSOR_FEATURES = [
    "ph_air",
    "suhu_air_c",
    "do_mg_l",
    "amonia_mg_l",
    "nitrit_mg_l",
    "nitrat_mg_l",
    "ec_ms_cm",
    "tds_ppm",
    "level_air_pct",
    "suhu_udara_c",
    "kelembapan_pct",
    "intensitas_cahaya_lux",
]

TIME_FEATURES = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
MODEL_FEATURES = SENSOR_FEATURES + TIME_FEATURES

LABEL_ORDER = [
    "Normal",
    "DO rendah",
    "Amonia tinggi",
    "Nitrit tinggi",
    "Nutrisi rendah",
    "pH tidak stabil",
    "Level air rendah",
    "Suhu air tinggi",
    "Risiko stres ikan",
]


@dataclass(frozen=True)
class DomainRanges:
    """Domain thresholds used for realistic simulation and hybrid labels."""

    ph_min_ok: float = 6.5
    ph_max_ok: float = 7.5
    ph_low_warning: float = 6.4
    ph_high_warning: float = 7.6
    ph_low_critical: float = 6.2
    ph_high_critical: float = 7.8
    water_temp_warning_c: float = 30.0
    water_temp_critical_c: float = 32.0
    do_warning_mg_l: float = 5.0
    do_critical_mg_l: float = 4.0
    ammonia_warning_mg_l: float = 0.50
    ammonia_critical_mg_l: float = 1.00
    nitrite_warning_mg_l: float = 0.50
    nitrite_critical_mg_l: float = 1.00
    nitrate_warning_mg_l: float = 20.0
    nitrate_critical_mg_l: float = 10.0
    ec_low_warning_ms_cm: float = 1.00
    ec_high_warning_ms_cm: float = 1.80
    ec_low_critical_ms_cm: float = 0.80
    ec_high_critical_ms_cm: float = 2.20
    tds_low_warning_ppm: float = 500.0
    tds_high_warning_ppm: float = 900.0
    tds_low_critical_ppm: float = 350.0
    tds_high_critical_ppm: float = 1100.0
    level_warning_pct: float = 80.0
    level_critical_pct: float = 60.0
    air_temp_warning_c: float = 32.0
    air_temp_critical_c: float = 35.0
    humidity_low_warning_pct: float = 50.0
    humidity_high_warning_pct: float = 80.0
    humidity_low_critical_pct: float = 40.0
    humidity_high_critical_pct: float = 90.0


RANGES = DomainRanges()

REFERENCE_NOTES = [
    "OSU Extension: aquaponic pH 6.5-7.5 should be maintained; fish pH target near neutral.",
    "OSU Extension: tilapia optimal water temperature listed around 74-80 F.",
    "UF/IFAS: dissolved oxygen is a primary fish-production water quality parameter.",
    "UF/IFAS: ammonia is a key intensive-system fish risk after oxygen; low oxygen can disrupt nitrification.",
    "FAO small-scale aquaponics: oxygen, pH, ammonia, nitrite, nitrate, and temperature are core monitoring variables.",
]

REFERENCE_LINKS = [
    "https://extension.okstate.edu/fact-sheets/principles-of-small-scale-aquaponics",
    "https://extension.okstate.edu/fact-sheets/aquaponics",
    "https://ask.ifas.ufl.edu/FA002",
    "https://edis.ifas.ufl.edu/publication/FA031",
    "https://www.fao.org/family-farming/detail/en/c/1743021/",
]
