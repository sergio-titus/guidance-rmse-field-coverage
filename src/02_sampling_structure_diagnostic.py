from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# STAGE 0C — SAMPLING STRUCTURE DIAGNOSTIC
# ============================================================
#
# Purpose:
#   Determine the actual structure of the released guidance
#   arrays BEFORE computing temporal/spatial autocorrelation.
#
# This script does NOT:
#   - sort the trajectories,
#   - interpolate them,
#   - resample them,
#   - fit any stochastic model,
#   - remove trends,
#   - modify the original observations.
#
# It examines:
#   - sample counts
#   - coordinate ranges
#   - consecutive Easting increments in acquisition order
#   - number of unique Easting values
#   - duplicate coordinates
#   - monotonicity
#   - implied duration if 5 Hz were applied directly
#   - implied mean speed under that interpretation
#
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

MAT_FILE = (
    PROJECT_DIR
    / "data"
    / "source"
    / "2_Guidance_tests"
    / "2A_Guidance_tests_data.mat"
)

RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Published experimental values
# ------------------------------------------------------------

PUBLISHED_FS_HZ = 5.0
PUBLISHED_SPEED_MPS = 1.0


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------

if not MAT_FILE.exists():
    raise FileNotFoundError(
        f"MAT file not found:\n{MAT_FILE}"
    )

print("=" * 90)
print("STAGE 0C — GUIDANCE DATA SAMPLING STRUCTURE DIAGNOSTIC")
print("=" * 90)

raw = loadmat(
    MAT_FILE,
    squeeze_me=True,
    struct_as_record=False
)

data = {
    key: value
    for key, value in raw.items()
    if not key.startswith("__")
}


# ------------------------------------------------------------
# 4. Parse variable names
# ------------------------------------------------------------

pattern = re.compile(
    r"Guidance_Conf_(C\d{2})_(.+)_Pass_(\d)_(Easting|Northing)$"
)

records = {}

for variable_name, values in data.items():

    match = pattern.match(variable_name)

    if not match:
        continue

    config = match.group(1)
    name = match.group(2)
    pass_number = int(match.group(3))
    coordinate = match.group(4)

    key = (config, name, pass_number)

    if key not in records:
        records[key] = {}

    records[key][coordinate] = np.asarray(
        values,
        dtype=float
    ).ravel()


# ------------------------------------------------------------
# 5. Validate
# ------------------------------------------------------------

if len(records) != 42:
    raise RuntimeError(
        f"Expected 42 configuration-pass combinations, "
        f"found {len(records)}."
    )

for key, coordinates in records.items():

    if "Easting" not in coordinates:
        raise RuntimeError(
            f"Missing Easting for {key}"
        )

    if "Northing" not in coordinates:
        raise RuntimeError(
            f"Missing Northing for {key}"
        )

    if (
        len(coordinates["Easting"])
        != len(coordinates["Northing"])
    ):
        raise RuntimeError(
            f"Coordinate length mismatch for {key}"
        )


# ------------------------------------------------------------
# 6. Utility
# ------------------------------------------------------------

def percentile_or_nan(values, q):

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return np.percentile(values, q)


# ------------------------------------------------------------
# 7. Diagnose each pass in ORIGINAL ARRAY ORDER
# ------------------------------------------------------------

rows = []

for key in sorted(records.keys()):

    config, name, pass_number = key

    east = records[key]["Easting"]
    north = records[key]["Northing"]

    finite_mask = (
        np.isfinite(east)
        & np.isfinite(north)
    )

    east = east[finite_mask]
    north = north[finite_mask]

    n = len(east)

    # Consecutive increments in ORIGINAL order.
    de = np.diff(east)
    dn = np.diff(north)

    step_2d = np.sqrt(
        de ** 2
        + dn ** 2
    )

    # Easting characteristics
    positive_de = de[de > 0]
    negative_de = de[de < 0]
    zero_de = de[de == 0]

    # Unique coordinate counts
    unique_east_count = len(np.unique(east))

    unique_pairs_count = len(
        np.unique(
            np.column_stack((east, north)),
            axis=0
        )
    )

    duplicate_east_fraction = (
        1.0
        - unique_east_count / n
        if n > 0
        else np.nan
    )

    duplicate_pair_fraction = (
        1.0
        - unique_pairs_count / n
        if n > 0
        else np.nan
    )

    # Monotonicity
    fraction_de_positive = (
        np.mean(de > 0)
        if len(de) > 0
        else np.nan
    )

    fraction_de_negative = (
        np.mean(de < 0)
        if len(de) > 0
        else np.nan
    )

    fraction_de_zero = (
        np.mean(de == 0)
        if len(de) > 0
        else np.nan
    )

    monotonic_non_decreasing = bool(
        np.all(de >= 0)
    )

    monotonic_strictly_increasing = bool(
        np.all(de > 0)
    )

    # Coordinate ranges
    east_range = (
        np.max(east) - np.min(east)
    )

    north_range = (
        np.max(north) - np.min(north)
    )

    # Path length from consecutive 2D coordinates
    path_length_2d = np.sum(step_2d)

    # If every stored row were literally a 5 Hz observation,
    # what duration and speed would that imply?
    implied_duration_s = (
        (n - 1) / PUBLISHED_FS_HZ
        if n > 1
        else np.nan
    )

    implied_duration_min = (
        implied_duration_s / 60.0
        if np.isfinite(implied_duration_s)
        else np.nan
    )

    implied_easting_speed = (
        east_range / implied_duration_s
        if implied_duration_s > 0
        else np.nan
    )

    implied_path_speed = (
        path_length_2d / implied_duration_s
        if implied_duration_s > 0
        else np.nan
    )

    # Expected sample count if this were one 300 m pass
    # at 1 m/s sampled at 5 Hz, using the ACTUAL Easting range.
    expected_duration_from_range_s = (
        east_range / PUBLISHED_SPEED_MPS
    )

    expected_n_from_range = (
        expected_duration_from_range_s
        * PUBLISHED_FS_HZ
    )

    sample_count_ratio = (
        n / expected_n_from_range
        if expected_n_from_range > 0
        else np.nan
    )

    rows.append({
        "Configuration": config,
        "Configuration_Name": name,
        "Pass": pass_number,

        "N_samples": n,

        "Easting_min_m": np.min(east),
        "Easting_max_m": np.max(east),
        "Easting_range_m": east_range,

        "Northing_min_m": np.min(north),
        "Northing_max_m": np.max(north),
        "Northing_range_m": north_range,

        "Unique_Easting_values": unique_east_count,
        "Unique_coordinate_pairs": unique_pairs_count,

        "Duplicate_Easting_fraction":
            duplicate_east_fraction,

        "Duplicate_coordinate_pair_fraction":
            duplicate_pair_fraction,

        "Fraction_dE_positive":
            fraction_de_positive,

        "Fraction_dE_negative":
            fraction_de_negative,

        "Fraction_dE_zero":
            fraction_de_zero,

        "Monotonic_non_decreasing":
            monotonic_non_decreasing,

        "Monotonic_strictly_increasing":
            monotonic_strictly_increasing,

        "dE_mean_m":
            np.mean(de),

        "dE_median_m":
            np.median(de),

        "dE_abs_median_m":
            np.median(np.abs(de)),

        "dE_abs_P05_m":
            percentile_or_nan(
                np.abs(de),
                5
            ),

        "dE_abs_P95_m":
            percentile_or_nan(
                np.abs(de),
                95
            ),

        "Positive_dE_median_m":
            np.median(positive_de)
            if len(positive_de) > 0
            else np.nan,

        "Negative_dE_median_m":
            np.median(negative_de)
            if len(negative_de) > 0
            else np.nan,

        "Step2D_median_m":
            np.median(step_2d),

        "Step2D_P95_m":
            percentile_or_nan(
                step_2d,
                95
            ),

        "Path_length_from_rows_m":
            path_length_2d,

        "Implied_duration_if_5Hz_s":
            implied_duration_s,

        "Implied_duration_if_5Hz_min":
            implied_duration_min,

        "Implied_speed_from_Easting_range_mps":
            implied_easting_speed,

        "Implied_speed_from_2D_path_mps":
            implied_path_speed,

        "Expected_samples_from_range_at_1mps_5Hz":
            expected_n_from_range,

        "Observed_to_expected_sample_ratio":
            sample_count_ratio
    })


# ------------------------------------------------------------
# 8. DataFrame
# ------------------------------------------------------------

diagnostic = pd.DataFrame(rows)


# ------------------------------------------------------------
# 9. Save complete table
# ------------------------------------------------------------

output_file = (
    TABLES_DIR
    / "guidance_sampling_structure_diagnostic.csv"
)

diagnostic.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# 10. Compact terminal table
# ------------------------------------------------------------

display_columns = [
    "Configuration",
    "Pass",
    "N_samples",
    "Easting_range_m",
    "Unique_Easting_values",
    "Fraction_dE_positive",
    "Fraction_dE_negative",
    "Fraction_dE_zero",
    "dE_median_m",
    "dE_abs_median_m",
    "Path_length_from_rows_m",
    "Implied_duration_if_5Hz_min",
    "Implied_speed_from_Easting_range_mps",
    "Observed_to_expected_sample_ratio"
]

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    240
)

pd.set_option(
    "display.max_rows",
    100
)

print("\n" + "=" * 90)
print("PER-PASS SAMPLING DIAGNOSTIC")
print("=" * 90)

print(
    diagnostic[
        display_columns
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 11. Dataset-wide summary
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("DATASET-WIDE SUMMARY")
print("=" * 90)

print(
    f"\nSample count per pass:"
    f"\n  min    = {diagnostic['N_samples'].min():.0f}"
    f"\n  median = {diagnostic['N_samples'].median():.0f}"
    f"\n  max    = {diagnostic['N_samples'].max():.0f}"
)

print(
    f"\nEasting range per pass:"
    f"\n  min    = {diagnostic['Easting_range_m'].min():.3f} m"
    f"\n  median = {diagnostic['Easting_range_m'].median():.3f} m"
    f"\n  max    = {diagnostic['Easting_range_m'].max():.3f} m"
)

print(
    f"\nMedian absolute consecutive Easting step:"
    f"\n  min    = {diagnostic['dE_abs_median_m'].min():.6f} m"
    f"\n  median = {diagnostic['dE_abs_median_m'].median():.6f} m"
    f"\n  max    = {diagnostic['dE_abs_median_m'].max():.6f} m"
)

print(
    f"\nObserved / expected sample-count ratio "
    f"(based on Easting range, 1 m/s, 5 Hz):"
    f"\n  min    = "
    f"{diagnostic['Observed_to_expected_sample_ratio'].min():.2f}"
    f"\n  median = "
    f"{diagnostic['Observed_to_expected_sample_ratio'].median():.2f}"
    f"\n  max    = "
    f"{diagnostic['Observed_to_expected_sample_ratio'].max():.2f}"
)

print(
    f"\nStrictly increasing Easting passes: "
    f"{diagnostic['Monotonic_strictly_increasing'].sum()} / "
    f"{len(diagnostic)}"
)

print(
    f"Non-decreasing Easting passes: "
    f"{diagnostic['Monotonic_non_decreasing'].sum()} / "
    f"{len(diagnostic)}"
)


# ------------------------------------------------------------
# 12. Example raw observations
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("FIRST 20 RAW OBSERVATIONS — C01 PASS 1")
print("=" * 90)

example_key = [
    key
    for key in sorted(records.keys())
    if key[0] == "C01"
    and key[2] == 1
][0]

example_e = records[
    example_key
]["Easting"]

example_n = records[
    example_key
]["Northing"]

example_table = pd.DataFrame({
    "Row": np.arange(
        1,
        min(21, len(example_e) + 1)
    ),
    "Easting_m": example_e[:20],
    "Northing_m": example_n[:20]
})

print(
    example_table.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 13. End
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)

print(
    f"\nFull diagnostic saved to:\n{output_file}"
)

print(
    "\nIMPORTANT:"
    "\nNo ACF, PSD, interpolation, resampling, or stochastic "
    "model should be interpreted until this sampling structure "
    "has been resolved."
)