from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# STAGE 0D — SPATIAL REPRESENTATION OF GUIDANCE TRAJECTORIES
# ============================================================
#
# Goal
# ----
# Convert the released guidance trajectories into a defensible
# spatial representation e(x), where:
#
#       x    = along-track Easting position
#       e(x) = signed lateral deviation from relative reference
#
# Scientific rationale
# --------------------
# The released arrays contain many repeated / near-repeated
# positions and cannot be interpreted directly as a regular
# 5 Hz time series.
#
# The experiment, however, reports:
#
#       tractor speed = 1 m/s
#       GNSS rate     = 5 Hz
#
# Therefore the nominal physical spacing between observations is:
#
#       dx = 1 / 5 = 0.2 m
#
# This script:
#
#   1. Preserves original raw observations.
#   2. Collapses EXACT duplicate Easting coordinates using the
#      median Northing value.
#   3. Quantifies ambiguity among duplicate positions.
#   4. Computes spatially weighted error metrics using numerical
#      integration, independent of storage density.
#   5. Interpolates trajectories onto uniform spatial grids:
#
#         0.05 m
#         0.10 m
#         0.20 m  <-- nominal experimental spacing
#         0.40 m
#
#   6. Quantifies metric convergence across resolutions.
#   7. Saves a canonical 0.20 m trajectory dataset for the next
#      ACF / PSD / stochastic-model stage.
#
# IMPORTANT
# ---------
# No stochastic model is fitted here.
# No AR coefficient is assumed.
# No temporal autocorrelation is calculated here.
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
FIGURES_DIR = RESULTS_DIR / "figures"
SPATIAL_FIGURES_DIR = FIGURES_DIR / "spatial_representation"

for directory in [
    RESULTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    SPATIAL_FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Experimental settings from source study
# ------------------------------------------------------------

TRACTOR_SPEED_MPS = 1.0
GNSS_RATE_HZ = 5.0

NOMINAL_DX_M = TRACTOR_SPEED_MPS / GNSS_RATE_HZ

# Authors' displayed straight-line evaluation region.
X_MIN = 25.0
X_MAX = 300.0

# Resolution sensitivity around nominal 0.2 m.
#
# These are powers-of-two fractions/multiples of the
# experiment-derived nominal spacing:
#
#   0.05 = 0.2 / 4
#   0.10 = 0.2 / 2
#   0.20 = nominal
#   0.40 = 0.2 * 2
#
GRID_SPACINGS_M = [
    NOMINAL_DX_M / 4.0,
    NOMINAL_DX_M / 2.0,
    NOMINAL_DX_M,
    NOMINAL_DX_M * 2.0,
]

CANONICAL_DX_M = NOMINAL_DX_M


# ------------------------------------------------------------
# 3. Utility functions
# ------------------------------------------------------------

def rms(values):
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(values ** 2))


def spatial_integral(y, x):
    """
    Compatibility wrapper for NumPy versions with/without
    np.trapezoid.
    """

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)

    return np.trapz(y, x)


def spatial_weighted_metrics(x, e):
    """
    Compute geometry-weighted metrics from irregular spatial
    samples using trapezoidal numerical integration.

    Unlike an ordinary sample mean, these statistics do not give
    extra weight to locations simply because they appear more
    often in the stored arrays.
    """

    x = np.asarray(x, dtype=float)
    e = np.asarray(e, dtype=float)

    if len(x) < 2:
        return {
            "Spatial_Mean_m": np.nan,
            "Spatial_MAE_m": np.nan,
            "Spatial_RMS_m": np.nan,
        }

    length = x[-1] - x[0]

    if length <= 0:
        return {
            "Spatial_Mean_m": np.nan,
            "Spatial_MAE_m": np.nan,
            "Spatial_RMS_m": np.nan,
        }

    mean_e = spatial_integral(e, x) / length

    mae = (
        spatial_integral(
            np.abs(e),
            x
        )
        / length
    )

    mean_square = (
        spatial_integral(
            e ** 2,
            x
        )
        / length
    )

    spatial_rms = np.sqrt(mean_square)

    return {
        "Spatial_Mean_m": mean_e,
        "Spatial_MAE_m": mae,
        "Spatial_RMS_m": spatial_rms,
    }


def regular_grid(x_min, x_max, dx):
    """
    Construct a common aligned grid without extrapolation.
    """

    start = np.ceil(x_min / dx) * dx
    stop = np.floor(x_max / dx) * dx

    if stop < start:
        return np.array([])

    number = int(
        np.floor(
            (stop - start) / dx
        )
    ) + 1

    return (
        start
        + np.arange(number) * dx
    )


def percentile_or_nan(values, q):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return np.percentile(values, q)


# ------------------------------------------------------------
# 4. Load MAT file
# ------------------------------------------------------------

if not MAT_FILE.exists():
    raise FileNotFoundError(
        f"MAT file not found:\n{MAT_FILE}"
    )

print("=" * 90)
print("STAGE 0D — SPATIAL REPRESENTATION")
print("=" * 90)

print(f"\nLoading:\n{MAT_FILE}")

raw_mat = loadmat(
    MAT_FILE,
    squeeze_me=True,
    struct_as_record=False
)

mat_variables = {
    key: value
    for key, value in raw_mat.items()
    if not key.startswith("__")
}


# ------------------------------------------------------------
# 5. Parse variables
# ------------------------------------------------------------

pattern = re.compile(
    r"Guidance_Conf_(C\d{2})_(.+)_Pass_(\d)_(Easting|Northing)$"
)

records = {}

for variable_name, values in mat_variables.items():

    match = pattern.match(variable_name)

    if not match:
        continue

    config_code = match.group(1)
    config_name = match.group(2)
    pass_number = int(match.group(3))
    coordinate = match.group(4)

    key = (
        config_code,
        config_name,
        pass_number
    )

    if key not in records:
        records[key] = {}

    records[key][coordinate] = np.asarray(
        values,
        dtype=float
    ).ravel()


# ------------------------------------------------------------
# 6. Validate
# ------------------------------------------------------------

if len(records) != 42:
    raise RuntimeError(
        f"Expected 42 configuration-pass records; "
        f"found {len(records)}."
    )

for key, coordinates in records.items():

    if (
        "Easting" not in coordinates
        or "Northing" not in coordinates
    ):
        raise RuntimeError(
            f"Incomplete coordinate pair for {key}"
        )

    if (
        len(coordinates["Easting"])
        != len(coordinates["Northing"])
    ):
        raise RuntimeError(
            f"Length mismatch for {key}"
        )


print(
    f"Validated {len(records)} configuration-pass records."
)

print(
    f"Nominal experiment-derived spatial spacing: "
    f"{NOMINAL_DX_M:.3f} m"
)


# ------------------------------------------------------------
# 7. Output containers
# ------------------------------------------------------------

duplicate_diagnostics_rows = []
spatial_metric_rows = []
resolution_rows = []
canonical_rows = []

collapsed_passes = {}
canonical_passes = {}


# ------------------------------------------------------------
# 8. Process each trajectory
# ------------------------------------------------------------

for key in sorted(records.keys()):

    config_code, config_name, pass_number = key

    east = np.asarray(
        records[key]["Easting"],
        dtype=float
    )

    north = np.asarray(
        records[key]["Northing"],
        dtype=float
    )

    # --------------------------------------------------------
    # 8.1 Restrict to valid study interval
    # --------------------------------------------------------

    valid = (
        np.isfinite(east)
        & np.isfinite(north)
        & (east >= X_MIN)
        & (east <= X_MAX)
    )

    x_raw = east[valid]
    e_raw = north[valid]

    if len(x_raw) < 2:
        raise RuntimeError(
            f"Insufficient observations for {key}"
        )

    raw_n = len(x_raw)


    # --------------------------------------------------------
    # 8.2 Exact-Easting grouping
    # --------------------------------------------------------
    #
    # We do not round coordinates here.
    #
    # Only positions that are numerically identical in the
    # released dataset are grouped.
    # --------------------------------------------------------

    raw_df = pd.DataFrame({
        "Easting_m": x_raw,
        "Northing_m": e_raw
    })

    grouped = (
        raw_df
        .groupby(
            "Easting_m",
            sort=True
        )["Northing_m"]
        .agg(
            [
                "count",
                "median",
                "mean",
                "std",
                "min",
                "max"
            ]
        )
        .reset_index()
    )

    grouped["range"] = (
        grouped["max"]
        - grouped["min"]
    )

    x_unique = grouped[
        "Easting_m"
    ].to_numpy(dtype=float)

    e_unique = grouped[
        "median"
    ].to_numpy(dtype=float)

    counts = grouped[
        "count"
    ].to_numpy(dtype=int)

    within_x_range = grouped[
        "range"
    ].to_numpy(dtype=float)

    unique_n = len(x_unique)


    # --------------------------------------------------------
    # 8.3 Duplicate diagnostics
    # --------------------------------------------------------

    duplicated_groups_mask = counts > 1

    n_duplicated_groups = int(
        np.sum(
            duplicated_groups_mask
        )
    )

    n_singleton_groups = int(
        np.sum(
            counts == 1
        )
    )

    if n_duplicated_groups > 0:

        duplicated_ranges = within_x_range[
            duplicated_groups_mask
        ]

        exact_duplicate_same_northing_fraction = np.mean(
            duplicated_ranges == 0
        )

        duplicated_range_median = np.median(
            duplicated_ranges
        )

        duplicated_range_p95 = np.percentile(
            duplicated_ranges,
            95
        )

        duplicated_range_max = np.max(
            duplicated_ranges
        )

    else:

        exact_duplicate_same_northing_fraction = np.nan
        duplicated_range_median = np.nan
        duplicated_range_p95 = np.nan
        duplicated_range_max = np.nan


    # --------------------------------------------------------
    # 8.4 Unique-position spacing
    # --------------------------------------------------------

    dx_unique = np.diff(
        x_unique
    )

    positive_dx_unique = dx_unique[
        dx_unique > 0
    ]

    if len(positive_dx_unique) > 0:

        unique_dx_median = np.median(
            positive_dx_unique
        )

        unique_dx_p05 = np.percentile(
            positive_dx_unique,
            5
        )

        unique_dx_p95 = np.percentile(
            positive_dx_unique,
            95
        )

    else:

        unique_dx_median = np.nan
        unique_dx_p05 = np.nan
        unique_dx_p95 = np.nan


    # --------------------------------------------------------
    # 8.5 Spatially weighted baseline metrics
    # --------------------------------------------------------

    spatial_metrics = spatial_weighted_metrics(
        x_unique,
        e_unique
    )

    raw_sample_rms = rms(
        e_raw
    )

    raw_sample_mae = np.mean(
        np.abs(e_raw)
    )

    collapsed_unweighted_rms = rms(
        e_unique
    )

    collapsed_unweighted_mae = np.mean(
        np.abs(e_unique)
    )


    # --------------------------------------------------------
    # 8.6 Save duplicate / spacing diagnostics
    # --------------------------------------------------------

    duplicate_diagnostics_rows.append({

        "Configuration":
            config_code,

        "Configuration_Name":
            config_name,

        "Pass":
            pass_number,

        "Raw_N":
            raw_n,

        "Unique_Easting_N":
            unique_n,

        "Compression_ratio_raw_to_unique":
            raw_n / unique_n,

        "Duplicated_Easting_groups":
            n_duplicated_groups,

        "Singleton_Easting_groups":
            n_singleton_groups,

        "Duplicate_groups_with_identical_Northing_fraction":
            exact_duplicate_same_northing_fraction,

        "Duplicate_Northing_range_median_m":
            duplicated_range_median,

        "Duplicate_Northing_range_P95_m":
            duplicated_range_p95,

        "Duplicate_Northing_range_max_m":
            duplicated_range_max,

        "Unique_Easting_spacing_P05_m":
            unique_dx_p05,

        "Unique_Easting_spacing_median_m":
            unique_dx_median,

        "Unique_Easting_spacing_P95_m":
            unique_dx_p95
    })


    # --------------------------------------------------------
    # 8.7 Save baseline metric comparison
    # --------------------------------------------------------

    spatial_metric_rows.append({

        "Configuration":
            config_code,

        "Configuration_Name":
            config_name,

        "Pass":
            pass_number,

        "Raw_sample_RMS_m":
            raw_sample_rms,

        "Raw_sample_MAE_m":
            raw_sample_mae,

        "Unique_unweighted_RMS_m":
            collapsed_unweighted_rms,

        "Unique_unweighted_MAE_m":
            collapsed_unweighted_mae,

        "Spatial_weighted_Mean_m":
            spatial_metrics[
                "Spatial_Mean_m"
            ],

        "Spatial_weighted_MAE_m":
            spatial_metrics[
                "Spatial_MAE_m"
            ],

        "Spatial_weighted_RMS_m":
            spatial_metrics[
                "Spatial_RMS_m"
            ]
    })


    # Store collapsed representation.
    collapsed_passes[key] = {
        "x": x_unique,
        "e": e_unique
    }


    # --------------------------------------------------------
    # 8.8 Resolution study
    # --------------------------------------------------------

    baseline_rms = spatial_metrics[
        "Spatial_RMS_m"
    ]

    baseline_mae = spatial_metrics[
        "Spatial_MAE_m"
    ]

    baseline_mean = spatial_metrics[
        "Spatial_Mean_m"
    ]


    for dx_grid in GRID_SPACINGS_M:

        grid = regular_grid(
            x_unique[0],
            x_unique[-1],
            dx_grid
        )

        if len(grid) < 2:
            continue

        # Linear interpolation on the spatial trajectory.
        #
        # No extrapolation occurs because regular_grid() remains
        # within the observed Easting support.
        e_grid = np.interp(
            grid,
            x_unique,
            e_unique
        )

        grid_rms = rms(
            e_grid
        )

        grid_mae = np.mean(
            np.abs(e_grid)
        )

        grid_mean = np.mean(
            e_grid
        )

        rms_absolute_difference = (
            grid_rms
            - baseline_rms
        )

        mae_absolute_difference = (
            grid_mae
            - baseline_mae
        )

        mean_absolute_difference = (
            grid_mean
            - baseline_mean
        )

        if baseline_rms != 0:

            rms_relative_difference_pct = (
                100
                * rms_absolute_difference
                / baseline_rms
            )

        else:

            rms_relative_difference_pct = np.nan


        if baseline_mae != 0:

            mae_relative_difference_pct = (
                100
                * mae_absolute_difference
                / baseline_mae
            )

        else:

            mae_relative_difference_pct = np.nan


        resolution_rows.append({

            "Configuration":
                config_code,

            "Configuration_Name":
                config_name,

            "Pass":
                pass_number,

            "Grid_spacing_m":
                dx_grid,

            "Grid_N":
                len(grid),

            "Grid_Mean_m":
                grid_mean,

            "Grid_MAE_m":
                grid_mae,

            "Grid_RMS_m":
                grid_rms,

            "Spatial_integral_Mean_m":
                baseline_mean,

            "Spatial_integral_MAE_m":
                baseline_mae,

            "Spatial_integral_RMS_m":
                baseline_rms,

            "Mean_difference_m":
                mean_absolute_difference,

            "MAE_difference_m":
                mae_absolute_difference,

            "RMS_difference_m":
                rms_absolute_difference,

            "MAE_relative_difference_pct":
                mae_relative_difference_pct,

            "RMS_relative_difference_pct":
                rms_relative_difference_pct
        })


        # ----------------------------------------------------
        # Canonical 0.20 m representation
        # ----------------------------------------------------

        if np.isclose(
            dx_grid,
            CANONICAL_DX_M
        ):

            canonical_passes[key] = {
                "x": grid,
                "e": e_grid
            }

            for spatial_index, (
                x_value,
                e_value
            ) in enumerate(
                zip(
                    grid,
                    e_grid
                )
            ):

                canonical_rows.append({

                    "Configuration":
                        config_code,

                    "Configuration_Name":
                        config_name,

                    "Pass":
                        pass_number,

                    "Spatial_Index":
                        spatial_index,

                    "Easting_m":
                        x_value,

                    "Signed_Lateral_Error_m":
                        e_value,

                    "Grid_Spacing_m":
                        CANONICAL_DX_M
                })


# ------------------------------------------------------------
# 9. Convert to DataFrames
# ------------------------------------------------------------

duplicate_diagnostics = pd.DataFrame(
    duplicate_diagnostics_rows
)

spatial_metrics_df = pd.DataFrame(
    spatial_metric_rows
)

resolution_df = pd.DataFrame(
    resolution_rows
)

canonical_df = pd.DataFrame(
    canonical_rows
)


# ------------------------------------------------------------
# 10. Save tables
# ------------------------------------------------------------

duplicate_file = (
    TABLES_DIR
    / "spatial_duplicate_diagnostics.csv"
)

metric_file = (
    TABLES_DIR
    / "spatial_metric_comparison.csv"
)

resolution_file = (
    TABLES_DIR
    / "spatial_resolution_sensitivity.csv"
)

canonical_file = (
    TABLES_DIR
    / "guidance_canonical_0p20m.csv"
)

duplicate_diagnostics.to_csv(
    duplicate_file,
    index=False
)

spatial_metrics_df.to_csv(
    metric_file,
    index=False
)

resolution_df.to_csv(
    resolution_file,
    index=False
)

canonical_df.to_csv(
    canonical_file,
    index=False
)


# ------------------------------------------------------------
# 11. Configuration-level canonical statistics
# ------------------------------------------------------------
#
# Pass 1 and Pass 3 are kept spatially independent, then their
# canonical values are combined only for descriptive comparison
# with the authors' short-term configuration statistics.
# ------------------------------------------------------------

configuration_rows = []

configuration_keys = sorted(
    set(
        (
            config_code,
            config_name
        )
        for (
            config_code,
            config_name,
            pass_number
        ) in canonical_passes.keys()
    )
)

for config_code, config_name in configuration_keys:

    pass1 = canonical_passes[
        (
            config_code,
            config_name,
            1
        )
    ]["e"]

    pass3 = canonical_passes[
        (
            config_code,
            config_name,
            3
        )
    ]["e"]

    combined = np.concatenate(
        [
            pass1,
            pass3
        ]
    )

    configuration_rows.append({

        "Configuration":
            config_code,

        "Configuration_Name":
            config_name,

        "Canonical_N_Pass1_Plus_Pass3":
            len(combined),

        "Canonical_0p20m_RMS_m":
            rms(combined),

        "Canonical_0p20m_MAE_m":
            np.mean(
                np.abs(combined)
            ),

        "Canonical_0p20m_Signed_Mean_m":
            np.mean(combined),

        "Canonical_0p20m_SD_m":
            np.std(
                combined,
                ddof=1
            ),

        "Canonical_0p20m_Abs_P95_m":
            np.percentile(
                np.abs(combined),
                95
            ),

        "Canonical_0p20m_Abs_P99_m":
            np.percentile(
                np.abs(combined),
                99
            )
    })


configuration_df = pd.DataFrame(
    configuration_rows
)

configuration_file = (
    TABLES_DIR
    / "canonical_0p20m_configuration_statistics.csv"
)

configuration_df.to_csv(
    configuration_file,
    index=False
)


# ------------------------------------------------------------
# 12. Plot canonical trajectories
# ------------------------------------------------------------

for (
    config_code,
    config_name
) in configuration_keys:

    plt.figure(
        figsize=(11, 5)
    )

    for pass_number in [
        1,
        2,
        3
    ]:

        key = (
            config_code,
            config_name,
            pass_number
        )

        x = canonical_passes[
            key
        ]["x"]

        e = canonical_passes[
            key
        ]["e"]

        plt.plot(
            x,
            e,
            linewidth=1,
            label=f"Pass {pass_number}"
        )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Along-track position (m)"
    )

    plt.ylabel(
        "Signed lateral guidance deviation (m)"
    )

    plt.title(
        f"{config_code} — "
        f"{config_name.replace('_', ' ')} "
        f"(0.20 m spatial grid)"
    )

    plt.legend()

    plt.tight_layout()

    figure_file = (
        SPATIAL_FIGURES_DIR
        / f"{config_code}_canonical_0p20m.png"
    )

    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 13. Resolution-convergence summaries
# ------------------------------------------------------------

resolution_summary = (
    resolution_df
    .groupby(
        "Grid_spacing_m"
    )
    .agg(
        Median_abs_RMS_difference_pct=(
            "RMS_relative_difference_pct",
            lambda x:
                np.nanmedian(
                    np.abs(x)
                )
        ),

        Max_abs_RMS_difference_pct=(
            "RMS_relative_difference_pct",
            lambda x:
                np.nanmax(
                    np.abs(x)
                )
        ),

        Median_abs_MAE_difference_pct=(
            "MAE_relative_difference_pct",
            lambda x:
                np.nanmedian(
                    np.abs(x)
                )
        ),

        Max_abs_MAE_difference_pct=(
            "MAE_relative_difference_pct",
            lambda x:
                np.nanmax(
                    np.abs(x)
                )
        )
    )
    .reset_index()
)

resolution_summary_file = (
    TABLES_DIR
    / "spatial_resolution_convergence_summary.csv"
)

resolution_summary.to_csv(
    resolution_summary_file,
    index=False
)


# ------------------------------------------------------------
# 14. Terminal output
# ------------------------------------------------------------

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
print("EXACT-DUPLICATE DIAGNOSTIC")
print("=" * 90)

duplicate_display_columns = [

    "Configuration",
    "Pass",

    "Raw_N",
    "Unique_Easting_N",

    "Compression_ratio_raw_to_unique",

    "Duplicate_groups_with_identical_Northing_fraction",

    "Duplicate_Northing_range_P95_m",

    "Unique_Easting_spacing_median_m"
]

print(
    duplicate_diagnostics[
        duplicate_display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("SPATIAL METRIC COMPARISON")
print("=" * 90)

metric_display_columns = [

    "Configuration",
    "Pass",

    "Raw_sample_RMS_m",

    "Unique_unweighted_RMS_m",

    "Spatial_weighted_RMS_m",

    "Raw_sample_MAE_m",

    "Spatial_weighted_MAE_m"
]

print(
    spatial_metrics_df[
        metric_display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("GRID RESOLUTION CONVERGENCE")
print("=" * 90)

print(
    resolution_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("CANONICAL 0.20 m CONFIGURATION STATISTICS")
print("=" * 90)

canonical_display_columns = [

    "Configuration",
    "Configuration_Name",

    "Canonical_0p20m_RMS_m",

    "Canonical_0p20m_MAE_m",

    "Canonical_0p20m_Signed_Mean_m",

    "Canonical_0p20m_Abs_P95_m"
]

print(
    configuration_df[
        canonical_display_columns
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 15. Key diagnostic summaries
# ------------------------------------------------------------

same_northing_values = (
    duplicate_diagnostics[
        "Duplicate_groups_with_identical_Northing_fraction"
    ]
    .dropna()
)

print("\n" + "=" * 90)
print("DATASET-WIDE SUMMARY")
print("=" * 90)

print(
    "\nRaw-to-unique compression ratio:"
)

print(
    f"  minimum = "
    f"{duplicate_diagnostics['Compression_ratio_raw_to_unique'].min():.3f}"
)

print(
    f"  median  = "
    f"{duplicate_diagnostics['Compression_ratio_raw_to_unique'].median():.3f}"
)

print(
    f"  maximum = "
    f"{duplicate_diagnostics['Compression_ratio_raw_to_unique'].max():.3f}"
)


if len(same_northing_values) > 0:

    print(
        "\nFraction of duplicated Easting groups whose repeated "
        "Northing values are exactly identical:"
    )

    print(
        f"  median = "
        f"{same_northing_values.median():.6f}"
    )

    print(
        f"  min    = "
        f"{same_northing_values.min():.6f}"
    )


print(
    "\nCanonical spatial spacing:"
)

print(
    f"  {CANONICAL_DX_M:.3f} m"
)

print(
    f"  derived from "
    f"{TRACTOR_SPEED_MPS:.1f} m/s / "
    f"{GNSS_RATE_HZ:.1f} Hz"
)


# ------------------------------------------------------------
# 16. Output locations
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("OUTPUT FILES")
print("=" * 90)

print(
    f"\nDuplicate diagnostics:\n"
    f"{duplicate_file}"
)

print(
    f"\nSpatial metric comparison:\n"
    f"{metric_file}"
)

print(
    f"\nResolution sensitivity:\n"
    f"{resolution_file}"
)

print(
    f"\nResolution convergence summary:\n"
    f"{resolution_summary_file}"
)

print(
    f"\nCanonical 0.20 m trajectory dataset:\n"
    f"{canonical_file}"
)

print(
    f"\nCanonical configuration statistics:\n"
    f"{configuration_file}"
)

print(
    f"\nCanonical trajectory figures:\n"
    f"{SPATIAL_FIGURES_DIR}"
)


print("\n" + "=" * 90)
print("STAGE 0D COMPLETE")
print("=" * 90)

print(
    "\nNo stochastic model or autocorrelation model has "
    "been fitted."
)

print(
    "The canonical 0.20 m trajectories are now ready for "
    "spatial ACF / PSD analysis, subject to inspection of the "
    "resolution-convergence diagnostics above."
)