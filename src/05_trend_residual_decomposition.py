from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis


# ============================================================
# STAGE 1B — TREND / STOCHASTIC RESIDUAL DECOMPOSITION
# ============================================================
#
# PURPOSE
# -------
# Investigate whether the observed long-range spatial dependence
# is partly produced by deterministic / slowly varying drift.
#
# For every canonical pass:
#
#       e(x) = a + b*x + u(x)
#
# where:
#
#       e(x) = observed signed lateral guidance error
#       a+b*x = fitted linear spatial trend
#       u(x) = residual component
#
# IMPORTANT:
# -------
# This does NOT assert that the true physical trend is linear.
#
# Linear detrending is used here as the simplest identifiable
# first-order diagnostic because the previous stage showed large
# linear R² values in several passes.
#
# We compare:
#
#   - original variance
#   - variance explained by linear drift
#   - residual variance
#   - original vs residual ACF
#   - original vs residual correlation scales
#   - residual distribution shape
#   - residual quarter stability
#   - original vs residual PSD
#
# No stochastic process is fitted yet.
#
# Primary calibration:
#       Pass 1 and Pass 3
#
# Pass 2:
#       diagnostic only
#
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "guidance_canonical_0p20m.csv"
)

TABLES_DIR = (
    PROJECT_DIR
    / "results"
    / "tables"
)

FIGURES_DIR = (
    PROJECT_DIR
    / "results"
    / "figures"
    / "trend_residual_decomposition"
)

TRAJECTORY_DIR = (
    FIGURES_DIR
    / "trajectory_decomposition"
)

ACF_DIR = (
    FIGURES_DIR
    / "acf_comparison"
)

PSD_DIR = (
    FIGURES_DIR
    / "psd_comparison"
)

for directory in [
    TABLES_DIR,
    FIGURES_DIR,
    TRAJECTORY_DIR,
    ACF_DIR,
    PSD_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ------------------------------------------------------------
# 2. Study settings
# ------------------------------------------------------------

DX_M = 0.20

PRIMARY_PASSES = [
    1,
    3
]


# ------------------------------------------------------------
# 3. Utility functions
# ------------------------------------------------------------

def rms(values):

    values = np.asarray(
        values,
        dtype=float
    )

    return np.sqrt(
        np.mean(
            values ** 2
        )
    )


def bounded_acf_fft(values):
    """
    Standard biased normalized ACF.

    Unlike the previous unbiased estimator, this version uses
    the same denominator at all lags. It is numerically stable
    and keeps the lag-zero value equal to 1.

    This is appropriate here for comparing empirical decay
    shapes between original and residual trajectories.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    values = (
        values
        - np.mean(values)
    )

    n = len(values)

    if n < 2:
        return np.array([
            np.nan
        ])

    sum_square = np.sum(
        values ** 2
    )

    if sum_square == 0:
        return np.full(
            n,
            np.nan
        )

    fft_length = (
        1
        << (
            2 * n - 1
        ).bit_length()
    )

    transformed = np.fft.rfft(
        values,
        n=fft_length
    )

    covariance = np.fft.irfft(
        transformed
        * np.conjugate(
            transformed
        ),
        n=fft_length
    )[:n]

    acf = (
        covariance
        / sum_square
    )

    acf[0] = 1.0

    return acf


def first_crossing(
    acf,
    threshold,
    dx
):

    acf = np.asarray(
        acf,
        dtype=float
    )

    if len(acf) < 2:
        return np.nan

    indices = np.where(
        acf[1:]
        <= threshold
    )[0]

    if len(indices) == 0:
        return np.nan

    index = (
        indices[0]
        + 1
    )

    return (
        index
        * dx
    )


def e_folding_distance(
    acf,
    dx
):

    return first_crossing(
        acf,
        np.exp(-1),
        dx
    )


def first_zero_crossing(
    acf,
    dx
):

    return first_crossing(
        acf,
        0.0,
        dx
    )


def integral_correlation_scale(
    acf,
    dx
):
    """
    Integrate the positive ACF from zero to its first
    zero crossing.
    """

    acf = np.asarray(
        acf,
        dtype=float
    )

    if len(acf) < 2:
        return np.nan

    zero_indices = np.where(
        acf[1:]
        <= 0
    )[0]

    if len(zero_indices) > 0:

        end_index = (
            zero_indices[0]
            + 1
        )

    else:

        end_index = (
            len(acf)
            - 1
        )

    local_acf = acf[
        :end_index + 1
    ]

    distances = (
        np.arange(
            len(local_acf)
        )
        * dx
    )

    if hasattr(
        np,
        "trapezoid"
    ):

        return np.trapezoid(
            local_acf,
            distances
        )

    return np.trapz(
        local_acf,
        distances
    )


def welch_psd(
    values,
    dx
):
    """
    Welch spatial PSD.

    The segment count is data-derived from trajectory length.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = (
        values
        - np.mean(values)
    )

    n = len(values)

    spatial_sampling_frequency = (
        1.0
        / dx
    )

    nperseg = max(
        64,
        n // 8
    )

    nperseg = min(
        nperseg,
        n
    )

    frequency, power = welch(
        values,
        fs=spatial_sampling_frequency,
        nperseg=nperseg,
        detrend="constant",
        scaling="density"
    )

    return (
        frequency,
        power
    )


def quarter_residual_diagnostic(
    x,
    residual
):

    boundaries = np.linspace(
        x.min(),
        x.max(),
        5
    )

    means = []
    sds = []

    output = []

    for quarter in range(4):

        lower = boundaries[
            quarter
        ]

        upper = boundaries[
            quarter + 1
        ]

        if quarter < 3:

            mask = (
                (x >= lower)
                & (x < upper)
            )

        else:

            mask = (
                (x >= lower)
                & (x <= upper)
            )

        values = residual[
            mask
        ]

        q_mean = np.mean(
            values
        )

        q_sd = np.std(
            values,
            ddof=1
        )

        means.append(
            q_mean
        )

        sds.append(
            q_sd
        )

        output.append(
            {
                "Quarter":
                    quarter + 1,

                "X_start_m":
                    lower,

                "X_end_m":
                    upper,

                "N":
                    len(values),

                "Residual_Mean_m":
                    q_mean,

                "Residual_SD_m":
                    q_sd,

                "Residual_RMS_m":
                    rms(
                        values
                    )
            }
        )

    return (
        output,
        np.ptp(
            means
        ),
        np.ptp(
            sds
        )
    )


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n"
        f"{INPUT_FILE}"
    )


print(
    "=" * 90
)

print(
    "STAGE 1B — TREND / STOCHASTIC RESIDUAL DECOMPOSITION"
)

print(
    "=" * 90
)


df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nLoaded:"
    f"\n{INPUT_FILE}"
)

print(
    f"\nRows: "
    f"{len(df):,}"
)


# ------------------------------------------------------------
# 5. Containers
# ------------------------------------------------------------

summary_rows = []

acf_rows = []

psd_rows = []

quarter_rows = []

residual_rows = []


# ------------------------------------------------------------
# 6. Analyze each trajectory
# ------------------------------------------------------------

group_columns = [
    "Configuration",
    "Configuration_Name",
    "Pass"
]


for (
    config,
    config_name,
    pass_number
), group in df.groupby(
    group_columns,
    sort=True
):

    group = group.sort_values(
        "Easting_m"
    )

    x = group[
        "Easting_m"
    ].to_numpy(
        dtype=float
    )

    error = group[
        "Signed_Lateral_Error_m"
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # 6.1 Center x to improve numerical conditioning
    # --------------------------------------------------------

    x_centered = (
        x
        - np.mean(x)
    )


    # --------------------------------------------------------
    # 6.2 Linear trend
    # --------------------------------------------------------

    slope, intercept_centered = np.polyfit(
        x_centered,
        error,
        1
    )

    trend = (
        intercept_centered
        + slope * x_centered
    )

    residual = (
        error
        - trend
    )


    # --------------------------------------------------------
    # 6.3 Equivalent intercept in original x-coordinate system
    # --------------------------------------------------------

    intercept_original = (
        intercept_centered
        - slope
        * np.mean(x)
    )


    # --------------------------------------------------------
    # 6.4 Variance decomposition
    # --------------------------------------------------------

    error_mean = np.mean(
        error
    )

    total_centered = (
        error
        - error_mean
    )

    total_variance = np.var(
        error,
        ddof=1
    )

    residual_variance = np.var(
        residual,
        ddof=1
    )

    trend_centered = (
        trend
        - np.mean(trend)
    )

    trend_variance = np.var(
        trend_centered,
        ddof=1
    )


    ss_total = np.sum(
        total_centered ** 2
    )

    ss_residual = np.sum(
        residual ** 2
    )

    if ss_total > 0:

        drift_r2 = (
            1
            - ss_residual
            / ss_total
        )

    else:

        drift_r2 = np.nan


    variance_removed_fraction = (

        1
        - residual_variance
        / total_variance

        if total_variance > 0

        else np.nan
    )


    # --------------------------------------------------------
    # 6.5 Magnitudes
    # --------------------------------------------------------

    total_rmse = rms(
        error
    )

    centered_rmse = rms(
        total_centered
    )

    trend_centered_rms = rms(
        trend_centered
    )

    residual_rms = rms(
        residual
    )


    # --------------------------------------------------------
    # 6.6 Original ACF
    # --------------------------------------------------------

    original_acf = bounded_acf_fft(
        error
    )

    original_lag1 = (
        original_acf[1]
        if len(original_acf) > 1
        else np.nan
    )

    original_e_fold = e_folding_distance(
        original_acf,
        DX_M
    )

    original_zero = first_zero_crossing(
        original_acf,
        DX_M
    )

    original_integral = integral_correlation_scale(
        original_acf,
        DX_M
    )


    # --------------------------------------------------------
    # 6.7 Residual ACF
    # --------------------------------------------------------

    residual_acf = bounded_acf_fft(
        residual
    )

    residual_lag1 = (
        residual_acf[1]
        if len(residual_acf) > 1
        else np.nan
    )

    residual_e_fold = e_folding_distance(
        residual_acf,
        DX_M
    )

    residual_zero = first_zero_crossing(
        residual_acf,
        DX_M
    )

    residual_integral = integral_correlation_scale(
        residual_acf,
        DX_M
    )


    # --------------------------------------------------------
    # 6.8 Save ACF curves
    # --------------------------------------------------------

    maximum_length = min(
        len(original_acf),
        len(residual_acf)
    )

    for index in range(
        maximum_length
    ):

        acf_rows.append(
            {
                "Configuration":
                    config,

                "Configuration_Name":
                    config_name,

                "Pass":
                    pass_number,

                "Primary_Calibration_Pass":
                    (
                        pass_number
                        in PRIMARY_PASSES
                    ),

                "Lag_m":
                    index * DX_M,

                "Original_ACF":
                    original_acf[index],

                "Linear_Detrended_Residual_ACF":
                    residual_acf[index]
            }
        )


    # --------------------------------------------------------
    # 6.9 PSD comparison
    # --------------------------------------------------------

    (
        original_frequency,
        original_power
    ) = welch_psd(
        error,
        DX_M
    )

    (
        residual_frequency,
        residual_power
    ) = welch_psd(
        residual,
        DX_M
    )


    common_n = min(
        len(original_frequency),
        len(residual_frequency)
    )


    for index in range(
        common_n
    ):

        psd_rows.append(
            {
                "Configuration":
                    config,

                "Configuration_Name":
                    config_name,

                "Pass":
                    pass_number,

                "Primary_Calibration_Pass":
                    (
                        pass_number
                        in PRIMARY_PASSES
                    ),

                "Spatial_Frequency_cycles_per_m":
                    original_frequency[
                        index
                    ],

                "Original_PSD":
                    original_power[
                        index
                    ],

                "Residual_PSD":
                    residual_power[
                        index
                    ]
            }
        )


    # --------------------------------------------------------
    # 6.10 Distribution of residuals
    # --------------------------------------------------------

    residual_sd = np.std(
        residual,
        ddof=1
    )

    residual_skew = skew(
        residual,
        bias=False
    )

    residual_excess_kurtosis = kurtosis(
        residual,
        fisher=True,
        bias=False
    )


    # --------------------------------------------------------
    # 6.11 Quarter stability after detrending
    # --------------------------------------------------------

    (
        quarter_stats,
        residual_quarter_mean_range,
        residual_quarter_sd_range
    ) = quarter_residual_diagnostic(
        x,
        residual
    )


    for item in quarter_stats:

        quarter_rows.append(
            {
                "Configuration":
                    config,

                "Configuration_Name":
                    config_name,

                "Pass":
                    pass_number,

                "Primary_Calibration_Pass":
                    (
                        pass_number
                        in PRIMARY_PASSES
                    ),

                **item
            }
        )


    # --------------------------------------------------------
    # 6.12 Residual trajectory table
    # --------------------------------------------------------

    for (
        x_value,
        error_value,
        trend_value,
        residual_value
    ) in zip(
        x,
        error,
        trend,
        residual
    ):

        residual_rows.append(
            {
                "Configuration":
                    config,

                "Configuration_Name":
                    config_name,

                "Pass":
                    pass_number,

                "Primary_Calibration_Pass":
                    (
                        pass_number
                        in PRIMARY_PASSES
                    ),

                "Easting_m":
                    x_value,

                "Observed_Error_m":
                    error_value,

                "Linear_Trend_m":
                    trend_value,

                "Residual_Error_m":
                    residual_value
            }
        )


    # --------------------------------------------------------
    # 6.13 Summary
    # --------------------------------------------------------

    summary_rows.append(
        {
            "Configuration":
                config,

            "Configuration_Name":
                config_name,

            "Pass":
                pass_number,

            "Primary_Calibration_Pass":
                (
                    pass_number
                    in PRIMARY_PASSES
                ),

            "N":
                len(error),

            "Length_m":
                (
                    x.max()
                    - x.min()
                ),

            "Observed_Mean_m":
                error_mean,

            "Observed_RMSE_m":
                total_rmse,

            "Observed_Centered_RMS_m":
                centered_rmse,

            "Linear_Trend_Slope_m_per_m":
                slope,

            "Linear_Trend_Intercept_m":
                intercept_original,

            "Trend_Centered_RMS_m":
                trend_centered_rms,

            "Residual_Mean_m":
                np.mean(
                    residual
                ),

            "Residual_SD_m":
                residual_sd,

            "Residual_RMS_m":
                residual_rms,

            "Drift_R2":
                drift_r2,

            "Variance_Removed_By_Linear_Drift_Fraction":
                variance_removed_fraction,

            "Original_ACF_Lag1":
                original_lag1,

            "Residual_ACF_Lag1":
                residual_lag1,

            "Original_E_Folding_m":
                original_e_fold,

            "Residual_E_Folding_m":
                residual_e_fold,

            "Original_First_Zero_Crossing_m":
                original_zero,

            "Residual_First_Zero_Crossing_m":
                residual_zero,

            "Original_Integral_Correlation_Scale_m":
                original_integral,

            "Residual_Integral_Correlation_Scale_m":
                residual_integral,

            "Residual_Skewness":
                residual_skew,

            "Residual_Excess_Kurtosis":
                residual_excess_kurtosis,

            "Residual_Quarter_Mean_Range_m":
                residual_quarter_mean_range,

            "Residual_Quarter_SD_Range_m":
                residual_quarter_sd_range
        }
    )


# ------------------------------------------------------------
# 7. DataFrames
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    summary_rows
)

acf_df = pd.DataFrame(
    acf_rows
)

psd_df = pd.DataFrame(
    psd_rows
)

quarter_df = pd.DataFrame(
    quarter_rows
)

residual_df = pd.DataFrame(
    residual_rows
)


# ------------------------------------------------------------
# 8. Save outputs
# ------------------------------------------------------------

summary_file = (
    TABLES_DIR
    / "trend_residual_decomposition_summary.csv"
)

acf_file = (
    TABLES_DIR
    / "original_vs_residual_spatial_acf.csv"
)

psd_file = (
    TABLES_DIR
    / "original_vs_residual_spatial_psd.csv"
)

quarter_file = (
    TABLES_DIR
    / "residual_quarter_stability.csv"
)

residual_file = (
    TABLES_DIR
    / "canonical_linear_detrended_residuals.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)

acf_df.to_csv(
    acf_file,
    index=False
)

psd_df.to_csv(
    psd_file,
    index=False
)

quarter_df.to_csv(
    quarter_file,
    index=False
)

residual_df.to_csv(
    residual_file,
    index=False
)


# ------------------------------------------------------------
# 9. Configuration-level primary summary
# ------------------------------------------------------------

primary = summary_df[
    summary_df[
        "Primary_Calibration_Pass"
    ]
].copy()


configuration_summary = (
    primary
    .groupby(
        [
            "Configuration",
            "Configuration_Name"
        ]
    )
    .agg(

        Mean_Drift_R2=(
            "Drift_R2",
            "mean"
        ),

        Mean_Variance_Removed=(
            "Variance_Removed_By_Linear_Drift_Fraction",
            "mean"
        ),

        Mean_Original_ACF_Lag1=(
            "Original_ACF_Lag1",
            "mean"
        ),

        Mean_Residual_ACF_Lag1=(
            "Residual_ACF_Lag1",
            "mean"
        ),

        Mean_Original_E_Folding_m=(
            "Original_E_Folding_m",
            "mean"
        ),

        Mean_Residual_E_Folding_m=(
            "Residual_E_Folding_m",
            "mean"
        ),

        Mean_Original_Integral_Scale_m=(
            "Original_Integral_Correlation_Scale_m",
            "mean"
        ),

        Mean_Residual_Integral_Scale_m=(
            "Residual_Integral_Correlation_Scale_m",
            "mean"
        ),

        Mean_Residual_Skewness=(
            "Residual_Skewness",
            "mean"
        ),

        Mean_Residual_Excess_Kurtosis=(
            "Residual_Excess_Kurtosis",
            "mean"
        )
    )
    .reset_index()
)


configuration_summary_file = (
    TABLES_DIR
    / "primary_trend_residual_configuration_summary.csv"
)


configuration_summary.to_csv(
    configuration_summary_file,
    index=False
)


# ------------------------------------------------------------
# 10. Decomposition figures
# ------------------------------------------------------------

for (
    config,
    config_name,
    pass_number
), group in residual_df.groupby(
    [
        "Configuration",
        "Configuration_Name",
        "Pass"
    ],
    sort=True
):

    if pass_number not in PRIMARY_PASSES:
        continue

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        group[
            "Easting_m"
        ],
        group[
            "Observed_Error_m"
        ],
        linewidth=1,
        label="Observed guidance error"
    )

    plt.plot(
        group[
            "Easting_m"
        ],
        group[
            "Linear_Trend_m"
        ],
        linewidth=2,
        label="Linear drift component"
    )

    plt.xlabel(
        "Along-track position (m)"
    )

    plt.ylabel(
        "Signed lateral guidance error (m)"
    )

    plt.title(
        f"{config} Pass {pass_number} — "
        f"observed error and linear drift"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        TRAJECTORY_DIR
        / (
            f"{config}_Pass{pass_number}_"
            f"trend_decomposition.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 11. ACF comparison figures
# ------------------------------------------------------------

for (
    config,
    config_name,
    pass_number
), group in acf_df.groupby(
    [
        "Configuration",
        "Configuration_Name",
        "Pass"
    ],
    sort=True
):

    if pass_number not in PRIMARY_PASSES:
        continue

    max_display_lag = min(
        120.0,
        group[
            "Lag_m"
        ].max()
    )

    shown = group[
        group[
            "Lag_m"
        ]
        <= max_display_lag
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        shown[
            "Lag_m"
        ],
        shown[
            "Original_ACF"
        ],
        linewidth=1.3,
        label="Original centered error"
    )

    plt.plot(
        shown[
            "Lag_m"
        ],
        shown[
            "Linear_Detrended_Residual_ACF"
        ],
        linewidth=1.3,
        label="After linear detrending"
    )

    plt.axhline(
        0,
        linestyle=":",
        linewidth=1
    )

    plt.axhline(
        np.exp(-1),
        linestyle="--",
        linewidth=1,
        label="1/e"
    )

    plt.xlabel(
        "Spatial lag (m)"
    )

    plt.ylabel(
        "Autocorrelation"
    )

    plt.title(
        f"{config} Pass {pass_number} — "
        f"ACF before and after detrending"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ACF_DIR
        / (
            f"{config}_Pass{pass_number}_"
            f"acf_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 12. PSD comparison figures
# ------------------------------------------------------------

for (
    config,
    config_name,
    pass_number
), group in psd_df.groupby(
    [
        "Configuration",
        "Configuration_Name",
        "Pass"
    ],
    sort=True
):

    if pass_number not in PRIMARY_PASSES:
        continue

    positive = group[
        group[
            "Spatial_Frequency_cycles_per_m"
        ]
        > 0
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.loglog(
        positive[
            "Spatial_Frequency_cycles_per_m"
        ],
        positive[
            "Original_PSD"
        ],
        linewidth=1.3,
        label="Original centered error"
    )

    plt.loglog(
        positive[
            "Spatial_Frequency_cycles_per_m"
        ],
        positive[
            "Residual_PSD"
        ],
        linewidth=1.3,
        label="Linear-detrended residual"
    )

    plt.xlabel(
        "Spatial frequency (cycles/m)"
    )

    plt.ylabel(
        "Power spectral density"
    )

    plt.title(
        f"{config} Pass {pass_number} — "
        f"PSD before and after detrending"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PSD_DIR
        / (
            f"{config}_Pass{pass_number}_"
            f"psd_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 13. Terminal tables
# ------------------------------------------------------------

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    280
)

pd.set_option(
    "display.max_rows",
    100
)


print(
    "\n"
    + "=" * 90
)

print(
    "PRIMARY PASSES — TREND / RESIDUAL DECOMPOSITION"
)

print(
    "=" * 90
)


display_columns = [

    "Configuration",
    "Pass",

    "Observed_RMSE_m",

    "Observed_Centered_RMS_m",

    "Trend_Centered_RMS_m",

    "Residual_RMS_m",

    "Drift_R2",

    "Original_ACF_Lag1",

    "Residual_ACF_Lag1",

    "Original_E_Folding_m",

    "Residual_E_Folding_m",

    "Original_Integral_Correlation_Scale_m",

    "Residual_Integral_Correlation_Scale_m",

    "Residual_Skewness",

    "Residual_Excess_Kurtosis"
]


print(
    primary[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 90
)

print(
    "CONFIGURATION-LEVEL PRIMARY-PASS SUMMARY"
)

print(
    "=" * 90
)


print(
    configuration_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 14. Dataset-wide summary
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DATASET-WIDE PRIMARY-PASS SUMMARY"
)

print(
    "=" * 90
)


for column, label in [

    (
        "Drift_R2",
        "Linear drift R²"
    ),

    (
        "Variance_Removed_By_Linear_Drift_Fraction",
        "Variance removed by linear detrending"
    ),

    (
        "Original_ACF_Lag1",
        "Original lag-1 ACF"
    ),

    (
        "Residual_ACF_Lag1",
        "Residual lag-1 ACF"
    ),

    (
        "Original_E_Folding_m",
        "Original e-folding distance (m)"
    ),

    (
        "Residual_E_Folding_m",
        "Residual e-folding distance (m)"
    ),

    (
        "Original_Integral_Correlation_Scale_m",
        "Original integral scale (m)"
    ),

    (
        "Residual_Integral_Correlation_Scale_m",
        "Residual integral scale (m)"
    )
]:

    values = primary[
        column
    ].dropna()

    print(
        f"\n{label}:"
    )

    print(
        f"  min    = "
        f"{values.min():.6f}"
    )

    print(
        f"  median = "
        f"{values.median():.6f}"
    )

    print(
        f"  max    = "
        f"{values.max():.6f}"
    )


# ------------------------------------------------------------
# 15. Correlation-scale change
# ------------------------------------------------------------

scale_comparison = primary[
    [
        "Original_Integral_Correlation_Scale_m",
        "Residual_Integral_Correlation_Scale_m"
    ]
].dropna()


if len(
    scale_comparison
) > 0:

    ratio = (
        scale_comparison[
            "Residual_Integral_Correlation_Scale_m"
        ]
        /
        scale_comparison[
            "Original_Integral_Correlation_Scale_m"
        ]
    )

    print(
        "\nResidual / original integral-correlation-scale ratio:"
    )

    print(
        f"  min    = "
        f"{ratio.min():.4f}"
    )

    print(
        f"  median = "
        f"{ratio.median():.4f}"
    )

    print(
        f"  max    = "
        f"{ratio.max():.4f}"
    )


# ------------------------------------------------------------
# 16. Output paths
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 90
)


print(
    f"\nDecomposition summary:\n"
    f"{summary_file}"
)

print(
    f"\nOriginal vs residual ACF:\n"
    f"{acf_file}"
)

print(
    f"\nOriginal vs residual PSD:\n"
    f"{psd_file}"
)

print(
    f"\nResidual quarter stability:\n"
    f"{quarter_file}"
)

print(
    f"\nCanonical residual trajectories:\n"
    f"{residual_file}"
)

print(
    f"\nConfiguration summary:\n"
    f"{configuration_summary_file}"
)

print(
    f"\nTrajectory decomposition figures:\n"
    f"{TRAJECTORY_DIR}"
)

print(
    f"\nACF comparison figures:\n"
    f"{ACF_DIR}"
)

print(
    f"\nPSD comparison figures:\n"
    f"{PSD_DIR}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 1B COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nNo stochastic-process family has been selected yet."
)

print(
    "The next stage will use the residual dependence structure "
    "to determine whether exponential/OU, Gaussian, Matérn, "
    "AR-type, or nonparametric models are supported."
)