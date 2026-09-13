from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.stats import skew, kurtosis, anderson


# ============================================================
# STAGE 1A — EMPIRICAL SPATIAL DEPENDENCE CHARACTERIZATION
# ============================================================
#
# PURPOSE
# -------
# Characterize the canonical 0.20 m lateral-guidance-error
# trajectories before fitting any stochastic model.
#
# PRIMARY CALIBRATION PASSES:
#     Pass 1 and Pass 3
#
# DIAGNOSTIC PASS:
#     Pass 2
#
# WHY:
# The original study's short-term guidance statistics were
# calculated from Pass 1 + Pass 3.
#
# This script DOES NOT fit:
#     - AR(1)
#     - OU / Gauss-Markov
#     - ARMA
#     - Gaussian process
#     - any other stochastic model
#
# It only measures empirical:
#     - signed bias
#     - spread
#     - skewness
#     - kurtosis
#     - quantiles
#     - spatial autocorrelation
#     - e-folding distance
#     - first zero crossing
#     - integral correlation scale
#     - PSD
#     - dominant spatial wavelength
#     - spectral centroid
#     - quarter-to-quarter stability
#
# ACF and PSD are computed after removing ONLY the pass mean.
# No linear detrending is imposed.
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

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
    / "empirical_dependence"
)

ACF_DIR = (
    FIGURES_DIR
    / "acf"
)

PSD_DIR = (
    FIGURES_DIR
    / "psd"
)

DISTRIBUTION_DIR = (
    FIGURES_DIR
    / "distributions"
)

QUARTER_DIR = (
    FIGURES_DIR
    / "quarter_stability"
)

for directory in [
    TABLES_DIR,
    FIGURES_DIR,
    ACF_DIR,
    PSD_DIR,
    DISTRIBUTION_DIR,
    QUARTER_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ------------------------------------------------------------
# 2. Canonical grid
# ------------------------------------------------------------

DX_M = 0.20

PRIMARY_PASSES = [
    1,
    3
]


# ------------------------------------------------------------
# 3. Utility functions
# ------------------------------------------------------------

def rms(x):

    x = np.asarray(
        x,
        dtype=float
    )

    return np.sqrt(
        np.mean(
            x ** 2
        )
    )


def empirical_acf_fft(x):
    """
    Unbiased empirical autocorrelation using FFT.

    Input is centered internally.

    Returns ACF for all non-negative lags.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    x = (
        x
        - np.mean(x)
    )

    n = len(x)

    if n < 2:
        return np.array([
            np.nan
        ])

    variance = np.var(
        x
    )

    if variance == 0:
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

    spectrum = np.fft.rfft(
        x,
        n=fft_length
    )

    autocov = np.fft.irfft(
        spectrum
        * np.conjugate(
            spectrum
        ),
        n=fft_length
    )[:n]

    normalization = np.arange(
        n,
        0,
        -1,
        dtype=float
    )

    autocov = (
        autocov
        / normalization
    )

    return (
        autocov
        / autocov[0]
    )


def first_threshold_crossing(
    values,
    threshold,
    dx
):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) <= 1:
        return np.nan

    indices = np.where(
        values[1:]
        <= threshold
    )[0]

    if len(indices) == 0:
        return np.nan

    lag = (
        indices[0]
        + 1
    )

    return (
        lag
        * dx
    )


def first_zero_crossing(
    acf,
    dx
):

    return first_threshold_crossing(
        acf,
        0.0,
        dx
    )


def e_folding_distance(
    acf,
    dx
):

    return first_threshold_crossing(
        acf,
        np.exp(-1),
        dx
    )


def integral_correlation_scale(
    acf,
    dx
):
    """
    Integral of positive ACF from lag zero to the first
    zero crossing.

    This does not assume an exponential correlation model.
    """

    acf = np.asarray(
        acf,
        dtype=float
    )

    if len(acf) < 2:
        return np.nan

    negative = np.where(
        acf[1:]
        <= 0
    )[0]

    if len(negative) > 0:

        last_index = (
            negative[0]
            + 1
        )

    else:

        last_index = (
            len(acf)
            - 1
        )

    positive_acf = acf[
        :last_index + 1
    ]

    distances = (
        np.arange(
            len(
                positive_acf
            )
        )
        * dx
    )

    if hasattr(
        np,
        "trapezoid"
    ):

        return np.trapezoid(
            positive_acf,
            distances
        )

    return np.trapz(
        positive_acf,
        distances
    )


def spectral_metrics(
    error,
    dx
):
    """
    Welch PSD in cycles per metre.

    Segment length is derived from record length:
    approximately eight segments, with a minimum practical
    segment size and no arbitrary physical frequency cutoff.
    """

    error = np.asarray(
        error,
        dtype=float
    )

    centered = (
        error
        - np.mean(error)
    )

    n = len(centered)

    spatial_fs = (
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
        centered,
        fs=spatial_fs,
        nperseg=nperseg,
        detrend="constant",
        scaling="density"
    )

    positive = (
        frequency > 0
    )

    f = frequency[
        positive
    ]

    p = power[
        positive
    ]

    if (
        len(f) == 0
        or np.sum(p) == 0
    ):

        return (
            frequency,
            power,
            np.nan,
            np.nan
        )

    dominant_index = np.argmax(
        p
    )

    dominant_frequency = f[
        dominant_index
    ]

    dominant_wavelength = (
        1.0
        / dominant_frequency
        if dominant_frequency > 0
        else np.nan
    )

    spectral_centroid = (
        np.sum(
            f * p
        )
        / np.sum(p)
    )

    return (
        frequency,
        power,
        dominant_wavelength,
        spectral_centroid
    )


def quarter_statistics(
    x,
    e
):
    """
    Divide the observed pass into four equal spatial quarters.

    This is a descriptive stationarity diagnostic.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    e = np.asarray(
        e,
        dtype=float
    )

    boundaries = np.linspace(
        x.min(),
        x.max(),
        5
    )

    output = []

    for q in range(4):

        lower = boundaries[
            q
        ]

        upper = boundaries[
            q + 1
        ]

        if q < 3:

            mask = (
                (x >= lower)
                & (x < upper)
            )

        else:

            mask = (
                (x >= lower)
                & (x <= upper)
            )

        values = e[
            mask
        ]

        output.append(
            {
                "Quarter": (
                    q + 1
                ),

                "X_start_m":
                    lower,

                "X_end_m":
                    upper,

                "N":
                    len(values),

                "Mean_m":
                    np.mean(
                        values
                    ),

                "SD_m":
                    np.std(
                        values,
                        ddof=1
                    ),

                "RMS_m":
                    rms(
                        values
                    ),

                "MAE_m":
                    np.mean(
                        np.abs(
                            values
                        )
                    )
            }
        )

    return output


# ------------------------------------------------------------
# 4. Load canonical dataset
# ------------------------------------------------------------

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Canonical file not found:\n"
        f"{INPUT_FILE}"
    )


print(
    "=" * 90
)

print(
    "STAGE 1A — EMPIRICAL SPATIAL DEPENDENCE CHARACTERIZATION"
)

print(
    "=" * 90
)


df = pd.read_csv(
    INPUT_FILE
)


required_columns = {

    "Configuration",
    "Configuration_Name",
    "Pass",
    "Easting_m",
    "Signed_Lateral_Error_m"

}


missing = (
    required_columns
    - set(
        df.columns
    )
)


if missing:

    raise RuntimeError(
        f"Missing columns: "
        f"{sorted(missing)}"
    )


print(
    f"\nLoaded canonical dataset:"
    f"\n{INPUT_FILE}"
)

print(
    f"\nRows: "
    f"{len(df):,}"
)

print(
    f"Configurations: "
    f"{df['Configuration'].nunique()}"
)

print(
    f"Passes: "
    f"{sorted(df['Pass'].unique())}"
)


# ------------------------------------------------------------
# 5. Containers
# ------------------------------------------------------------

summary_rows = []

acf_rows = []

psd_rows = []

quarter_rows = []


# ------------------------------------------------------------
# 6. Analyze each pass separately
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

    e = group[
        "Signed_Lateral_Error_m"
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # Distribution statistics
    # --------------------------------------------------------

    signed_mean = np.mean(
        e
    )

    signed_sd = np.std(
        e,
        ddof=1
    )

    rmse_value = rms(
        e
    )

    mae_value = np.mean(
        np.abs(
            e
        )
    )

    centered = (
        e
        - signed_mean
    )

    skew_value = skew(
        centered,
        bias=False
    )

    excess_kurtosis = kurtosis(
        centered,
        fisher=True,
        bias=False
    )

    abs_error = np.abs(
        e
    )


    # --------------------------------------------------------
    # Anderson-Darling normality statistic
    # --------------------------------------------------------
    #
    # We store the statistic only.
    #
    # For large N, formal hypothesis-test decisions can become
    # over-sensitive. Model choice later will therefore use
    # several diagnostics, not a single p-value.
    # --------------------------------------------------------

    if signed_sd > 0:

        standardized = (
            centered
            / signed_sd
        )

        ad_result = anderson(
            standardized,
            dist="norm"
        )

        ad_statistic = (
            ad_result.statistic
        )

    else:

        ad_statistic = np.nan


    # --------------------------------------------------------
    # Spatial ACF
    # --------------------------------------------------------

    acf = empirical_acf_fft(
        e
    )

    lag_distance = (
        np.arange(
            len(acf)
        )
        * DX_M
    )

    e_fold = e_folding_distance(
        acf,
        DX_M
    )

    zero_cross = first_zero_crossing(
        acf,
        DX_M
    )

    integral_scale = integral_correlation_scale(
        acf,
        DX_M
    )


    for lag_m, value in zip(
        lag_distance,
        acf
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
                    lag_m,

                "ACF":
                    value
            }
        )


    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------

    (
        frequency,
        power,
        dominant_wavelength,
        spectral_centroid
    ) = spectral_metrics(
        e,
        DX_M
    )


    for f, p in zip(
        frequency,
        power
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
                    f,

                "PSD":
                    p
            }
        )


    # --------------------------------------------------------
    # Quarter stability
    # --------------------------------------------------------

    quarters = quarter_statistics(
        x,
        e
    )

    quarter_means = []
    quarter_sds = []

    for quarter in quarters:

        quarter_means.append(
            quarter[
                "Mean_m"
            ]
        )

        quarter_sds.append(
            quarter[
                "SD_m"
            ]
        )

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

                **quarter
            }
        )


    quarter_mean_range = (
        np.max(
            quarter_means
        )
        - np.min(
            quarter_means
        )
    )

    quarter_sd_range = (
        np.max(
            quarter_sds
        )
        - np.min(
            quarter_sds
        )
    )


    # --------------------------------------------------------
    # Linear drift diagnostic
    # --------------------------------------------------------

    slope, intercept = np.polyfit(
        x,
        e,
        1
    )

    fitted = (
        slope * x
        + intercept
    )

    residual = (
        e
        - fitted
    )

    ss_res = np.sum(
        residual ** 2
    )

    ss_tot = np.sum(
        (
            e
            - np.mean(e)
        ) ** 2
    )

    linear_r2 = (
        1
        - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Summary
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
                len(e),

            "Length_m":
                (
                    x.max()
                    - x.min()
                ),

            "Signed_Mean_m":
                signed_mean,

            "Signed_SD_m":
                signed_sd,

            "RMSE_m":
                rmse_value,

            "MAE_m":
                mae_value,

            "Abs_P50_m":
                np.percentile(
                    abs_error,
                    50
                ),

            "Abs_P90_m":
                np.percentile(
                    abs_error,
                    90
                ),

            "Abs_P95_m":
                np.percentile(
                    abs_error,
                    95
                ),

            "Abs_P99_m":
                np.percentile(
                    abs_error,
                    99
                ),

            "Skewness_centered":
                skew_value,

            "Excess_Kurtosis_centered":
                excess_kurtosis,

            "Anderson_Darling_Normality_Statistic":
                ad_statistic,

            "ACF_Lag1":
                (
                    acf[1]
                    if len(acf) > 1
                    else np.nan
                ),

            "E_Folding_Distance_m":
                e_fold,

            "First_Zero_Crossing_m":
                zero_cross,

            "Integral_Correlation_Scale_m":
                integral_scale,

            "Dominant_Wavelength_m":
                dominant_wavelength,

            "Spectral_Centroid_cycles_per_m":
                spectral_centroid,

            "Linear_Drift_Slope_m_per_m":
                slope,

            "Linear_Drift_R2":
                linear_r2,

            "Quarter_Mean_Range_m":
                quarter_mean_range,

            "Quarter_SD_Range_m":
                quarter_sd_range
        }
    )


# ------------------------------------------------------------
# 7. Tables
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


summary_file = (
    TABLES_DIR
    / "empirical_spatial_dependence_summary.csv"
)

acf_file = (
    TABLES_DIR
    / "empirical_spatial_acf_0p20m.csv"
)

psd_file = (
    TABLES_DIR
    / "empirical_spatial_psd_0p20m.csv"
)

quarter_file = (
    TABLES_DIR
    / "empirical_quarter_stability.csv"
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


# ------------------------------------------------------------
# 8. Configuration-level primary-pass summary
# ------------------------------------------------------------

primary = summary_df[
    summary_df[
        "Primary_Calibration_Pass"
    ]
].copy()


config_summary = (
    primary
    .groupby(
        [
            "Configuration",
            "Configuration_Name"
        ]
    )
    .agg(

        Pass1_3_RMSE_Mean_m=(
            "RMSE_m",
            "mean"
        ),

        Pass1_3_Bias_Mean_m=(
            "Signed_Mean_m",
            "mean"
        ),

        Pass1_3_ACF_Lag1_Mean=(
            "ACF_Lag1",
            "mean"
        ),

        Pass1_3_E_Folding_Mean_m=(
            "E_Folding_Distance_m",
            "mean"
        ),

        Pass1_3_Integral_Scale_Mean_m=(
            "Integral_Correlation_Scale_m",
            "mean"
        ),

        Pass1_3_Skewness_Mean=(
            "Skewness_centered",
            "mean"
        ),

        Pass1_3_Excess_Kurtosis_Mean=(
            "Excess_Kurtosis_centered",
            "mean"
        ),

        Pass1_3_Linear_Drift_R2_Mean=(
            "Linear_Drift_R2",
            "mean"
        )
    )
    .reset_index()
)


config_summary_file = (
    TABLES_DIR
    / "primary_pass_configuration_dependence_summary.csv"
)


config_summary.to_csv(
    config_summary_file,
    index=False
)


# ------------------------------------------------------------
# 9. ACF figures
# ------------------------------------------------------------

for (
    config,
    config_name
), group in acf_df.groupby(
    [
        "Configuration",
        "Configuration_Name"
    ],
    sort=True
):

    plt.figure(
        figsize=(10, 5)
    )

    for pass_number in [
        1,
        2,
        3
    ]:

        pass_data = group[
            group[
                "Pass"
            ]
            == pass_number
        ]

        # Display only first half of each trajectory.
        #
        # This is purely a plotting restriction.
        # Full ACF values remain in the saved CSV.
        max_lag = (
            pass_data[
                "Lag_m"
            ].max()
            / 2
        )

        shown = pass_data[
            pass_data[
                "Lag_m"
            ]
            <= max_lag
        ]

        plt.plot(
            shown[
                "Lag_m"
            ],
            shown[
                "ACF"
            ],
            linewidth=1.2,
            label=(
                f"Pass {pass_number}"
            )
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
        f"{config} — spatial autocorrelation"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ACF_DIR
        / f"{config}_spatial_acf.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 10. PSD figures
# ------------------------------------------------------------

for (
    config,
    config_name
), group in psd_df.groupby(
    [
        "Configuration",
        "Configuration_Name"
    ],
    sort=True
):

    plt.figure(
        figsize=(10, 5)
    )

    for pass_number in [
        1,
        2,
        3
    ]:

        pass_data = group[
            (
                group[
                    "Pass"
                ]
                == pass_number
            )
            &
            (
                group[
                    "Spatial_Frequency_cycles_per_m"
                ]
                > 0
            )
        ]

        plt.loglog(
            pass_data[
                "Spatial_Frequency_cycles_per_m"
            ],
            pass_data[
                "PSD"
            ],
            linewidth=1.2,
            label=(
                f"Pass {pass_number}"
            )
        )

    plt.xlabel(
        "Spatial frequency (cycles/m)"
    )

    plt.ylabel(
        "Power spectral density"
    )

    plt.title(
        f"{config} — spatial power spectrum"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PSD_DIR
        / f"{config}_spatial_psd.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 11. Distribution figures
# ------------------------------------------------------------

for (
    config,
    config_name
), group in df.groupby(
    [
        "Configuration",
        "Configuration_Name"
    ],
    sort=True
):

    plt.figure(
        figsize=(10, 5)
    )

    for pass_number in [
        1,
        3
    ]:

        values = group[
            group[
                "Pass"
            ]
            == pass_number
        ][
            "Signed_Lateral_Error_m"
        ].to_numpy()

        plt.hist(
            values,
            bins=40,
            density=True,
            histtype="step",
            linewidth=1.3,
            label=(
                f"Pass {pass_number}"
            )
        )

    plt.xlabel(
        "Signed lateral guidance error (m)"
    )

    plt.ylabel(
        "Probability density"
    )

    plt.title(
        f"{config} — empirical error distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DISTRIBUTION_DIR
        / f"{config}_error_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 12. Quarter-stability figures
# ------------------------------------------------------------

for (
    config,
    config_name
), group in quarter_df.groupby(
    [
        "Configuration",
        "Configuration_Name"
    ],
    sort=True
):

    plt.figure(
        figsize=(9, 5)
    )

    for pass_number in [
        1,
        3
    ]:

        pass_data = group[
            group[
                "Pass"
            ]
            == pass_number
        ]

        plt.plot(
            pass_data[
                "Quarter"
            ],
            pass_data[
                "Mean_m"
            ],
            marker="o",
            label=(
                f"Pass {pass_number}"
            )
        )

    plt.axhline(
        0,
        linestyle=":",
        linewidth=1
    )

    plt.xticks(
        [
            1,
            2,
            3,
            4
        ]
    )

    plt.xlabel(
        "Spatial quarter"
    )

    plt.ylabel(
        "Mean signed error (m)"
    )

    plt.title(
        f"{config} — spatial mean stability"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        QUARTER_DIR
        / f"{config}_quarter_mean.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 13. Terminal output
# ------------------------------------------------------------

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    260
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
    "PRIMARY CALIBRATION PASSES — PASS 1 AND PASS 3"
)

print(
    "=" * 90
)


display_columns = [

    "Configuration",
    "Pass",

    "RMSE_m",
    "Signed_Mean_m",

    "Skewness_centered",
    "Excess_Kurtosis_centered",

    "ACF_Lag1",

    "E_Folding_Distance_m",

    "First_Zero_Crossing_m",

    "Integral_Correlation_Scale_m",

    "Dominant_Wavelength_m",

    "Linear_Drift_R2"
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
    config_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 14. Dataset-wide diagnostic
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DATASET-WIDE PRIMARY-PASS DIAGNOSTICS"
)

print(
    "=" * 90
)


print(
    "\nLag-1 spatial ACF:"
)

print(
    f"  min    = "
    f"{primary['ACF_Lag1'].min():.4f}"
)

print(
    f"  median = "
    f"{primary['ACF_Lag1'].median():.4f}"
)

print(
    f"  max    = "
    f"{primary['ACF_Lag1'].max():.4f}"
)


valid_efold = primary[
    "E_Folding_Distance_m"
].dropna()


if len(valid_efold) > 0:

    print(
        "\nEmpirical e-folding distance:"
    )

    print(
        f"  min    = "
        f"{valid_efold.min():.3f} m"
    )

    print(
        f"  median = "
        f"{valid_efold.median():.3f} m"
    )

    print(
        f"  max    = "
        f"{valid_efold.max():.3f} m"
    )


valid_integral = primary[
    "Integral_Correlation_Scale_m"
].dropna()


if len(valid_integral) > 0:

    print(
        "\nIntegral correlation scale:"
    )

    print(
        f"  min    = "
        f"{valid_integral.min():.3f} m"
    )

    print(
        f"  median = "
        f"{valid_integral.median():.3f} m"
    )

    print(
        f"  max    = "
        f"{valid_integral.max():.3f} m"
    )


print(
    "\nCentered skewness:"
)

print(
    f"  min    = "
    f"{primary['Skewness_centered'].min():.3f}"
)

print(
    f"  median = "
    f"{primary['Skewness_centered'].median():.3f}"
)

print(
    f"  max    = "
    f"{primary['Skewness_centered'].max():.3f}"
)


print(
    "\nCentered excess kurtosis:"
)

print(
    f"  min    = "
    f"{primary['Excess_Kurtosis_centered'].min():.3f}"
)

print(
    f"  median = "
    f"{primary['Excess_Kurtosis_centered'].median():.3f}"
)

print(
    f"  max    = "
    f"{primary['Excess_Kurtosis_centered'].max():.3f}"
)


print(
    "\nLinear-drift R²:"
)

print(
    f"  min    = "
    f"{primary['Linear_Drift_R2'].min():.4f}"
)

print(
    f"  median = "
    f"{primary['Linear_Drift_R2'].median():.4f}"
)

print(
    f"  max    = "
    f"{primary['Linear_Drift_R2'].max():.4f}"
)


# ------------------------------------------------------------
# 15. Saved outputs
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
    f"\nEmpirical summary:\n"
    f"{summary_file}"
)

print(
    f"\nFull ACF table:\n"
    f"{acf_file}"
)

print(
    f"\nFull PSD table:\n"
    f"{psd_file}"
)

print(
    f"\nQuarter stability:\n"
    f"{quarter_file}"
)

print(
    f"\nPrimary-pass configuration summary:\n"
    f"{config_summary_file}"
)

print(
    f"\nACF figures:\n"
    f"{ACF_DIR}"
)

print(
    f"\nPSD figures:\n"
    f"{PSD_DIR}"
)

print(
    f"\nDistribution figures:\n"
    f"{DISTRIBUTION_DIR}"
)

print(
    f"\nQuarter-stability figures:\n"
    f"{QUARTER_DIR}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 1A COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nNo stochastic process has been assumed or fitted."
)

print(
    "The next step is model-family selection using these "
    "empirical marginal, ACF and PSD characteristics."
)