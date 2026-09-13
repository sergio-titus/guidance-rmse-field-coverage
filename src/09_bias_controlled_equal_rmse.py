from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 1F — BIAS-CONTROLLED EQUAL-RMS EMPIRICAL STRUCTURES
# ============================================================
#
# PURPOSE
# -------
# Build a controlled empirical experiment that separates:
#
#   A. TOTAL signed guidance-error structure
#      (already created in Stage 1E)
#
# from
#
#   B. CENTERED spatial structure
#      where all trajectories have:
#
#          mean = 0
#          identical RMS
#
# while preserving each trajectory's normalized spatial ACF
# and standardized marginal shape.
#
# ------------------------------------------------------------
# WHY THIS IS NEEDED
# ------------------------------------------------------------
#
# Scaling the original signed trajectory:
#
#       e*(x) = c e(x)
#
# preserves BOTH:
#
#       - spatial dependence
#       - relative signed bias
#
# Therefore Stage 1E represents realistic total guidance-error
# profiles, but cannot attribute outcome differences to spatial
# dependence alone.
#
# Here we define:
#
#       u(x) = e(x) - mean[e(x)]
#
# and then:
#
#       u*(x) = u(x) * R_target / RMS[u(x)]
#
# Hence:
#
#       mean[u*] = 0
#       RMS[u*]   = R_target
#
# while normalized ACF, e-folding distance, integral scale,
# skewness and excess kurtosis are unchanged.
#
# ------------------------------------------------------------
# TARGET MAGNITUDES
# ------------------------------------------------------------
#
# To avoid arbitrary magnitude choices, the target centered-RMS
# levels are the 28 EMPIRICALLY OBSERVED centered RMS values.
#
# This produces:
#
#       28 empirical structures × 28 empirical centered-RMS levels
#       = 784 controlled combinations
#
# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# No field geometry is introduced here.
# No stochastic covariance model is imposed.
# No arbitrary agronomic threshold is used.
#
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

CANONICAL_FILE = (
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
    / "bias_controlled_equal_rmse"
)

for directory in [
    TABLES_DIR,
    FIGURES_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


PRIMARY_PASSES = [
    1,
    3
]

DX_M = 0.20


# ------------------------------------------------------------
# 2. Utility functions
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


def normalized_acf(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    values = (
        values
        - np.mean(
            values
        )
    )

    n = len(
        values
    )

    if n < 2:

        return np.array([
            np.nan
        ])

    denominator = np.sum(
        values ** 2
    )

    if denominator <= 0:

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
        / denominator
    )

    acf[0] = 1.0

    return acf


def first_crossing_distance(
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

    return (
        (
            indices[0]
            + 1
        )
        * dx
    )


def e_folding_distance(
    acf,
    dx
):

    return first_crossing_distance(
        acf,
        np.exp(-1),
        dx
    )


def integral_correlation_scale(
    acf,
    dx
):

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

    if len(
        zero_indices
    ) > 0:

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

    lag = (
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
            lag
        )

    return np.trapz(
        local_acf,
        lag
    )


def standardized_moments(values):

    values = np.asarray(
        values,
        dtype=float
    )

    centered = (
        values
        - np.mean(
            values
        )
    )

    sd = np.std(
        centered,
        ddof=0
    )

    if sd <= 0:

        return (
            np.nan,
            np.nan
        )

    z = (
        centered
        / sd
    )

    skewness = np.mean(
        z ** 3
    )

    excess_kurtosis = (
        np.mean(
            z ** 4
        )
        - 3.0
    )

    return (
        skewness,
        excess_kurtosis
    )


# ------------------------------------------------------------
# 3. Load empirical primary trajectories
# ------------------------------------------------------------

if not CANONICAL_FILE.exists():

    raise FileNotFoundError(
        f"Canonical trajectory file not found:\n"
        f"{CANONICAL_FILE}"
    )


print(
    "=" * 90
)

print(
    "STAGE 1F — BIAS-CONTROLLED EQUAL-RMS EMPIRICAL STRUCTURES"
)

print(
    "=" * 90
)


df = pd.read_csv(
    CANONICAL_FILE
)


primary = df[
    df[
        "Pass"
    ].isin(
        PRIMARY_PASSES
    )
].copy()


print(
    f"\nLoaded:"
    f"\n{CANONICAL_FILE}"
)

print(
    f"\nPrimary rows: "
    f"{len(primary):,}"
)

print(
    f"Primary trajectories: "
    f"{primary.groupby(['Configuration', 'Pass']).ngroups}"
)


# ------------------------------------------------------------
# 4. Build centered structure catalog
# ------------------------------------------------------------

structure_rows = []

trajectory_store = {}


group_columns = [
    "Configuration",
    "Configuration_Name",
    "Pass"
]


for (
    config,
    config_name,
    pass_number
), group in primary.groupby(
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

    signed_error = group[
        "Signed_Lateral_Error_m"
    ].to_numpy(
        dtype=float
    )


    mean_error = np.mean(
        signed_error
    )


    centered_error = (
        signed_error
        - mean_error
    )


    total_rms = rms(
        signed_error
    )


    centered_rms = rms(
        centered_error
    )


    if centered_rms <= 0:

        raise RuntimeError(
            f"Zero centered RMS for "
            f"{config} Pass {pass_number}"
        )


    bias_fraction_of_total_rms = (
        abs(
            mean_error
        )
        / total_rms
        if total_rms > 0
        else np.nan
    )


    acf = normalized_acf(
        centered_error
    )


    e_fold = e_folding_distance(
        acf,
        DX_M
    )


    integral_scale = integral_correlation_scale(
        acf,
        DX_M
    )


    (
        skewness,
        excess_kurtosis
    ) = standardized_moments(
        centered_error
    )


    trajectory_id = (
        f"{config}_P{pass_number}"
    )


    trajectory_store[
        trajectory_id
    ] = {
        "Configuration":
            config,

        "Configuration_Name":
            config_name,

        "Pass":
            pass_number,

        "X":
            x,

        "Signed_Error":
            signed_error,

        "Mean_Error":
            mean_error,

        "Centered_Error":
            centered_error,

        "Total_RMS":
            total_rms,

        "Centered_RMS":
            centered_rms,

        "ACF":
            acf,

        "E_Folding":
            e_fold,

        "Integral_Scale":
            integral_scale,

        "Skewness":
            skewness,

        "Excess_Kurtosis":
            excess_kurtosis
    }


    structure_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Configuration":
                config,

            "Configuration_Name":
                config_name,

            "Pass":
                pass_number,

            "Original_Total_RMS_m":
                total_rms,

            "Original_Bias_m":
                mean_error,

            "Absolute_Bias_to_Total_RMS_Ratio":
                bias_fraction_of_total_rms,

            "Original_Centered_RMS_m":
                centered_rms,

            "E_Folding_m":
                e_fold,

            "Integral_Correlation_Scale_m":
                integral_scale,

            "Centered_Skewness":
                skewness,

            "Centered_Excess_Kurtosis":
                excess_kurtosis
        }
    )


structure_df = pd.DataFrame(
    structure_rows
)


# ------------------------------------------------------------
# 5. Empirical centered-RMS levels
# ------------------------------------------------------------

target_rms_table = (
    structure_df[
        [
            "Trajectory_ID",
            "Configuration",
            "Pass",
            "Original_Centered_RMS_m"
        ]
    ]
    .copy()
    .sort_values(
        "Original_Centered_RMS_m"
    )
    .reset_index(
        drop=True
    )
)


target_rms_table[
    "Centered_RMS_Level_Rank"
] = (
    np.arange(
        len(
            target_rms_table
        )
    )
    + 1
)


target_rms_table[
    "Target_Centered_RMS_ID"
] = (
    "CRMS_"
    + target_rms_table[
        "Centered_RMS_Level_Rank"
    ]
    .astype(str)
    .str.zfill(2)
)


target_rms_table = target_rms_table[
    [
        "Target_Centered_RMS_ID",
        "Centered_RMS_Level_Rank",
        "Trajectory_ID",
        "Configuration",
        "Pass",
        "Original_Centered_RMS_m"
    ]
].rename(
    columns={
        "Trajectory_ID":
            "Source_RMS_Trajectory_ID",

        "Configuration":
            "Source_RMS_Configuration",

        "Pass":
            "Source_RMS_Pass",

        "Original_Centered_RMS_m":
            "Target_Centered_RMS_m"
    }
)


# ------------------------------------------------------------
# 6. Complete structure × centered-RMS matrix
# ------------------------------------------------------------

controlled_rows = []

verification_rows = []


for trajectory_id, record in trajectory_store.items():

    x = record[
        "X"
    ]

    centered_error = record[
        "Centered_Error"
    ]

    original_centered_rms = record[
        "Centered_RMS"
    ]

    original_acf = record[
        "ACF"
    ]


    for _, target_record in target_rms_table.iterrows():

        target_id = target_record[
            "Target_Centered_RMS_ID"
        ]

        target_rms = float(
            target_record[
                "Target_Centered_RMS_m"
            ]
        )


        scale_factor = (
            target_rms
            / original_centered_rms
        )


        controlled_error = (
            centered_error
            * scale_factor
        )


        achieved_mean = np.mean(
            controlled_error
        )


        achieved_rms = rms(
            controlled_error
        )


        controlled_acf = normalized_acf(
            controlled_error
        )


        controlled_e_fold = e_folding_distance(
            controlled_acf,
            DX_M
        )


        controlled_integral_scale = integral_correlation_scale(
            controlled_acf,
            DX_M
        )


        (
            controlled_skewness,
            controlled_excess_kurtosis
        ) = standardized_moments(
            controlled_error
        )


        common_length = min(
            len(
                original_acf
            ),
            len(
                controlled_acf
            )
        )


        max_acf_difference = np.nanmax(
            np.abs(
                original_acf[
                    :common_length
                ]
                -
                controlled_acf[
                    :common_length
                ]
            )
        )


        verification_rows.append(
            {
                "Structure_Trajectory_ID":
                    trajectory_id,

                "Structure_Configuration":
                    record[
                        "Configuration"
                    ],

                "Structure_Pass":
                    record[
                        "Pass"
                    ],

                "Target_Centered_RMS_ID":
                    target_id,

                "Target_Centered_RMS_m":
                    target_rms,

                "Original_Centered_RMS_m":
                    original_centered_rms,

                "Scale_Factor":
                    scale_factor,

                "Achieved_Mean_m":
                    achieved_mean,

                "Achieved_Centered_RMS_m":
                    achieved_rms,

                "Absolute_Centered_RMS_Error_m":
                    abs(
                        achieved_rms
                        - target_rms
                    ),

                "Original_E_Folding_m":
                    record[
                        "E_Folding"
                    ],

                "Controlled_E_Folding_m":
                    controlled_e_fold,

                "Original_Integral_Scale_m":
                    record[
                        "Integral_Scale"
                    ],

                "Controlled_Integral_Scale_m":
                    controlled_integral_scale,

                "Original_Skewness":
                    record[
                        "Skewness"
                    ],

                "Controlled_Skewness":
                    controlled_skewness,

                "Original_Excess_Kurtosis":
                    record[
                        "Excess_Kurtosis"
                    ],

                "Controlled_Excess_Kurtosis":
                    controlled_excess_kurtosis,

                "Maximum_Absolute_ACF_Difference":
                    max_acf_difference
            }
        )


        for (
            x_value,
            controlled_value
        ) in zip(
            x,
            controlled_error
        ):

            controlled_rows.append(
                {
                    "Structure_Trajectory_ID":
                        trajectory_id,

                    "Structure_Configuration":
                        record[
                            "Configuration"
                        ],

                    "Structure_Configuration_Name":
                        record[
                            "Configuration_Name"
                        ],

                    "Structure_Pass":
                        record[
                            "Pass"
                        ],

                    "Target_Centered_RMS_ID":
                        target_id,

                    "Target_Centered_RMS_m":
                        target_rms,

                    "Scale_Factor":
                        scale_factor,

                    "Easting_m":
                        x_value,

                    "Zero_Mean_Equal_RMS_Error_m":
                        controlled_value
                }
            )


# ------------------------------------------------------------
# 7. Build output DataFrames
# ------------------------------------------------------------

controlled_df = pd.DataFrame(
    controlled_rows
)

verification_df = pd.DataFrame(
    verification_rows
)


# ------------------------------------------------------------
# 8. Save tables
# ------------------------------------------------------------

structure_file = (
    TABLES_DIR
    / "bias_controlled_structure_catalog.csv"
)

target_file = (
    TABLES_DIR
    / "empirical_centered_rms_levels.csv"
)

verification_file = (
    TABLES_DIR
    / "bias_controlled_equal_rmse_verification.csv"
)

controlled_file = (
    TABLES_DIR
    / "bias_controlled_equal_rmse_trajectory_matrix.csv"
)


structure_df.to_csv(
    structure_file,
    index=False
)

target_rms_table.to_csv(
    target_file,
    index=False
)

verification_df.to_csv(
    verification_file,
    index=False
)

controlled_df.to_csv(
    controlled_file,
    index=False
)


# ------------------------------------------------------------
# 9. Numerical invariance diagnostics
# ------------------------------------------------------------

max_abs_mean = verification_df[
    "Achieved_Mean_m"
].abs().max()


max_rms_error = verification_df[
    "Absolute_Centered_RMS_Error_m"
].max()


max_acf_difference = verification_df[
    "Maximum_Absolute_ACF_Difference"
].max()


max_e_fold_difference = (
    verification_df[
        "Controlled_E_Folding_m"
    ]
    -
    verification_df[
        "Original_E_Folding_m"
    ]
).abs().max()


max_integral_difference = (
    verification_df[
        "Controlled_Integral_Scale_m"
    ]
    -
    verification_df[
        "Original_Integral_Scale_m"
    ]
).abs().max()


max_skew_difference = (
    verification_df[
        "Controlled_Skewness"
    ]
    -
    verification_df[
        "Original_Skewness"
    ]
).abs().max()


max_kurtosis_difference = (
    verification_df[
        "Controlled_Excess_Kurtosis"
    ]
    -
    verification_df[
        "Original_Excess_Kurtosis"
    ]
).abs().max()


# ------------------------------------------------------------
# 10. Bias diagnostics from original trajectories
# ------------------------------------------------------------

bias_summary = structure_df[
    [
        "Trajectory_ID",
        "Original_Total_RMS_m",
        "Original_Bias_m",
        "Absolute_Bias_to_Total_RMS_Ratio",
        "Original_Centered_RMS_m",
        "Integral_Correlation_Scale_m"
    ]
].copy()


bias_summary = bias_summary.sort_values(
    "Absolute_Bias_to_Total_RMS_Ratio",
    ascending=False
)


bias_file = (
    TABLES_DIR
    / "empirical_bias_contribution_summary.csv"
)


bias_summary.to_csv(
    bias_file,
    index=False
)


# ------------------------------------------------------------
# 11. Structural diversity after bias control
# ------------------------------------------------------------

diversity_rows = []


for _, target_record in target_rms_table.iterrows():

    target_id = target_record[
        "Target_Centered_RMS_ID"
    ]

    target_rms = float(
        target_record[
            "Target_Centered_RMS_m"
        ]
    )


    subset = verification_df[
        verification_df[
            "Target_Centered_RMS_ID"
        ]
        == target_id
    ]


    diversity_rows.append(
        {
            "Target_Centered_RMS_ID":
                target_id,

            "Target_Centered_RMS_m":
                target_rms,

            "Number_of_Structures":
                len(
                    subset
                ),

            "Minimum_E_Folding_m":
                subset[
                    "Controlled_E_Folding_m"
                ].min(),

            "Median_E_Folding_m":
                subset[
                    "Controlled_E_Folding_m"
                ].median(),

            "Maximum_E_Folding_m":
                subset[
                    "Controlled_E_Folding_m"
                ].max(),

            "Minimum_Integral_Scale_m":
                subset[
                    "Controlled_Integral_Scale_m"
                ].min(),

            "Median_Integral_Scale_m":
                subset[
                    "Controlled_Integral_Scale_m"
                ].median(),

            "Maximum_Integral_Scale_m":
                subset[
                    "Controlled_Integral_Scale_m"
                ].max()
        }
    )


diversity_df = pd.DataFrame(
    diversity_rows
)


diversity_file = (
    TABLES_DIR
    / "bias_controlled_structure_diversity.csv"
)


diversity_df.to_csv(
    diversity_file,
    index=False
)


# ------------------------------------------------------------
# 12. Select representative controlled RMS level
# ------------------------------------------------------------

median_target_index = (
    len(
        target_rms_table
    )
    // 2
)


representative_target = target_rms_table.iloc[
    median_target_index
]


representative_target_id = representative_target[
    "Target_Centered_RMS_ID"
]


representative_target_rms = float(
    representative_target[
        "Target_Centered_RMS_m"
    ]
)


representative_verification = (
    verification_df[
        verification_df[
            "Target_Centered_RMS_ID"
        ]
        == representative_target_id
    ]
    .sort_values(
        "Controlled_Integral_Scale_m"
    )
    .reset_index(
        drop=True
    )
)


minimum_row = representative_verification.iloc[
    0
]


maximum_row = representative_verification.iloc[
    -1
]


median_integral = representative_verification[
    "Controlled_Integral_Scale_m"
].median()


median_index = (
    representative_verification[
        "Controlled_Integral_Scale_m"
    ]
    - median_integral
).abs().idxmin()


median_row = representative_verification.loc[
    median_index
]


selected_ids = [
    minimum_row[
        "Structure_Trajectory_ID"
    ],
    median_row[
        "Structure_Trajectory_ID"
    ],
    maximum_row[
        "Structure_Trajectory_ID"
    ]
]


# ------------------------------------------------------------
# 13. Representative zero-mean equal-RMS figure
# ------------------------------------------------------------

representative_data = controlled_df[
    (
        controlled_df[
            "Target_Centered_RMS_ID"
        ]
        == representative_target_id
    )
    &
    (
        controlled_df[
            "Structure_Trajectory_ID"
        ]
        .isin(
            selected_ids
        )
    )
]


plt.figure(
    figsize=(12, 6)
)


for trajectory_id in selected_ids:

    group = representative_data[
        representative_data[
            "Structure_Trajectory_ID"
        ]
        == trajectory_id
    ].sort_values(
        "Easting_m"
    )


    row = representative_verification[
        representative_verification[
            "Structure_Trajectory_ID"
        ]
        == trajectory_id
    ].iloc[
        0
    ]


    integral_scale = float(
        row[
            "Controlled_Integral_Scale_m"
        ]
    )


    plt.plot(
        group[
            "Easting_m"
        ],
        group[
            "Zero_Mean_Equal_RMS_Error_m"
        ],
        linewidth=1.2,
        label=(
            f"{trajectory_id}, "
            f"L_int={integral_scale:.1f} m"
        )
    )


plt.axhline(
    0,
    linestyle=":",
    linewidth=1
)


plt.xlabel(
    "Along-track position (m)"
)

plt.ylabel(
    "Zero-mean lateral error (m)"
)

plt.title(
    "Bias-controlled empirical structures at identical "
    f"centered RMS ({representative_target_rms:.4f} m)"
)

plt.legend()

plt.tight_layout()


representative_figure = (
    FIGURES_DIR
    / "representative_bias_controlled_equal_rmse_structures.png"
)


plt.savefig(
    representative_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 14. Bias contribution figure
# ------------------------------------------------------------

plot_bias = structure_df.sort_values(
    "Absolute_Bias_to_Total_RMS_Ratio"
).reset_index(
    drop=True
)


plt.figure(
    figsize=(12, 5)
)


plt.bar(
    plot_bias[
        "Trajectory_ID"
    ],
    plot_bias[
        "Absolute_Bias_to_Total_RMS_Ratio"
    ]
)


plt.xticks(
    rotation=90
)

plt.xlabel(
    "Empirical primary trajectory"
)

plt.ylabel(
    "|bias| / total RMS"
)

plt.title(
    "Relative contribution of signed bias to empirical guidance RMSE"
)

plt.tight_layout()


bias_figure = (
    FIGURES_DIR
    / "empirical_bias_fraction_of_total_rmse.png"
)


plt.savefig(
    bias_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 15. Terminal formatting
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


# ------------------------------------------------------------
# 16. Print structure catalog
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "EMPIRICAL BIAS / CENTERED-STRUCTURE CATALOG"
)

print(
    "=" * 90
)


print(
    structure_df[
        [
            "Trajectory_ID",
            "Original_Total_RMS_m",
            "Original_Bias_m",
            "Absolute_Bias_to_Total_RMS_Ratio",
            "Original_Centered_RMS_m",
            "E_Folding_m",
            "Integral_Correlation_Scale_m",
            "Centered_Skewness",
            "Centered_Excess_Kurtosis"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 17. Print centered RMS levels
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "EMPIRICALLY OBSERVED CENTERED-RMS LEVELS"
)

print(
    "=" * 90
)


print(
    target_rms_table[
        [
            "Target_Centered_RMS_ID",
            "Centered_RMS_Level_Rank",
            "Source_RMS_Trajectory_ID",
            "Target_Centered_RMS_m"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 18. Invariance checks
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "BIAS-CONTROLLED NORMALIZATION — NUMERICAL INVARIANCE CHECK"
)

print(
    "=" * 90
)


print(
    f"\nTotal structure × centered-RMS combinations: "
    f"{len(verification_df):,}"
)


print(
    f"\nMaximum absolute achieved mean:"
    f"\n  {max_abs_mean:.12e} m"
)


print(
    f"\nMaximum absolute centered-RMS error:"
    f"\n  {max_rms_error:.12e} m"
)


print(
    f"\nMaximum absolute ACF difference:"
    f"\n  {max_acf_difference:.12e}"
)


print(
    f"\nMaximum absolute e-folding-distance difference:"
    f"\n  {max_e_fold_difference:.12e} m"
)


print(
    f"\nMaximum absolute integral-scale difference:"
    f"\n  {max_integral_difference:.12e} m"
)


print(
    f"\nMaximum absolute skewness difference:"
    f"\n  {max_skew_difference:.12e}"
)


print(
    f"\nMaximum absolute excess-kurtosis difference:"
    f"\n  {max_kurtosis_difference:.12e}"
)


# ------------------------------------------------------------
# 19. Original bias contribution
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "ORIGINAL BIAS CONTRIBUTION TO TOTAL RMSE"
)

print(
    "=" * 90
)


bias_ratio = structure_df[
    "Absolute_Bias_to_Total_RMS_Ratio"
]


print(
    f"\n|bias| / total RMS:"
)

print(
    f"  min    = "
    f"{bias_ratio.min():.6f}"
)

print(
    f"  median = "
    f"{bias_ratio.median():.6f}"
)

print(
    f"  max    = "
    f"{bias_ratio.max():.6f}"
)


print(
    "\nFive most bias-dominated empirical trajectories:"
)


print(
    bias_summary.head(
        5
    ).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 20. Controlled structural diversity
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "STRUCTURAL DIVERSITY AFTER ZERO-MEAN BIAS CONTROL"
)

print(
    "=" * 90
)


print(
    diversity_df[
        [
            "Target_Centered_RMS_ID",
            "Target_Centered_RMS_m",
            "Number_of_Structures",
            "Minimum_E_Folding_m",
            "Median_E_Folding_m",
            "Maximum_E_Folding_m",
            "Minimum_Integral_Scale_m",
            "Median_Integral_Scale_m",
            "Maximum_Integral_Scale_m"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 21. Representative controlled structures
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "REPRESENTATIVE ZERO-MEAN EQUAL-RMS STRUCTURES"
)

print(
    "=" * 90
)


print(
    f"\nRepresentative centered-RMS level:"
)

print(
    f"  {representative_target_id}"
    f" = "
    f"{representative_target_rms:.6f} m"
)


print(
    "\nSelected empirical structures:"
)


for trajectory_id in selected_ids:

    row = representative_verification[
        representative_verification[
            "Structure_Trajectory_ID"
        ]
        == trajectory_id
    ].iloc[
        0
    ]

    print(
        f"  {trajectory_id}: "
        f"e-folding = "
        f"{row['Controlled_E_Folding_m']:.3f} m, "
        f"integral scale = "
        f"{row['Controlled_Integral_Scale_m']:.3f} m"
    )


# ------------------------------------------------------------
# 22. Output paths
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
    f"\nBias-controlled structure catalog:\n"
    f"{structure_file}"
)

print(
    f"\nEmpirical centered-RMS levels:\n"
    f"{target_file}"
)

print(
    f"\nNormalization verification:\n"
    f"{verification_file}"
)

print(
    f"\nBias-controlled trajectory matrix:\n"
    f"{controlled_file}"
)

print(
    f"\nBias contribution summary:\n"
    f"{bias_file}"
)

print(
    f"\nStructural diversity:\n"
    f"{diversity_file}"
)

print(
    f"\nRepresentative controlled figure:\n"
    f"{representative_figure}"
)

print(
    f"\nBias contribution figure:\n"
    f"{bias_figure}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 1F COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nWe now have two complementary empirical experiments:"
)

print(
    "  A. Stage 1E: equal TOTAL RMS, preserving realistic bias + structure."
)

print(
    "  B. Stage 1F: equal CENTERED RMS with exactly zero mean, "
    "isolating variation in centered empirical structure."
)

print(
    "\nNo field geometry or agronomic threshold has yet been imposed."
)

print(
    "The next stage can therefore define field-operation geometry "
    "without confounding magnitude with systematic bias."
)