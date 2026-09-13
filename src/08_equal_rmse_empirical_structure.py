from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 1E — EQUAL-RMSE EMPIRICAL STRUCTURE EXPERIMENT
# ============================================================
#
# PURPOSE
# -------
# Test the central scientific premise:
#
#   trajectories can have the SAME RMS magnitude while retaining
#   DIFFERENT empirically observed spatial structures.
#
# This stage does NOT perform field-coverage simulation yet.
#
# Instead, it constructs controlled equal-RMS versions of the
# primary empirical guidance trajectories and verifies that
# normalization preserves:
#
#   - ACF
#   - integral correlation scale
#   - e-folding distance
#   - skewness
#   - excess kurtosis
#
# ------------------------------------------------------------
# IMPORTANT DESIGN PRINCIPLE
# ------------------------------------------------------------
#
# For each empirical signed trajectory e(x):
#
#           e*(x) = e(x) * R_target / RMS[e(x)]
#
# Multiplication by a positive constant changes magnitude but
# does NOT change the normalized autocorrelation structure.
#
# Therefore equal-RMS normalization provides a controlled way
# to isolate the effect of STRUCTURE from the effect of RMS.
#
# ------------------------------------------------------------
# TARGET RMS
# ------------------------------------------------------------
#
# No arbitrary target RMS is imposed here.
#
# Every unique empirical primary-pass RMS becomes a possible
# target level.
#
# Thus the experiment generates a complete matrix:
#
#       empirical structure × empirical RMS level
#
# This allows the later coverage stage to ask:
#
#       At the same RMS, how much do operational consequences
#       vary solely because the spatial error structure differs?
#
# ------------------------------------------------------------
# PRIMARY DATA
# ------------------------------------------------------------
#
# Pass 1 and Pass 3 only.
#
# Pass 2 remains diagnostic because it contributed to the
# relative reference construction in the source study.
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
    / "equal_rmse_empirical_structure"
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

def rms(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    return np.sqrt(
        np.mean(
            values ** 2
        )
    )


def normalized_acf(
    values
):

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

    if denominator == 0:

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

    acf[
        0
    ] = 1.0

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

    if len(
        acf
    ) < 2:

        return np.nan

    indices = np.where(
        acf[
            1:
        ]
        <= threshold
    )[0]

    if len(
        indices
    ) == 0:

        return np.nan

    return (
        (
            indices[
                0
            ]
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
        np.exp(
            -1
        ),
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

    if len(
        acf
    ) < 2:

        return np.nan

    zero_indices = np.where(
        acf[
            1:
        ]
        <= 0
    )[0]

    if len(
        zero_indices
    ) > 0:

        end_index = (
            zero_indices[
                0
            ]
            + 1
        )

    else:

        end_index = (
            len(
                acf
            )
            - 1
        )

    local_acf = acf[
        :end_index + 1
    ]

    lag = (
        np.arange(
            len(
                local_acf
            )
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


def standardized_moments(
    values
):

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

    if sd == 0:

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
# 3. Load canonical trajectories
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
    "STAGE 1E — EQUAL-RMSE EMPIRICAL STRUCTURE EXPERIMENT"
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
    f"\nPrimary trajectory rows: "
    f"{len(primary):,}"
)

print(
    f"Configurations: "
    f"{primary['Configuration'].nunique()}"
)

print(
    f"Primary empirical trajectories: "
    f"{primary.groupby(['Configuration', 'Pass']).ngroups}"
)


# ------------------------------------------------------------
# 4. Characterize original empirical structures
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

    error = group[
        "Signed_Lateral_Error_m"
    ].to_numpy(
        dtype=float
    )


    trajectory_id = (
        f"{config}_P{pass_number}"
    )


    original_rms = rms(
        error
    )


    acf = normalized_acf(
        error
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
        error
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

        "Error":
            error,

        "Original_RMS":
            original_rms,

        "Original_ACF":
            acf,

        "Original_E_Folding":
            e_fold,

        "Original_Integral_Scale":
            integral_scale,

        "Original_Skewness":
            skewness,

        "Original_Excess_Kurtosis":
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

            "Original_RMS_m":
                original_rms,

            "Original_Mean_m":
                np.mean(
                    error
                ),

            "Original_SD_m":
                np.std(
                    error,
                    ddof=1
                ),

            "Original_E_Folding_m":
                e_fold,

            "Original_Integral_Correlation_Scale_m":
                integral_scale,

            "Original_Skewness":
                skewness,

            "Original_Excess_Kurtosis":
                excess_kurtosis
        }
    )


structure_df = pd.DataFrame(
    structure_rows
)


# ------------------------------------------------------------
# 5. Empirical target RMS levels
# ------------------------------------------------------------

target_rms_table = (
    structure_df[
        [
            "Trajectory_ID",
            "Configuration",
            "Pass",
            "Original_RMS_m"
        ]
    ]
    .copy()
    .sort_values(
        "Original_RMS_m"
    )
    .reset_index(
        drop=True
    )
)


target_rms_table[
    "RMS_Level_Rank"
] = (
    np.arange(
        len(
            target_rms_table
        )
    )
    + 1
)


target_rms_table[
    "Target_RMS_ID"
] = (
    "RMS_"
    + target_rms_table[
        "RMS_Level_Rank"
    ]
    .astype(str)
    .str.zfill(2)
)


target_rms_table = target_rms_table[
    [
        "Target_RMS_ID",
        "RMS_Level_Rank",
        "Trajectory_ID",
        "Configuration",
        "Pass",
        "Original_RMS_m"
    ]
].rename(
    columns={
        "Trajectory_ID":
            "Source_RMS_Trajectory_ID",

        "Configuration":
            "Source_RMS_Configuration",

        "Pass":
            "Source_RMS_Pass",

        "Original_RMS_m":
            "Target_RMS_m"
    }
)


# ------------------------------------------------------------
# 6. Generate complete structure × RMS matrix
# ------------------------------------------------------------

normalized_rows = []

verification_rows = []


for trajectory_id, record in trajectory_store.items():

    x = record[
        "X"
    ]

    original_error = record[
        "Error"
    ]

    original_rms = record[
        "Original_RMS"
    ]

    original_acf = record[
        "Original_ACF"
    ]

    original_e_fold = record[
        "Original_E_Folding"
    ]

    original_integral_scale = record[
        "Original_Integral_Scale"
    ]

    original_skewness = record[
        "Original_Skewness"
    ]

    original_excess_kurtosis = record[
        "Original_Excess_Kurtosis"
    ]


    if original_rms <= 0:

        raise RuntimeError(
            f"Non-positive RMS for "
            f"{trajectory_id}"
        )


    for _, target_record in target_rms_table.iterrows():

        target_rms_id = target_record[
            "Target_RMS_ID"
        ]

        target_rms = float(
            target_record[
                "Target_RMS_m"
            ]
        )


        scale_factor = (
            target_rms
            / original_rms
        )


        normalized_error = (
            original_error
            * scale_factor
        )


        achieved_rms = rms(
            normalized_error
        )


        normalized_acf_values = normalized_acf(
            normalized_error
        )


        normalized_e_fold = e_folding_distance(
            normalized_acf_values,
            DX_M
        )


        normalized_integral_scale = integral_correlation_scale(
            normalized_acf_values,
            DX_M
        )


        (
            normalized_skewness,
            normalized_excess_kurtosis
        ) = standardized_moments(
            normalized_error
        )


        common_length = min(
            len(
                original_acf
            ),
            len(
                normalized_acf_values
            )
        )


        acf_max_abs_difference = np.nanmax(
            np.abs(
                original_acf[
                    :common_length
                ]
                -
                normalized_acf_values[
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

                "Target_RMS_ID":
                    target_rms_id,

                "Target_RMS_m":
                    target_rms,

                "Original_Structure_RMS_m":
                    original_rms,

                "Scale_Factor":
                    scale_factor,

                "Achieved_RMS_m":
                    achieved_rms,

                "Absolute_RMS_Error_m":
                    abs(
                        achieved_rms
                        - target_rms
                    ),

                "Original_E_Folding_m":
                    original_e_fold,

                "Normalized_E_Folding_m":
                    normalized_e_fold,

                "Absolute_E_Folding_Difference_m":
                    (
                        abs(
                            normalized_e_fold
                            - original_e_fold
                        )
                        if (
                            np.isfinite(
                                normalized_e_fold
                            )
                            and np.isfinite(
                                original_e_fold
                            )
                        )
                        else np.nan
                    ),

                "Original_Integral_Scale_m":
                    original_integral_scale,

                "Normalized_Integral_Scale_m":
                    normalized_integral_scale,

                "Absolute_Integral_Scale_Difference_m":
                    (
                        abs(
                            normalized_integral_scale
                            - original_integral_scale
                        )
                        if (
                            np.isfinite(
                                normalized_integral_scale
                            )
                            and np.isfinite(
                                original_integral_scale
                            )
                        )
                        else np.nan
                    ),

                "Original_Skewness":
                    original_skewness,

                "Normalized_Skewness":
                    normalized_skewness,

                "Absolute_Skewness_Difference":
                    abs(
                        normalized_skewness
                        - original_skewness
                    ),

                "Original_Excess_Kurtosis":
                    original_excess_kurtosis,

                "Normalized_Excess_Kurtosis":
                    normalized_excess_kurtosis,

                "Absolute_Excess_Kurtosis_Difference":
                    abs(
                        normalized_excess_kurtosis
                        - original_excess_kurtosis
                    ),

                "Maximum_Absolute_ACF_Difference":
                    acf_max_abs_difference
            }
        )


        for (
            x_value,
            normalized_value
        ) in zip(
            x,
            normalized_error
        ):

            normalized_rows.append(
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

                    "Target_RMS_ID":
                        target_rms_id,

                    "Target_RMS_m":
                        target_rms,

                    "Scale_Factor":
                        scale_factor,

                    "Easting_m":
                        x_value,

                    "Equal_RMS_Error_m":
                        normalized_value
                }
            )


# ------------------------------------------------------------
# 7. DataFrames
# ------------------------------------------------------------

normalized_df = pd.DataFrame(
    normalized_rows
)

verification_df = pd.DataFrame(
    verification_rows
)


# ------------------------------------------------------------
# 8. Save outputs
# ------------------------------------------------------------

structure_file = (
    TABLES_DIR
    / "empirical_primary_structure_catalog.csv"
)

target_file = (
    TABLES_DIR
    / "empirical_target_rms_levels.csv"
)

verification_file = (
    TABLES_DIR
    / "equal_rmse_structure_preservation_verification.csv"
)

normalized_file = (
    TABLES_DIR
    / "equal_rmse_empirical_trajectory_matrix.csv"
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

normalized_df.to_csv(
    normalized_file,
    index=False
)


# ------------------------------------------------------------
# 9. Numerical verification summary
# ------------------------------------------------------------

max_rms_error = verification_df[
    "Absolute_RMS_Error_m"
].max()


max_acf_difference = verification_df[
    "Maximum_Absolute_ACF_Difference"
].max()


max_integral_difference = verification_df[
    "Absolute_Integral_Scale_Difference_m"
].max()


max_e_fold_difference = verification_df[
    "Absolute_E_Folding_Difference_m"
].max()


max_skew_difference = verification_df[
    "Absolute_Skewness_Difference"
].max()


max_kurtosis_difference = verification_df[
    "Absolute_Excess_Kurtosis_Difference"
].max()


# ------------------------------------------------------------
# 10. Structure diversity at equal RMS
# ------------------------------------------------------------

structure_diversity_rows = []


for _, target_record in target_rms_table.iterrows():

    target_rms_id = target_record[
        "Target_RMS_ID"
    ]

    target_rms = float(
        target_record[
            "Target_RMS_m"
        ]
    )


    subset = verification_df[
        verification_df[
            "Target_RMS_ID"
        ]
        == target_rms_id
    ]


    structure_diversity_rows.append(
        {
            "Target_RMS_ID":
                target_rms_id,

            "Target_RMS_m":
                target_rms,

            "Number_of_Structures":
                len(
                    subset
                ),

            "Minimum_E_Folding_m":
                subset[
                    "Normalized_E_Folding_m"
                ].min(),

            "Median_E_Folding_m":
                subset[
                    "Normalized_E_Folding_m"
                ].median(),

            "Maximum_E_Folding_m":
                subset[
                    "Normalized_E_Folding_m"
                ].max(),

            "Minimum_Integral_Scale_m":
                subset[
                    "Normalized_Integral_Scale_m"
                ].min(),

            "Median_Integral_Scale_m":
                subset[
                    "Normalized_Integral_Scale_m"
                ].median(),

            "Maximum_Integral_Scale_m":
                subset[
                    "Normalized_Integral_Scale_m"
                ].max()
        }
    )


structure_diversity_df = pd.DataFrame(
    structure_diversity_rows
)


structure_diversity_file = (
    TABLES_DIR
    / "equal_rmse_structure_diversity.csv"
)


structure_diversity_df.to_csv(
    structure_diversity_file,
    index=False
)


# ------------------------------------------------------------
# 11. Representative figure selection
# ------------------------------------------------------------
#
# Figure uses the median empirical RMS level.
#
# At that SAME RMS we plot structures having approximately:
#
#   minimum integral scale
#   median integral scale
#   maximum integral scale
#
# Selection is data-driven.
#
# ------------------------------------------------------------

median_target_index = (
    len(
        target_rms_table
    )
    // 2
)


median_target_record = target_rms_table.iloc[
    median_target_index
]


representative_rms_id = median_target_record[
    "Target_RMS_ID"
]


representative_rms = float(
    median_target_record[
        "Target_RMS_m"
    ]
)


representative_verification = verification_df[
    verification_df[
        "Target_RMS_ID"
    ]
    == representative_rms_id
].sort_values(
    "Normalized_Integral_Scale_m"
).reset_index(
    drop=True
)


minimum_row = representative_verification.iloc[
    0
]


maximum_row = representative_verification.iloc[
    -1
]


median_integral_value = representative_verification[
    "Normalized_Integral_Scale_m"
].median()


median_row_index = (
    representative_verification[
        "Normalized_Integral_Scale_m"
    ]
    - median_integral_value
).abs().idxmin()


median_row = representative_verification.loc[
    median_row_index
]


selected_structure_ids = [
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


representative_df = normalized_df[
    (
        normalized_df[
            "Target_RMS_ID"
        ]
        == representative_rms_id
    )
    &
    (
        normalized_df[
            "Structure_Trajectory_ID"
        ]
        .isin(
            selected_structure_ids
        )
    )
]


plt.figure(
    figsize=(12, 6)
)


for trajectory_id in selected_structure_ids:

    group = representative_df[
        representative_df[
            "Structure_Trajectory_ID"
        ]
        == trajectory_id
    ].sort_values(
        "Easting_m"
    )

    integral_value = float(
        representative_verification[
            representative_verification[
                "Structure_Trajectory_ID"
            ]
            == trajectory_id
        ][
            "Normalized_Integral_Scale_m"
        ].iloc[
            0
        ]
    )


    plt.plot(
        group[
            "Easting_m"
        ],
        group[
            "Equal_RMS_Error_m"
        ],
        linewidth=1.2,
        label=(
            f"{trajectory_id}, "
            f"L_int={integral_value:.1f} m"
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
    "Signed lateral guidance error (m)"
)

plt.title(
    "Empirical guidance-error structures at identical RMS "
    f"({representative_rms:.4f} m)"
)

plt.legend()

plt.tight_layout()


representative_figure_file = (
    FIGURES_DIR
    / "representative_equal_rmse_structures.png"
)


plt.savefig(
    representative_figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 12. RMS distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


ordered_rms = target_rms_table.sort_values(
    "Target_RMS_m"
)


plt.plot(
    np.arange(
        1,
        len(
            ordered_rms
        )
        + 1
    ),
    ordered_rms[
        "Target_RMS_m"
    ],
    marker="o",
    linewidth=1
)


plt.xlabel(
    "Empirical RMS level rank"
)

plt.ylabel(
    "Target RMS (m)"
)

plt.title(
    "Empirically observed RMS levels used in the "
    "structure × magnitude experiment"
)

plt.tight_layout()


rms_figure_file = (
    FIGURES_DIR
    / "empirical_rms_levels.png"
)


plt.savefig(
    rms_figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 13. Terminal formatting
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
# 14. Print empirical structure catalog
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "EMPIRICAL PRIMARY STRUCTURE CATALOG"
)

print(
    "=" * 90
)


print(
    structure_df[
        [
            "Trajectory_ID",
            "Original_RMS_m",
            "Original_Mean_m",
            "Original_SD_m",
            "Original_E_Folding_m",
            "Original_Integral_Correlation_Scale_m",
            "Original_Skewness",
            "Original_Excess_Kurtosis"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 15. Print target RMS levels
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "EMPIRICALLY OBSERVED RMS LEVELS"
)

print(
    "=" * 90
)


print(
    target_rms_table[
        [
            "Target_RMS_ID",
            "RMS_Level_Rank",
            "Source_RMS_Trajectory_ID",
            "Target_RMS_m"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 16. Numerical invariance verification
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "EQUAL-RMS NORMALIZATION — NUMERICAL INVARIANCE CHECK"
)

print(
    "=" * 90
)


print(
    f"\nTotal structure × RMS combinations: "
    f"{len(verification_df):,}"
)


print(
    f"\nMaximum absolute target-RMS error:"
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
# 17. Equal-RMS structural diversity
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "STRUCTURAL DIVERSITY AVAILABLE AT EVERY IDENTICAL RMS LEVEL"
)

print(
    "=" * 90
)


print(
    structure_diversity_df[
        [
            "Target_RMS_ID",
            "Target_RMS_m",
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
# 18. Representative structures
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DATA-DRIVEN REPRESENTATIVE EQUAL-RMS STRUCTURES"
)

print(
    "=" * 90
)


print(
    f"\nRepresentative target RMS:"
    f"\n  {representative_rms_id}"
    f" = {representative_rms:.6f} m"
)


print(
    "\nSelected structures:"
)


for trajectory_id in selected_structure_ids:

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
        f"{row['Normalized_E_Folding_m']:.3f} m, "
        f"integral scale = "
        f"{row['Normalized_Integral_Scale_m']:.3f} m"
    )


# ------------------------------------------------------------
# 19. Output paths
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
    f"\nEmpirical structure catalog:\n"
    f"{structure_file}"
)

print(
    f"\nEmpirical RMS levels:\n"
    f"{target_file}"
)

print(
    f"\nNormalization verification:\n"
    f"{verification_file}"
)

print(
    f"\nEqual-RMS trajectory matrix:\n"
    f"{normalized_file}"
)

print(
    f"\nStructural diversity table:\n"
    f"{structure_diversity_file}"
)

print(
    f"\nRepresentative figure:\n"
    f"{representative_figure_file}"
)

print(
    f"\nRMS-level figure:\n"
    f"{rms_figure_file}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 1E COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nNo field-coverage assumptions have been introduced yet."
)

print(
    "This stage isolates error magnitude from empirically "
    "observed spatial structure."
)

print(
    "\nThe next stage will define the field-operation geometry "
    "and perform numerical-convergence testing before Monte "
    "Carlo coverage experiments."
)