from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# STAGE 4E — FINAL GEOMETRY + ALIGNMENT AUDIT
# =====================================================================
#
# PURPOSE
# -------
# This is a documentation / consistency audit only.
#
# It introduces:
#   - no new model,
#   - no new stochastic assumption,
#   - no new simulation experiment,
#   - no new threshold.
#
# It formally documents:
#
#   A. Stage 4B analytical coverage geometry versus exact Shapely
#      polygon geometry.
#
#   B. The distinction between:
#
#        K_aligned
#          = discrepancy at the actually recorded spatial alignment
#            of two empirical trajectory profiles;
#
#        K_phase
#          = uniform relative-phase DESIGN average used in the
#            operational tolerance framework.
#
#   C. Whether the distribution of actually aligned empirical pairs
#      appears systematically different from the phase-design value.
#
# IMPORTANT
# ---------
# The 378 trajectory pairs are dependent combinations because each
# trajectory occurs in multiple pairs.
#
# Therefore:
#   - aligned-pair summaries are DESCRIPTIVE;
#   - no p-values are calculated;
#   - no CI is calculated by pretending that the 378 pairs are
#     independent observations.
#
# Configuration-cluster uncertainty for the headline coefficients was
# already handled separately in Stage 4D.
#
# =====================================================================


# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

TABLES_DIR = (
    PROJECT_DIR
    / "results"
    / "tables"
)

FIGURES_DIR = (
    PROJECT_DIR
    / "results"
    / "figures"
    / "final_geometry_alignment_audit"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Stage 4B
GEOMETRY_FILE = (
    TABLES_DIR
    / "geometric_coverage_validation_all_pairs.csv"
)

GEOMETRY_REPRESENTATIVE_FILE = (
    TABLES_DIR
    / "geometric_coverage_validation_representative_cases.csv"
)

RASTER_FILE = (
    TABLES_DIR
    / "geometric_coverage_raster_convergence.csv"
)


# Stage 4D
ALIGNED_FILE = (
    TABLES_DIR
    / "corrected_aligned_pair_persistence.csv"
)

FINAL_COEFFICIENT_FILE = (
    TABLES_DIR
    / "corrected_final_coefficient_summary.csv"
)

FINAL_OPERATIONAL_FILE = (
    TABLES_DIR
    / "corrected_final_operational_coefficients.csv"
)


# ---------------------------------------------------------------------
# 2. Validate files
# ---------------------------------------------------------------------

required_files = [
    GEOMETRY_FILE,
    ALIGNED_FILE,
    FINAL_COEFFICIENT_FILE,
    FINAL_OPERATIONAL_FILE
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file_path}"
        )


print(
    "=" * 100
)

print(
    "STAGE 4E — FINAL GEOMETRY + ALIGNMENT AUDIT"
)

print(
    "=" * 100
)


# ---------------------------------------------------------------------
# 3. Load files
# ---------------------------------------------------------------------

geometry_df = pd.read_csv(
    GEOMETRY_FILE
)

aligned_df = pd.read_csv(
    ALIGNED_FILE
)

coefficient_df = pd.read_csv(
    FINAL_COEFFICIENT_FILE
)

operational_df = pd.read_csv(
    FINAL_OPERATIONAL_FILE
)


print(
    f"\nStage 4B geometry rows:"
    f"\n  {len(geometry_df):,}"
)

print(
    f"\nStage 4D aligned-persistence rows:"
    f"\n  {len(aligned_df):,}"
)


# ---------------------------------------------------------------------
# 4. Column-discovery helper
# ---------------------------------------------------------------------

def find_column(
    df,
    candidates,
    description
):

    for candidate in candidates:

        if candidate in df.columns:

            return candidate


    raise KeyError(
        f"\nCould not identify {description}.\n"
        f"Candidates:\n"
        f"{candidates}\n\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


# ---------------------------------------------------------------------
# 5. Stage 4B geometry columns
# ---------------------------------------------------------------------

analytic_skip_col = find_column(
    geometry_df,
    [
        "Analytical_Skip_Fraction",
        "Analytic_Skip_Fraction",
        "Analytical_Inside_Skip_Fraction",
        "Analytic_Inside_Skip_Fraction"
    ],
    "analytical skip fraction"
)


shapely_skip_col = find_column(
    geometry_df,
    [
        "Shapely_Skip_Fraction",
        "Exact_Shapely_Skip_Fraction",
        "Shapely_Inside_Skip_Fraction"
    ],
    "Shapely skip fraction"
)


analytic_overlap_col = find_column(
    geometry_df,
    [
        "Analytical_Overlap_Fraction",
        "Analytic_Overlap_Fraction",
        "Analytical_Inside_Overlap_Fraction",
        "Analytic_Inside_Overlap_Fraction"
    ],
    "analytical overlap fraction"
)


shapely_overlap_col = find_column(
    geometry_df,
    [
        "Shapely_Overlap_Fraction",
        "Exact_Shapely_Overlap_Fraction",
        "Shapely_Inside_Overlap_Fraction"
    ],
    "Shapely overlap fraction"
)


analytic_outside_col = find_column(
    geometry_df,
    [
        "Analytical_Outside_Application_Fraction",
        "Analytic_Outside_Application_Fraction",
        "Analytical_Outside_Fraction",
        "Analytic_Outside_Fraction"
    ],
    "analytical outside-application fraction"
)


shapely_outside_col = find_column(
    geometry_df,
    [
        "Shapely_Outside_Application_Fraction",
        "Exact_Shapely_Outside_Application_Fraction",
        "Shapely_Outside_Fraction"
    ],
    "Shapely outside-application fraction"
)


analytic_total_col = find_column(
    geometry_df,
    [
        "Analytical_Total_Discrepancy_Fraction",
        "Analytic_Total_Discrepancy_Fraction",
        "Analytical_Total_Fraction",
        "Analytic_Total_Fraction"
    ],
    "analytical total-discrepancy fraction"
)


shapely_total_col = find_column(
    geometry_df,
    [
        "Shapely_Total_Discrepancy_Fraction",
        "Exact_Shapely_Total_Discrepancy_Fraction",
        "Shapely_Total_Fraction"
    ],
    "Shapely total-discrepancy fraction"
)


# ---------------------------------------------------------------------
# 6. Compute Stage 4B closure errors independently
# ---------------------------------------------------------------------

geometry_df[
    "Audit_Abs_Skip_Error"
] = np.abs(
    geometry_df[
        analytic_skip_col
    ]
    -
    geometry_df[
        shapely_skip_col
    ]
)


geometry_df[
    "Audit_Abs_Overlap_Error"
] = np.abs(
    geometry_df[
        analytic_overlap_col
    ]
    -
    geometry_df[
        shapely_overlap_col
    ]
)


geometry_df[
    "Audit_Abs_Outside_Error"
] = np.abs(
    geometry_df[
        analytic_outside_col
    ]
    -
    geometry_df[
        shapely_outside_col
    ]
)


geometry_df[
    "Audit_Abs_Total_Error"
] = np.abs(
    geometry_df[
        analytic_total_col
    ]
    -
    geometry_df[
        shapely_total_col
    ]
)


geometry_summary_rows = []


for metric, column in [
    (
        "Skip fraction",
        "Audit_Abs_Skip_Error"
    ),
    (
        "Overlap fraction",
        "Audit_Abs_Overlap_Error"
    ),
    (
        "Outside-application fraction",
        "Audit_Abs_Outside_Error"
    ),
    (
        "Total coverage-count discrepancy",
        "Audit_Abs_Total_Error"
    )
]:

    geometry_summary_rows.append(
        {
            "Metric":
                metric,

            "Number_of_Pairs":
                len(
                    geometry_df
                ),

            "Median_Absolute_Error":
                geometry_df[
                    column
                ].median(),

            "Maximum_Absolute_Error":
                geometry_df[
                    column
                ].max()
        }
    )


geometry_summary_df = pd.DataFrame(
    geometry_summary_rows
)


# ---------------------------------------------------------------------
# 7. Verify number of geometry pairs
# ---------------------------------------------------------------------

if len(
    geometry_df
) != 378:

    print(
        "\nWARNING:"
        f" Stage 4B contains {len(geometry_df)} rows,"
        " not 378."
    )


# ---------------------------------------------------------------------
# 8. Stage 4D aligned-pair data
# ---------------------------------------------------------------------

required_aligned_columns = [
    "Marginal_Type",
    "Trajectory_1",
    "Trajectory_2",
    "K_Aligned_Integrated"
]


for column in required_aligned_columns:

    if column not in aligned_df.columns:

        raise KeyError(
            f"\nMissing Stage 4D aligned column:\n"
            f"{column}\n\n"
            f"Available columns:\n"
            f"{list(aligned_df.columns)}"
        )


original_aligned_df = aligned_df[
    aligned_df[
        "Marginal_Type"
    ]
    == "Original_Centered"
].copy()


common_aligned_df = aligned_df[
    aligned_df[
        "Marginal_Type"
    ]
    == "Common_Marginal"
].copy()


print(
    "\nAligned pair counts:"
)

print(
    f"  original centered = "
    f"{len(original_aligned_df):,}"
)

print(
    f"  common marginal   = "
    f"{len(common_aligned_df):,}"
)


if len(
    original_aligned_df
) != 378:

    raise RuntimeError(
        "\nExpected 378 original-centered aligned pairs."
    )


if len(
    common_aligned_df
) != 378:

    raise RuntimeError(
        "\nExpected 378 common-marginal aligned pairs."
    )


# ---------------------------------------------------------------------
# 9. Retrieve final corrected phase-design K
# ---------------------------------------------------------------------

if "Coefficient" not in coefficient_df.columns:

    raise KeyError(
        "\nFinal coefficient file has no 'Coefficient' column."
    )


if "Corrected_Point_Estimate" not in coefficient_df.columns:

    raise KeyError(
        "\nFinal coefficient file has no "
        "'Corrected_Point_Estimate' column."
    )


K_row = coefficient_df[
    coefficient_df[
        "Coefficient"
    ].astype(
        str
    ).str.upper()
    == "K"
]


if len(
    K_row
) != 1:

    raise RuntimeError(
        "\nCould not uniquely retrieve final corrected K."
    )


K_PHASE_ORIGINAL = float(
    K_row[
        "Corrected_Point_Estimate"
    ].iloc[
        0
    ]
)


# Corrected common-marginal K is not necessarily stored in the final
# coefficient summary, so use the Stage 4D value if no explicit column
# exists.

COMMON_K_CANDIDATES = [
    "Corrected_Common_Marginal_K",
    "Common_Marginal_K",
    "K_Common"
]


K_PHASE_COMMON = None


for candidate in COMMON_K_CANDIDATES:

    if candidate in coefficient_df.columns:

        value = coefficient_df[
            candidate
        ].dropna()

        if len(
            value
        ) > 0:

            K_PHASE_COMMON = float(
                value.iloc[
                    0
                ]
            )

            break


# Stage 4D console result.
# This is used only if the CSV does not explicitly store the value.

if K_PHASE_COMMON is None:

    K_PHASE_COMMON = 1.128416958486


# ---------------------------------------------------------------------
# 10. Descriptive aligned-K summary
# ---------------------------------------------------------------------

def descriptive_summary(
    values,
    marginal_type,
    phase_K
):

    values = np.asarray(
        values,
        dtype=float
    )


    mean_value = float(
        np.mean(
            values
        )
    )


    median_value = float(
        np.median(
            values
        )
    )


    return {
        "Marginal_Type":
            marginal_type,

        "Number_of_Dependent_Combinations":
            len(
                values
            ),

        "Minimum_Aligned_K":
            float(
                np.min(
                    values
                )
            ),

        "Q05_Aligned_K":
            float(
                np.quantile(
                    values,
                    0.05
                )
            ),

        "Q25_Aligned_K":
            float(
                np.quantile(
                    values,
                    0.25
                )
            ),

        "Mean_Aligned_K":
            mean_value,

        "Median_Aligned_K":
            median_value,

        "Q75_Aligned_K":
            float(
                np.quantile(
                    values,
                    0.75
                )
            ),

        "Q95_Aligned_K":
            float(
                np.quantile(
                    values,
                    0.95
                )
            ),

        "Maximum_Aligned_K":
            float(
                np.max(
                    values
                )
            ),

        "SD_Aligned_K_Descriptive":
            float(
                np.std(
                    values,
                    ddof=1
                )
            ),

        "Phase_Design_K":
            phase_K,

        "Mean_minus_Phase_K":
            mean_value
            - phase_K,

        "Mean_Relative_Difference_from_Phase":
            (
                mean_value
                - phase_K
            )
            / phase_K,

        "Median_minus_Phase_K":
            median_value
            - phase_K,

        "Median_Relative_Difference_from_Phase":
            (
                median_value
                - phase_K
            )
            / phase_K
    }


aligned_summary_df = pd.DataFrame(
    [
        descriptive_summary(
            original_aligned_df[
                "K_Aligned_Integrated"
            ],
            "Original_Centered",
            K_PHASE_ORIGINAL
        ),

        descriptive_summary(
            common_aligned_df[
                "K_Aligned_Integrated"
            ],
            "Common_Marginal",
            K_PHASE_COMMON
        )
    ]
)


# ---------------------------------------------------------------------
# 11. Position of phase-design K inside aligned distribution
# ---------------------------------------------------------------------
#
# This is a descriptive empirical percentile/rank only.
# It is NOT a probability statement.
#
# ---------------------------------------------------------------------

phase_position_rows = []


for marginal_type, subset, phase_K in [
    (
        "Original_Centered",
        original_aligned_df,
        K_PHASE_ORIGINAL
    ),
    (
        "Common_Marginal",
        common_aligned_df,
        K_PHASE_COMMON
    )
]:

    values = subset[
        "K_Aligned_Integrated"
    ].to_numpy(
        dtype=float
    )


    fraction_below = np.mean(
        values
        <= phase_K
    )


    phase_position_rows.append(
        {
            "Marginal_Type":
                marginal_type,

            "Phase_Design_K":
                phase_K,

            "Fraction_of_Aligned_Pairs_at_or_below_Phase_K":
                fraction_below,

            "Descriptive_Percentile_Position":
                100.0
                * fraction_below
        }
    )


phase_position_df = pd.DataFrame(
    phase_position_rows
)


# ---------------------------------------------------------------------
# 12. Identify aligned extremes
# ---------------------------------------------------------------------

extreme_rows = []


for marginal_type, subset, phase_K in [
    (
        "Original_Centered",
        original_aligned_df,
        K_PHASE_ORIGINAL
    ),
    (
        "Common_Marginal",
        common_aligned_df,
        K_PHASE_COMMON
    )
]:

    minimum_row = subset.loc[
        subset[
            "K_Aligned_Integrated"
        ].idxmin()
    ]


    maximum_row = subset.loc[
        subset[
            "K_Aligned_Integrated"
        ].idxmax()
    ]


    closest_index = (
        subset[
            "K_Aligned_Integrated"
        ]
        - phase_K
    ).abs().idxmin()


    closest_row = subset.loc[
        closest_index
    ]


    for label, row in [
        (
            "Minimum aligned K",
            minimum_row
        ),
        (
            "Closest aligned pair to phase-design K",
            closest_row
        ),
        (
            "Maximum aligned K",
            maximum_row
        )
    ]:

        extreme_rows.append(
            {
                "Marginal_Type":
                    marginal_type,

                "Case":
                    label,

                "Trajectory_1":
                    row[
                        "Trajectory_1"
                    ],

                "Trajectory_2":
                    row[
                        "Trajectory_2"
                    ],

                "Aligned_K":
                    row[
                        "K_Aligned_Integrated"
                    ],

                "Phase_Design_K":
                    phase_K,

                "Difference_from_Phase_K":
                    row[
                        "K_Aligned_Integrated"
                    ]
                    - phase_K
            }
        )


extreme_df = pd.DataFrame(
    extreme_rows
)


# ---------------------------------------------------------------------
# 13. Optional Stage 4B representative/raster documentation
# ---------------------------------------------------------------------

raster_summary_df = pd.DataFrame()


if RASTER_FILE.exists():

    raster_df = pd.read_csv(
        RASTER_FILE
    )


    # Preserve the full raster table as an audit artifact.
    raster_summary_df = raster_df.copy()


representative_geometry_df = pd.DataFrame()


if GEOMETRY_REPRESENTATIVE_FILE.exists():

    representative_geometry_df = pd.read_csv(
        GEOMETRY_REPRESENTATIVE_FILE
    )


# ---------------------------------------------------------------------
# 14. Manuscript-ready audit table
# ---------------------------------------------------------------------

original_summary = aligned_summary_df[
    aligned_summary_df[
        "Marginal_Type"
    ]
    == "Original_Centered"
].iloc[
    0
]


common_summary = aligned_summary_df[
    aligned_summary_df[
        "Marginal_Type"
    ]
    == "Common_Marginal"
].iloc[
    0
]


manuscript_rows = [
    {
        "Audit_Item":
            "Stage 4B analytical-vs-polygon skip closure",

        "Result":
            geometry_df[
                "Audit_Abs_Skip_Error"
            ].max(),

        "Interpretation":
            "Maximum absolute error across 378 pairs"
    },

    {
        "Audit_Item":
            "Stage 4B analytical-vs-polygon overlap closure",

        "Result":
            geometry_df[
                "Audit_Abs_Overlap_Error"
            ].max(),

        "Interpretation":
            "Maximum absolute error across 378 pairs"
    },

    {
        "Audit_Item":
            "Stage 4B analytical-vs-polygon outside-application closure",

        "Result":
            geometry_df[
                "Audit_Abs_Outside_Error"
            ].max(),

        "Interpretation":
            "Maximum absolute error across 378 pairs"
    },

    {
        "Audit_Item":
            "Stage 4B analytical-vs-polygon total closure",

        "Result":
            geometry_df[
                "Audit_Abs_Total_Error"
            ].max(),

        "Interpretation":
            "Maximum absolute error across 378 pairs"
    },

    {
        "Audit_Item":
            "Original empirical mean aligned K",

        "Result":
            original_summary[
                "Mean_Aligned_K"
            ],

        "Interpretation":
            "Descriptive mean of 378 dependent aligned trajectory pairs"
    },

    {
        "Audit_Item":
            "Original empirical median aligned K",

        "Result":
            original_summary[
                "Median_Aligned_K"
            ],

        "Interpretation":
            "Descriptive median of 378 dependent aligned trajectory pairs"
    },

    {
        "Audit_Item":
            "Original empirical phase-design K",

        "Result":
            K_PHASE_ORIGINAL,

        "Interpretation":
            "Uniform relative-phase design coefficient used in tolerance framework"
    },

    {
        "Audit_Item":
            "Original mean aligned-vs-phase relative difference",

        "Result":
            original_summary[
                "Mean_Relative_Difference_from_Phase"
            ],

        "Interpretation":
            "(mean aligned K - phase-design K) / phase-design K"
    },

    {
        "Audit_Item":
            "Common-marginal mean aligned K",

        "Result":
            common_summary[
                "Mean_Aligned_K"
            ],

        "Interpretation":
            "Descriptive mean after exact common-marginal control"
    },

    {
        "Audit_Item":
            "Common-marginal phase-design K",

        "Result":
            K_PHASE_COMMON,

        "Interpretation":
            "Uniform relative-phase common-marginal design coefficient"
    },

    {
        "Audit_Item":
            "Common-marginal mean aligned-vs-phase relative difference",

        "Result":
            common_summary[
                "Mean_Relative_Difference_from_Phase"
            ],

        "Interpretation":
            "(mean aligned K - phase-design K) / phase-design K"
    }
]


manuscript_audit_df = pd.DataFrame(
    manuscript_rows
)


# ---------------------------------------------------------------------
# 15. Save all audit artifacts
# ---------------------------------------------------------------------

geometry_audit_file = (
    TABLES_DIR
    / "final_geometry_closure_audit.csv"
)


geometry_summary_file = (
    TABLES_DIR
    / "final_geometry_closure_summary.csv"
)


aligned_summary_file = (
    TABLES_DIR
    / "final_aligned_vs_phase_K_summary.csv"
)


phase_position_file = (
    TABLES_DIR
    / "final_phase_K_position_in_aligned_distribution.csv"
)


extreme_file = (
    TABLES_DIR
    / "final_aligned_K_representative_pairs.csv"
)


manuscript_file = (
    TABLES_DIR
    / "final_manuscript_validation_audit.csv"
)


geometry_df.to_csv(
    geometry_audit_file,
    index=False
)


geometry_summary_df.to_csv(
    geometry_summary_file,
    index=False
)


aligned_summary_df.to_csv(
    aligned_summary_file,
    index=False
)


phase_position_df.to_csv(
    phase_position_file,
    index=False
)


extreme_df.to_csv(
    extreme_file,
    index=False
)


manuscript_audit_df.to_csv(
    manuscript_file,
    index=False
)


# ---------------------------------------------------------------------
# 16. Figure — analytical vs exact polygon geometry
# ---------------------------------------------------------------------

plt.figure(
    figsize=(
        8,
        7
    )
)


plt.scatter(
    geometry_df[
        shapely_total_col
    ],
    geometry_df[
        analytic_total_col
    ],
    s=18,
    alpha=0.65
)


minimum_value = min(
    geometry_df[
        shapely_total_col
    ].min(),
    geometry_df[
        analytic_total_col
    ].min()
)


maximum_value = max(
    geometry_df[
        shapely_total_col
    ].max(),
    geometry_df[
        analytic_total_col
    ].max()
)


plt.plot(
    [
        minimum_value,
        maximum_value
    ],
    [
        minimum_value,
        maximum_value
    ],
    linestyle="--",
    linewidth=1.5,
    label="1:1 agreement"
)


plt.xlabel(
    "Exact polygon total discrepancy fraction"
)


plt.ylabel(
    "Analytical total discrepancy fraction"
)


plt.title(
    "Independent geometric closure across 378 trajectory pairs"
)


plt.legend()


plt.tight_layout()


geometry_figure = (
    FIGURES_DIR
    / "final_analytic_vs_exact_polygon_validation.png"
)


plt.savefig(
    geometry_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ---------------------------------------------------------------------
# 17. Figure — aligned K versus phase-design K
# ---------------------------------------------------------------------

plt.figure(
    figsize=(
        9,
        6
    )
)


plt.hist(
    original_aligned_df[
        "K_Aligned_Integrated"
    ],
    bins=30,
    alpha=0.75,
    label="Recorded aligned empirical pairs"
)


plt.axvline(
    K_PHASE_ORIGINAL,
    linestyle="--",
    linewidth=2,
    label=(
        f"Uniform phase-design K = "
        f"{K_PHASE_ORIGINAL:.3f}"
    )
)


plt.axvline(
    original_aligned_df[
        "K_Aligned_Integrated"
    ].mean(),
    linestyle=":",
    linewidth=2,
    label=(
        "Mean recorded aligned K"
    )
)


plt.xlabel(
    "Integrated normalized discrepancy K"
)


plt.ylabel(
    "Number of dependent trajectory-pair combinations"
)


plt.title(
    "Recorded alignment versus uniform phase-design discrepancy"
)


plt.legend()


plt.tight_layout()


aligned_figure = (
    FIGURES_DIR
    / "final_recorded_aligned_vs_phase_design_K.png"
)


plt.savefig(
    aligned_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ---------------------------------------------------------------------
# 18. Console report
# ---------------------------------------------------------------------

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    360
)


print(
    "\n"
    + "=" * 100
)

print(
    "A. STAGE 4B GEOMETRIC CLOSURE"
)

print(
    "=" * 100
)


print(
    geometry_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.12e}"
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "B. RECORDED ALIGNED K VS UNIFORM PHASE-DESIGN K"
)

print(
    "=" * 100
)


print(
    aligned_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}"
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "C. DESCRIPTIVE POSITION OF PHASE-DESIGN K"
)

print(
    "=" * 100
)


print(
    phase_position_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}"
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "D. REPRESENTATIVE ALIGNED PAIRS"
)

print(
    "=" * 100
)


print(
    extreme_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}"
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "E. MANUSCRIPT-READY VALIDATION AUDIT"
)

print(
    "=" * 100
)


print(
    manuscript_audit_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.12e}"
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "INTERPRETATION"
)

print(
    "=" * 100
)


print(
    "\n1. Stage 4B is an independent geometric closure test:"
)


print(
    "   analytical skip/overlap/outside-application areas are compared"
)


print(
    "   directly against exact polygon geometry."
)


print(
    "\n2. K_aligned and K_phase are intentionally different quantities."
)


print(
    "\n   K_aligned:"
)


print(
    "      discrepancy at the recorded relative spatial alignment of"
)


print(
    "      two empirical trajectory profiles."
)


print(
    "\n   K_phase:"
)


print(
    "      uniform relative-phase design average used when constructing"
)


print(
    "      the general operational tolerance transfer."
)


print(
    "\n3. The 378 aligned pair combinations are dependent."
)


print(
    "   Their mean, median, quantiles and SD are descriptive only."
)


print(
    "\n4. No p-value or independent-pair confidence interval is used."
)


print(
    "\n5. Configuration-level uncertainty of the headline coefficient"
)


print(
    "   remains the cluster-bootstrap interval from Stage 4D."
)


print(
    "\n"
    + "=" * 100
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 100
)


print(
    f"\nFull geometry closure audit:\n"
    f"{geometry_audit_file}"
)


print(
    f"\nGeometry closure summary:\n"
    f"{geometry_summary_file}"
)


print(
    f"\nAligned-vs-phase K summary:\n"
    f"{aligned_summary_file}"
)


print(
    f"\nPhase-design K position:\n"
    f"{phase_position_file}"
)


print(
    f"\nRepresentative aligned pairs:\n"
    f"{extreme_file}"
)


print(
    f"\nManuscript validation audit:\n"
    f"{manuscript_file}"
)


print(
    f"\nFigures:\n"
    f"{FIGURES_DIR}"
)


print(
    "\n"
    + "=" * 100
)

print(
    "STAGE 4E COMPLETE"
)

print(
    "=" * 100
)