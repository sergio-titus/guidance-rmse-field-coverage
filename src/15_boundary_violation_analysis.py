from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import ndtri


# ============================================================
# STAGE 3B — OUTER-BOUNDARY VIOLATION AND EDGE-GAP ANALYSIS
# ============================================================
#
# PURPOSE
# -------
# Internal skip/overlap between adjacent passes and outer-field
# boundary effects are geometrically different.
#
# This stage therefore analyzes OUTER FIELD EDGES separately.
#
#
# LEFT FIELD EDGE
# ---------------
#
# Nominal first-pass centerline:
#
#       y = W / 2
#
# Guidance error:
#
#       e(x)
#
# Actual implement left edge:
#
#       y_left_edge = e(x)
#
# Relative to field boundary y = 0:
#
#       outward violation width = max(-e, 0)
#       inward uncovered gap    = max( e, 0)
#
#
# RIGHT FIELD EDGE
# ----------------
#
# Nominal final-pass centerline:
#
#       y = M W - W/2
#
# Actual right implement edge:
#
#       M W + e(x)
#
# Therefore:
#
#       outward violation width = max( e, 0)
#       inward uncovered gap    = max(-e, 0)
#
#
# DIMENSIONLESS FORM
# ------------------
#
# Write:
#
#       e(x) = R z(x)
#
# with unit-RMS z and:
#
#       eta = R / W
#
# Then:
#
#       boundary_width / W
#           = eta * max(+/- z, 0)
#
#
# IMPORTANT
# ---------
#
# We use the SAME common-marginal unit-RMS structure control
# established in Stages 2C and 2D.
#
# Consequently:
#
#   - RMS is identical,
#   - marginal distribution is identical,
#   - mean boundary magnitude is identical,
#
# while:
#
#   - clustering,
#   - persistence,
#   - number of excursions,
#   - largest contiguous boundary event
#
# can differ because of spatial organization.
#
#
# NO CIRCULAR PHASE SHIFT IS USED HERE.
#
# Boundary runs are measured directly over the common empirical
# 271.4 m support. This avoids interpreting wrapped trajectories
# as literal physical field edges.
#
#
# FIELD-LEVEL SCALING
# -------------------
#
# For M passes and two external boundaries:
#
#       boundary contribution / eta
#
#       = (B_left + B_right) / M
#
# where B is mean positive/negative unit-RMS excursion.
#
# If both boundaries have the same common marginal:
#
#       expected total boundary factor
#
#       = 2 B / M
#
# Thus boundary effects decay approximately as 1/M.
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
    / "bias_controlled_equal_rmse_trajectory_matrix.csv"
)

MULTIPASS_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "analytical_multipass_uncertainty_propagation.csv"
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
    / "boundary_violation"
)

for directory in [
    TABLES_DIR,
    FIGURES_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


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


def common_grid_for_all(
    trajectory_dict,
    spacing
):

    starts = [
        np.min(
            record["X"]
        )
        for record in trajectory_dict.values()
    ]

    ends = [
        np.max(
            record["X"]
        )
        for record in trajectory_dict.values()
    ]

    start = max(
        starts
    )

    end = min(
        ends
    )

    if end <= start:

        raise RuntimeError(
            "No common spatial support."
        )

    number_of_steps = int(
        np.floor(
            (
                end
                - start
            )
            / spacing
        )
    )

    grid = (
        start
        + np.arange(
            number_of_steps + 1
        )
        * spacing
    )

    if len(grid) < 2:

        raise RuntimeError(
            "Too few points on common grid."
        )

    return grid


def gaussian_rank_transform(
    values,
    common_scores
):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) != len(common_scores):

        raise RuntimeError(
            "Rank-transform length mismatch."
        )

    order = np.argsort(
        values,
        kind="mergesort"
    )

    ranks = np.empty(
        len(values),
        dtype=int
    )

    ranks[order] = np.arange(
        len(values)
    )

    return common_scores[ranks]


def linear_runs(
    condition
):

    condition = np.asarray(
        condition,
        dtype=bool
    )

    padded = np.concatenate(
        (
            [False],
            condition,
            [False]
        )
    )

    transitions = np.diff(
        padded.astype(int)
    )

    starts = np.where(
        transitions == 1
    )[0]

    ends = np.where(
        transitions == -1
    )[0]

    return list(
        zip(
            starts,
            ends
        )
    )


def excursion_statistics(
    values,
    dx
):

    values = np.asarray(
        values,
        dtype=float
    )

    positive_values = np.maximum(
        values,
        0.0
    )

    condition = (
        values > 0
    )

    runs = linear_runs(
        condition
    )

    run_lengths = []

    run_area_factors = []

    peak_values = []


    for start, end in runs:

        local = positive_values[
            start:end
        ]

        run_lengths.append(
            len(local)
            * dx
        )

        run_area_factors.append(
            np.sum(local)
            * dx
        )

        peak_values.append(
            np.max(local)
        )


    if len(runs) == 0:

        longest_run = 0.0
        largest_area = 0.0
        largest_peak = 0.0

    else:

        longest_run = max(
            run_lengths
        )

        largest_area = max(
            run_area_factors
        )

        largest_peak = max(
            peak_values
        )


    return {
        "Mean_Excursion_Factor":
            np.mean(
                positive_values
            ),

        "Fraction_of_Field_in_Excursion":
            np.mean(
                condition.astype(float)
            ),

        "Number_of_Excursion_Clusters":
            len(runs),

        "Longest_Excursion_Run_m":
            longest_run,

        "Largest_Excursion_Cluster_Area_Factor_m":
            largest_area,

        "Maximum_Instantaneous_Excursion_Factor":
            largest_peak
    }


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------

for file_path in [
    INPUT_FILE,
    MULTIPASS_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing file:\n"
            f"{file_path}"
        )


print(
    "=" * 90
)

print(
    "STAGE 3B — OUTER-BOUNDARY VIOLATION AND EDGE-GAP ANALYSIS"
)

print(
    "=" * 90
)


df = pd.read_csv(
    INPUT_FILE
)

multipass_df = pd.read_csv(
    MULTIPASS_FILE
)


print(
    f"\nLoaded trajectory matrix:"
    f"\n{INPUT_FILE}"
)

print(
    f"\nRows: "
    f"{len(df):,}"
)


# ------------------------------------------------------------
# 4. Retrieve one controlled structure set
# ------------------------------------------------------------

rms_levels = (
    df[
        [
            "Target_Centered_RMS_ID",
            "Target_Centered_RMS_m"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "Target_Centered_RMS_m"
    )
    .reset_index(
        drop=True
    )
)


selected_index = (
    len(rms_levels)
    // 2
)


selected_level = rms_levels.iloc[
    selected_index
]


SELECTED_RMS_ID = selected_level[
    "Target_Centered_RMS_ID"
]


working_df = df[
    df[
        "Target_Centered_RMS_ID"
    ]
    == SELECTED_RMS_ID
].copy()


trajectory_ids = sorted(
    working_df[
        "Structure_Trajectory_ID"
    ].unique()
)


print(
    f"\nEmpirical structures:"
    f"\n  {len(trajectory_ids)}"
)


# ------------------------------------------------------------
# 5. Load structures on their native canonical representation
# ------------------------------------------------------------

raw_store = {}


for trajectory_id in trajectory_ids:

    group = working_df[
        working_df[
            "Structure_Trajectory_ID"
        ]
        == trajectory_id
    ].sort_values(
        "Easting_m"
    )


    x = group[
        "Easting_m"
    ].to_numpy(
        dtype=float
    )


    error = group[
        "Zero_Mean_Equal_RMS_Error_m"
    ].to_numpy(
        dtype=float
    )


    finite = (
        np.isfinite(x)
        &
        np.isfinite(error)
    )


    x = x[finite]

    error = error[finite]


    unique_x, unique_indices = np.unique(
        x,
        return_index=True
    )


    raw_store[
        trajectory_id
    ] = {
        "X":
            unique_x,

        "Error":
            error[
                unique_indices
            ]
    }


# ------------------------------------------------------------
# 6. Common empirical field support
# ------------------------------------------------------------

grid = common_grid_for_all(
    raw_store,
    DX_M
)


N = len(grid)


FIELD_LENGTH_M = (
    grid[-1]
    - grid[0]
)


print(
    "\nCommon empirical boundary length:"
)

print(
    f"  start  = "
    f"{grid[0]:.3f} m"
)

print(
    f"  end    = "
    f"{grid[-1]:.3f} m"
)

print(
    f"  length = "
    f"{FIELD_LENGTH_M:.3f} m"
)

print(
    f"  points = "
    f"{N:,}"
)


# ------------------------------------------------------------
# 7. Exact shared marginal
# ------------------------------------------------------------

probabilities = (
    (
        np.arange(N)
        + 0.5
    )
    / N
)


common_scores = ndtri(
    probabilities
)


common_scores = (
    common_scores
    - np.mean(
        common_scores
    )
)


common_scores = (
    common_scores
    / rms(
        common_scores
    )
)


# ------------------------------------------------------------
# 8. Build common-marginal structures
# ------------------------------------------------------------

structure_store = {}


for trajectory_id in trajectory_ids:

    record = raw_store[
        trajectory_id
    ]


    interpolated = np.interp(
        grid,
        record["X"],
        record["Error"]
    )


    transformed = gaussian_rank_transform(
        interpolated,
        common_scores
    )


    transformed = (
        transformed
        - np.mean(
            transformed
        )
    )


    transformed = (
        transformed
        / rms(
            transformed
        )
    )


    structure_store[
        trajectory_id
    ] = transformed


# ------------------------------------------------------------
# 9. Verify exact common marginal
# ------------------------------------------------------------

reference_sorted = np.sort(
    structure_store[
        trajectory_ids[0]
    ]
)


max_marginal_difference = 0.0


for trajectory_id in trajectory_ids[1:]:

    current_sorted = np.sort(
        structure_store[
            trajectory_id
        ]
    )

    difference = np.max(
        np.abs(
            current_sorted
            - reference_sorted
        )
    )

    max_marginal_difference = max(
        max_marginal_difference,
        difference
    )


# ------------------------------------------------------------
# 10. Analyze both boundary directions for every structure
# ------------------------------------------------------------
#
# Direction "+":
#
#       excursion = max(z, 0)
#
# This is:
#       right-boundary outward violation
#       OR left-boundary inward gap
#
#
# Direction "-":
#
#       excursion = max(-z, 0)
#
# This is:
#       left-boundary outward violation
#       OR right-boundary inward gap
#
# ------------------------------------------------------------

boundary_rows = []


for trajectory_id in trajectory_ids:

    z = structure_store[
        trajectory_id
    ]


    positive_stats = excursion_statistics(
        z,
        DX_M
    )


    negative_stats = excursion_statistics(
        -z,
        DX_M
    )


    boundary_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Direction":
                "Positive",

            "Physical_Interpretation":
                "Right outward / Left inward",

            **positive_stats
        }
    )


    boundary_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Direction":
                "Negative",

            "Physical_Interpretation":
                "Left outward / Right inward",

            **negative_stats
        }
    )


boundary_df = pd.DataFrame(
    boundary_rows
)


# ------------------------------------------------------------
# 11. Common mean boundary factor
# ------------------------------------------------------------

positive_base = np.maximum(
    common_scores,
    0.0
)


negative_base = np.maximum(
    -common_scores,
    0.0
)


B_POSITIVE = np.mean(
    positive_base
)


B_NEGATIVE = np.mean(
    negative_base
)


B_TWO_EDGES = (
    B_POSITIVE
    + B_NEGATIVE
)


# ------------------------------------------------------------
# 12. Structure-level persistence summary
# ------------------------------------------------------------

structure_summary_rows = []


for trajectory_id in trajectory_ids:

    subset = boundary_df[
        boundary_df[
            "Trajectory_ID"
        ]
        == trajectory_id
    ]


    positive_row = subset[
        subset[
            "Direction"
        ]
        == "Positive"
    ].iloc[
        0
    ]


    negative_row = subset[
        subset[
            "Direction"
        ]
        == "Negative"
    ].iloc[
        0
    ]


    structure_summary_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Positive_Longest_Run_m":
                positive_row[
                    "Longest_Excursion_Run_m"
                ],

            "Negative_Longest_Run_m":
                negative_row[
                    "Longest_Excursion_Run_m"
                ],

            "Maximum_of_Two_Longest_Runs_m":
                max(
                    positive_row[
                        "Longest_Excursion_Run_m"
                    ],
                    negative_row[
                        "Longest_Excursion_Run_m"
                    ]
                ),

            "Positive_Number_of_Clusters":
                positive_row[
                    "Number_of_Excursion_Clusters"
                ],

            "Negative_Number_of_Clusters":
                negative_row[
                    "Number_of_Excursion_Clusters"
                ],

            "Positive_Largest_Cluster_Area_Factor_m":
                positive_row[
                    "Largest_Excursion_Cluster_Area_Factor_m"
                ],

            "Negative_Largest_Cluster_Area_Factor_m":
                negative_row[
                    "Largest_Excursion_Cluster_Area_Factor_m"
                ],

            "Maximum_of_Two_Largest_Cluster_Areas_m":
                max(
                    positive_row[
                        "Largest_Excursion_Cluster_Area_Factor_m"
                    ],
                    negative_row[
                        "Largest_Excursion_Cluster_Area_Factor_m"
                    ]
                )
        }
    )


structure_summary_df = pd.DataFrame(
    structure_summary_rows
)


# ------------------------------------------------------------
# 13. Multi-pass boundary scaling
# ------------------------------------------------------------
#
# For a field with M passes:
#
# total outer-boundary applied-area contribution:
#
#       D_boundary / eta
#           = (B_left + B_right) / M
#
# Under the common marginal:
#
#       = (B_positive + B_negative) / M
#
#
# Compare to internal mean:
#
#       D_internal / eta
#
# already obtained in Stage 3A.
#
# ------------------------------------------------------------

boundary_scaling_rows = []


for _, row in multipass_df.iterrows():

    m = int(
        row[
            "Number_of_Passes_M"
        ]
    )


    boundary_mean_factor = (
        B_TWO_EDGES
        / m
    )


    internal_mean_factor = float(
        row[
            "Mean_D_over_eta"
        ]
    )


    total_mean_factor = (
        internal_mean_factor
        + boundary_mean_factor
    )


    boundary_to_internal_ratio = (
        boundary_mean_factor
        / internal_mean_factor
    )


    boundary_share_of_total = (
        boundary_mean_factor
        / total_mean_factor
    )


    boundary_scaling_rows.append(
        {
            "Number_of_Passes_M":
                m,

            "Internal_Mean_D_over_eta":
                internal_mean_factor,

            "Two_Edge_Boundary_Mean_D_over_eta":
                boundary_mean_factor,

            "Combined_Internal_plus_Boundary_Mean_over_eta":
                total_mean_factor,

            "Boundary_to_Internal_Mean_Ratio":
                boundary_to_internal_ratio,

            "Boundary_Share_of_Combined_Mean":
                boundary_share_of_total
        }
    )


boundary_scaling_df = pd.DataFrame(
    boundary_scaling_rows
)


# ------------------------------------------------------------
# 14. Large-M analytical limits
# ------------------------------------------------------------

internal_large_m_limit = float(
    multipass_df[
        "Mean_D_over_eta"
    ].iloc[
        -1
    ]
)


boundary_large_m_limit = 0.0


# ------------------------------------------------------------
# 15. Numerical identities
# ------------------------------------------------------------

mean_factor_range = (
    boundary_df[
        "Mean_Excursion_Factor"
    ].max()
    -
    boundary_df[
        "Mean_Excursion_Factor"
    ].min()
)


positive_negative_factor_difference = abs(
    B_POSITIVE
    - B_NEGATIVE
)


# ------------------------------------------------------------
# 16. Save tables
# ------------------------------------------------------------

boundary_file = (
    TABLES_DIR
    / "boundary_excursion_structure_metrics.csv"
)

structure_summary_file = (
    TABLES_DIR
    / "boundary_structure_persistence_summary.csv"
)

boundary_scaling_file = (
    TABLES_DIR
    / "multipass_boundary_scaling.csv"
)


boundary_df.to_csv(
    boundary_file,
    index=False
)

structure_summary_df.to_csv(
    structure_summary_file,
    index=False
)

boundary_scaling_df.to_csv(
    boundary_scaling_file,
    index=False
)


# ------------------------------------------------------------
# 17. Figure — longest boundary-event runs
# ------------------------------------------------------------

sorted_summary = structure_summary_df.sort_values(
    "Maximum_of_Two_Longest_Runs_m",
    ascending=True
)


plt.figure(
    figsize=(11, 8)
)


plt.barh(
    sorted_summary[
        "Trajectory_ID"
    ],
    sorted_summary[
        "Maximum_of_Two_Longest_Runs_m"
    ]
)


plt.xlabel(
    "Longest positive/negative boundary excursion run (m)"
)

plt.ylabel(
    "Empirical structure"
)

plt.title(
    "Boundary-event persistence at equal RMS and identical marginal distribution"
)

plt.tight_layout()


run_figure = (
    FIGURES_DIR
    / "boundary_longest_excursion_by_structure.png"
)


plt.savefig(
    run_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 18. Figure — internal versus boundary scaling
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    boundary_scaling_df[
        "Number_of_Passes_M"
    ],
    boundary_scaling_df[
        "Internal_Mean_D_over_eta"
    ],
    linewidth=2,
    label="Internal interfaces"
)


plt.plot(
    boundary_scaling_df[
        "Number_of_Passes_M"
    ],
    boundary_scaling_df[
        "Two_Edge_Boundary_Mean_D_over_eta"
    ],
    linewidth=2,
    label="Two outer boundaries"
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "Expected normalized discrepancy / eta"
)

plt.title(
    "Internal-interface versus outer-boundary scaling"
)

plt.legend()

plt.tight_layout()


scaling_figure = (
    FIGURES_DIR
    / "internal_vs_boundary_scaling.png"
)


plt.savefig(
    scaling_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 19. Figure — boundary share of total
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    boundary_scaling_df[
        "Number_of_Passes_M"
    ],
    100
    * boundary_scaling_df[
        "Boundary_Share_of_Combined_Mean"
    ],
    linewidth=2
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "Boundary share of combined mean discrepancy (%)"
)

plt.title(
    "Declining contribution of outer boundaries as field width increases"
)

plt.tight_layout()


share_figure = (
    FIGURES_DIR
    / "boundary_share_vs_pass_count.png"
)


plt.savefig(
    share_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 20. Figure — cluster count versus longest run
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    boundary_df[
        "Number_of_Excursion_Clusters"
    ],
    boundary_df[
        "Longest_Excursion_Run_m"
    ]
)


plt.xlabel(
    "Number of boundary excursion clusters"
)

plt.ylabel(
    "Longest boundary excursion run (m)"
)

plt.title(
    "Boundary fragmentation versus persistence"
)

plt.tight_layout()


cluster_figure = (
    FIGURES_DIR
    / "boundary_cluster_count_vs_longest_run.png"
)


plt.savefig(
    cluster_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 21. Descriptive correlations
# ------------------------------------------------------------

correlation_columns = [
    "Mean_Excursion_Factor",
    "Fraction_of_Field_in_Excursion",
    "Number_of_Excursion_Clusters",
    "Longest_Excursion_Run_m",
    "Largest_Excursion_Cluster_Area_Factor_m",
    "Maximum_Instantaneous_Excursion_Factor"
]


correlation_matrix = boundary_df[
    correlation_columns
].corr(
    method="spearman"
)


correlation_file = (
    TABLES_DIR
    / "boundary_metric_spearman_matrix.csv"
)


correlation_matrix.to_csv(
    correlation_file
)


# ------------------------------------------------------------
# 22. Terminal display
# ------------------------------------------------------------

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    300
)

pd.set_option(
    "display.max_rows",
    120
)


# ------------------------------------------------------------
# 23. Verification
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "COMMON-MARGINAL BOUNDARY VERIFICATION"
)

print(
    "=" * 90
)


print(
    "\nMaximum difference between sorted transformed marginals:"
)

print(
    f"  {max_marginal_difference:.12e}"
)


print(
    "\nRange of mean excursion factor across all structures/directions:"
)

print(
    f"  {mean_factor_range:.12e}"
)


print(
    "\nPositive common-marginal mean excursion factor:"
)

print(
    f"  {B_POSITIVE:.12f}"
)


print(
    "\nNegative common-marginal mean excursion factor:"
)

print(
    f"  {B_NEGATIVE:.12f}"
)


print(
    "\nAbsolute positive-negative difference:"
)

print(
    f"  {positive_negative_factor_difference:.12e}"
)


print(
    "\nCombined two-edge factor:"
)

print(
    f"  {B_TWO_EDGES:.12f}"
)


# ------------------------------------------------------------
# 24. Persistence summary
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "BOUNDARY PERSISTENCE ACROSS EMPIRICAL STRUCTURES"
)

print(
    "=" * 90
)


for column, label in [
    (
        "Longest_Excursion_Run_m",
        "Longest boundary excursion run"
    ),
    (
        "Largest_Excursion_Cluster_Area_Factor_m",
        "Largest boundary cluster area factor"
    ),
    (
        "Number_of_Excursion_Clusters",
        "Number of boundary excursion clusters"
    )
]:

    values = boundary_df[
        column
    ]


    print(
        f"\n{label}:"
    )

    print(
        f"  min    = "
        f"{values.min():.6f}"
    )

    print(
        f"  Q25    = "
        f"{values.quantile(0.25):.6f}"
    )

    print(
        f"  median = "
        f"{values.median():.6f}"
    )

    print(
        f"  Q75    = "
        f"{values.quantile(0.75):.6f}"
    )

    print(
        f"  max    = "
        f"{values.max():.6f}"
    )


# ------------------------------------------------------------
# 25. Most persistent structures
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "STRUCTURES WITH LONGEST BOUNDARY EVENTS"
)

print(
    "=" * 90
)


print(
    structure_summary_df.sort_values(
        "Maximum_of_Two_Longest_Runs_m",
        ascending=False
    ).head(
        15
    ).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 26. Multi-pass scaling examples
# ------------------------------------------------------------

example_pass_counts = [
    2,
    3,
    5,
    10,
    20,
    50,
    100
]


example_df = boundary_scaling_df[
    boundary_scaling_df[
        "Number_of_Passes_M"
    ].isin(
        example_pass_counts
    )
]


print(
    "\n"
    + "=" * 90
)

print(
    "INTERNAL VERSUS OUTER-BOUNDARY SCALING"
)

print(
    "=" * 90
)


print(
    example_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 27. Descriptive associations
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "SPEARMAN ASSOCIATIONS AMONG BOUNDARY METRICS"
)

print(
    "=" * 90
)


print(
    correlation_matrix.to_string()
)


# ------------------------------------------------------------
# 28. Analytical interpretation
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "ANALYTICAL BOUNDARY RELATIONSHIP"
)

print(
    "=" * 90
)


print(
    "\nFor one outer boundary:"
)

print(
    "  normalized mean outward violation"
)

print(
    f"    = eta * "
    f"{B_POSITIVE:.12f}"
)


print(
    "\nFor two outer boundaries in an M-pass field:"
)

print(
    "  mean boundary contribution / eta"
)

print(
    f"    = "
    f"{B_TWO_EDGES:.12f}"
    f" / M"
)


print(
    "\nTherefore:"
)

print(
    "  outer-boundary contribution -> 0 as M -> infinity"
)

print(
    "while internal-interface contribution approaches a "
    "non-zero limit."
)


# ------------------------------------------------------------
# 29. Output paths
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
    f"\nBoundary metrics:\n"
    f"{boundary_file}"
)

print(
    f"\nStructure persistence summary:\n"
    f"{structure_summary_file}"
)

print(
    f"\nMulti-pass boundary scaling:\n"
    f"{boundary_scaling_file}"
)

print(
    f"\nBoundary Spearman matrix:\n"
    f"{correlation_file}"
)

print(
    f"\nPersistence figure:\n"
    f"{run_figure}"
)

print(
    f"\nInternal/boundary scaling figure:\n"
    f"{scaling_figure}"
)

print(
    f"\nBoundary-share figure:\n"
    f"{share_figure}"
)

print(
    f"\nCluster/persistence figure:\n"
    f"{cluster_figure}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 3B COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nOuter boundaries are now separated from internal "
    "neighboring-pass skip/overlap."
)

print(
    "At identical RMS and marginal distribution, mean boundary "
    "magnitude is fixed, while event persistence and clustering "
    "remain structure-dependent."
)

print(
    "\nThe next stage can restore the ORIGINAL empirical marginal "
    "distributions as a sensitivity analysis before constructing "
    "the final operational tolerance curves."
)