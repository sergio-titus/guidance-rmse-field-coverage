from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 3C — ORIGINAL EMPIRICAL MARGINAL SENSITIVITY
# ============================================================
#
# PURPOSE
# -------
# Stages 2C, 2D, and 3B imposed one exact common Gaussian-score
# marginal distribution in order to isolate spatial ordering.
#
# This stage RESTORES the original empirical marginal shape of
# every centered guidance-error trajectory while retaining:
#
#       mean = 0
#       RMS  = 1
#
# Therefore error magnitude remains controlled, but empirical:
#
#       - skewness,
#       - tail shape,
#       - positive/negative occupancy,
#       - absolute-error distribution
#
# are restored.
#
#
# QUESTION
# --------
#
# How sensitive are the principal operational coefficients to
# the common-marginal control?
#
#
# PART A — SINGLE-STRUCTURE MARGINAL METRICS
# ------------------------------------------
#
# For each original standardized trajectory:
#
#       B+ = mean(max(z, 0))
#       B- = mean(max(-z, 0))
#
# Since each trajectory is exactly centered:
#
#       B+ = B- = 0.5 * mean(|z|)
#
# up to numerical precision.
#
# Unlike the common-marginal case, however, B may differ BETWEEN
# structures because their empirical marginal distributions differ.
#
#
# PART B — EXACT ALL-PHASE PAIRWISE K
# -----------------------------------
#
# For a pair z_i and z_j:
#
#       K(s) = mean(|z_j(x+s) - z_i(x)|)
#
# Averaging over ALL circular shifts gives:
#
#       mean_s K(s)
#           = mean_{a,b} |z_i[a] - z_j[b]|
#
# Therefore the all-phase mean depends on the two marginal
# distributions but NOT their spatial ordering.
#
# We compute this exactly without enumerating every circular
# shift.
#
#
# PART C — BOUNDARY PERSISTENCE
# -----------------------------
#
# Original spatial ordering is retained and direct, non-wrapped
# boundary persistence is measured over the common empirical
# field support.
#
# This tests whether conclusions from Stage 3B remain similar
# after restoring the observed marginal shapes.
#
#
# IMPORTANT
# ---------
#
# This is a SENSITIVITY ANALYSIS.
#
# The 28 empirical trajectories are not treated as independent
# population samples and ordinary inferential p-values are not
# used.
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

COMMON_PHASE_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "exhaustive_phase_clustering_scenarios.csv"
)

COMMON_BOUNDARY_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "boundary_excursion_structure_metrics.csv"
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
    / "original_marginal_sensitivity"
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
            "No common empirical spatial support."
        )

    n_steps = int(
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
            n_steps + 1
        )
        * spacing
    )

    if len(grid) < 2:

        raise RuntimeError(
            "Common grid contains too few samples."
        )

    return grid


def linear_runs(
    condition
):

    condition = np.asarray(
        condition,
        dtype=bool
    )

    padded = np.concatenate(
        (
            np.array(
                [False]
            ),
            condition,
            np.array(
                [False]
            )
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

    positive = np.maximum(
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

    run_areas = []

    run_peaks = []


    for start, end in runs:

        local = positive[
            start:end
        ]

        run_lengths.append(
            len(local)
            * dx
        )

        run_areas.append(
            np.sum(local)
            * dx
        )

        run_peaks.append(
            np.max(local)
        )


    if len(runs) == 0:

        longest_run = 0.0
        largest_area = 0.0
        maximum_peak = 0.0

    else:

        longest_run = max(
            run_lengths
        )

        largest_area = max(
            run_areas
        )

        maximum_peak = max(
            run_peaks
        )


    return {
        "Mean_Excursion_Factor":
            np.mean(
                positive
            ),

        "Excursion_Fraction":
            np.mean(
                condition.astype(float)
            ),

        "Number_of_Clusters":
            len(runs),

        "Longest_Run_m":
            longest_run,

        "Largest_Cluster_Area_Factor_m":
            largest_area,

        "Maximum_Excursion_Factor":
            maximum_peak
    }


def standardized_moments(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    mean_value = np.mean(
        values
    )

    centered = (
        values
        - mean_value
    )

    sd = np.sqrt(
        np.mean(
            centered ** 2
        )
    )

    if sd <= 0:

        return (
            np.nan,
            np.nan
        )

    standardized = (
        centered
        / sd
    )

    skewness = np.mean(
        standardized ** 3
    )

    excess_kurtosis = (
        np.mean(
            standardized ** 4
        )
        - 3.0
    )

    return (
        skewness,
        excess_kurtosis
    )


def mean_absolute_difference_between_marginals(
    x,
    y
):

    # --------------------------------------------------------
    # Exact O(N log N) calculation of:
    #
    #       mean_{a,b} |x[a] - y[b]|
    #
    # rather than allocating an N x N difference matrix.
    # --------------------------------------------------------

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    y_sorted = np.sort(
        y
    )

    n_y = len(
        y_sorted
    )

    prefix = np.concatenate(
        (
            np.array(
                [0.0]
            ),
            np.cumsum(
                y_sorted
            )
        )
    )

    total_y = prefix[
        -1
    ]

    total_absolute_difference = 0.0


    for value in x:

        split = np.searchsorted(
            y_sorted,
            value,
            side="right"
        )

        n_left = split

        n_right = (
            n_y
            - split
        )

        sum_left = prefix[
            split
        ]

        sum_right = (
            total_y
            - sum_left
        )

        left_difference = (
            value
            * n_left
            - sum_left
        )

        right_difference = (
            sum_right
            - value
            * n_right
        )

        total_absolute_difference += (
            left_difference
            + right_difference
        )


    return (
        total_absolute_difference
        / (
            len(x)
            * len(y)
        )
    )


# ------------------------------------------------------------
# 3. Load files
# ------------------------------------------------------------

for file_path in [
    INPUT_FILE,
    COMMON_PHASE_FILE,
    COMMON_BOUNDARY_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n"
            f"{file_path}"
        )


print(
    "=" * 90
)

print(
    "STAGE 3C — ORIGINAL EMPIRICAL MARGINAL SENSITIVITY"
)

print(
    "=" * 90
)


df = pd.read_csv(
    INPUT_FILE
)

common_phase_df = pd.read_csv(
    COMMON_PHASE_FILE
)

common_boundary_df = pd.read_csv(
    COMMON_BOUNDARY_FILE
)


print(
    f"\nLoaded trajectory matrix:"
    f"\n{INPUT_FILE}"
)

print(
    f"\nRows:"
    f"\n  {len(df):,}"
)


# ------------------------------------------------------------
# 4. Retrieve one copy of each structure
# ------------------------------------------------------------
#
# The physical RMS level is irrelevant because every trajectory
# is re-standardized to unit RMS after interpolation.
#
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


SELECTED_RMS_M = float(
    selected_level[
        "Target_Centered_RMS_m"
    ]
)


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
    f"\nControlled level used to retrieve structures:"
)

print(
    f"  {SELECTED_RMS_ID}"
    f" = "
    f"{SELECTED_RMS_M:.9f} m"
)


print(
    f"\nStructures:"
    f"\n  {len(trajectory_ids)}"
)


# ------------------------------------------------------------
# 5. Native structures
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


    x = x[
        finite
    ]

    error = error[
        finite
    ]


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
# 6. Common empirical spatial support
# ------------------------------------------------------------

grid = common_grid_for_all(
    raw_store,
    DX_M
)


N = len(
    grid
)


FIELD_LENGTH_M = (
    grid[-1]
    - grid[0]
)


print(
    "\nCommon empirical spatial support:"
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
# 7. Build ORIGINAL-MARGINAL standardized structure library
# ------------------------------------------------------------

structure_store = {}

structure_rows = []


for trajectory_id in trajectory_ids:

    record = raw_store[
        trajectory_id
    ]


    z = np.interp(
        grid,
        record[
            "X"
        ],
        record[
            "Error"
        ]
    )


    # Re-center after common-domain interpolation.
    z = (
        z
        - np.mean(
            z
        )
    )


    z = (
        z
        / rms(
            z
        )
    )


    structure_store[
        trajectory_id
    ] = z


    positive = np.maximum(
        z,
        0.0
    )

    negative = np.maximum(
        -z,
        0.0
    )


    skewness, excess_kurtosis = standardized_moments(
        z
    )


    structure_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Mean":
                np.mean(
                    z
                ),

            "RMS":
                rms(
                    z
                ),

            "MAE_Factor":
                np.mean(
                    np.abs(
                        z
                    )
                ),

            "Positive_Boundary_Factor":
                np.mean(
                    positive
                ),

            "Negative_Boundary_Factor":
                np.mean(
                    negative
                ),

            "Positive_Fraction":
                np.mean(
                    z > 0
                ),

            "Negative_Fraction":
                np.mean(
                    z < 0
                ),

            "Skewness":
                skewness,

            "Excess_Kurtosis":
                excess_kurtosis,

            "Minimum":
                np.min(
                    z
                ),

            "Maximum":
                np.max(
                    z
                )
        }
    )


structure_df = pd.DataFrame(
    structure_rows
)


# ------------------------------------------------------------
# 8. Unit-standardization verification
# ------------------------------------------------------------

maximum_absolute_mean = (
    structure_df[
        "Mean"
    ].abs().max()
)


maximum_rms_error = (
    structure_df[
        "RMS"
    ]
    - 1.0
).abs().max()


positive_negative_boundary_error = (
    structure_df[
        "Positive_Boundary_Factor"
    ]
    -
    structure_df[
        "Negative_Boundary_Factor"
    ]
).abs().max()


mae_identity_error = (
    structure_df[
        "MAE_Factor"
    ]
    -
    2.0
    * structure_df[
        "Positive_Boundary_Factor"
    ]
).abs().max()


# ------------------------------------------------------------
# 9. Exact all-phase K for every unordered structure pair
# ------------------------------------------------------------

pair_rows = []


for pair_number, (
    trajectory_1,
    trajectory_2
) in enumerate(
    itertools.combinations(
        trajectory_ids,
        2
    ),
    start=1
):

    z1 = structure_store[
        trajectory_1
    ]

    z2 = structure_store[
        trajectory_2
    ]


    k_phase_averaged = (
        mean_absolute_difference_between_marginals(
            z1,
            z2
        )
    )


    k_zero_phase = np.mean(
        np.abs(
            z2
            - z1
        )
    )


    row_1 = structure_df[
        structure_df[
            "Trajectory_ID"
        ]
        == trajectory_1
    ].iloc[
        0
    ]


    row_2 = structure_df[
        structure_df[
            "Trajectory_ID"
        ]
        == trajectory_2
    ].iloc[
        0
    ]


    pair_rows.append(
        {
            "Pair_ID":
                f"PAIR_{pair_number:03d}",

            "Trajectory_1":
                trajectory_1,

            "Trajectory_2":
                trajectory_2,

            "Zero_Phase_K_L1":
                k_zero_phase,

            "Exact_All_Phase_Mean_K_L1":
                k_phase_averaged,

            "Absolute_MAE_Factor_Difference":
                abs(
                    row_1[
                        "MAE_Factor"
                    ]
                    -
                    row_2[
                        "MAE_Factor"
                    ]
                ),

            "Absolute_Skewness_Difference":
                abs(
                    row_1[
                        "Skewness"
                    ]
                    -
                    row_2[
                        "Skewness"
                    ]
                ),

            "Absolute_Excess_Kurtosis_Difference":
                abs(
                    row_1[
                        "Excess_Kurtosis"
                    ]
                    -
                    row_2[
                        "Excess_Kurtosis"
                    ]
                )
        }
    )


pair_df = pd.DataFrame(
    pair_rows
)


# ------------------------------------------------------------
# 10. Common-marginal reference values
# ------------------------------------------------------------

COMMON_K = np.mean(
    common_phase_df[
        "K_L1"
    ].to_numpy(
        dtype=float
    )
)


COMMON_BOUNDARY_FACTOR = np.mean(
    common_boundary_df[
        "Mean_Excursion_Factor"
    ].to_numpy(
        dtype=float
    )
)


print(
    "\nCommon-marginal reference coefficients:"
)

print(
    f"  K_all_phase = "
    f"{COMMON_K:.12f}"
)

print(
    f"  B_one_edge  = "
    f"{COMMON_BOUNDARY_FACTOR:.12f}"
)


# ------------------------------------------------------------
# 11. Original empirical-marginal summary coefficients
# ------------------------------------------------------------

ORIGINAL_K_MEAN = np.mean(
    pair_df[
        "Exact_All_Phase_Mean_K_L1"
    ]
)


ORIGINAL_K_MEDIAN = np.median(
    pair_df[
        "Exact_All_Phase_Mean_K_L1"
    ]
)


ORIGINAL_K_MIN = np.min(
    pair_df[
        "Exact_All_Phase_Mean_K_L1"
    ]
)


ORIGINAL_K_MAX = np.max(
    pair_df[
        "Exact_All_Phase_Mean_K_L1"
    ]
)


ORIGINAL_B_MEAN = np.mean(
    structure_df[
        "Positive_Boundary_Factor"
    ]
)


ORIGINAL_B_MEDIAN = np.median(
    structure_df[
        "Positive_Boundary_Factor"
    ]
)


ORIGINAL_B_MIN = np.min(
    structure_df[
        "Positive_Boundary_Factor"
    ]
)


ORIGINAL_B_MAX = np.max(
    structure_df[
        "Positive_Boundary_Factor"
    ]
)


# ------------------------------------------------------------
# 12. Boundary persistence with ORIGINAL marginals
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

            **positive_stats
        }
    )


    boundary_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Direction":
                "Negative",

            **negative_stats
        }
    )


original_boundary_df = pd.DataFrame(
    boundary_rows
)


# ------------------------------------------------------------
# 13. Match original versus common-marginal boundary metrics
# ------------------------------------------------------------

common_boundary_reduced = common_boundary_df[
    [
        "Trajectory_ID",
        "Direction",
        "Mean_Excursion_Factor",
        "Fraction_of_Field_in_Excursion",
        "Number_of_Excursion_Clusters",
        "Longest_Excursion_Run_m",
        "Largest_Excursion_Cluster_Area_Factor_m",
        "Maximum_Instantaneous_Excursion_Factor"
    ]
].copy()


common_boundary_reduced = common_boundary_reduced.rename(
    columns={
        "Mean_Excursion_Factor":
            "Common_Mean_Excursion_Factor",

        "Fraction_of_Field_in_Excursion":
            "Common_Excursion_Fraction",

        "Number_of_Excursion_Clusters":
            "Common_Number_of_Clusters",

        "Longest_Excursion_Run_m":
            "Common_Longest_Run_m",

        "Largest_Excursion_Cluster_Area_Factor_m":
            "Common_Largest_Cluster_Area_Factor_m",

        "Maximum_Instantaneous_Excursion_Factor":
            "Common_Maximum_Excursion_Factor"
    }
)


boundary_compare_df = original_boundary_df.merge(
    common_boundary_reduced,
    on=[
        "Trajectory_ID",
        "Direction"
    ],
    how="left"
)


boundary_compare_df[
    "Longest_Run_Change_m"
] = (
    boundary_compare_df[
        "Longest_Run_m"
    ]
    -
    boundary_compare_df[
        "Common_Longest_Run_m"
    ]
)


boundary_compare_df[
    "Largest_Cluster_Area_Change_m"
] = (
    boundary_compare_df[
        "Largest_Cluster_Area_Factor_m"
    ]
    -
    boundary_compare_df[
        "Common_Largest_Cluster_Area_Factor_m"
    ]
)


# ------------------------------------------------------------
# 14. Sensitivity coefficient table
# ------------------------------------------------------------

sensitivity_df = pd.DataFrame(
    [
        {
            "Operational_Coefficient":
                "All-phase internal K",

            "Common_Marginal_Value":
                COMMON_K,

            "Original_Marginal_Ensemble_Mean":
                ORIGINAL_K_MEAN,

            "Original_Marginal_Min":
                ORIGINAL_K_MIN,

            "Original_Marginal_Median":
                ORIGINAL_K_MEDIAN,

            "Original_Marginal_Max":
                ORIGINAL_K_MAX,

            "Mean_Relative_Change":
                (
                    ORIGINAL_K_MEAN
                    - COMMON_K
                )
                / COMMON_K
        },

        {
            "Operational_Coefficient":
                "One-edge boundary B",

            "Common_Marginal_Value":
                COMMON_BOUNDARY_FACTOR,

            "Original_Marginal_Ensemble_Mean":
                ORIGINAL_B_MEAN,

            "Original_Marginal_Min":
                ORIGINAL_B_MIN,

            "Original_Marginal_Median":
                ORIGINAL_B_MEDIAN,

            "Original_Marginal_Max":
                ORIGINAL_B_MAX,

            "Mean_Relative_Change":
                (
                    ORIGINAL_B_MEAN
                    - COMMON_BOUNDARY_FACTOR
                )
                / COMMON_BOUNDARY_FACTOR
        }
    ]
)


# ------------------------------------------------------------
# 15. Multi-pass coefficient sensitivity
# ------------------------------------------------------------
#
# These are descriptive coefficient curves.
#
# No claim is made that empirical trajectory pairs constitute an
# independent random population.
#
# ------------------------------------------------------------

pass_counts = np.arange(
    2,
    101
)


multipass_rows = []


for m in pass_counts:

    common_internal = (
        (
            m - 1
        )
        / m
        * COMMON_K
    )


    original_internal = (
        (
            m - 1
        )
        / m
        * ORIGINAL_K_MEAN
    )


    common_boundary = (
        2.0
        * COMMON_BOUNDARY_FACTOR
        / m
    )


    original_boundary = (
        2.0
        * ORIGINAL_B_MEAN
        / m
    )


    multipass_rows.append(
        {
            "Number_of_Passes_M":
                m,

            "Common_Internal_over_eta":
                common_internal,

            "Original_Marginal_Internal_over_eta":
                original_internal,

            "Common_Two_Edge_Boundary_over_eta":
                common_boundary,

            "Original_Marginal_Two_Edge_Boundary_over_eta":
                original_boundary,

            "Common_Combined_over_eta":
                common_internal
                + common_boundary,

            "Original_Marginal_Combined_over_eta":
                original_internal
                + original_boundary
        }
    )


multipass_sensitivity_df = pd.DataFrame(
    multipass_rows
)


# ------------------------------------------------------------
# 16. Descriptive associations with all-phase K
# ------------------------------------------------------------

association_rows = []


for descriptor in [
    "Absolute_MAE_Factor_Difference",
    "Absolute_Skewness_Difference",
    "Absolute_Excess_Kurtosis_Difference"
]:

    rho = pair_df[
        [
            descriptor,
            "Exact_All_Phase_Mean_K_L1"
        ]
    ].corr(
        method="spearman"
    ).iloc[
        0,
        1
    ]


    association_rows.append(
        {
            "Descriptor":
                descriptor,

            "Spearman_Rho_with_All_Phase_K":
                rho
        }
    )


association_df = pd.DataFrame(
    association_rows
)


# ------------------------------------------------------------
# 17. Save tables
# ------------------------------------------------------------

structure_file = (
    TABLES_DIR
    / "original_marginal_structure_metrics.csv"
)

pair_file = (
    TABLES_DIR
    / "original_marginal_all_phase_pair_factors.csv"
)

boundary_file = (
    TABLES_DIR
    / "original_marginal_boundary_metrics.csv"
)

boundary_compare_file = (
    TABLES_DIR
    / "original_vs_common_marginal_boundary_sensitivity.csv"
)

sensitivity_file = (
    TABLES_DIR
    / "operational_coefficient_marginal_sensitivity.csv"
)

multipass_file = (
    TABLES_DIR
    / "multipass_original_vs_common_marginal_sensitivity.csv"
)

association_file = (
    TABLES_DIR
    / "original_marginal_descriptor_associations.csv"
)


structure_df.to_csv(
    structure_file,
    index=False
)

pair_df.to_csv(
    pair_file,
    index=False
)

original_boundary_df.to_csv(
    boundary_file,
    index=False
)

boundary_compare_df.to_csv(
    boundary_compare_file,
    index=False
)

sensitivity_df.to_csv(
    sensitivity_file,
    index=False
)

multipass_sensitivity_df.to_csv(
    multipass_file,
    index=False
)

association_df.to_csv(
    association_file,
    index=False
)


# ------------------------------------------------------------
# 18. Figure — internal K sensitivity
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    pair_df[
        "Exact_All_Phase_Mean_K_L1"
    ],
    bins="auto"
)


plt.axvline(
    COMMON_K,
    linestyle="--",
    linewidth=2,
    label="Common-marginal value"
)


plt.xlabel(
    "Exact all-phase mean K_L1"
)

plt.ylabel(
    "Number of empirical trajectory pairs"
)

plt.title(
    "Sensitivity of mean internal discrepancy to empirical marginal shape"
)

plt.legend()

plt.tight_layout()


k_figure = (
    FIGURES_DIR
    / "original_marginal_all_phase_K_distribution.png"
)


plt.savefig(
    k_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 19. Figure — boundary-factor sensitivity
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    structure_df[
        "Positive_Boundary_Factor"
    ],
    bins="auto"
)


plt.axvline(
    COMMON_BOUNDARY_FACTOR,
    linestyle="--",
    linewidth=2,
    label="Common-marginal value"
)


plt.xlabel(
    "One-sided unit-RMS boundary factor B"
)

plt.ylabel(
    "Number of empirical structures"
)

plt.title(
    "Sensitivity of mean boundary magnitude to empirical marginal shape"
)

plt.legend()

plt.tight_layout()


boundary_factor_figure = (
    FIGURES_DIR
    / "original_marginal_boundary_factor_distribution.png"
)


plt.savefig(
    boundary_factor_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 20. Figure — boundary persistence comparison
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    boundary_compare_df[
        "Common_Longest_Run_m"
    ],
    boundary_compare_df[
        "Longest_Run_m"
    ]
)


minimum_axis = min(
    boundary_compare_df[
        "Common_Longest_Run_m"
    ].min(),
    boundary_compare_df[
        "Longest_Run_m"
    ].min()
)


maximum_axis = max(
    boundary_compare_df[
        "Common_Longest_Run_m"
    ].max(),
    boundary_compare_df[
        "Longest_Run_m"
    ].max()
)


plt.plot(
    [
        minimum_axis,
        maximum_axis
    ],
    [
        minimum_axis,
        maximum_axis
    ],
    linestyle="--"
)


plt.xlabel(
    "Common-marginal longest boundary run (m)"
)

plt.ylabel(
    "Original-marginal longest boundary run (m)"
)

plt.title(
    "Effect of marginal restoration on boundary-event persistence"
)

plt.tight_layout()


persistence_figure = (
    FIGURES_DIR
    / "original_vs_common_boundary_persistence.png"
)


plt.savefig(
    persistence_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 21. Figure — combined multi-pass sensitivity
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    multipass_sensitivity_df[
        "Number_of_Passes_M"
    ],
    multipass_sensitivity_df[
        "Common_Combined_over_eta"
    ],
    linewidth=2,
    label="Common marginal"
)


plt.plot(
    multipass_sensitivity_df[
        "Number_of_Passes_M"
    ],
    multipass_sensitivity_df[
        "Original_Marginal_Combined_over_eta"
    ],
    linewidth=2,
    label="Original empirical marginals"
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "Mean combined discrepancy / eta"
)

plt.title(
    "Sensitivity of dimensionless field coefficient to marginal model"
)

plt.legend()

plt.tight_layout()


multipass_figure = (
    FIGURES_DIR
    / "multipass_original_vs_common_marginal.png"
)


plt.savefig(
    multipass_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 22. Terminal formatting
# ------------------------------------------------------------

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    320
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
    "UNIT-RMS ORIGINAL-MARGINAL VERIFICATION"
)

print(
    "=" * 90
)


print(
    "\nMaximum absolute structure mean:"
)

print(
    f"  {maximum_absolute_mean:.12e}"
)


print(
    "\nMaximum |RMS - 1|:"
)

print(
    f"  {maximum_rms_error:.12e}"
)


print(
    "\nMaximum |B_positive - B_negative|:"
)

print(
    f"  {positive_negative_boundary_error:.12e}"
)


print(
    "\nMaximum |MAE - 2B|:"
)

print(
    f"  {mae_identity_error:.12e}"
)


# ------------------------------------------------------------
# 24. Structure marginal diversity
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "ORIGINAL EMPIRICAL MARGINAL DIVERSITY"
)

print(
    "=" * 90
)


for column, label in [
    (
        "MAE_Factor",
        "Unit-RMS MAE factor"
    ),
    (
        "Positive_Boundary_Factor",
        "One-sided boundary factor B"
    ),
    (
        "Positive_Fraction",
        "Positive occupancy fraction"
    ),
    (
        "Skewness",
        "Skewness"
    ),
    (
        "Excess_Kurtosis",
        "Excess kurtosis"
    )
]:

    values = structure_df[
        column
    ]


    print(
        f"\n{label}:"
    )

    print(
        f"  min    = "
        f"{values.min():.9f}"
    )

    print(
        f"  median = "
        f"{values.median():.9f}"
    )

    print(
        f"  max    = "
        f"{values.max():.9f}"
    )


# ------------------------------------------------------------
# 25. Exact all-phase internal coefficient
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "EXACT ALL-PHASE INTERNAL K — ORIGINAL EMPIRICAL MARGINALS"
)

print(
    "=" * 90
)


all_phase_values = pair_df[
    "Exact_All_Phase_Mean_K_L1"
]


print(
    f"\nNumber of empirical pairs:"
    f"\n  {len(pair_df)}"
)


print(
    "\nOriginal-marginal all-phase K:"
)

print(
    f"  min    = "
    f"{all_phase_values.min():.9f}"
)

print(
    f"  Q25    = "
    f"{all_phase_values.quantile(0.25):.9f}"
)

print(
    f"  median = "
    f"{all_phase_values.median():.9f}"
)

print(
    f"  Q75    = "
    f"{all_phase_values.quantile(0.75):.9f}"
)

print(
    f"  max    = "
    f"{all_phase_values.max():.9f}"
)

print(
    f"  mean   = "
    f"{all_phase_values.mean():.9f}"
)


print(
    "\nCommon-marginal reference:"
)

print(
    f"  {COMMON_K:.9f}"
)


print(
    "\nRelative change in grand mean:"
)

print(
    f"  "
    f"{100.0 * (ORIGINAL_K_MEAN - COMMON_K) / COMMON_K:.6f}%"
)


# ------------------------------------------------------------
# 26. Boundary-factor sensitivity
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "BOUNDARY MAGNITUDE — ORIGINAL EMPIRICAL MARGINALS"
)

print(
    "=" * 90
)


boundary_values = structure_df[
    "Positive_Boundary_Factor"
]


print(
    "\nOne-edge B factor:"
)

print(
    f"  min    = "
    f"{boundary_values.min():.9f}"
)

print(
    f"  Q25    = "
    f"{boundary_values.quantile(0.25):.9f}"
)

print(
    f"  median = "
    f"{boundary_values.median():.9f}"
)

print(
    f"  Q75    = "
    f"{boundary_values.quantile(0.75):.9f}"
)

print(
    f"  max    = "
    f"{boundary_values.max():.9f}"
)

print(
    f"  mean   = "
    f"{boundary_values.mean():.9f}"
)


print(
    "\nCommon-marginal reference:"
)

print(
    f"  {COMMON_BOUNDARY_FACTOR:.9f}"
)


print(
    "\nRelative change in ensemble mean:"
)

print(
    f"  "
    f"{100.0 * (ORIGINAL_B_MEAN - COMMON_BOUNDARY_FACTOR) / COMMON_BOUNDARY_FACTOR:.6f}%"
)


# ------------------------------------------------------------
# 27. Boundary persistence sensitivity
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "BOUNDARY PERSISTENCE SENSITIVITY"
)

print(
    "=" * 90
)


print(
    "\nOriginal-marginal longest boundary runs:"
)

print(
    f"  min    = "
    f"{original_boundary_df['Longest_Run_m'].min():.3f} m"
)

print(
    f"  median = "
    f"{original_boundary_df['Longest_Run_m'].median():.3f} m"
)

print(
    f"  max    = "
    f"{original_boundary_df['Longest_Run_m'].max():.3f} m"
)


print(
    "\nChange relative to common-marginal runs:"
)

print(
    f"  minimum change = "
    f"{boundary_compare_df['Longest_Run_Change_m'].min():.3f} m"
)

print(
    f"  median change  = "
    f"{boundary_compare_df['Longest_Run_Change_m'].median():.3f} m"
)

print(
    f"  maximum change = "
    f"{boundary_compare_df['Longest_Run_Change_m'].max():.3f} m"
)


# ------------------------------------------------------------
# 28. Sensitivity coefficient table
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "OPERATIONAL COEFFICIENT SENSITIVITY"
)

print(
    "=" * 90
)


print(
    sensitivity_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 29. Descriptive marginal associations
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DESCRIPTIVE MARGINAL ASSOCIATIONS WITH ALL-PHASE K"
)

print(
    "=" * 90
)


print(
    association_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 30. Multi-pass examples
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


example_df = multipass_sensitivity_df[
    multipass_sensitivity_df[
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
    "MULTI-PASS ORIGINAL-VERSUS-COMMON MARGINAL SENSITIVITY"
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
# 31. Output files
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
    f"\nOriginal-marginal structure metrics:\n"
    f"{structure_file}"
)

print(
    f"\nExact all-phase pair factors:\n"
    f"{pair_file}"
)

print(
    f"\nOriginal boundary metrics:\n"
    f"{boundary_file}"
)

print(
    f"\nOriginal/common boundary comparison:\n"
    f"{boundary_compare_file}"
)

print(
    f"\nOperational coefficient sensitivity:\n"
    f"{sensitivity_file}"
)

print(
    f"\nMulti-pass marginal sensitivity:\n"
    f"{multipass_file}"
)

print(
    f"\nMarginal descriptor associations:\n"
    f"{association_file}"
)

print(
    f"\nK sensitivity figure:\n"
    f"{k_figure}"
)

print(
    f"\nBoundary-factor sensitivity figure:\n"
    f"{boundary_factor_figure}"
)

print(
    f"\nBoundary-persistence comparison:\n"
    f"{persistence_figure}"
)

print(
    f"\nMulti-pass sensitivity figure:\n"
    f"{multipass_figure}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 3C COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nThe common-marginal experiment isolates spatial "
    "organization; this stage restores the empirical marginal "
    "shapes while keeping every structure centered and unit RMS."
)

print(
    "\nIf the principal dimensionless coefficients remain close "
    "to the common-marginal values, the main operational "
    "conclusions are robust to marginal-distribution choice."
)

print(
    "If they change substantially, marginal shape must be "
    "retained explicitly in the final tolerance curves."
)