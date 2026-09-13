from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import ndtri


# ============================================================
# STAGE 2D — EXHAUSTIVE FINITE-FIELD PHASE / CLUSTERING ANALYSIS
# ============================================================
#
# PURPOSE
# -------
# Stage 2C established that when:
#
#   - centered RMS is identical,
#   - marginal distribution is identical,
#   - relative spatial phase is unrestricted,
#
# the long-run mean pointwise L1 discrepancy is common to all
# empirical structure pairs.
#
# Therefore spatial organization should not be interpreted as
# changing the infinite-domain expected local mean directly.
#
# The appropriate operational consequences are instead:
#
#   - realization-to-realization variability,
#   - persistence,
#   - clustering,
#   - contiguous skip episodes,
#   - contiguous overlap episodes,
#   - concentration of discrepancy into spatially extended events.
#
#
# THIS STAGE
# ----------
#
# We exhaustively evaluate:
#
#       28 structures
#       -> 378 unordered pairs
#
# and for every pair:
#
#       ALL possible circular relative shifts.
#
# With N common spatial samples:
#
#       number of scenarios = 378 * N
#
# There is therefore:
#
#       NO Monte Carlo sample count,
#       NO random seed,
#       NO arbitrary sampling frequency.
#
#
# COMMON MARGINAL
# ---------------
#
# As in Stage 2C, every structure is rank-transformed to the
# exact same deterministic Gaussian-score marginal.
#
# Therefore differences in the metrics below arise from:
#
#       spatial ordering + relative phase
#
# rather than RMS or marginal-distribution differences.
#
#
# FIELD LENGTH
# ------------
#
# The analysis uses the complete common empirical spatial support
# determined directly from the released trajectories.
#
# No synthetic field length is introduced.
#
#
# LOCAL GEOMETRY
# --------------
#
# Let:
#
#       delta(x) = z2(x) - z1(x)
#
# Then:
#
#       delta > 0  -> local skip
#       delta < 0  -> local overlap
#
# No magnitude threshold is imposed.
#
#
# CLUSTER METRICS
# ---------------
#
# For every pair × phase scenario:
#
#   1. K_L1
#      mean absolute unit-RMS spacing discrepancy.
#
#   2. Longest skip run
#      longest contiguous region where delta > 0.
#
#   3. Longest overlap run
#      longest contiguous region where delta < 0.
#
#   4. Largest skip-cluster area factor
#      maximum integral of positive delta within one contiguous
#      positive cluster.
#
#   5. Largest overlap-cluster area factor
#      corresponding negative cluster magnitude.
#
#   6. Number of skip clusters.
#
#   7. Number of overlap clusters.
#
#
# DIMENSIONLESS INTERPRETATION
# ----------------------------
#
# If physical centered RMS is R and working width is W:
#
#       eta = R / W
#
# then:
#
#   normalized local discrepancy = eta * delta
#
# Hence:
#
#   normalized cluster-area effects scale linearly with eta.
#
# Cluster LENGTHS do not depend on R or W.
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

TABLES_DIR = (
    PROJECT_DIR
    / "results"
    / "tables"
)

FIGURES_DIR = (
    PROJECT_DIR
    / "results"
    / "figures"
    / "exhaustive_phase_clustering"
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
            record[
                "X"
            ]
        )
        for record
        in trajectory_dict.values()
    ]

    ends = [
        np.max(
            record[
                "X"
            ]
        )
        for record
        in trajectory_dict.values()
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

    if len(
        grid
    ) < 2:

        raise RuntimeError(
            "Common grid contains fewer than two samples."
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

    if len(
        values
    ) != len(
        common_scores
    ):

        raise RuntimeError(
            "Rank-transform length mismatch."
        )

    order = np.argsort(
        values,
        kind="mergesort"
    )

    ranks = np.empty(
        len(
            values
        ),
        dtype=int
    )

    ranks[
        order
    ] = np.arange(
        len(
            values
        )
    )

    return common_scores[
        ranks
    ]


def linear_runs(
    condition
):

    condition = np.asarray(
        condition,
        dtype=bool
    )

    n = len(
        condition
    )

    if n == 0:

        return []


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
        padded.astype(
            int
        )
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


def run_statistics(
    delta,
    dx
):

    delta = np.asarray(
        delta,
        dtype=float
    )


    positive = (
        delta
        > 0
    )

    negative = (
        delta
        < 0
    )


    positive_runs = linear_runs(
        positive
    )

    negative_runs = linear_runs(
        negative
    )


    # --------------------------------------------
    # Skip clusters
    # --------------------------------------------

    skip_lengths = []

    skip_area_factors = []


    for start, end in positive_runs:

        run_values = delta[
            start:end
        ]


        run_length = (
            len(
                run_values
            )
            * dx
        )


        run_area_factor = (
            np.sum(
                run_values
            )
            * dx
        )


        skip_lengths.append(
            run_length
        )

        skip_area_factors.append(
            run_area_factor
        )


    # --------------------------------------------
    # Overlap clusters
    # --------------------------------------------

    overlap_lengths = []

    overlap_area_factors = []


    for start, end in negative_runs:

        run_values = (
            -delta[
                start:end
            ]
        )


        run_length = (
            len(
                run_values
            )
            * dx
        )


        run_area_factor = (
            np.sum(
                run_values
            )
            * dx
        )


        overlap_lengths.append(
            run_length
        )

        overlap_area_factors.append(
            run_area_factor
        )


    longest_skip = (
        max(
            skip_lengths
        )
        if len(
            skip_lengths
        ) > 0
        else 0.0
    )


    longest_overlap = (
        max(
            overlap_lengths
        )
        if len(
            overlap_lengths
        ) > 0
        else 0.0
    )


    largest_skip_area = (
        max(
            skip_area_factors
        )
        if len(
            skip_area_factors
        ) > 0
        else 0.0
    )


    largest_overlap_area = (
        max(
            overlap_area_factors
        )
        if len(
            overlap_area_factors
        ) > 0
        else 0.0
    )


    return {
        "Number_of_Skip_Clusters":
            len(
                skip_lengths
            ),

        "Number_of_Overlap_Clusters":
            len(
                overlap_lengths
            ),

        "Longest_Skip_Run_m":
            longest_skip,

        "Longest_Overlap_Run_m":
            longest_overlap,

        "Largest_Skip_Cluster_Area_Factor_m":
            largest_skip_area,

        "Largest_Overlap_Cluster_Area_Factor_m":
            largest_overlap_area
    }


def quantile_summary(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    return {
        "Min":
            np.min(
                values
            ),

        "Q05":
            np.quantile(
                values,
                0.05
            ),

        "Q25":
            np.quantile(
                values,
                0.25
            ),

        "Median":
            np.median(
                values
            ),

        "Q75":
            np.quantile(
                values,
                0.75
            ),

        "Q95":
            np.quantile(
                values,
                0.95
            ),

        "Max":
            np.max(
                values
            ),

        "Mean":
            np.mean(
                values
            ),

        "SD":
            np.std(
                values,
                ddof=0
            )
    }


# ------------------------------------------------------------
# 3. Load input
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
    "STAGE 2D — EXHAUSTIVE FINITE-FIELD PHASE / CLUSTERING ANALYSIS"
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
# 4. Retrieve one copy of all controlled structures
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
    len(
        rms_levels
    )
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
# 5. Load raw controlled structures
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
        np.isfinite(
            x
        )
        &
        np.isfinite(
            error
        )
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


    x = unique_x

    error = error[
        unique_indices
    ]


    raw_store[
        trajectory_id
    ] = {
        "X":
            x,

        "Error":
            error
    }


# ------------------------------------------------------------
# 6. Common empirical support
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
    "\nCommon empirical field support:"
)

print(
    f"  start       = "
    f"{grid[0]:.3f} m"
)

print(
    f"  end         = "
    f"{grid[-1]:.3f} m"
)

print(
    f"  field length= "
    f"{FIELD_LENGTH_M:.3f} m"
)

print(
    f"  grid points = "
    f"{N:,}"
)

print(
    f"  spacing     = "
    f"{DX_M:.3f} m"
)


# ------------------------------------------------------------
# 7. Exact common marginal
# ------------------------------------------------------------

probabilities = (
    (
        np.arange(
            N
        )
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
# 8. Build common-marginal unit-RMS structure library
# ------------------------------------------------------------

structure_store = {}


for trajectory_id in trajectory_ids:

    record = raw_store[
        trajectory_id
    ]


    interpolated = np.interp(
        grid,
        record[
            "X"
        ],
        record[
            "Error"
        ]
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
# 9. Verify common marginal
# ------------------------------------------------------------

reference_sorted = np.sort(
    structure_store[
        trajectory_ids[0]
    ]
)


max_marginal_difference = 0.0


for trajectory_id in trajectory_ids[1:]:

    difference = np.max(
        np.abs(
            np.sort(
                structure_store[
                    trajectory_id
                ]
            )
            -
            reference_sorted
        )
    )


    max_marginal_difference = max(
        max_marginal_difference,
        difference
    )


# ------------------------------------------------------------
# 10. Enumerate every pair and every phase
# ------------------------------------------------------------

pairs = list(
    itertools.combinations(
        trajectory_ids,
        2
    )
)


number_of_pairs = len(
    pairs
)


number_of_scenarios = (
    number_of_pairs
    * N
)


print(
    f"\nUnordered structure pairs:"
    f"\n  {number_of_pairs}"
)

print(
    f"\nCircular phases per pair:"
    f"\n  {N}"
)

print(
    f"\nTotal exhaustive scenarios:"
    f"\n  {number_of_scenarios:,}"
)


scenario_rows = []


for pair_index, (
    trajectory_1,
    trajectory_2
) in enumerate(
    pairs,
    start=1
):

    z1 = structure_store[
        trajectory_1
    ]

    z2_original = structure_store[
        trajectory_2
    ]


    for shift_index in range(
        N
    ):

        z2 = np.roll(
            z2_original,
            shift_index
        )


        delta = (
            z2
            - z1
        )


        k_l1 = np.mean(
            np.abs(
                delta
            )
        )


        k_skip = np.mean(
            np.maximum(
                delta,
                0.0
            )
        )


        k_overlap = np.mean(
            np.maximum(
                -delta,
                0.0
            )
        )


        cluster_metrics = run_statistics(
            delta,
            DX_M
        )


        scenario_rows.append(
            {
                "Pair_ID":
                    f"PAIR_{pair_index:03d}",

                "Trajectory_1":
                    trajectory_1,

                "Trajectory_2":
                    trajectory_2,

                "Shift_Index":
                    shift_index,

                "Shift_Distance_m":
                    shift_index
                    * DX_M,

                "K_L1":
                    k_l1,

                "K_Skip":
                    k_skip,

                "K_Overlap":
                    k_overlap,

                **cluster_metrics
            }
        )


    if (
        pair_index % 25 == 0
        or pair_index
        == number_of_pairs
    ):

        print(
            f"Processed pairs: "
            f"{pair_index}/{number_of_pairs}"
        )


scenario_df = pd.DataFrame(
    scenario_rows
)


# ------------------------------------------------------------
# 11. Add normalized run lengths
# ------------------------------------------------------------

scenario_df[
    "Longest_Skip_Run_Fraction"
] = (
    scenario_df[
        "Longest_Skip_Run_m"
    ]
    / FIELD_LENGTH_M
)


scenario_df[
    "Longest_Overlap_Run_Fraction"
] = (
    scenario_df[
        "Longest_Overlap_Run_m"
    ]
    / FIELD_LENGTH_M
)


# ------------------------------------------------------------
# 12. Pair-level phase summaries
# ------------------------------------------------------------

pair_summary_rows = []


metric_columns = [
    "K_L1",
    "Longest_Skip_Run_m",
    "Longest_Overlap_Run_m",
    "Largest_Skip_Cluster_Area_Factor_m",
    "Largest_Overlap_Cluster_Area_Factor_m",
    "Number_of_Skip_Clusters",
    "Number_of_Overlap_Clusters"
]


for pair_id, group in scenario_df.groupby(
    "Pair_ID",
    sort=True
):

    first = group.iloc[
        0
    ]


    summary_record = {
        "Pair_ID":
            pair_id,

        "Trajectory_1":
            first[
                "Trajectory_1"
            ],

        "Trajectory_2":
            first[
                "Trajectory_2"
            ]
    }


    for metric in metric_columns:

        stats = quantile_summary(
            group[
                metric
            ].to_numpy(
                dtype=float
            )
        )


        for stat_name, stat_value in stats.items():

            summary_record[
                f"{metric}_{stat_name}"
            ] = stat_value


    pair_summary_rows.append(
        summary_record
    )


pair_summary_df = pd.DataFrame(
    pair_summary_rows
)


# ------------------------------------------------------------
# 13. Dataset-wide scenario summaries
# ------------------------------------------------------------

dataset_summary_rows = []


dataset_metrics = [
    "K_L1",
    "Longest_Skip_Run_m",
    "Longest_Overlap_Run_m",
    "Longest_Skip_Run_Fraction",
    "Longest_Overlap_Run_Fraction",
    "Largest_Skip_Cluster_Area_Factor_m",
    "Largest_Overlap_Cluster_Area_Factor_m",
    "Number_of_Skip_Clusters",
    "Number_of_Overlap_Clusters"
]


for metric in dataset_metrics:

    stats = quantile_summary(
        scenario_df[
            metric
        ].to_numpy(
            dtype=float
        )
    )


    row = {
        "Metric":
            metric
    }


    row.update(
        stats
    )


    dataset_summary_rows.append(
        row
    )


dataset_summary_df = pd.DataFrame(
    dataset_summary_rows
)


# ------------------------------------------------------------
# 14. Phase sensitivity ranking
# ------------------------------------------------------------

phase_ranking_df = pair_summary_df[
    [
        "Pair_ID",
        "Trajectory_1",
        "Trajectory_2",

        "K_L1_SD",

        "Longest_Skip_Run_m_SD",
        "Longest_Overlap_Run_m_SD",

        "Largest_Skip_Cluster_Area_Factor_m_SD",
        "Largest_Overlap_Cluster_Area_Factor_m_SD"
    ]
].copy()


phase_ranking_df[
    "Mean_Longest_Run_SD_m"
] = (
    (
        phase_ranking_df[
            "Longest_Skip_Run_m_SD"
        ]
        +
        phase_ranking_df[
            "Longest_Overlap_Run_m_SD"
        ]
    )
    / 2.0
)


phase_ranking_df = phase_ranking_df.sort_values(
    "Mean_Longest_Run_SD_m",
    ascending=False
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# 15. Correlation of mean discrepancy with persistence
# ------------------------------------------------------------

association_columns = [
    "K_L1",
    "Longest_Skip_Run_m",
    "Longest_Overlap_Run_m",
    "Largest_Skip_Cluster_Area_Factor_m",
    "Largest_Overlap_Cluster_Area_Factor_m",
    "Number_of_Skip_Clusters",
    "Number_of_Overlap_Clusters"
]


association_matrix = scenario_df[
    association_columns
].corr(
    method="spearman"
)


association_file = (
    TABLES_DIR
    / "phase_clustering_spearman_matrix.csv"
)


association_matrix.to_csv(
    association_file
)


# ------------------------------------------------------------
# 16. Numerical identities
# ------------------------------------------------------------

identity_error = np.abs(
    (
        scenario_df[
            "K_Skip"
        ]
        +
        scenario_df[
            "K_Overlap"
        ]
    )
    -
    scenario_df[
        "K_L1"
    ]
)


max_identity_error = identity_error.max()


all_pair_phase_mean = (
    scenario_df
    .groupby(
        "Pair_ID"
    )[
        "K_L1"
    ]
    .mean()
)


max_pair_phase_mean_difference = (
    all_pair_phase_mean.max()
    -
    all_pair_phase_mean.min()
)


# ------------------------------------------------------------
# 17. Save tables
# ------------------------------------------------------------

scenario_file = (
    TABLES_DIR
    / "exhaustive_phase_clustering_scenarios.csv"
)

pair_summary_file = (
    TABLES_DIR
    / "exhaustive_phase_clustering_pair_summary.csv"
)

dataset_summary_file = (
    TABLES_DIR
    / "exhaustive_phase_clustering_dataset_summary.csv"
)

ranking_file = (
    TABLES_DIR
    / "exhaustive_phase_clustering_pair_ranking.csv"
)


scenario_df.to_csv(
    scenario_file,
    index=False
)

pair_summary_df.to_csv(
    pair_summary_file,
    index=False
)

dataset_summary_df.to_csv(
    dataset_summary_file,
    index=False
)

phase_ranking_df.to_csv(
    ranking_file,
    index=False
)


# ------------------------------------------------------------
# 18. Figure — longest skip-run distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    scenario_df[
        "Longest_Skip_Run_m"
    ],
    bins="auto"
)


plt.xlabel(
    "Longest contiguous skip run (m)"
)

plt.ylabel(
    "Number of pair-phase scenarios"
)

plt.title(
    "Finite-field persistence of skip episodes"
)

plt.tight_layout()


skip_run_figure = (
    FIGURES_DIR
    / "longest_skip_run_distribution.png"
)


plt.savefig(
    skip_run_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 19. Figure — longest overlap-run distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    scenario_df[
        "Longest_Overlap_Run_m"
    ],
    bins="auto"
)


plt.xlabel(
    "Longest contiguous overlap run (m)"
)

plt.ylabel(
    "Number of pair-phase scenarios"
)

plt.title(
    "Finite-field persistence of overlap episodes"
)

plt.tight_layout()


overlap_run_figure = (
    FIGURES_DIR
    / "longest_overlap_run_distribution.png"
)


plt.savefig(
    overlap_run_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 20. Figure — K versus longest cluster
# ------------------------------------------------------------

scenario_df[
    "Longest_Event_Run_m"
] = np.maximum(
    scenario_df[
        "Longest_Skip_Run_m"
    ],
    scenario_df[
        "Longest_Overlap_Run_m"
    ]
)


plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    scenario_df[
        "K_L1"
    ],
    scenario_df[
        "Longest_Event_Run_m"
    ],
    s=4,
    alpha=0.15
)


plt.xlabel(
    "Mean dimensionless discrepancy K_L1"
)

plt.ylabel(
    "Longest contiguous skip/overlap run (m)"
)

plt.title(
    "Mean discrepancy versus finite-field spatial persistence"
)

plt.tight_layout()


k_vs_persistence_figure = (
    FIGURES_DIR
    / "K_vs_longest_event_persistence.png"
)


plt.savefig(
    k_vs_persistence_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 21. Figure — phase sensitivity ranking
# ------------------------------------------------------------

top_phase_sensitive = phase_ranking_df.head(
    20
).copy()


labels = (
    top_phase_sensitive[
        "Trajectory_1"
    ]
    +
    " | "
    +
    top_phase_sensitive[
        "Trajectory_2"
    ]
)


plt.figure(
    figsize=(11, 7)
)


plt.barh(
    labels[::-1],
    top_phase_sensitive[
        "Mean_Longest_Run_SD_m"
    ][::-1]
)


plt.xlabel(
    "Mean SD of longest skip/overlap run across phase (m)"
)

plt.ylabel(
    "Empirical structure pair"
)

plt.title(
    "Pairs most sensitive to relative spatial phase"
)

plt.tight_layout()


ranking_figure = (
    FIGURES_DIR
    / "phase_sensitivity_ranking.png"
)


plt.savefig(
    ranking_figure,
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
    100
)


# ------------------------------------------------------------
# 23. Print verification
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "COMMON-MARGINAL / EXHAUSTIVE-PHASE VERIFICATION"
)

print(
    "=" * 90
)


print(
    f"\nMaximum common-marginal discrepancy:"
)

print(
    f"  {max_marginal_difference:.12e}"
)


print(
    f"\nMaximum error in "
    f"K_skip + K_overlap = K_L1:"
)

print(
    f"  {max_identity_error:.12e}"
)


print(
    "\nRange of pair-specific all-phase mean K_L1:"
)

print(
    f"  minimum = "
    f"{all_pair_phase_mean.min():.12f}"
)

print(
    f"  maximum = "
    f"{all_pair_phase_mean.max():.12f}"
)

print(
    f"  range   = "
    f"{max_pair_phase_mean_difference:.12e}"
)


# ------------------------------------------------------------
# 24. Print dataset-wide clustering results
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DATASET-WIDE EXHAUSTIVE FINITE-FIELD DISTRIBUTIONS"
)

print(
    "=" * 90
)


print(
    dataset_summary_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 25. Longest-run specific interpretation
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "FINITE-FIELD PERSISTENCE"
)

print(
    "=" * 90
)


for metric, label in [
    (
        "Longest_Skip_Run_m",
        "Longest skip run"
    ),
    (
        "Longest_Overlap_Run_m",
        "Longest overlap run"
    )
]:

    values = scenario_df[
        metric
    ]


    print(
        f"\n{label}:"
    )

    print(
        f"  min    = "
        f"{values.min():.3f} m"
    )

    print(
        f"  median = "
        f"{values.median():.3f} m"
    )

    print(
        f"  Q95    = "
        f"{values.quantile(0.95):.3f} m"
    )

    print(
        f"  max    = "
        f"{values.max():.3f} m"
    )


# ------------------------------------------------------------
# 26. Phase-sensitive pairs
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "MOST PHASE-SENSITIVE STRUCTURE PAIRS"
)

print(
    "=" * 90
)


print(
    phase_ranking_df.head(
        15
    ).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 27. Associations
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "SPEARMAN ASSOCIATIONS AMONG FINITE-FIELD METRICS"
)

print(
    "=" * 90
)


print(
    association_matrix.to_string()
)


# ------------------------------------------------------------
# 28. Outputs
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
    f"\nAll exhaustive scenarios:\n"
    f"{scenario_file}"
)

print(
    f"\nPair-level phase summaries:\n"
    f"{pair_summary_file}"
)

print(
    f"\nDataset summary:\n"
    f"{dataset_summary_file}"
)

print(
    f"\nPhase sensitivity ranking:\n"
    f"{ranking_file}"
)

print(
    f"\nSpearman matrix:\n"
    f"{association_file}"
)

print(
    f"\nLongest-skip figure:\n"
    f"{skip_run_figure}"
)

print(
    f"\nLongest-overlap figure:\n"
    f"{overlap_run_figure}"
)

print(
    f"\nK versus persistence figure:\n"
    f"{k_vs_persistence_figure}"
)

print(
    f"\nPhase-sensitivity figure:\n"
    f"{ranking_figure}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 2D COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nThis stage exhaustively evaluates finite-field persistence "
    "without introducing a Monte Carlo sample count or an "
    "agronomic magnitude threshold."
)

print(
    "\nThe next stage can extend from one neighboring-pass "
    "interface to a full multi-pass field and evaluate boundary "
    "violations and accumulated coverage effects."
)