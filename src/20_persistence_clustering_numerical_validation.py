from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm


# =====================================================================
# STAGE 4D — PERSISTENCE / CLUSTERING NUMERICAL VALIDATION
#             + SPATIALLY INTEGRATED COEFFICIENT CORRECTION
# =====================================================================
#
# PURPOSE
# -------
#
# This final numerical-validation stage addresses four issues identified
# after Stages 2D–4C:
#
#   1. exact ties in the empirical trajectories before rank-Gaussian
#      common-marginal transformation;
#
#   2. previous run lengths based on number_of_samples * dx rather than
#      exact continuous excursion length;
#
#   3. previous cluster areas based on rectangular sums rather than exact
#      integration of the piecewise-linear error signal;
#
#   4. circular shifts create an artificial seam and therefore should not
#      be interpreted as literal physical persistence realizations.
#
#
# WHAT THIS SCRIPT DOES
# ---------------------
#
# A. Reconstructs the 28 centered unit-RMS empirical trajectories on the
#    common 0.20-m linear field support.
#
# B. Diagnoses exact ties in each empirical trajectory.
#
# C. Reconstructs the common-marginal Gaussian-score control and verifies
#    that every transformed structure has the same marginal multiset.
#
# D. Calculates boundary excursion persistence using exact piecewise-
#    linear geometry:
#
#       - total positive/negative excursion length
#       - longest excursion run
#       - largest excursion-cluster area
#       - number of excursion clusters
#       - exact integrated one-edge boundary factor B
#
# E. Calculates aligned neighboring-pass persistence for all 378 pairs on
#    the REAL LINEAR DOMAIN ONLY:
#
#       delta(x) = z2(x) - z1(x)
#
#    No circular shift is used for physical persistence.
#
# F. Recomputes the uniform circular-phase DESIGN coefficient K using
#    trapezoidal spatial weights.
#
#    Circular phase averaging remains a mathematical phase-control
#    experiment for mean discrepancy ONLY. It is NOT used to construct
#    persistence/run-length distributions.
#
# G. Recomputes the 14-configuration cluster bootstrap using:
#
#       corrected spatially integrated K
#       corrected spatially integrated B
#
#    and compares the corrected results with Stage 4C.
#
#
# IMPORTANT INTERPRETATION
# ------------------------
#
# Persistence metrics from this script refer to the observed linear
# 271.4-m common support and aligned empirical profiles.
#
# Previous circular-shift run-length distributions should therefore be
# described only as "phase-control sensitivity", not as literal field
# persistence probabilities.
#
# =====================================================================


# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "bias_controlled_equal_rmse_trajectory_matrix.csv"
)

STAGE4C_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "bootstrap_coefficient_summary.csv"
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
    / "persistence_clustering_validation"
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------
# 2. Numerical settings
# ---------------------------------------------------------------------

DX = 0.20

BOOTSTRAP_REPLICATES = 5000

RANDOM_SEED = 20260913

PASS_COUNTS = np.array(
    [2, 3, 5, 10, 20, 50, 100],
    dtype=int
)

OLD_STAGE4C_K = 1.122528585367

OLD_STAGE4C_B = 0.398524115117


# ---------------------------------------------------------------------
# 3. Utility functions
# ---------------------------------------------------------------------

def config_from_trajectory(
    trajectory_id
):

    return str(
        trajectory_id
    ).split("_")[0]


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


def gaussian_score_transform(
    values
):

    """
    Reproduce the deterministic ordinal rank-Gaussian control.

    Stable sorting is used so that the transformation is fully
    reproducible.

    IMPORTANT:
    exact ties are diagnosed separately because assigning distinct
    Gaussian scores within an exact tie necessarily imposes an ordering.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    n = len(values)

    order = np.argsort(
        values,
        kind="mergesort"
    )

    probabilities = (
        (
            np.arange(n)
            + 0.5
        )
        / n
    )

    scores = norm.ppf(
        probabilities
    )

    transformed = np.empty(
        n,
        dtype=float
    )

    transformed[
        order
    ] = scores

    transformed -= np.mean(
        transformed
    )

    transformed /= rms(
        transformed
    )

    return transformed


def tie_diagnostics(
    values
):

    values = np.asarray(
        values
    )

    unique_values, counts = np.unique(
        values,
        return_counts=True
    )

    tied_groups = counts[
        counts > 1
    ]

    if len(
        tied_groups
    ) == 0:

        tied_samples = 0

        max_multiplicity = 1

        tied_group_count = 0

    else:

        tied_samples = int(
            np.sum(
                tied_groups
            )
        )

        max_multiplicity = int(
            np.max(
                tied_groups
            )
        )

        tied_group_count = int(
            len(
                tied_groups
            )
        )

    return {
        "Number_of_Samples":
            len(values),

        "Number_of_Unique_Values":
            len(unique_values),

        "Unique_Fraction":
            len(unique_values)
            / len(values),

        "Number_of_Tied_Groups":
            tied_group_count,

        "Number_of_Samples_in_Tied_Groups":
            tied_samples,

        "Tied_Sample_Fraction":
            tied_samples
            / len(values),

        "Maximum_Tie_Multiplicity":
            max_multiplicity
    }


# ---------------------------------------------------------------------
# 4. Exact piecewise-linear positive-excursion analysis
# ---------------------------------------------------------------------

def positive_excursion_clusters(
    x,
    f
):

    """
    Exact positive-excursion geometry for a piecewise-linear signal.

    Returns:
      total positive length
      total positive area
      number of positive clusters
      longest positive cluster length
      largest positive cluster area

    Zero crossings are solved analytically within each line segment.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    f = np.asarray(
        f,
        dtype=float
    )

    if len(x) != len(f):

        raise ValueError(
            "x and f must have equal length."
        )

    if len(x) < 2:

        raise ValueError(
            "At least two samples are required."
        )

    if np.any(
        np.diff(x) <= 0
    ):

        raise ValueError(
            "x must be strictly increasing."
        )


    # Each item:
    # [start, end, area]
    intervals = []


    for i in range(
        len(x) - 1
    ):

        x0 = x[i]

        x1 = x[i + 1]

        f0 = f[i]

        f1 = f[i + 1]


        # Entire segment non-positive.
        if (
            f0 <= 0
            and
            f1 <= 0
        ):

            continue


        # Entire segment non-negative, with at least some positive part.
        if (
            f0 >= 0
            and
            f1 >= 0
        ):

            if (
                f0 == 0
                and
                f1 == 0
            ):

                continue

            area = (
                0.5
                * (
                    f0
                    + f1
                )
                * (
                    x1
                    - x0
                )
            )

            intervals.append(
                [
                    x0,
                    x1,
                    area
                ]
            )

            continue


        # Crossing.
        x_cross = (
            x0
            - f0
            * (
                x1
                - x0
            )
            / (
                f1
                - f0
            )
        )


        # Positive on left part.
        if (
            f0 > 0
            and
            f1 < 0
        ):

            area = (
                0.5
                * f0
                * (
                    x_cross
                    - x0
                )
            )

            intervals.append(
                [
                    x0,
                    x_cross,
                    area
                ]
            )


        # Positive on right part.
        elif (
            f0 < 0
            and
            f1 > 0
        ):

            area = (
                0.5
                * f1
                * (
                    x1
                    - x_cross
                )
            )

            intervals.append(
                [
                    x_cross,
                    x1,
                    area
                ]
            )


    if len(
        intervals
    ) == 0:

        return {
            "Total_Excursion_Length":
                0.0,

            "Total_Excursion_Area":
                0.0,

            "Number_of_Clusters":
                0,

            "Longest_Cluster_Length":
                0.0,

            "Largest_Cluster_Area":
                0.0
        }


    # Merge contiguous positive intervals.
    clusters = []

    current_start = intervals[
        0
    ][0]

    current_end = intervals[
        0
    ][1]

    current_area = intervals[
        0
    ][2]


    tolerance = 1e-12


    for start, end, area in intervals[
        1:
    ]:

        if abs(
            start
            - current_end
        ) <= tolerance:

            current_end = end

            current_area += area

        else:

            clusters.append(
                [
                    current_start,
                    current_end,
                    current_area
                ]
            )

            current_start = start

            current_end = end

            current_area = area


    clusters.append(
        [
            current_start,
            current_end,
            current_area
        ]
    )


    lengths = np.array(
        [
            end
            - start
            for start, end, area
            in clusters
        ],
        dtype=float
    )


    areas = np.array(
        [
            area
            for start, end, area
            in clusters
        ],
        dtype=float
    )


    return {
        "Total_Excursion_Length":
            float(
                np.sum(
                    lengths
                )
            ),

        "Total_Excursion_Area":
            float(
                np.sum(
                    areas
                )
            ),

        "Number_of_Clusters":
            int(
                len(
                    clusters
                )
            ),

        "Longest_Cluster_Length":
            float(
                np.max(
                    lengths
                )
            ),

        "Largest_Cluster_Area":
            float(
                np.max(
                    areas
                )
            )
    }


def bidirectional_excursion_metrics(
    x,
    f
):

    positive = positive_excursion_clusters(
        x,
        f
    )

    negative = positive_excursion_clusters(
        x,
        -f
    )

    return positive, negative


# ---------------------------------------------------------------------
# 5. Exact weighted all-phase L1 coefficient
# ---------------------------------------------------------------------

def weighted_mean_absolute_difference_to_distribution(
    x_values,
    y_values,
    x_weights
):

    """
    Computes:

      sum_i w_i * mean_j |x_i - y_j| / sum_i w_i

    efficiently by sorting y_values.

    Used for the trapezoid-weighted circular-phase design average.
    """

    x_values = np.asarray(
        x_values,
        dtype=float
    )

    y_values = np.asarray(
        y_values,
        dtype=float
    )

    x_weights = np.asarray(
        x_weights,
        dtype=float
    )


    y_sorted = np.sort(
        y_values
    )

    n = len(
        y_sorted
    )

    prefix = np.concatenate(
        [
            [0.0],
            np.cumsum(
                y_sorted
            )
        ]
    )


    distances = np.zeros(
        len(
            x_values
        ),
        dtype=float
    )


    for index, value in enumerate(
        x_values
    ):

        p = np.searchsorted(
            y_sorted,
            value,
            side="right"
        )

        left_sum = prefix[
            p
        ]

        right_sum = (
            prefix[
                n
            ]
            - prefix[
                p
            ]
        )

        left_distance = (
            value
            * p
            - left_sum
        )

        right_count = (
            n
            - p
        )

        right_distance = (
            right_sum
            - value
            * right_count
        )

        distances[
            index
        ] = (
            left_distance
            + right_distance
        ) / n


    return float(
        np.sum(
            x_weights
            * distances
        )
        / np.sum(
            x_weights
        )
    )


def symmetric_trapezoidal_phase_K(
    z1,
    z2
):

    """
    Circular relative-phase DESIGN average with physical trapezoidal
    longitudinal weighting.

    Orientation 1:
      z1 held on the fixed field coordinates and z2 shifted.

    Orientation 2:
      z2 held fixed and z1 shifted.

    The reported coefficient is their symmetric average.

    This avoids assigning special physical meaning to which trajectory
    happens to occupy the fixed endpoint-weighted coordinate system.
    """

    n = len(
        z1
    )

    weights = np.ones(
        n,
        dtype=float
    )

    weights[
        0
    ] = 0.5

    weights[
        -1
    ] = 0.5


    k12 = weighted_mean_absolute_difference_to_distribution(
        z1,
        z2,
        weights
    )


    k21 = weighted_mean_absolute_difference_to_distribution(
        z2,
        z1,
        weights
    )


    return (
        k12
        + k21
    ) / 2.0


# ---------------------------------------------------------------------
# 6. Load trajectories
# ---------------------------------------------------------------------

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nMissing input file:\n"
        f"{INPUT_FILE}"
    )


print(
    "=" * 96
)

print(
    "STAGE 4D — PERSISTENCE / CLUSTERING NUMERICAL VALIDATION"
)

print(
    "=" * 96
)


df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nLoaded:\n"
    f"{INPUT_FILE}"
)


print(
    f"\nRows:"
    f"\n  {len(df):,}"
)


required_columns = [
    "Target_Centered_RMS_ID",
    "Target_Centered_RMS_m",
    "Structure_Trajectory_ID",
    "Easting_m",
    "Zero_Mean_Equal_RMS_Error_m"
]


for column in required_columns:

    if column not in df.columns:

        raise KeyError(
            f"\nMissing required column:\n"
            f"{column}"
        )


# ---------------------------------------------------------------------
# 7. Retrieve representative structure level
# ---------------------------------------------------------------------

levels = (
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


selected_level = levels.iloc[
    len(levels) // 2
]


SELECTED_ID = selected_level[
    "Target_Centered_RMS_ID"
]


SELECTED_RMS = float(
    selected_level[
        "Target_Centered_RMS_m"
    ]
)


working_df = df[
    df[
        "Target_Centered_RMS_ID"
    ]
    == SELECTED_ID
].copy()


trajectory_ids = sorted(
    working_df[
        "Structure_Trajectory_ID"
    ].unique()
)


print(
    "\nStructure retrieval level:"
)

print(
    f"  {SELECTED_ID}"
    f" = "
    f"{SELECTED_RMS:.9f} m"
)


print(
    "\nTrajectories:"
)

print(
    f"  {len(trajectory_ids)}"
)


# ---------------------------------------------------------------------
# 8. Raw trajectories and common support
# ---------------------------------------------------------------------

raw_store = {}


for trajectory_id in trajectory_ids:

    subset = working_df[
        working_df[
            "Structure_Trajectory_ID"
        ]
        == trajectory_id
    ].sort_values(
        "Easting_m"
    )


    x = subset[
        "Easting_m"
    ].to_numpy(
        dtype=float
    )


    e = subset[
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
            e
        )
    )


    x = x[
        finite
    ]

    e = e[
        finite
    ]


    temp = pd.DataFrame(
        {
            "x":
                x,

            "e":
                e
        }
    )


    temp = (
        temp
        .groupby(
            "x",
            as_index=False
        )[
            "e"
        ]
        .median()
        .sort_values(
            "x"
        )
    )


    raw_store[
        trajectory_id
    ] = {
        "x":
            temp[
                "x"
            ].to_numpy(
                dtype=float
            ),

        "e":
            temp[
                "e"
            ].to_numpy(
                dtype=float
            )
    }


common_start = max(
    record[
        "x"
    ].min()
    for record
    in raw_store.values()
)


common_end = min(
    record[
        "x"
    ].max()
    for record
    in raw_store.values()
)


n_steps = int(
    np.floor(
        (
            common_end
            - common_start
        )
        / DX
    )
)


grid = (
    common_start
    + np.arange(
        n_steps + 1
    )
    * DX
)


FIELD_LENGTH = (
    grid[-1]
    - grid[0]
)


print(
    "\nCommon linear field support:"
)

print(
    f"  start   = "
    f"{grid[0]:.3f} m"
)

print(
    f"  end     = "
    f"{grid[-1]:.3f} m"
)

print(
    f"  length  = "
    f"{FIELD_LENGTH:.3f} m"
)

print(
    f"  points  = "
    f"{len(grid):,}"
)


# ---------------------------------------------------------------------
# 9. Centered unit-RMS empirical trajectories
# ---------------------------------------------------------------------

original_store = {}

common_store = {}

tie_rows = []


for trajectory_id in trajectory_ids:

    record = raw_store[
        trajectory_id
    ]


    z = np.interp(
        grid,
        record[
            "x"
        ],
        record[
            "e"
        ]
    )


    z -= np.mean(
        z
    )

    z /= rms(
        z
    )


    original_store[
        trajectory_id
    ] = z


    diagnostics = tie_diagnostics(
        z
    )


    tie_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Configuration":
                config_from_trajectory(
                    trajectory_id
                ),

            **diagnostics
        }
    )


    common_store[
        trajectory_id
    ] = gaussian_score_transform(
        z
    )


tie_df = pd.DataFrame(
    tie_rows
)


# ---------------------------------------------------------------------
# 10. Common-marginal verification
# ---------------------------------------------------------------------

reference_sorted = np.sort(
    common_store[
        trajectory_ids[
            0
        ]
    ]
)


max_common_marginal_difference = 0.0


for trajectory_id in trajectory_ids:

    difference = np.max(
        np.abs(
            np.sort(
                common_store[
                    trajectory_id
                ]
            )
            - reference_sorted
        )
    )


    max_common_marginal_difference = max(
        max_common_marginal_difference,
        difference
    )


print(
    "\n"
    + "=" * 96
)

print(
    "TIE DIAGNOSTICS"
)

print(
    "=" * 96
)


print(
    "\nUnique-value fraction across trajectories:"
)

print(
    f"  min    = "
    f"{tie_df['Unique_Fraction'].min():.9f}"
)

print(
    f"  median = "
    f"{tie_df['Unique_Fraction'].median():.9f}"
)

print(
    f"  max    = "
    f"{tie_df['Unique_Fraction'].max():.9f}"
)


print(
    "\nTied-sample fraction:"
)

print(
    f"  min    = "
    f"{tie_df['Tied_Sample_Fraction'].min():.9f}"
)

print(
    f"  median = "
    f"{tie_df['Tied_Sample_Fraction'].median():.9f}"
)

print(
    f"  max    = "
    f"{tie_df['Tied_Sample_Fraction'].max():.9f}"
)


print(
    "\nMaximum tie multiplicity:"
)

print(
    f"  {tie_df['Maximum_Tie_Multiplicity'].max()}"
)


print(
    "\nMaximum common-marginal discrepancy:"
)

print(
    f"  {max_common_marginal_difference:.12e}"
)


# ---------------------------------------------------------------------
# 11. Corrected boundary persistence
# ---------------------------------------------------------------------

boundary_rows = []


for marginal_type, store in [
    (
        "Original_Centered",
        original_store
    ),
    (
        "Common_Marginal",
        common_store
    )
]:

    for trajectory_id in trajectory_ids:

        z = store[
            trajectory_id
        ]


        positive, negative = bidirectional_excursion_metrics(
            grid,
            z
        )


        B_positive = (
            positive[
                "Total_Excursion_Area"
            ]
            / FIELD_LENGTH
        )


        B_negative = (
            negative[
                "Total_Excursion_Area"
            ]
            / FIELD_LENGTH
        )


        boundary_rows.append(
            {
                "Marginal_Type":
                    marginal_type,

                "Trajectory_ID":
                    trajectory_id,

                "Configuration":
                    config_from_trajectory(
                        trajectory_id
                    ),

                "Direction":
                    "Positive",

                "B_Integrated":
                    B_positive,

                "Excursion_Length_m":
                    positive[
                        "Total_Excursion_Length"
                    ],

                "Excursion_Length_Fraction":
                    positive[
                        "Total_Excursion_Length"
                    ]
                    / FIELD_LENGTH,

                "Number_of_Clusters":
                    positive[
                        "Number_of_Clusters"
                    ],

                "Longest_Cluster_Length_m":
                    positive[
                        "Longest_Cluster_Length"
                    ],

                "Largest_Cluster_Area_Factor_m":
                    positive[
                        "Largest_Cluster_Area"
                    ]
            }
        )


        boundary_rows.append(
            {
                "Marginal_Type":
                    marginal_type,

                "Trajectory_ID":
                    trajectory_id,

                "Configuration":
                    config_from_trajectory(
                        trajectory_id
                    ),

                "Direction":
                    "Negative",

                "B_Integrated":
                    B_negative,

                "Excursion_Length_m":
                    negative[
                        "Total_Excursion_Length"
                    ],

                "Excursion_Length_Fraction":
                    negative[
                        "Total_Excursion_Length"
                    ]
                    / FIELD_LENGTH,

                "Number_of_Clusters":
                    negative[
                        "Number_of_Clusters"
                    ],

                "Longest_Cluster_Length_m":
                    negative[
                        "Longest_Cluster_Length"
                    ],

                "Largest_Cluster_Area_Factor_m":
                    negative[
                        "Largest_Cluster_Area"
                    ]
            }
        )


boundary_df = pd.DataFrame(
    boundary_rows
)


# Per-trajectory symmetric one-edge B.

trajectory_B_rows = []


for marginal_type in [
    "Original_Centered",
    "Common_Marginal"
]:

    subset = boundary_df[
        boundary_df[
            "Marginal_Type"
        ]
        == marginal_type
    ]


    for trajectory_id in trajectory_ids:

        temp = subset[
            subset[
                "Trajectory_ID"
            ]
            == trajectory_id
        ]


        B_value = temp[
            "B_Integrated"
        ].mean()


        trajectory_B_rows.append(
            {
                "Marginal_Type":
                    marginal_type,

                "Trajectory_ID":
                    trajectory_id,

                "Configuration":
                    config_from_trajectory(
                        trajectory_id
                    ),

                "B_One_Edge_Integrated":
                    B_value
            }
        )


trajectory_B_df = pd.DataFrame(
    trajectory_B_rows
)


# ---------------------------------------------------------------------
# 12. Corrected aligned internal persistence — 378 pairs
# ---------------------------------------------------------------------

pair_rows = []


pairs = list(
    itertools.combinations(
        trajectory_ids,
        2
    )
)


for marginal_type, store in [
    (
        "Original_Centered",
        original_store
    ),
    (
        "Common_Marginal",
        common_store
    )
]:

    for pair_number, (
        trajectory_1,
        trajectory_2
    ) in enumerate(
        pairs,
        start=1
    ):

        z1 = store[
            trajectory_1
        ]

        z2 = store[
            trajectory_2
        ]


        delta = (
            z2
            - z1
        )


        skip, overlap = bidirectional_excursion_metrics(
            grid,
            delta
        )


        K_aligned_integrated = (
            skip[
                "Total_Excursion_Area"
            ]
            +
            overlap[
                "Total_Excursion_Area"
            ]
        ) / FIELD_LENGTH


        pair_rows.append(
            {
                "Marginal_Type":
                    marginal_type,

                "Pair_ID":
                    f"PAIR_{pair_number:03d}",

                "Trajectory_1":
                    trajectory_1,

                "Trajectory_2":
                    trajectory_2,

                "K_Aligned_Integrated":
                    K_aligned_integrated,

                "Skip_Total_Length_m":
                    skip[
                        "Total_Excursion_Length"
                    ],

                "Skip_Longest_Cluster_m":
                    skip[
                        "Longest_Cluster_Length"
                    ],

                "Skip_Largest_Cluster_Area_Factor_m":
                    skip[
                        "Largest_Cluster_Area"
                    ],

                "Skip_Number_of_Clusters":
                    skip[
                        "Number_of_Clusters"
                    ],

                "Overlap_Total_Length_m":
                    overlap[
                        "Total_Excursion_Length"
                    ],

                "Overlap_Longest_Cluster_m":
                    overlap[
                        "Longest_Cluster_Length"
                    ],

                "Overlap_Largest_Cluster_Area_Factor_m":
                    overlap[
                        "Largest_Cluster_Area"
                    ],

                "Overlap_Number_of_Clusters":
                    overlap[
                        "Number_of_Clusters"
                    ]
            }
        )


pair_persistence_df = pd.DataFrame(
    pair_rows
)


# ---------------------------------------------------------------------
# 13. Corrected trapezoidal uniform-phase K matrix
# ---------------------------------------------------------------------

n_trajectories = len(
    trajectory_ids
)


trajectory_index = {
    trajectory_id:
        index
    for index, trajectory_id
    in enumerate(
        trajectory_ids
    )
}


K_matrix = np.zeros(
    (
        n_trajectories,
        n_trajectories
    ),
    dtype=float
)


print(
    "\n"
    + "=" * 96
)

print(
    "BUILDING TRAPEZOID-WEIGHTED PHASE-DESIGN K MATRIX"
)

print(
    "=" * 96
)


for i in range(
    n_trajectories
):

    for j in range(
        i,
        n_trajectories
    ):

        value = symmetric_trapezoidal_phase_K(
            original_store[
                trajectory_ids[
                    i
                ]
            ],
            original_store[
                trajectory_ids[
                    j
                ]
            ]
        )


        K_matrix[
            i,
            j
        ] = value

        K_matrix[
            j,
            i
        ] = value


upper_i, upper_j = np.triu_indices(
    n_trajectories,
    k=1
)


CORRECTED_K = float(
    np.mean(
        K_matrix[
            upper_i,
            upper_j
        ]
    )
)


original_B_subset = trajectory_B_df[
    trajectory_B_df[
        "Marginal_Type"
    ]
    == "Original_Centered"
]


CORRECTED_B = float(
    original_B_subset[
        "B_One_Edge_Integrated"
    ].mean()
)


common_B_subset = trajectory_B_df[
    trajectory_B_df[
        "Marginal_Type"
    ]
    == "Common_Marginal"
]


CORRECTED_COMMON_B = float(
    common_B_subset[
        "B_One_Edge_Integrated"
    ].mean()
)


# Common-marginal corrected K.

common_K_matrix = np.zeros(
    (
        n_trajectories,
        n_trajectories
    ),
    dtype=float
)


for i in range(
    n_trajectories
):

    for j in range(
        i,
        n_trajectories
    ):

        value = symmetric_trapezoidal_phase_K(
            common_store[
                trajectory_ids[
                    i
                ]
            ],
            common_store[
                trajectory_ids[
                    j
                ]
            ]
        )


        common_K_matrix[
            i,
            j
        ] = value

        common_K_matrix[
            j,
            i
        ] = value


CORRECTED_COMMON_K = float(
    np.mean(
        common_K_matrix[
            upper_i,
            upper_j
        ]
    )
)


# ---------------------------------------------------------------------
# 14. Cluster membership
# ---------------------------------------------------------------------

configurations = sorted(
    {
        config_from_trajectory(
            trajectory_id
        )
        for trajectory_id
        in trajectory_ids
    }
)


if len(
    configurations
) != 14:

    raise RuntimeError(
        f"Expected 14 configurations; "
        f"found {len(configurations)}."
    )


config_to_indices = {}


for configuration in configurations:

    members = [
        trajectory_index[
            trajectory_id
        ]
        for trajectory_id
        in trajectory_ids
        if config_from_trajectory(
            trajectory_id
        )
        == configuration
    ]


    if len(
        members
    ) != 2:

        raise RuntimeError(
            f"{configuration} does not contain exactly 2 primary passes."
        )


    config_to_indices[
        configuration
    ] = members


B_vector = np.array(
    [
        original_B_subset.loc[
            original_B_subset[
                "Trajectory_ID"
            ]
            == trajectory_id,
            "B_One_Edge_Integrated"
        ].iloc[
            0
        ]
        for trajectory_id
        in trajectory_ids
    ],
    dtype=float
)


# ---------------------------------------------------------------------
# 15. Corrected configuration-cluster bootstrap
# ---------------------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_SEED
)


bootstrap_rows = []


for replicate in range(
    BOOTSTRAP_REPLICATES
):

    sampled_configurations = rng.choice(
        configurations,
        size=len(
            configurations
        ),
        replace=True
    )


    selected_indices = []


    for configuration in sampled_configurations:

        selected_indices.extend(
            config_to_indices[
                configuration
            ]
        )


    selected_indices = np.asarray(
        selected_indices,
        dtype=int
    )


    bi, bj = np.triu_indices(
        len(
            selected_indices
        ),
        k=1
    )


    K_boot = float(
        np.mean(
            K_matrix[
                selected_indices[
                    bi
                ],
                selected_indices[
                    bj
                ]
            ]
        )
    )


    B_boot = float(
        np.mean(
            B_vector[
                selected_indices
            ]
        )
    )


    row = {
        "Replicate":
            replicate + 1,

        "K":
            K_boot,

        "B":
            B_boot
    }


    for M in PASS_COUNTS:

        C_M = (
            (
                M - 1
            )
            / M
            * K_boot
            +
            2.0
            * B_boot
            / M
        )


        row[
            f"C_M_{M}"
        ] = C_M


        row[
            f"T_M_{M}"
        ] = (
            1.0
            / C_M
        )


    bootstrap_rows.append(
        row
    )


    if (
        replicate + 1
    ) % 500 == 0:

        print(
            f"  bootstrap "
            f"{replicate + 1:,}"
            f" / "
            f"{BOOTSTRAP_REPLICATES:,}"
        )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


# ---------------------------------------------------------------------
# 16. Confidence intervals
# ---------------------------------------------------------------------

def ci95(
    values
):

    return (
        float(
            np.percentile(
                values,
                2.5
            )
        ),
        float(
            np.percentile(
                values,
                97.5
            )
        )
    )


K_low, K_high = ci95(
    bootstrap_df[
        "K"
    ]
)


B_low, B_high = ci95(
    bootstrap_df[
        "B"
    ]
)


coefficient_summary_df = pd.DataFrame(
    [
        {
            "Coefficient":
                "K",

            "Corrected_Point_Estimate":
                CORRECTED_K,

            "Bootstrap_Mean":
                bootstrap_df[
                    "K"
                ].mean(),

            "Bootstrap_SD":
                bootstrap_df[
                    "K"
                ].std(
                    ddof=1
                ),

            "CI95_Lower":
                K_low,

            "CI95_Upper":
                K_high,

            "Previous_Stage4C_Value":
                OLD_STAGE4C_K,

            "Relative_Change_from_Stage4C":
                (
                    CORRECTED_K
                    - OLD_STAGE4C_K
                )
                / OLD_STAGE4C_K
        },

        {
            "Coefficient":
                "B",

            "Corrected_Point_Estimate":
                CORRECTED_B,

            "Bootstrap_Mean":
                bootstrap_df[
                    "B"
                ].mean(),

            "Bootstrap_SD":
                bootstrap_df[
                    "B"
                ].std(
                    ddof=1
                ),

            "CI95_Lower":
                B_low,

            "CI95_Upper":
                B_high,

            "Previous_Stage4C_Value":
                OLD_STAGE4C_B,

            "Relative_Change_from_Stage4C":
                (
                    CORRECTED_B
                    - OLD_STAGE4C_B
                )
                / OLD_STAGE4C_B
        }
    ]
)


# ---------------------------------------------------------------------
# 17. Corrected operational coefficients
# ---------------------------------------------------------------------

operational_rows = []


for M in PASS_COUNTS:

    C_column = (
        f"C_M_{M}"
    )

    T_column = (
        f"T_M_{M}"
    )


    C_low, C_high = ci95(
        bootstrap_df[
            C_column
        ]
    )


    T_low, T_high = ci95(
        bootstrap_df[
            T_column
        ]
    )


    point_C = (
        (
            M - 1
        )
        / M
        * CORRECTED_K
        +
        2.0
        * CORRECTED_B
        / M
    )


    point_T = (
        1.0
        / point_C
    )


    common_C = (
        (
            M - 1
        )
        / M
        * CORRECTED_COMMON_K
        +
        2.0
        * CORRECTED_COMMON_B
        / M
    )


    operational_rows.append(
        {
            "M":
                M,

            "Corrected_C_Point":
                point_C,

            "C_CI95_Lower":
                C_low,

            "C_CI95_Upper":
                C_high,

            "Tolerance_Factor_Point":
                point_T,

            "Tolerance_Factor_CI95_Lower":
                T_low,

            "Tolerance_Factor_CI95_Upper":
                T_high,

            "Corrected_Common_Marginal_C":
                common_C
        }
    )


operational_df = pd.DataFrame(
    operational_rows
)


# ---------------------------------------------------------------------
# 18. Persistence summaries
# ---------------------------------------------------------------------

persistence_summary_rows = []


for marginal_type in [
    "Original_Centered",
    "Common_Marginal"
]:

    subset = boundary_df[
        boundary_df[
            "Marginal_Type"
        ]
        == marginal_type
    ]


    persistence_summary_rows.append(
        {
            "Marginal_Type":
                marginal_type,

            "Metric":
                "Boundary_Longest_Cluster_Length_m",

            "Minimum":
                subset[
                    "Longest_Cluster_Length_m"
                ].min(),

            "Q25":
                subset[
                    "Longest_Cluster_Length_m"
                ].quantile(
                    0.25
                ),

            "Median":
                subset[
                    "Longest_Cluster_Length_m"
                ].median(),

            "Q75":
                subset[
                    "Longest_Cluster_Length_m"
                ].quantile(
                    0.75
                ),

            "Maximum":
                subset[
                    "Longest_Cluster_Length_m"
                ].max()
        }
    )


    persistence_summary_rows.append(
        {
            "Marginal_Type":
                marginal_type,

            "Metric":
                "Boundary_Largest_Cluster_Area_Factor_m",

            "Minimum":
                subset[
                    "Largest_Cluster_Area_Factor_m"
                ].min(),

            "Q25":
                subset[
                    "Largest_Cluster_Area_Factor_m"
                ].quantile(
                    0.25
                ),

            "Median":
                subset[
                    "Largest_Cluster_Area_Factor_m"
                ].median(),

            "Q75":
                subset[
                    "Largest_Cluster_Area_Factor_m"
                ].quantile(
                    0.75
                ),

            "Maximum":
                subset[
                    "Largest_Cluster_Area_Factor_m"
                ].max()
        }
    )


persistence_summary_df = pd.DataFrame(
    persistence_summary_rows
)


# ---------------------------------------------------------------------
# 19. Save tables
# ---------------------------------------------------------------------

tie_file = (
    TABLES_DIR
    / "corrected_tie_diagnostics.csv"
)

boundary_file = (
    TABLES_DIR
    / "corrected_linear_boundary_persistence.csv"
)

trajectory_B_file = (
    TABLES_DIR
    / "corrected_integrated_boundary_coefficients.csv"
)

pair_file = (
    TABLES_DIR
    / "corrected_aligned_pair_persistence.csv"
)

K_matrix_file = (
    TABLES_DIR
    / "corrected_trapezoidal_phase_K_matrix.csv"
)

bootstrap_file = (
    TABLES_DIR
    / "corrected_configuration_bootstrap_replicates.csv"
)

coefficient_file = (
    TABLES_DIR
    / "corrected_final_coefficient_summary.csv"
)

operational_file = (
    TABLES_DIR
    / "corrected_final_operational_coefficients.csv"
)

persistence_summary_file = (
    TABLES_DIR
    / "corrected_persistence_summary.csv"
)


tie_df.to_csv(
    tie_file,
    index=False
)

boundary_df.to_csv(
    boundary_file,
    index=False
)

trajectory_B_df.to_csv(
    trajectory_B_file,
    index=False
)

pair_persistence_df.to_csv(
    pair_file,
    index=False
)

pd.DataFrame(
    K_matrix,
    index=trajectory_ids,
    columns=trajectory_ids
).to_csv(
    K_matrix_file
)

bootstrap_df.to_csv(
    bootstrap_file,
    index=False
)

coefficient_summary_df.to_csv(
    coefficient_file,
    index=False
)

operational_df.to_csv(
    operational_file,
    index=False
)

persistence_summary_df.to_csv(
    persistence_summary_file,
    index=False
)


# ---------------------------------------------------------------------
# 20. Figures
# ---------------------------------------------------------------------

# Boundary longest-cluster distribution.

plt.figure(
    figsize=(
        9,
        6
    )
)


for marginal_type in [
    "Original_Centered",
    "Common_Marginal"
]:

    values = boundary_df.loc[
        boundary_df[
            "Marginal_Type"
        ]
        == marginal_type,
        "Longest_Cluster_Length_m"
    ]


    plt.hist(
        values,
        bins=15,
        alpha=0.5,
        label=marginal_type
    )


plt.xlabel(
    "Longest boundary excursion cluster length (m)"
)

plt.ylabel(
    "Number of trajectory-direction cases"
)

plt.title(
    "Corrected linear-domain boundary persistence"
)

plt.legend()

plt.tight_layout()

boundary_figure = (
    FIGURES_DIR
    / "corrected_boundary_longest_cluster_distribution.png"
)

plt.savefig(
    boundary_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Operational coefficient CI.

plt.figure(
    figsize=(
        9,
        6
    )
)


plt.plot(
    operational_df[
        "M"
    ],
    operational_df[
        "Corrected_C_Point"
    ],
    marker="o",
    label="Corrected empirical coefficient"
)


plt.fill_between(
    operational_df[
        "M"
    ],
    operational_df[
        "C_CI95_Lower"
    ],
    operational_df[
        "C_CI95_Upper"
    ],
    alpha=0.25,
    label="95% configuration-bootstrap interval"
)


plt.plot(
    operational_df[
        "M"
    ],
    operational_df[
        "Corrected_Common_Marginal_C"
    ],
    linestyle="--",
    marker="s",
    label="Corrected common-marginal control"
)


plt.xscale(
    "log"
)

plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "Operational coefficient C(M)"
)

plt.title(
    "Final spatially integrated operational coefficient"
)

plt.legend()

plt.tight_layout()

coefficient_figure = (
    FIGURES_DIR
    / "corrected_final_operational_coefficient.png"
)

plt.savefig(
    coefficient_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------------------
# 21. Console report
# ---------------------------------------------------------------------

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    320
)


print(
    "\n"
    + "=" * 96
)

print(
    "CORRECTED BOUNDARY PERSISTENCE"
)

print(
    "=" * 96
)


print(
    persistence_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}"
    )
)


print(
    "\n"
    + "=" * 96
)

print(
    "CORRECTED OPERATIONAL COEFFICIENTS"
)

print(
    "=" * 96
)


print(
    "\nOriginal centered empirical marginal:"
)

print(
    f"  K corrected = "
    f"{CORRECTED_K:.12f}"
)

print(
    f"  B corrected = "
    f"{CORRECTED_B:.12f}"
)


print(
    "\nCommon-marginal control:"
)

print(
    f"  K corrected = "
    f"{CORRECTED_COMMON_K:.12f}"
)

print(
    f"  B corrected = "
    f"{CORRECTED_COMMON_B:.12f}"
)


print(
    "\nChanges relative to Stage 4C:"
)

print(
    f"  K relative change = "
    f"{100.0 * (CORRECTED_K - OLD_STAGE4C_K) / OLD_STAGE4C_K:.6f}%"
)

print(
    f"  B relative change = "
    f"{100.0 * (CORRECTED_B - OLD_STAGE4C_B) / OLD_STAGE4C_B:.6f}%"
)


print(
    "\n"
    + "=" * 96
)

print(
    "CORRECTED CONFIGURATION-CLUSTER BOOTSTRAP"
)

print(
    "=" * 96
)


print(
    coefficient_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}"
    )
)


print(
    "\n"
    + "=" * 96
)

print(
    "FINAL MULTI-PASS COEFFICIENTS"
)

print(
    "=" * 96
)


print(
    operational_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}"
    )
)


print(
    "\n"
    + "=" * 96
)

print(
    "INTERPRETATION RULE FOR THE MANUSCRIPT"
)

print(
    "=" * 96
)


print(
    "\n1. Corrected persistence values above are computed on the real "
    "linear 271.4-m empirical support."
)


print(
    "\n2. No circularly shifted signal is used to define physical "
    "run lengths or cluster areas."
)


print(
    "\n3. Circular phase remains only a mathematical control used to "
    "characterize mean discrepancy sensitivity to relative alignment."
)


print(
    "\n4. Previous circular-shift persistence distributions must not "
    "be described as field-risk probabilities."
)


print(
    "\n5. The final tolerance relation remains:"
)


print(
    "\n       R_max = W q / C(M)"
)


print(
    "\n   with C(M) and its confidence interval now based on the "
    "spatially integrated corrected coefficients."
)


print(
    "\n"
    + "=" * 96
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 96
)


print(
    f"\nTie diagnostics:\n"
    f"{tie_file}"
)

print(
    f"\nCorrected boundary persistence:\n"
    f"{boundary_file}"
)

print(
    f"\nCorrected boundary B:\n"
    f"{trajectory_B_file}"
)

print(
    f"\nCorrected aligned-pair persistence:\n"
    f"{pair_file}"
)

print(
    f"\nCorrected K matrix:\n"
    f"{K_matrix_file}"
)

print(
    f"\nCorrected bootstrap replicates:\n"
    f"{bootstrap_file}"
)

print(
    f"\nFinal coefficient summary:\n"
    f"{coefficient_file}"
)

print(
    f"\nFinal operational coefficients:\n"
    f"{operational_file}"
)

print(
    f"\nPersistence summary:\n"
    f"{persistence_summary_file}"
)

print(
    f"\nFigures:\n"
    f"{FIGURES_DIR}"
)


print(
    "\n"
    + "=" * 96
)

print(
    "STAGE 4D COMPLETE"
)

print(
    "=" * 96
)