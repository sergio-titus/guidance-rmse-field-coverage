from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# STAGE 4C — CONFIGURATION-CLUSTER BOOTSTRAP
# =====================================================================
#
# PURPOSE
# -------
# Quantify sampling uncertainty in the empirically derived operational
# coefficients:
#
#       K  = all-phase internal discrepancy coefficient
#       B  = one-edge boundary magnitude coefficient
#
# using a configuration-level cluster bootstrap.
#
#
# BOOTSTRAP UNIT
# --------------
#
# The dataset contains:
#
#       14 GNSS/guidance configurations
#       2 primary passes per configuration: P1 and P3
#
# P1 and P3 from the same configuration are therefore kept together.
#
# Each bootstrap replicate:
#
#       1. resamples 14 configurations with replacement;
#       2. carries both P1 and P3 for every selected configuration;
#       3. reconstructs the resulting 28-trajectory multiset;
#       4. recomputes the mean pairwise all-phase K;
#       5. recomputes mean one-edge B;
#       6. propagates these into C(M) and 1/C(M).
#
#
# IMPORTANT BOOTSTRAP DETAIL
# --------------------------
#
# Because configurations are sampled WITH replacement, the same empirical
# trajectory may appear more than once in a bootstrap sample.
#
# Such duplicate copies are distinct bootstrap observations. Therefore
# pairs between duplicated copies must be included.
#
# This requires K(i,i), the all-phase kernel of a trajectory with itself.
#
# The original 378-pair table contains only distinct trajectory pairs and
# therefore cannot supply these diagonal terms.
#
# For that reason this script reconstructs the full 28 x 28 K matrix
# directly from the empirical trajectories.
#
#
# ALL-PHASE K
# -----------
#
# For equal-length trajectories z_i and z_j:
#
#       K_ij
#
# is the mean L1 discrepancy averaged over every circular relative phase.
#
# Averaging over all circular shifts is mathematically equivalent to:
#
#       mean_{a,b} |z_i[a] - z_j[b]|
#
# because every sample of z_i is paired exactly once with every sample
# of z_j across the complete shift ensemble.
#
# The value is calculated exactly using sorting and prefix sums.
#
#
# BOUNDARY FACTOR
# ---------------
#
# For a centered unit-RMS trajectory z:
#
#       B+ = mean(max(z,0))
#       B- = mean(max(-z,0))
#
# Under numerical centering:
#
#       B+ = B-
#
# The one-edge coefficient is:
#
#       B = (B+ + B-) / 2
#
#
# OPERATIONAL COEFFICIENT
# -----------------------
#
# For M parallel passes:
#
#       C(M)
#         = (M-1)/M * K
#           + 2B/M
#
#
# Mean normalized discrepancy:
#
#       D = eta * C(M)
#
# where:
#
#       eta = R/W
#
#
# TOLERANCE TRANSFER
# ------------------
#
# For externally supplied allowable discrepancy q:
#
#       eta_max = q / C(M)
#
#       R_max = W q / C(M)
#
# Hence:
#
#       R_max / (Wq) = 1/C(M)
#
#
# CONFIDENCE INTERVAL
# -------------------
#
# Percentile 95% cluster-bootstrap intervals are reported.
#
# No assumption of independence among the 378 pairwise combinations is
# made.
#
# =====================================================================


# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAJECTORY_FILE = (
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
    / "configuration_cluster_bootstrap"
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

BOOTSTRAP_REPLICATES = 5000

RANDOM_SEED = 20260913

COMMON_GRID_SPACING_M = 0.20

PASS_COUNTS = np.array(
    [
        2,
        3,
        5,
        10,
        20,
        50,
        100
    ],
    dtype=int
)


# Deterministic common-marginal references from previous controlled stage.

COMMON_K = 1.128664783206

COMMON_B = 0.399070030980


# ---------------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------------

def configuration_from_trajectory(
    trajectory_id
):

    return str(
        trajectory_id
    ).split(
        "_"
    )[0]


def exact_all_phase_l1(
    x,
    y
):

    """
    Exact mean |X-Y| over all Cartesian pairings between x and y.

    For equal-length circular phase averaging, this is exactly the
    unrestricted all-phase mean L1 discrepancy.
    """

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


    prefix = np.concatenate(
        (
            [0.0],
            np.cumsum(
                y_sorted
            )
        )
    )


    n = len(
        y_sorted
    )


    total = 0.0


    for value in x:

        position = np.searchsorted(
            y_sorted,
            value,
            side="right"
        )


        left_count = position

        right_count = (
            n
            - position
        )


        left_sum = prefix[
            position
        ]

        right_sum = (
            prefix[
                n
            ]
            - prefix[
                position
            ]
        )


        left_distance = (
            value
            * left_count
            - left_sum
        )


        right_distance = (
            right_sum
            - value
            * right_count
        )


        total += (
            left_distance
            + right_distance
        )


    return (
        total
        / (
            len(x)
            * len(y)
        )
    )


def percentile_interval(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )


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


# ---------------------------------------------------------------------
# 4. Load empirical trajectory matrix
# ---------------------------------------------------------------------

if not TRAJECTORY_FILE.exists():

    raise FileNotFoundError(
        f"\nRequired file not found:\n"
        f"{TRAJECTORY_FILE}"
    )


print(
    "=" * 94
)

print(
    "STAGE 4C — CONFIGURATION-CLUSTER BOOTSTRAP"
)

print(
    "=" * 94
)


df = pd.read_csv(
    TRAJECTORY_FILE
)


print(
    f"\nLoaded:\n"
    f"{TRAJECTORY_FILE}"
)


print(
    f"\nRows:"
    f"\n  {len(df):,}"
)


# ---------------------------------------------------------------------
# 5. Validate required columns
# ---------------------------------------------------------------------

required_columns = [
    "Target_Centered_RMS_ID",
    "Target_Centered_RMS_m",
    "Structure_Trajectory_ID",
    "Easting_m",
    "Zero_Mean_Equal_RMS_Error_m"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise KeyError(
        "\nMissing required columns:\n"
        + "\n".join(
            missing_columns
        )
        + "\n\nAvailable columns:\n"
        + "\n".join(
            df.columns
        )
    )


# ---------------------------------------------------------------------
# 6. Retrieve one complete empirical structure set
# ---------------------------------------------------------------------
#
# The equal-RMS matrix repeats the same normalized structure at every
# target centered-RMS level.
#
# We select the median level only as a convenient way of retrieving one
# copy of each of the 28 structures.
#
# The trajectories are subsequently re-centered and normalized to
# unit RMS, so the selected level has no effect on K or B.
#
# ---------------------------------------------------------------------

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


SELECTED_RMS_ID = (
    selected_level[
        "Target_Centered_RMS_ID"
    ]
)


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
    "\nStructure-retrieval level:"
)

print(
    f"  {SELECTED_RMS_ID}"
    f" = "
    f"{SELECTED_RMS_M:.9f} m"
)


print(
    "\nTrajectories:"
)

print(
    f"  {len(trajectory_ids)}"
)


# ---------------------------------------------------------------------
# 7. Retrieve raw trajectory arrays
# ---------------------------------------------------------------------

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


    # The Stage-1 canonical matrix should already have unique x values.
    # This check prevents silent ambiguity.

    unique_x, unique_counts = np.unique(
        x,
        return_counts=True
    )


    if np.any(
        unique_counts > 1
    ):

        # If duplicates unexpectedly remain, aggregate by median rather
        # than retaining an arbitrary first occurrence.

        temp = pd.DataFrame(
            {
                "x":
                    x,

                "error":
                    error
            }
        )


        aggregated = (
            temp
            .groupby(
                "x",
                as_index=False
            )[
                "error"
            ]
            .median()
            .sort_values(
                "x"
            )
        )


        x = aggregated[
            "x"
        ].to_numpy(
            dtype=float
        )


        error = aggregated[
            "error"
        ].to_numpy(
            dtype=float
        )


    raw_store[
        trajectory_id
    ] = {
        "x":
            x,

        "error":
            error
    }


# ---------------------------------------------------------------------
# 8. Common empirical support
# ---------------------------------------------------------------------

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
        / COMMON_GRID_SPACING_M
    )
)


grid = (
    common_start
    + np.arange(
        n_steps + 1
    )
    * COMMON_GRID_SPACING_M
)


field_length = (
    grid[-1]
    - grid[0]
)


print(
    "\nCommon empirical support:"
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
    f"{field_length:.3f} m"
)

print(
    f"  points  = "
    f"{len(grid):,}"
)


# ---------------------------------------------------------------------
# 9. Create centered unit-RMS empirical structures
# ---------------------------------------------------------------------

structure_store = {}

verification_rows = []


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
            "error"
        ]
    )


    z = (
        z
        - np.mean(
            z
        )
    )


    rms = np.sqrt(
        np.mean(
            z ** 2
        )
    )


    if rms <= 0:

        raise RuntimeError(
            f"Zero RMS encountered for "
            f"{trajectory_id}"
        )


    z = (
        z
        / rms
    )


    mean_z = np.mean(
        z
    )


    rms_z = np.sqrt(
        np.mean(
            z ** 2
        )
    )


    positive_B = np.mean(
        np.maximum(
            z,
            0.0
        )
    )


    negative_B = np.mean(
        np.maximum(
            -z,
            0.0
        )
    )


    B = (
        positive_B
        + negative_B
    ) / 2.0


    structure_store[
        trajectory_id
    ] = z


    verification_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Configuration":
                configuration_from_trajectory(
                    trajectory_id
                ),

            "Mean":
                mean_z,

            "RMS":
                rms_z,

            "B_Positive":
                positive_B,

            "B_Negative":
                negative_B,

            "B_One_Edge":
                B
        }
    )


verification_df = pd.DataFrame(
    verification_rows
)


max_abs_mean = np.max(
    np.abs(
        verification_df[
            "Mean"
        ]
    )
)


max_rms_error = np.max(
    np.abs(
        verification_df[
            "RMS"
        ]
        - 1.0
    )
)


max_B_asymmetry = np.max(
    np.abs(
        verification_df[
            "B_Positive"
        ]
        -
        verification_df[
            "B_Negative"
        ]
    )
)


print(
    "\n"
    + "=" * 94
)

print(
    "UNIT-RMS VERIFICATION"
)

print(
    "=" * 94
)


print(
    "\nMaximum absolute mean:"
)

print(
    f"  {max_abs_mean:.12e}"
)


print(
    "\nMaximum |RMS - 1|:"
)

print(
    f"  {max_rms_error:.12e}"
)


print(
    "\nMaximum |B_positive - B_negative|:"
)

print(
    f"  {max_B_asymmetry:.12e}"
)


# ---------------------------------------------------------------------
# 10. Configuration membership
# ---------------------------------------------------------------------

verification_df[
    "Configuration"
] = verification_df[
    "Trajectory_ID"
].apply(
    configuration_from_trajectory
)


configurations = sorted(
    verification_df[
        "Configuration"
    ].unique()
)


if len(configurations) != 14:

    raise RuntimeError(
        f"\nExpected 14 configurations, "
        f"found {len(configurations)}."
    )


config_to_trajectories = {}


for configuration in configurations:

    members = sorted(
        verification_df.loc[
            verification_df[
                "Configuration"
            ]
            == configuration,
            "Trajectory_ID"
        ].tolist()
    )


    if len(members) != 2:

        raise RuntimeError(
            f"\nConfiguration {configuration} "
            f"contains {len(members)} trajectories; "
            f"expected exactly 2."
        )


    config_to_trajectories[
        configuration
    ] = members


print(
    "\n"
    + "=" * 94
)

print(
    "BOOTSTRAP CLUSTERS"
)

print(
    "=" * 94
)


for configuration in configurations:

    print(
        f"  {configuration}: "
        f"{config_to_trajectories[configuration]}"
    )


print(
    f"\nConfigurations:"
    f"\n  {len(configurations)}"
)


print(
    "\nPasses retained within every selected cluster:"
)

print(
    "  P1 + P3"
)


# ---------------------------------------------------------------------
# 11. Build full 28 x 28 all-phase K matrix
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
    + "=" * 94
)

print(
    "BUILDING COMPLETE ALL-PHASE K MATRIX"
)

print(
    "=" * 94
)


for i in range(
    n_trajectories
):

    z_i = structure_store[
        trajectory_ids[
            i
        ]
    ]


    for j in range(
        i,
        n_trajectories
    ):

        z_j = structure_store[
            trajectory_ids[
                j
            ]
        ]


        K_value = exact_all_phase_l1(
            z_i,
            z_j
        )


        K_matrix[
            i,
            j
        ] = K_value


        K_matrix[
            j,
            i
        ] = K_value


print(
    "\nK matrix complete."
)


print(
    f"\nShape:"
    f"\n  {K_matrix.shape}"
)


print(
    "\nDiagonal self-pair K:"
)

print(
    f"  min    = "
    f"{np.min(np.diag(K_matrix)):.9f}"
)

print(
    f"  median = "
    f"{np.median(np.diag(K_matrix)):.9f}"
)

print(
    f"  max    = "
    f"{np.max(np.diag(K_matrix)):.9f}"
)


# ---------------------------------------------------------------------
# 12. Original empirical point estimates
# ---------------------------------------------------------------------

upper_i, upper_j = np.triu_indices(
    n_trajectories,
    k=1
)


original_pair_K = K_matrix[
    upper_i,
    upper_j
]


POINT_K = float(
    np.mean(
        original_pair_K
    )
)


trajectory_B = np.array(
    [
        verification_df.loc[
            verification_df[
                "Trajectory_ID"
            ]
            == trajectory_id,
            "B_One_Edge"
        ].iloc[
            0
        ]
        for trajectory_id
        in trajectory_ids
    ],
    dtype=float
)


POINT_B = float(
    np.mean(
        trajectory_B
    )
)


print(
    "\n"
    + "=" * 94
)

print(
    "RECONSTRUCTED EMPIRICAL POINT ESTIMATES"
)

print(
    "=" * 94
)


print(
    "\nEmpirical all-phase K:"
)

print(
    f"  {POINT_K:.12f}"
)


print(
    "\nEmpirical one-edge B:"
)

print(
    f"  {POINT_B:.12f}"
)


print(
    "\nPrevious Stage 3C values:"
)

print(
    "  K = 1.122528585367"
)

print(
    "  B = 0.398524115117"
)


print(
    "\nDifferences:"
)

print(
    f"  K difference = "
    f"{POINT_K - 1.122528585367:.12e}"
)

print(
    f"  B difference = "
    f"{POINT_B - 0.398524115117:.12e}"
)


# ---------------------------------------------------------------------
# 13. Map each configuration to numerical trajectory indices
# ---------------------------------------------------------------------

config_to_indices = {}


for configuration in configurations:

    config_to_indices[
        configuration
    ] = [
        trajectory_index[
            trajectory_id
        ]
        for trajectory_id
        in config_to_trajectories[
            configuration
        ]
    ]


# ---------------------------------------------------------------------
# 14. Configuration-cluster bootstrap
# ---------------------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_SEED
)


bootstrap_rows = []


n_configurations = len(
    configurations
)


print(
    "\n"
    + "=" * 94
)

print(
    "RUNNING CONFIGURATION-CLUSTER BOOTSTRAP"
)

print(
    "=" * 94
)


print(
    f"\nBootstrap replicates:"
    f"\n  {BOOTSTRAP_REPLICATES:,}"
)


print(
    f"\nRandom seed:"
    f"\n  {RANDOM_SEED}"
)


for replicate in range(
    BOOTSTRAP_REPLICATES
):

    selected_configs = rng.choice(
        configurations,
        size=n_configurations,
        replace=True
    )


    selected_indices = []


    for configuration in selected_configs:

        selected_indices.extend(
            config_to_indices[
                configuration
            ]
        )


    selected_indices = np.asarray(
        selected_indices,
        dtype=int
    )


    # -------------------------------------------------------------
    # Pairwise K for bootstrap observation copies
    # -------------------------------------------------------------
    #
    # selected_indices may contain the same underlying empirical
    # trajectory several times.
    #
    # We treat each occurrence as a distinct bootstrap observation.
    # Thus combinations are formed across bootstrap positions, not
    # across unique empirical IDs.
    #
    # K_matrix includes the required diagonal K(i,i).
    #
    # -------------------------------------------------------------

    bootstrap_i, bootstrap_j = np.triu_indices(
        len(
            selected_indices
        ),
        k=1
    )


    K_values = K_matrix[
        selected_indices[
            bootstrap_i
        ],
        selected_indices[
            bootstrap_j
        ]
    ]


    K_boot = float(
        np.mean(
            K_values
        )
    )


    # -------------------------------------------------------------
    # Boundary B
    # -------------------------------------------------------------

    B_values = trajectory_B[
        selected_indices
    ]


    B_boot = float(
        np.mean(
            B_values
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


    # -------------------------------------------------------------
    # Operational propagation
    # -------------------------------------------------------------

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


        tolerance_factor = (
            1.0
            / C_M
        )


        row[
            f"C_M_{M}"
        ] = C_M


        row[
            f"Tolerance_Factor_M_{M}"
        ] = tolerance_factor


    bootstrap_rows.append(
        row
    )


    if (
        replicate + 1
    ) % 500 == 0:

        print(
            f"  completed "
            f"{replicate + 1:,}"
            f" / "
            f"{BOOTSTRAP_REPLICATES:,}"
        )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


# ---------------------------------------------------------------------
# 15. K and B uncertainty summaries
# ---------------------------------------------------------------------

K_low, K_high = percentile_interval(
    bootstrap_df[
        "K"
    ]
)


B_low, B_high = percentile_interval(
    bootstrap_df[
        "B"
    ]
)


coefficient_summary_df = pd.DataFrame(
    [
        {
            "Coefficient":
                "K",

            "Point_Estimate":
                POINT_K,

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
                K_high
        },

        {
            "Coefficient":
                "B",

            "Point_Estimate":
                POINT_B,

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
                B_high
        }
    ]
)


# ---------------------------------------------------------------------
# 16. Operational coefficient uncertainty
# ---------------------------------------------------------------------

operational_rows = []


for M in PASS_COUNTS:

    C_column = (
        f"C_M_{M}"
    )


    T_column = (
        f"Tolerance_Factor_M_{M}"
    )


    C_low, C_high = percentile_interval(
        bootstrap_df[
            C_column
        ]
    )


    T_low, T_high = percentile_interval(
        bootstrap_df[
            T_column
        ]
    )


    point_C = (
        (
            M - 1
        )
        / M
        * POINT_K
        +
        2.0
        * POINT_B
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
        * COMMON_K
        +
        2.0
        * COMMON_B
        / M
    )


    common_T = (
        1.0
        / common_C
    )


    operational_rows.append(
        {
            "Number_of_Passes_M":
                M,

            "Empirical_C_Point":
                point_C,

            "Empirical_C_Bootstrap_Mean":
                bootstrap_df[
                    C_column
                ].mean(),

            "Empirical_C_Bootstrap_SD":
                bootstrap_df[
                    C_column
                ].std(
                    ddof=1
                ),

            "Empirical_C_CI95_Lower":
                C_low,

            "Empirical_C_CI95_Upper":
                C_high,

            "Tolerance_Factor_Point":
                point_T,

            "Tolerance_Factor_CI95_Lower":
                T_low,

            "Tolerance_Factor_CI95_Upper":
                T_high,

            "Common_Marginal_C":
                common_C,

            "Common_Marginal_Tolerance_Factor":
                common_T
        }
    )


operational_df = pd.DataFrame(
    operational_rows
)


# ---------------------------------------------------------------------
# 17. Large-M uncertainty
# ---------------------------------------------------------------------
#
# As M -> infinity:
#
#       C(M) -> K
#
#       tolerance factor -> 1/K
#
# ---------------------------------------------------------------------

large_M_tolerance = (
    1.0
    / bootstrap_df[
        "K"
    ]
)


large_T_low, large_T_high = percentile_interval(
    large_M_tolerance
)


large_limit_df = pd.DataFrame(
    [
        {
            "Quantity":
                "Large-M C limit",

            "Point_Estimate":
                POINT_K,

            "CI95_Lower":
                K_low,

            "CI95_Upper":
                K_high
        },

        {
            "Quantity":
                "Large-M tolerance factor 1/K",

            "Point_Estimate":
                1.0
                / POINT_K,

            "CI95_Lower":
                large_T_low,

            "CI95_Upper":
                large_T_high
        }
    ]
)


# ---------------------------------------------------------------------
# 18. Bootstrap correlation between K and B
# ---------------------------------------------------------------------
#
# This is NOT the problematic 378-pair Spearman analysis.
#
# It simply describes how the two cluster-bootstrap estimators co-vary
# across resampled configuration sets.
#
# ---------------------------------------------------------------------

bootstrap_KB_correlation = np.corrcoef(
    bootstrap_df[
        "K"
    ],
    bootstrap_df[
        "B"
    ]
)[
    0,
    1
]


# ---------------------------------------------------------------------
# 19. Save output tables
# ---------------------------------------------------------------------

verification_file = (
    TABLES_DIR
    / "bootstrap_structure_verification.csv"
)


K_matrix_file = (
    TABLES_DIR
    / "bootstrap_complete_K_matrix.csv"
)


replicate_file = (
    TABLES_DIR
    / "bootstrap_configuration_replicates.csv"
)


coefficient_summary_file = (
    TABLES_DIR
    / "bootstrap_coefficient_summary.csv"
)


operational_file = (
    TABLES_DIR
    / "bootstrap_operational_coefficients.csv"
)


large_limit_file = (
    TABLES_DIR
    / "bootstrap_large_field_limits.csv"
)


verification_df.to_csv(
    verification_file,
    index=False
)


K_matrix_df = pd.DataFrame(
    K_matrix,
    index=trajectory_ids,
    columns=trajectory_ids
)


K_matrix_df.to_csv(
    K_matrix_file,
    index=True
)


bootstrap_df.to_csv(
    replicate_file,
    index=False
)


coefficient_summary_df.to_csv(
    coefficient_summary_file,
    index=False
)


operational_df.to_csv(
    operational_file,
    index=False
)


large_limit_df.to_csv(
    large_limit_file,
    index=False
)


# ---------------------------------------------------------------------
# 20. Figure — K bootstrap distribution
# ---------------------------------------------------------------------

plt.figure(
    figsize=(
        9,
        6
    )
)


plt.hist(
    bootstrap_df[
        "K"
    ],
    bins=50
)


plt.axvline(
    POINT_K,
    linestyle="--",
    linewidth=2,
    label="Empirical point estimate"
)


plt.axvline(
    K_low,
    linestyle=":",
    linewidth=2,
    label="95% bootstrap interval"
)


plt.axvline(
    K_high,
    linestyle=":",
    linewidth=2
)


plt.axvline(
    COMMON_K,
    linestyle="-.",
    linewidth=2,
    label="Common-marginal reference"
)


plt.xlabel(
    "All-phase internal coefficient K"
)


plt.ylabel(
    "Bootstrap replicate count"
)


plt.title(
    "Configuration-cluster bootstrap distribution of K"
)


plt.legend()


plt.tight_layout()


K_figure = (
    FIGURES_DIR
    / "bootstrap_K_distribution.png"
)


plt.savefig(
    K_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ---------------------------------------------------------------------
# 21. Figure — B bootstrap distribution
# ---------------------------------------------------------------------

plt.figure(
    figsize=(
        9,
        6
    )
)


plt.hist(
    bootstrap_df[
        "B"
    ],
    bins=50
)


plt.axvline(
    POINT_B,
    linestyle="--",
    linewidth=2,
    label="Empirical point estimate"
)


plt.axvline(
    B_low,
    linestyle=":",
    linewidth=2,
    label="95% bootstrap interval"
)


plt.axvline(
    B_high,
    linestyle=":",
    linewidth=2
)


plt.axvline(
    COMMON_B,
    linestyle="-.",
    linewidth=2,
    label="Common-marginal reference"
)


plt.xlabel(
    "One-edge boundary coefficient B"
)


plt.ylabel(
    "Bootstrap replicate count"
)


plt.title(
    "Configuration-cluster bootstrap distribution of B"
)


plt.legend()


plt.tight_layout()


B_figure = (
    FIGURES_DIR
    / "bootstrap_B_distribution.png"
)


plt.savefig(
    B_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ---------------------------------------------------------------------
# 22. Figure — C(M) with cluster-bootstrap interval
# ---------------------------------------------------------------------

plt.figure(
    figsize=(
        9,
        6
    )
)


plt.plot(
    operational_df[
        "Number_of_Passes_M"
    ],
    operational_df[
        "Empirical_C_Point"
    ],
    marker="o",
    label="Empirical point estimate"
)


plt.fill_between(
    operational_df[
        "Number_of_Passes_M"
    ],
    operational_df[
        "Empirical_C_CI95_Lower"
    ],
    operational_df[
        "Empirical_C_CI95_Upper"
    ],
    alpha=0.25,
    label="95% configuration-bootstrap interval"
)


plt.plot(
    operational_df[
        "Number_of_Passes_M"
    ],
    operational_df[
        "Common_Marginal_C"
    ],
    linestyle="--",
    marker="s",
    label="Common-marginal control"
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
    "Operational coefficient with configuration-level uncertainty"
)


plt.legend()


plt.tight_layout()


C_figure = (
    FIGURES_DIR
    / "bootstrap_operational_coefficient_C.png"
)


plt.savefig(
    C_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ---------------------------------------------------------------------
# 23. Figure — tolerance transfer
# ---------------------------------------------------------------------

plt.figure(
    figsize=(
        9,
        6
    )
)


plt.plot(
    operational_df[
        "Number_of_Passes_M"
    ],
    operational_df[
        "Tolerance_Factor_Point"
    ],
    marker="o",
    label="Empirical point estimate"
)


plt.fill_between(
    operational_df[
        "Number_of_Passes_M"
    ],
    operational_df[
        "Tolerance_Factor_CI95_Lower"
    ],
    operational_df[
        "Tolerance_Factor_CI95_Upper"
    ],
    alpha=0.25,
    label="95% configuration-bootstrap interval"
)


plt.plot(
    operational_df[
        "Number_of_Passes_M"
    ],
    operational_df[
        "Common_Marginal_Tolerance_Factor"
    ],
    linestyle="--",
    marker="s",
    label="Common-marginal control"
)


plt.xscale(
    "log"
)


plt.xlabel(
    "Number of parallel passes M"
)


plt.ylabel(
    "Tolerance transfer factor 1/C(M)"
)


plt.title(
    "Uncertainty in the operational guidance-tolerance transfer"
)


plt.legend()


plt.tight_layout()


T_figure = (
    FIGURES_DIR
    / "bootstrap_tolerance_transfer_factor.png"
)


plt.savefig(
    T_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ---------------------------------------------------------------------
# 24. Terminal report
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
    + "=" * 94
)

print(
    "CONFIGURATION-CLUSTER BOOTSTRAP RESULTS"
)

print(
    "=" * 94
)


print(
    "\nBootstrap design:"
)


print(
    f"  sampling unit       = configuration"
)


print(
    f"  configurations      = "
    f"{n_configurations}"
)


print(
    f"  trajectories/config = 2"
)


print(
    f"  passes retained     = P1 + P3"
)


print(
    f"  replicates          = "
    f"{BOOTSTRAP_REPLICATES:,}"
)


print(
    f"  random seed         = "
    f"{RANDOM_SEED}"
)


print(
    "\n"
    + "=" * 94
)

print(
    "EMPIRICAL COEFFICIENT UNCERTAINTY"
)

print(
    "=" * 94
)


print(
    coefficient_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}"
    )
)


print(
    "\nBootstrap K-B correlation:"
)

print(
    f"  {bootstrap_KB_correlation:.6f}"
)


print(
    "\n"
    + "=" * 94
)

print(
    "OPERATIONAL COEFFICIENT AND TOLERANCE UNCERTAINTY"
)

print(
    "=" * 94
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
    + "=" * 94
)

print(
    "LARGE-FIELD LIMIT"
)

print(
    "=" * 94
)


print(
    large_limit_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}"
    )
)


print(
    "\n"
    + "=" * 94
)

print(
    "FINAL TOLERANCE INTERPRETATION"
)

print(
    "=" * 94
)


print(
    "\nFor an externally specified allowable discrepancy q:"
)


print(
    "\n  R_max = W q / C(M)"
)


print(
    "\nThe table reports the bootstrap uncertainty of:"
)


print(
    "\n  1/C(M) = R_max / (W q)"
)


print(
    "\nThus no arbitrary implement width W or agronomic threshold q "
    "is required to quantify uncertainty."
)


print(
    "\n"
    + "=" * 94
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 94
)


print(
    f"\nStructure verification:\n"
    f"{verification_file}"
)


print(
    f"\nComplete K matrix:\n"
    f"{K_matrix_file}"
)


print(
    f"\nBootstrap replicate table:\n"
    f"{replicate_file}"
)


print(
    f"\nCoefficient uncertainty:\n"
    f"{coefficient_summary_file}"
)


print(
    f"\nOperational uncertainty:\n"
    f"{operational_file}"
)


print(
    f"\nLarge-field uncertainty:\n"
    f"{large_limit_file}"
)


print(
    f"\nK bootstrap figure:\n"
    f"{K_figure}"
)


print(
    f"\nB bootstrap figure:\n"
    f"{B_figure}"
)


print(
    f"\nOperational coefficient figure:\n"
    f"{C_figure}"
)


print(
    f"\nTolerance transfer figure:\n"
    f"{T_figure}"
)


print(
    "\n"
    + "=" * 94
)

print(
    "STAGE 4C COMPLETE"
)

print(
    "=" * 94
)