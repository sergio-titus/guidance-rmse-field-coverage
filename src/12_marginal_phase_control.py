from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import ndtri


# ============================================================
# STAGE 2C — COMMON-MARGINAL AND PHASE CONTROL
# ============================================================
#
# PURPOSE
# -------
# Stage 2B showed that equal RMS does not uniquely determine
# realized neighboring-pass coverage discrepancy.
#
# However, its pairwise structure factor:
#
#       K = mean(|z2 - z1|)
#
# can depend simultaneously on:
#
#   - spatial ordering / dependence,
#   - marginal distribution,
#   - relative spatial phase,
#   - higher-order structure.
#
# This stage removes the marginal-distribution confounder and
# explicitly examines relative phase.
#
#
# PART A — COMMON MARGINAL
# ------------------------
#
# Every empirical centered trajectory is transformed by its
# within-trajectory rank to the SAME deterministic Gaussian
# score set.
#
# Therefore every transformed trajectory has exactly the same:
#
#       marginal values,
#       mean,
#       RMS,
#       empirical marginal distribution.
#
# What differs is the spatial ORDER of those common values.
#
# This preserves spatial rank ordering while removing
# differences in skewness, kurtosis, and amplitude distribution.
#
#
# PART B — PHASE CONTROL
# ----------------------
#
# A fixed x-to-x comparison can depend on arbitrary relative
# alignment between two empirical trajectories.
#
# For representative pairs we therefore evaluate EVERY possible
# circular spatial shift.
#
# No random number of shifts is chosen.
#
#
# PART C — EXACT PHASE RESULT
# ---------------------------
#
# Because all transformed trajectories contain exactly the same
# multiset of values, averaging mean absolute difference over
# ALL possible circular shifts is equivalent to averaging over
# every possible pair of marginal values.
#
# Thus the phase-averaged L1 structure factor is analytically
# common to all trajectory pairs.
#
# This distinguishes:
#
#   expected local magnitude
#
# from:
#
#   spatial clustering / finite-field variability.
#
#
# IMPORTANT INTERPRETATION
# ------------------------
#
# If common-marginal, random-phase expected K is identical across
# structures, then spatial autocorrelation should NOT be claimed
# to change the infinite-domain expected pointwise skip/overlap.
#
# Instead, spatial dependence is expected to matter through:
#
#   - finite-field variability,
#   - clustering of skip/overlap,
#   - persistence lengths,
#   - boundary violations,
#   - tail probabilities,
#   - realization-to-realization uncertainty.
#
# Those are the appropriate targets for the later Monte Carlo
# field simulation.
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
    / "marginal_phase_control"
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


def integrate_mean(
    values,
    x
):

    values = np.asarray(
        values,
        dtype=float
    )

    x = np.asarray(
        x,
        dtype=float
    )

    length = (
        x[-1]
        - x[0]
    )

    if length <= 0:

        raise RuntimeError(
            "Non-positive integration length."
        )

    if hasattr(
        np,
        "trapezoid"
    ):

        integral = np.trapezoid(
            values,
            x
        )

    else:

        integral = np.trapz(
            values,
            x
        )

    return (
        integral
        / length
    )


def normalized_acf(
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

    denominator = np.sum(
        centered ** 2
    )

    n = len(
        centered
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
        centered,
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


def e_folding_distance(
    acf,
    dx
):

    threshold = np.exp(
        -1
    )

    indices = np.where(
        acf[1:]
        <= threshold
    )[0]

    if len(
        indices
    ) == 0:

        return np.nan

    return (
        (
            indices[0]
            + 1
        )
        * dx
    )


def integral_scale(
    acf,
    dx
):

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


def common_grid_for_all(
    trajectory_dict,
    spacing
):

    starts = []

    ends = []

    for record in trajectory_dict.values():

        starts.append(
            np.min(
                record[
                    "X"
                ]
            )
        )

        ends.append(
            np.max(
                record[
                    "X"
                ]
            )
        )

    start = max(
        starts
    )

    end = min(
        ends
    )

    if end <= start:

        raise RuntimeError(
            "No common spatial support for all trajectories."
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

    return grid


def gaussian_rank_transform(
    values,
    base_scores
):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(
        values
    ) != len(
        base_scores
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

    transformed = base_scores[
        ranks
    ]

    return transformed


def circular_k_l1(
    z1,
    z2
):

    return np.mean(
        np.abs(
            z2
            - z1
        )
    )


def circular_k_l2(
    z1,
    z2
):

    return np.sqrt(
        np.mean(
            (
                z2
                - z1
            ) ** 2
        )
    )


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing input file:\n"
        f"{INPUT_FILE}"
    )


print(
    "=" * 90
)

print(
    "STAGE 2C — COMMON-MARGINAL AND PHASE CONTROL"
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
# 4. Retrieve one complete structure set
# ------------------------------------------------------------
#
# Physical RMS magnitude is irrelevant here because all
# structures will be transformed and normalized.
#
# Median empirical centered-RMS level is used only to retrieve
# one copy of all 28 structures.
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
    f"\nStructures:"
    f"\n  {len(trajectory_ids)}"
)


# ------------------------------------------------------------
# 5. Load native trajectory representations
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
# 6. Common spatial support
# ------------------------------------------------------------

grid = common_grid_for_all(
    raw_store,
    DX_M
)


N = len(
    grid
)


print(
    f"\nCommon spatial domain:"
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
    f"{grid[-1] - grid[0]:.3f} m"
)

print(
    f"  points = "
    f"{N:,}"
)


# ------------------------------------------------------------
# 7. Construct one EXACT common marginal
# ------------------------------------------------------------
#
# Gaussian plotting positions:
#
#       p_r = (r + 0.5) / N
#
# for r = 0,...,N-1.
#
# These are deterministic numerical quantiles, not an assumed
# stochastic model for the original GNSS data.
#
# Every structure receives exactly this same multiset.
#
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


base_scores = ndtri(
    probabilities
)


base_scores = (
    base_scores
    - np.mean(
        base_scores
    )
)


base_scores = (
    base_scores
    / rms(
        base_scores
    )
)


print(
    "\nCommon Gaussian-score marginal:"
)

print(
    f"  mean = "
    f"{np.mean(base_scores):.12e}"
)

print(
    f"  RMS  = "
    f"{rms(base_scores):.12f}"
)


# ------------------------------------------------------------
# 8. Rank-Gaussianize every spatial structure
# ------------------------------------------------------------

structure_store = {}

structure_rows = []


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
        base_scores
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


    acf = normalized_acf(
        transformed
    )


    e_fold = e_folding_distance(
        acf,
        DX_M
    )


    int_scale = integral_scale(
        acf,
        DX_M
    )


    structure_store[
        trajectory_id
    ] = transformed


    structure_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Mean":
                np.mean(
                    transformed
                ),

            "RMS":
                rms(
                    transformed
                ),

            "E_Folding_m":
                e_fold,

            "Integral_Scale_m":
                int_scale
        }
    )


structure_df = pd.DataFrame(
    structure_rows
)


# ------------------------------------------------------------
# 9. Verify identical marginal distributions
# ------------------------------------------------------------

reference_sorted = np.sort(
    structure_store[
        trajectory_ids[0]
    ]
)


maximum_marginal_difference = 0.0


for trajectory_id in trajectory_ids[1:]:

    sorted_values = np.sort(
        structure_store[
            trajectory_id
        ]
    )

    difference = np.max(
        np.abs(
            sorted_values
            - reference_sorted
        )
    )

    maximum_marginal_difference = max(
        maximum_marginal_difference,
        difference
    )


# ------------------------------------------------------------
# 10. Pairwise zero-phase K factors
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


    delta = (
        z2
        - z1
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


    k_l1 = np.mean(
        np.abs(
            delta
        )
    )


    k_l2 = np.sqrt(
        np.mean(
            delta ** 2
        )
    )


    structure_1_row = structure_df[
        structure_df[
            "Trajectory_ID"
        ]
        == trajectory_1
    ].iloc[
        0
    ]


    structure_2_row = structure_df[
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

            "K_Skip_Zero_Phase":
                k_skip,

            "K_Overlap_Zero_Phase":
                k_overlap,

            "K_L1_Zero_Phase":
                k_l1,

            "K_L2_Zero_Phase":
                k_l2,

            "Integral_Scale_1_m":
                structure_1_row[
                    "Integral_Scale_m"
                ],

            "Integral_Scale_2_m":
                structure_2_row[
                    "Integral_Scale_m"
                ],

            "Absolute_Integral_Scale_Difference_m":
                abs(
                    structure_1_row[
                        "Integral_Scale_m"
                    ]
                    -
                    structure_2_row[
                        "Integral_Scale_m"
                    ]
                ),

            "E_Folding_1_m":
                structure_1_row[
                    "E_Folding_m"
                ],

            "E_Folding_2_m":
                structure_2_row[
                    "E_Folding_m"
                ],

            "Absolute_E_Folding_Difference_m":
                abs(
                    structure_1_row[
                        "E_Folding_m"
                    ]
                    -
                    structure_2_row[
                        "E_Folding_m"
                    ]
                )
        }
    )


pair_df = pd.DataFrame(
    pair_rows
)


# ------------------------------------------------------------
# 11. Exact phase-averaged L1 factor
# ------------------------------------------------------------
#
# Averaging across every circular phase and every spatial point
# evaluates every ordered combination of marginal values once.
#
# Since all trajectories now share the SAME marginal multiset,
# the phase-averaged expected K_L1 is identical for all pairs.
#
# ------------------------------------------------------------

pairwise_absolute_marginal_difference = np.abs(
    base_scores[
        :,
        None
    ]
    -
    base_scores[
        None,
        :
    ]
)


EXACT_PHASE_AVERAGED_K_L1 = np.mean(
    pairwise_absolute_marginal_difference
)


EXACT_PHASE_AVERAGED_K_SKIP = (
    EXACT_PHASE_AVERAGED_K_L1
    / 2.0
)


EXACT_PHASE_AVERAGED_K_OVERLAP = (
    EXACT_PHASE_AVERAGED_K_L1
    / 2.0
)


# ------------------------------------------------------------
# 12. Exact phase-distribution of L2 for EVERY pair
# ------------------------------------------------------------
#
# For circular shifts:
#
#   mean[(z2_s - z1)^2]
#       = 2 - 2 * circular_cross_correlation(s)
#
# because each sequence has mean 0 and RMS 1.
#
# FFT gives every circular shift exactly.
#
# ------------------------------------------------------------

phase_l2_rows = []


for _, row in pair_df.iterrows():

    trajectory_1 = row[
        "Trajectory_1"
    ]

    trajectory_2 = row[
        "Trajectory_2"
    ]


    z1 = structure_store[
        trajectory_1
    ]

    z2 = structure_store[
        trajectory_2
    ]


    fft_1 = np.fft.fft(
        z1
    )

    fft_2 = np.fft.fft(
        z2
    )


    circular_cross_correlation = np.real(
        np.fft.ifft(
            np.conjugate(
                fft_1
            )
            * fft_2
        )
    ) / N


    mean_squared_difference = (
        2.0
        -
        2.0
        * circular_cross_correlation
    )


    mean_squared_difference = np.maximum(
        mean_squared_difference,
        0.0
    )


    k_l2_all_shifts = np.sqrt(
        mean_squared_difference
    )


    phase_l2_rows.append(
        {
            "Pair_ID":
                row[
                    "Pair_ID"
                ],

            "Trajectory_1":
                trajectory_1,

            "Trajectory_2":
                trajectory_2,

            "K_L2_Shift_Min":
                np.min(
                    k_l2_all_shifts
                ),

            "K_L2_Shift_Q25":
                np.quantile(
                    k_l2_all_shifts,
                    0.25
                ),

            "K_L2_Shift_Median":
                np.median(
                    k_l2_all_shifts
                ),

            "K_L2_Shift_Q75":
                np.quantile(
                    k_l2_all_shifts,
                    0.75
                ),

            "K_L2_Shift_Max":
                np.max(
                    k_l2_all_shifts
                ),

            "K_L2_Shift_SD":
                np.std(
                    k_l2_all_shifts,
                    ddof=0
                ),

            "Mean_Squared_Difference_Across_Shifts":
                np.mean(
                    mean_squared_difference
                )
        }
    )


phase_l2_df = pd.DataFrame(
    phase_l2_rows
)


# ------------------------------------------------------------
# 13. Select representative zero-phase L1 pairs
# ------------------------------------------------------------

minimum_pair = pair_df.loc[
    pair_df[
        "K_L1_Zero_Phase"
    ].idxmin()
]


maximum_pair = pair_df.loc[
    pair_df[
        "K_L1_Zero_Phase"
    ].idxmax()
]


median_k = pair_df[
    "K_L1_Zero_Phase"
].median()


median_pair_index = (
    pair_df[
        "K_L1_Zero_Phase"
    ]
    -
    median_k
).abs().idxmin()


median_pair = pair_df.loc[
    median_pair_index
]


representative_pairs = [
    (
        "Minimum",
        minimum_pair
    ),
    (
        "Median",
        median_pair
    ),
    (
        "Maximum",
        maximum_pair
    )
]


# ------------------------------------------------------------
# 14. Exact L1 circular-shift distributions for representative
#     pairs
# ------------------------------------------------------------

phase_l1_rows = []


for representative_label, row in representative_pairs:

    trajectory_1 = row[
        "Trajectory_1"
    ]

    trajectory_2 = row[
        "Trajectory_2"
    ]


    z1 = structure_store[
        trajectory_1
    ]

    z2 = structure_store[
        trajectory_2
    ]


    shift_values = []


    for shift_index in range(
        N
    ):

        shifted_z2 = np.roll(
            z2,
            shift_index
        )


        k_l1 = circular_k_l1(
            z1,
            shifted_z2
        )


        shift_values.append(
            k_l1
        )


        phase_l1_rows.append(
            {
                "Representative_Group":
                    representative_label,

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
                    k_l1
            }
        )


    shift_values = np.asarray(
        shift_values
    )


    print(
        "\n"
        + "-" * 90
    )

    print(
        f"{representative_label.upper()} ZERO-PHASE PAIR — "
        f"ALL CIRCULAR SHIFTS"
    )

    print(
        "-" * 90
    )

    print(
        f"\n{trajectory_1} vs {trajectory_2}"
    )

    print(
        f"\nZero-phase K_L1:"
        f"\n  {row['K_L1_Zero_Phase']:.9f}"
    )

    print(
        "\nAll-shift K_L1 distribution:"
    )

    print(
        f"  min    = "
        f"{shift_values.min():.9f}"
    )

    print(
        f"  Q25    = "
        f"{np.quantile(shift_values, 0.25):.9f}"
    )

    print(
        f"  median = "
        f"{np.median(shift_values):.9f}"
    )

    print(
        f"  Q75    = "
        f"{np.quantile(shift_values, 0.75):.9f}"
    )

    print(
        f"  max    = "
        f"{shift_values.max():.9f}"
    )

    print(
        f"  mean   = "
        f"{shift_values.mean():.9f}"
    )

    print(
        f"  SD     = "
        f"{shift_values.std(ddof=0):.9f}"
    )

    print(
        f"\nDifference from exact common "
        f"phase-averaged expectation:"
    )

    print(
        f"  "
        f"{abs(shift_values.mean() - EXACT_PHASE_AVERAGED_K_L1):.12e}"
    )


phase_l1_df = pd.DataFrame(
    phase_l1_rows
)


# ------------------------------------------------------------
# 15. Descriptive associations after marginal control
# ------------------------------------------------------------

association_rows = []


for variable in [
    "Absolute_Integral_Scale_Difference_m",
    "Absolute_E_Folding_Difference_m"
]:

    rho = pair_df[
        [
            variable,
            "K_L1_Zero_Phase"
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
                variable,

            "Spearman_Rho_with_Zero_Phase_K_L1":
                rho
        }
    )


association_df = pd.DataFrame(
    association_rows
)


# ------------------------------------------------------------
# 16. Save tables
# ------------------------------------------------------------

structure_file = (
    TABLES_DIR
    / "common_marginal_structure_catalog.csv"
)

pair_file = (
    TABLES_DIR
    / "common_marginal_zero_phase_pairwise_factors.csv"
)

phase_l2_file = (
    TABLES_DIR
    / "common_marginal_all_pair_phase_l2_summary.csv"
)

phase_l1_file = (
    TABLES_DIR
    / "common_marginal_representative_phase_l1.csv"
)

association_file = (
    TABLES_DIR
    / "common_marginal_structure_associations.csv"
)


structure_df.to_csv(
    structure_file,
    index=False
)

pair_df.to_csv(
    pair_file,
    index=False
)

phase_l2_df.to_csv(
    phase_l2_file,
    index=False
)

phase_l1_df.to_csv(
    phase_l1_file,
    index=False
)

association_df.to_csv(
    association_file,
    index=False
)


# ------------------------------------------------------------
# 17. Figure — original zero-phase spread after common marginal
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    pair_df[
        "K_L1_Zero_Phase"
    ],
    bins="auto"
)


plt.axvline(
    EXACT_PHASE_AVERAGED_K_L1,
    linestyle="--",
    label=(
        "All-phase common expectation"
    )
)


plt.xlabel(
    "Zero-phase common-marginal K_L1"
)

plt.ylabel(
    "Number of empirical structure pairs"
)

plt.title(
    "Aligned pairwise discrepancy after exact marginal control"
)

plt.legend()

plt.tight_layout()


zero_phase_figure = (
    FIGURES_DIR
    / "common_marginal_zero_phase_K_distribution.png"
)


plt.savefig(
    zero_phase_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 18. Figure — phase distributions for representative pairs
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


for representative_label, group in phase_l1_df.groupby(
    "Representative_Group",
    sort=False
):

    plt.hist(
        group[
            "K_L1"
        ],
        bins="auto",
        alpha=0.45,
        label=representative_label
    )


plt.axvline(
    EXACT_PHASE_AVERAGED_K_L1,
    linestyle="--",
    linewidth=2,
    label="Exact all-phase mean"
)


plt.xlabel(
    "K_L1 across circular spatial shifts"
)

plt.ylabel(
    "Number of shifts"
)

plt.title(
    "Effect of relative spatial phase after identical-marginal control"
)

plt.legend()

plt.tight_layout()


phase_distribution_figure = (
    FIGURES_DIR
    / "representative_all_phase_KL1_distributions.png"
)


plt.savefig(
    phase_distribution_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 19. Figure — phase variability versus correlation-scale
#      difference
# ------------------------------------------------------------

phase_merge = phase_l2_df.merge(
    pair_df[
        [
            "Pair_ID",
            "Absolute_Integral_Scale_Difference_m"
        ]
    ],
    on="Pair_ID",
    how="left"
)


plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    phase_merge[
        "Absolute_Integral_Scale_Difference_m"
    ],
    phase_merge[
        "K_L2_Shift_SD"
    ],
    alpha=0.7
)


plt.xlabel(
    "Absolute difference in transformed integral scale (m)"
)

plt.ylabel(
    "SD of K_L2 across all circular shifts"
)

plt.title(
    "Relative-phase sensitivity versus spatial-scale difference"
)

plt.tight_layout()


phase_variability_figure = (
    FIGURES_DIR
    / "phase_variability_vs_integral_scale_difference.png"
)


plt.savefig(
    phase_variability_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 20. Terminal formatting
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
    100
)


# ------------------------------------------------------------
# 21. Verification
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "COMMON-MARGINAL VERIFICATION"
)

print(
    "=" * 90
)


print(
    f"\nMaximum absolute marginal-value difference "
    f"between transformed structures:"
)

print(
    f"  {maximum_marginal_difference:.12e}"
)


print(
    "\nTransformed structure mean:"
)

print(
    f"  maximum |mean| = "
    f"{structure_df['Mean'].abs().max():.12e}"
)


print(
    "\nTransformed structure RMS:"
)

print(
    f"  maximum |RMS - 1| = "
    f"{(structure_df['RMS'] - 1.0).abs().max():.12e}"
)


# ------------------------------------------------------------
# 22. Zero-phase result
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "ZERO-PHASE K_L1 AFTER IDENTICAL-MARGINAL CONTROL"
)

print(
    "=" * 90
)


zero_k = pair_df[
    "K_L1_Zero_Phase"
]


print(
    f"\nNumber of structure pairs:"
    f"\n  {len(pair_df)}"
)


print(
    "\nK_L1:"
)

print(
    f"  min    = "
    f"{zero_k.min():.9f}"
)

print(
    f"  Q25    = "
    f"{zero_k.quantile(0.25):.9f}"
)

print(
    f"  median = "
    f"{zero_k.median():.9f}"
)

print(
    f"  Q75    = "
    f"{zero_k.quantile(0.75):.9f}"
)

print(
    f"  max    = "
    f"{zero_k.max():.9f}"
)


print(
    "\nZero-phase max/min ratio:"
)

print(
    f"  "
    f"{zero_k.max() / zero_k.min():.6f}"
)


# ------------------------------------------------------------
# 23. Exact all-phase result
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "EXACT RANDOM-PHASE LOCAL COVERAGE EXPECTATION"
)

print(
    "=" * 90
)


print(
    "\nBecause every transformed structure has the same "
    "empirical marginal distribution:"
)


print(
    "\nExact phase-averaged K_L1:"
)

print(
    f"  "
    f"{EXACT_PHASE_AVERAGED_K_L1:.9f}"
)


print(
    "\nExact phase-averaged K_skip:"
)

print(
    f"  "
    f"{EXACT_PHASE_AVERAGED_K_SKIP:.9f}"
)


print(
    "\nExact phase-averaged K_overlap:"
)

print(
    f"  "
    f"{EXACT_PHASE_AVERAGED_K_OVERLAP:.9f}"
)


print(
    "\nThese values are common to ALL structure pairs."
)


print(
    "\nTherefore the long-run expected pointwise skip/overlap "
    "cannot be attributed to spatial correlation alone when "
    "marginal distribution and RMS are held fixed and relative "
    "phase is unrestricted."
)


# ------------------------------------------------------------
# 24. L2 all-pair phase sensitivity
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "ALL-PAIR RELATIVE-PHASE SENSITIVITY — K_L2"
)

print(
    "=" * 90
)


for column, label in [
    (
        "K_L2_Shift_SD",
        "SD across shifts"
    ),
    (
        "K_L2_Shift_Min",
        "minimum across shifts"
    ),
    (
        "K_L2_Shift_Max",
        "maximum across shifts"
    )
]:

    values = phase_l2_df[
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
# 25. Structural descriptor associations
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "ZERO-PHASE ASSOCIATIONS AFTER MARGINAL CONTROL"
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
# 26. Representative pairs
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "REPRESENTATIVE COMMON-MARGINAL ZERO-PHASE PAIRS"
)

print(
    "=" * 90
)


for label, row in representative_pairs:

    print(
        f"\n{label}:"
    )

    print(
        f"  {row['Trajectory_1']} "
        f"vs "
        f"{row['Trajectory_2']}"
    )

    print(
        f"  zero-phase K_L1 = "
        f"{row['K_L1_Zero_Phase']:.9f}"
    )

    print(
        f"  integral-scale difference = "
        f"{row['Absolute_Integral_Scale_Difference_m']:.6f} m"
    )


# ------------------------------------------------------------
# 27. Output paths
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
    f"\nCommon-marginal structure catalog:\n"
    f"{structure_file}"
)

print(
    f"\nZero-phase pairwise factors:\n"
    f"{pair_file}"
)

print(
    f"\nAll-pair phase L2 summary:\n"
    f"{phase_l2_file}"
)

print(
    f"\nRepresentative all-phase L1:\n"
    f"{phase_l1_file}"
)

print(
    f"\nAssociations:\n"
    f"{association_file}"
)

print(
    f"\nZero-phase K figure:\n"
    f"{zero_phase_figure}"
)

print(
    f"\nRepresentative phase-distribution figure:\n"
    f"{phase_distribution_figure}"
)

print(
    f"\nPhase-variability figure:\n"
    f"{phase_variability_figure}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 2C COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nThis stage separates three concepts:"
)

print(
    "  1. error magnitude (already controlled),"
)

print(
    "  2. marginal distribution (now controlled),"
)

print(
    "  3. spatial ordering / relative phase."
)

print(
    "\nThe next stage should use finite field windows and multiple "
    "passes to quantify the operational consequences of spatial "
    "persistence through variability, clustering, and boundary "
    "events rather than attributing long-run mean skip/overlap "
    "directly to correlation length."
)