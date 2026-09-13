from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 2B — DIMENSIONLESS EMPIRICAL STRUCTURE FACTOR
# ============================================================
#
# PURPOSE
# -------
# Quantify how much neighboring-pass coverage performance can
# differ at IDENTICAL centered RMS solely because the empirical
# centered error profiles differ.
#
# No implement width is assumed.
# No Monte Carlo sampling is introduced.
# No agronomic threshold is introduced.
#
#
# THEORY
# ------
#
# Let z_i(x) and z_j(x) be zero-mean empirical guidance-error
# structures normalized to unit RMS.
#
# At centered RMS R:
#
#       e_i(x) = R z_i(x)
#       e_j(x) = R z_j(x)
#
# For adjacent nominal passes separated by working width W:
#
#       delta(x) = e_j(x) - e_i(x)
#
# Local skip and overlap widths are:
#
#       skip(x)    = max(delta(x), 0)
#       overlap(x) = max(-delta(x), 0)
#
# Define:
#
#       eta = R / W
#
# and the dimensionless pairwise structure factors:
#
#       K_skip    = mean[max(z_j-z_i, 0)]
#       K_overlap = mean[max(z_i-z_j, 0)]
#       K_total   = mean[|z_j-z_i|]
#
# Therefore:
#
#       mean_skip / W    = eta * K_skip
#       mean_overlap / W = eta * K_overlap
#
# and:
#
#       (mean_skip + mean_overlap) / W
#           = eta * K_total
#
#
# Because the trajectories are centered, K_skip and K_overlap
# should be nearly identical over a common domain.
#
#
# IMPORTANT
# ---------
#
# This is an empirical structure experiment, not a claim that
# correlation length alone causes the result.
#
# The normalized structures preserve differences in:
#
#   - spatial dependence,
#   - smoothness,
#   - marginal shape,
#   - skewness,
#   - kurtosis,
#   - other empirically observed centered structure.
#
#
# INPUT
# -----
#
# bias_controlled_equal_rmse_trajectory_matrix.csv
#
# We select the median empirical centered-RMS level only as a
# convenient source of all 28 already-controlled structures.
#
# We then divide each trajectory by that common RMS, producing
# unit-RMS structures.
#
# Thus K is independent of the selected physical RMS magnitude.
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

STRUCTURE_CATALOG_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "bias_controlled_structure_catalog.csv"
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
    / "dimensionless_structure_factor"
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

    path_length = (
        x[-1]
        - x[0]
    )

    if path_length <= 0:

        raise RuntimeError(
            "Non-positive common path length."
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
        / path_length
    )


def common_grid(
    x1,
    x2,
    spacing
):

    start = max(
        np.min(x1),
        np.min(x2)
    )

    end = min(
        np.max(x1),
        np.max(x2)
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
            "Common grid has fewer than two points."
        )

    return grid


# ------------------------------------------------------------
# 3. Load inputs
# ------------------------------------------------------------

for file_path in [
    INPUT_FILE,
    STRUCTURE_CATALOG_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing input file:\n"
            f"{file_path}"
        )


print(
    "=" * 90
)

print(
    "STAGE 2B — DIMENSIONLESS EMPIRICAL STRUCTURE FACTOR"
)

print(
    "=" * 90
)


df = pd.read_csv(
    INPUT_FILE
)

catalog = pd.read_csv(
    STRUCTURE_CATALOG_FILE
)


print(
    f"\nLoaded trajectory matrix:"
    f"\n{INPUT_FILE}"
)

print(
    f"\nRows: {len(df):,}"
)


# ------------------------------------------------------------
# 4. Select one controlled RMS level
# ------------------------------------------------------------
#
# K is invariant to the common RMS scaling.
#
# We use the median available empirical centered-RMS level only
# to retrieve one complete set of 28 controlled structures.
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


median_index = (
    len(rms_levels)
    // 2
)


selected_level = rms_levels.iloc[
    median_index
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


print(
    f"\nControlled RMS level used to retrieve structures:"
)

print(
    f"  {SELECTED_RMS_ID}"
    f" = "
    f"{SELECTED_RMS_M:.9f} m"
)


# ------------------------------------------------------------
# 5. Build unit-RMS structure library
# ------------------------------------------------------------

structure_store = {}

normalization_rows = []


for trajectory_id, group in working_df.groupby(
    "Structure_Trajectory_ID",
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
        "Zero_Mean_Equal_RMS_Error_m"
    ].to_numpy(
        dtype=float
    )


    finite = (
        np.isfinite(x)
        & np.isfinite(error)
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


    observed_mean = np.mean(
        error
    )


    observed_rms = rms(
        error
    )


    if observed_rms <= 0:

        raise RuntimeError(
            f"Non-positive RMS for {trajectory_id}"
        )


    unit_structure = (
        error
        / observed_rms
    )


    unit_mean = np.mean(
        unit_structure
    )


    unit_rms = rms(
        unit_structure
    )


    structure_store[
        trajectory_id
    ] = {
        "X":
            x,

        "Z":
            unit_structure
    }


    normalization_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Source_Mean_m":
                observed_mean,

            "Source_RMS_m":
                observed_rms,

            "Unit_Structure_Mean":
                unit_mean,

            "Unit_Structure_RMS":
                unit_rms
        }
    )


normalization_df = pd.DataFrame(
    normalization_rows
)


trajectory_ids = sorted(
    structure_store.keys()
)


print(
    f"\nUnit-RMS empirical structures:"
    f"\n  {len(trajectory_ids)}"
)


# ------------------------------------------------------------
# 6. Pairwise structure factors
# ------------------------------------------------------------

pair_rows = []


pair_counter = 0


for trajectory_1, trajectory_2 in itertools.combinations(
    trajectory_ids,
    2
):

    pair_counter += 1


    record_1 = structure_store[
        trajectory_1
    ]

    record_2 = structure_store[
        trajectory_2
    ]


    grid = common_grid(
        record_1[
            "X"
        ],
        record_2[
            "X"
        ],
        DX_M
    )


    z1 = np.interp(
        grid,
        record_1[
            "X"
        ],
        record_1[
            "Z"
        ]
    )


    z2 = np.interp(
        grid,
        record_2[
            "X"
        ],
        record_2[
            "Z"
        ]
    )


    delta_z = (
        z2
        - z1
    )


    positive = np.maximum(
        delta_z,
        0.0
    )


    negative = np.maximum(
        -delta_z,
        0.0
    )


    absolute = np.abs(
        delta_z
    )


    k_skip = integrate_mean(
        positive,
        grid
    )


    k_overlap = integrate_mean(
        negative,
        grid
    )


    k_total = integrate_mean(
        absolute,
        grid
    )


    signed_mean_delta = integrate_mean(
        delta_z,
        grid
    )


    identity_error = abs(
        (
            k_skip
            + k_overlap
        )
        - k_total
    )


    pair_rows.append(
        {
            "Pair_ID":
                f"PAIR_{pair_counter:03d}",

            "Trajectory_1":
                trajectory_1,

            "Trajectory_2":
                trajectory_2,

            "Common_Length_m":
                (
                    grid[-1]
                    - grid[0]
                ),

            "K_Skip":
                k_skip,

            "K_Overlap":
                k_overlap,

            "K_Total":
                k_total,

            "Signed_Mean_Unit_Spacing_Error":
                signed_mean_delta,

            "Skip_Overlap_Difference":
                abs(
                    k_skip
                    - k_overlap
                ),

            "Identity_Error":
                identity_error
        }
    )


pair_df = pd.DataFrame(
    pair_rows
)


# ------------------------------------------------------------
# 7. Add structural descriptors for both members
# ------------------------------------------------------------

descriptor_columns = [
    "Trajectory_ID",
    "E_Folding_m",
    "Integral_Correlation_Scale_m",
    "Centered_Skewness",
    "Centered_Excess_Kurtosis"
]


descriptor_df = catalog[
    descriptor_columns
].copy()


descriptor_1 = descriptor_df.rename(
    columns={
        "Trajectory_ID":
            "Trajectory_1",

        "E_Folding_m":
            "Trajectory_1_E_Folding_m",

        "Integral_Correlation_Scale_m":
            "Trajectory_1_Integral_Scale_m",

        "Centered_Skewness":
            "Trajectory_1_Skewness",

        "Centered_Excess_Kurtosis":
            "Trajectory_1_Excess_Kurtosis"
    }
)


descriptor_2 = descriptor_df.rename(
    columns={
        "Trajectory_ID":
            "Trajectory_2",

        "E_Folding_m":
            "Trajectory_2_E_Folding_m",

        "Integral_Correlation_Scale_m":
            "Trajectory_2_Integral_Scale_m",

        "Centered_Skewness":
            "Trajectory_2_Skewness",

        "Centered_Excess_Kurtosis":
            "Trajectory_2_Excess_Kurtosis"
    }
)


pair_df = pair_df.merge(
    descriptor_1,
    on="Trajectory_1",
    how="left"
)


pair_df = pair_df.merge(
    descriptor_2,
    on="Trajectory_2",
    how="left"
)


pair_df[
    "Absolute_Integral_Scale_Difference_m"
] = np.abs(
    pair_df[
        "Trajectory_1_Integral_Scale_m"
    ]
    -
    pair_df[
        "Trajectory_2_Integral_Scale_m"
    ]
)


pair_df[
    "Absolute_E_Folding_Difference_m"
] = np.abs(
    pair_df[
        "Trajectory_1_E_Folding_m"
    ]
    -
    pair_df[
        "Trajectory_2_E_Folding_m"
    ]
)


pair_df[
    "Absolute_Skewness_Difference"
] = np.abs(
    pair_df[
        "Trajectory_1_Skewness"
    ]
    -
    pair_df[
        "Trajectory_2_Skewness"
    ]
)


pair_df[
    "Absolute_Excess_Kurtosis_Difference"
] = np.abs(
    pair_df[
        "Trajectory_1_Excess_Kurtosis"
    ]
    -
    pair_df[
        "Trajectory_2_Excess_Kurtosis"
    ]
)


# ------------------------------------------------------------
# 8. Exact eta-response representation
# ------------------------------------------------------------
#
# Since response is analytically linear in eta, we do NOT need
# to choose arbitrary eta values to establish the relationship.
#
# For every pair:
#
#       normalized total coverage error = K_Total * eta
#
# We store the slope K directly.
#
# ------------------------------------------------------------

response_df = pair_df[
    [
        "Pair_ID",
        "Trajectory_1",
        "Trajectory_2",
        "K_Skip",
        "K_Overlap",
        "K_Total"
    ]
].copy()


response_df[
    "Normalized_Skip_Response"
] = (
    "K_Skip * eta"
)


response_df[
    "Normalized_Overlap_Response"
] = (
    "K_Overlap * eta"
)


response_df[
    "Normalized_Total_Response"
] = (
    "K_Total * eta"
)


# ------------------------------------------------------------
# 9. Structure-level participation summary
# ------------------------------------------------------------

participation_rows = []


for trajectory_id in trajectory_ids:

    subset = pair_df[
        (
            pair_df[
                "Trajectory_1"
            ]
            == trajectory_id
        )
        |
        (
            pair_df[
                "Trajectory_2"
            ]
            == trajectory_id
        )
    ]


    catalog_row = catalog[
        catalog[
            "Trajectory_ID"
        ]
        == trajectory_id
    ].iloc[
        0
    ]


    participation_rows.append(
        {
            "Trajectory_ID":
                trajectory_id,

            "Number_of_Pairs":
                len(
                    subset
                ),

            "Mean_K_Total":
                subset[
                    "K_Total"
                ].mean(),

            "Median_K_Total":
                subset[
                    "K_Total"
                ].median(),

            "Minimum_K_Total":
                subset[
                    "K_Total"
                ].min(),

            "Maximum_K_Total":
                subset[
                    "K_Total"
                ].max(),

            "E_Folding_m":
                catalog_row[
                    "E_Folding_m"
                ],

            "Integral_Correlation_Scale_m":
                catalog_row[
                    "Integral_Correlation_Scale_m"
                ],

            "Centered_Skewness":
                catalog_row[
                    "Centered_Skewness"
                ],

            "Centered_Excess_Kurtosis":
                catalog_row[
                    "Centered_Excess_Kurtosis"
                ]
        }
    )


participation_df = pd.DataFrame(
    participation_rows
)


# ------------------------------------------------------------
# 10. Basic nonparametric associations
# ------------------------------------------------------------
#
# Spearman correlation is used descriptively.
# No causal interpretation is made.
#
# ------------------------------------------------------------

association_variables = [
    "Absolute_Integral_Scale_Difference_m",
    "Absolute_E_Folding_Difference_m",
    "Absolute_Skewness_Difference",
    "Absolute_Excess_Kurtosis_Difference"
]


association_rows = []


for variable in association_variables:

    rho = pair_df[
        [
            variable,
            "K_Total"
        ]
    ].corr(
        method="spearman"
    ).iloc[
        0,
        1
    ]


    association_rows.append(
        {
            "Structural_Descriptor_Difference":
                variable,

            "Spearman_Rho_with_K_Total":
                rho
        }
    )


association_df = pd.DataFrame(
    association_rows
)


# ------------------------------------------------------------
# 11. Numerical checks
# ------------------------------------------------------------

max_unit_mean = normalization_df[
    "Unit_Structure_Mean"
].abs().max()


max_unit_rms_error = (
    normalization_df[
        "Unit_Structure_RMS"
    ]
    - 1.0
).abs().max()


max_identity_error = pair_df[
    "Identity_Error"
].max()


max_skip_overlap_difference = pair_df[
    "Skip_Overlap_Difference"
].max()


# ------------------------------------------------------------
# 12. Save tables
# ------------------------------------------------------------

normalization_file = (
    TABLES_DIR
    / "unit_rms_empirical_structure_verification.csv"
)

pair_file = (
    TABLES_DIR
    / "dimensionless_pairwise_structure_factors.csv"
)

response_file = (
    TABLES_DIR
    / "dimensionless_exact_response_equations.csv"
)

participation_file = (
    TABLES_DIR
    / "dimensionless_structure_participation_summary.csv"
)

association_file = (
    TABLES_DIR
    / "dimensionless_structure_factor_associations.csv"
)


normalization_df.to_csv(
    normalization_file,
    index=False
)

pair_df.to_csv(
    pair_file,
    index=False
)

response_df.to_csv(
    response_file,
    index=False
)

participation_df.to_csv(
    participation_file,
    index=False
)

association_df.to_csv(
    association_file,
    index=False
)


# ------------------------------------------------------------
# 13. Figure — K distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    pair_df[
        "K_Total"
    ],
    bins="auto"
)


plt.xlabel(
    "Dimensionless structure factor K"
)

plt.ylabel(
    "Number of empirical structure pairs"
)

plt.title(
    "Distribution of equal-RMS empirical coverage structure factors"
)

plt.tight_layout()


distribution_figure = (
    FIGURES_DIR
    / "dimensionless_structure_factor_distribution.png"
)


plt.savefig(
    distribution_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 14. Figure — K versus integral-scale difference
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    pair_df[
        "Absolute_Integral_Scale_Difference_m"
    ],
    pair_df[
        "K_Total"
    ],
    alpha=0.7
)


plt.xlabel(
    "Absolute difference in integral correlation scale (m)"
)

plt.ylabel(
    "Dimensionless structure factor K"
)

plt.title(
    "Coverage structure factor versus correlation-scale difference"
)

plt.tight_layout()


integral_figure = (
    FIGURES_DIR
    / "structure_factor_vs_integral_scale_difference.png"
)


plt.savefig(
    integral_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 15. Figure — exact normalized response envelope
# ------------------------------------------------------------
#
# eta is shown here only on a normalized plotting coordinate
# from 0 to 1.
#
# It is NOT an imposed agricultural operating range.
#
# This plot illustrates the exact family:
#
#       response = K * eta
#
# ------------------------------------------------------------

eta_plot = np.linspace(
    0.0,
    1.0,
    201
)


k_min = pair_df[
    "K_Total"
].min()


k_median = pair_df[
    "K_Total"
].median()


k_max = pair_df[
    "K_Total"
].max()


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    eta_plot,
    k_min * eta_plot,
    label=(
        f"Minimum K = {k_min:.3f}"
    )
)


plt.plot(
    eta_plot,
    k_median * eta_plot,
    label=(
        f"Median K = {k_median:.3f}"
    )
)


plt.plot(
    eta_plot,
    k_max * eta_plot,
    label=(
        f"Maximum K = {k_max:.3f}"
    )
)


plt.xlabel(
    "Normalized error magnitude η = RMS / working width"
)

plt.ylabel(
    "Normalized total skip + overlap"
)

plt.title(
    "Exact dimensionless coverage-response envelope"
)

plt.legend()

plt.tight_layout()


response_figure = (
    FIGURES_DIR
    / "dimensionless_exact_response_envelope.png"
)


plt.savefig(
    response_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 16. Terminal formatting
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
# 17. Print normalization checks
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "UNIT-RMS NORMALIZATION CHECK"
)

print(
    "=" * 90
)


print(
    f"\nMaximum absolute unit-structure mean:"
    f"\n  {max_unit_mean:.12e}"
)


print(
    f"\nMaximum absolute unit-RMS error:"
    f"\n  {max_unit_rms_error:.12e}"
)


# ------------------------------------------------------------
# 18. Pairwise K distribution
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "PAIRWISE DIMENSIONLESS STRUCTURE FACTOR"
)

print(
    "=" * 90
)


print(
    f"\nNumber of unordered structure pairs:"
    f"\n  {len(pair_df)}"
)


for column, label in [
    (
        "K_Skip",
        "K_skip"
    ),
    (
        "K_Overlap",
        "K_overlap"
    ),
    (
        "K_Total",
        "K_total"
    )
]:

    values = pair_df[
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
        f"  Q25    = "
        f"{values.quantile(0.25):.9f}"
    )

    print(
        f"  median = "
        f"{values.median():.9f}"
    )

    print(
        f"  Q75    = "
        f"{values.quantile(0.75):.9f}"
    )

    print(
        f"  max    = "
        f"{values.max():.9f}"
    )


print(
    "\nK_total max / min ratio:"
)

print(
    f"  "
    f"{pair_df['K_Total'].max() / pair_df['K_Total'].min():.6f}"
)


# ------------------------------------------------------------
# 19. Geometry identities
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DIMENSIONLESS GEOMETRY CHECK"
)

print(
    "=" * 90
)


print(
    f"\nMaximum error in "
    f"K_skip + K_overlap = K_total:"
)

print(
    f"  {max_identity_error:.12e}"
)


print(
    f"\nMaximum |K_skip - K_overlap|:"
)

print(
    f"  {max_skip_overlap_difference:.12e}"
)


# ------------------------------------------------------------
# 20. Extreme pairs
# ------------------------------------------------------------

minimum_pair = pair_df.loc[
    pair_df[
        "K_Total"
    ].idxmin()
]


maximum_pair = pair_df.loc[
    pair_df[
        "K_Total"
    ].idxmax()
]


median_k = pair_df[
    "K_Total"
].median()


median_pair_index = (
    pair_df[
        "K_Total"
    ]
    - median_k
).abs().idxmin()


median_pair = pair_df.loc[
    median_pair_index
]


print(
    "\n"
    + "=" * 90
)

print(
    "DATA-DRIVEN REPRESENTATIVE PAIRS"
)

print(
    "=" * 90
)


for label, row in [
    (
        "Minimum-K pair",
        minimum_pair
    ),
    (
        "Median-K pair",
        median_pair
    ),
    (
        "Maximum-K pair",
        maximum_pair
    )
]:

    print(
        f"\n{label}:"
    )

    print(
        f"  "
        f"{row['Trajectory_1']} "
        f"vs "
        f"{row['Trajectory_2']}"
    )

    print(
        f"  K_total = "
        f"{row['K_Total']:.9f}"
    )

    print(
        f"  integral-scale difference = "
        f"{row['Absolute_Integral_Scale_Difference_m']:.6f} m"
    )

    print(
        f"  e-folding difference = "
        f"{row['Absolute_E_Folding_Difference_m']:.6f} m"
    )


# ------------------------------------------------------------
# 21. Associations
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DESCRIPTIVE SPEARMAN ASSOCIATIONS WITH K_TOTAL"
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
# 22. Structure participation
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "STRUCTURES RANKED BY MEAN PAIRWISE K_TOTAL"
)

print(
    "=" * 90
)


print(
    participation_df.sort_values(
        "Mean_K_Total",
        ascending=False
    )[
        [
            "Trajectory_ID",
            "Mean_K_Total",
            "Median_K_Total",
            "Minimum_K_Total",
            "Maximum_K_Total",
            "Integral_Correlation_Scale_m",
            "E_Folding_m"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 23. Exact analytical result
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "EXACT DIMENSIONLESS COVERAGE RELATIONSHIP"
)

print(
    "=" * 90
)


print(
    "\nFor every empirical pair:"
)

print(
    "  normalized mean skip"
    "    = K_skip * eta"
)

print(
    "  normalized mean overlap"
    " = K_overlap * eta"
)

print(
    "  normalized total coverage discrepancy"
    " = K_total * eta"
)

print(
    "\nwhere:"
)

print(
    "  eta = centered RMS / implement working width"
)


print(
    "\nTherefore equal eta does NOT imply equal coverage "
    "performance unless K is also equal."
)


# ------------------------------------------------------------
# 24. Output paths
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
    f"\nUnit-RMS verification:\n"
    f"{normalization_file}"
)

print(
    f"\nPairwise structure factors:\n"
    f"{pair_file}"
)

print(
    f"\nExact response equations:\n"
    f"{response_file}"
)

print(
    f"\nStructure participation summary:\n"
    f"{participation_file}"
)

print(
    f"\nDescriptor associations:\n"
    f"{association_file}"
)

print(
    f"\nK distribution figure:\n"
    f"{distribution_figure}"
)

print(
    f"\nK versus integral-scale figure:\n"
    f"{integral_figure}"
)

print(
    f"\nExact response-envelope figure:\n"
    f"{response_figure}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 2B COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nNo implement width, agronomic threshold, or Monte Carlo "
    "sample count was assumed."
)

print(
    "The empirical structure factor K now separates guidance-error "
    "magnitude from the operational effect of error-process structure."
)

print(
    "\nThe next stage can add finite-field geometry and multiple "
    "parallel passes, where boundary violations and accumulated "
    "coverage effects make Monte Carlo simulation necessary."
)