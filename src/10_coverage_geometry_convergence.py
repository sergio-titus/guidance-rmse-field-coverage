from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 2A — DIMENSIONLESS COVERAGE GEOMETRY AND
#            SPATIAL-DISCRETIZATION CONVERGENCE
# ============================================================
#
# PURPOSE
# -------
# Before Monte Carlo field simulations, verify that the
# along-track spatial resolution used to integrate local
# skip/overlap metrics is numerically adequate.
#
# ------------------------------------------------------------
# GEOMETRY
# ------------------------------------------------------------
#
# For two adjacent parallel passes:
#
#       y1(x) = e1(x)
#       y2(x) = W + e2(x)
#
# where W is implement working width.
#
# Their centerline spacing is:
#
#       d(x) = W + e2(x) - e1(x)
#
# Therefore:
#
#       skip(x)    = max(d(x) - W, 0)
#                  = max(e2(x) - e1(x), 0)
#
#       overlap(x) = max(W - d(x), 0)
#                  = max(e1(x) - e2(x), 0)
#
# Hence absolute local skip/overlap widths depend on the
# DIFFERENCE between neighboring guidance errors.
#
# Dimensionless fractions are:
#
#       skip / W
#       overlap / W
#
# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# No implement width is assumed here.
#
# Convergence is evaluated using error differences directly.
# Division by W is only a constant scaling and therefore cannot
# change the numerical convergence conclusion.
#
# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------
#
# Stage 1F bias-controlled trajectories:
#
# bias_controlled_equal_rmse_trajectory_matrix.csv
#
# We use ONE data-driven representative centered-RMS level:
# the median empirical centered-RMS level.
#
# All 28 empirical structures are retained.
#
# Every unordered pair of distinct structures is evaluated:
#
#       C(28,2) = 378 structure pairs
#
# This avoids selecting a favorable example.
#
# ------------------------------------------------------------
# RESOLUTION TEST
# ------------------------------------------------------------
#
# Native empirical grid:
#
#       0.20 m
#
# Candidate integration spacings:
#
#       0.025 m
#       0.050 m
#       0.100 m
#       0.200 m
#       0.400 m
#
# The 0.025 m result acts as the numerical reference for this
# convergence experiment.
#
# This is NOT a claim that the underlying observations possess
# 0.025 m information.
#
# Finer grids are created only by linear interpolation of the
# already established 0.20 m empirical spatial representation,
# to quantify numerical integration error.
#
# ------------------------------------------------------------
# OUTPUT METRICS
# ------------------------------------------------------------
#
# For each pair and spacing:
#
#   mean skip width
#   mean overlap width
#   total skip area per unit transverse scaling
#   total overlap area per unit transverse scaling
#
# Since:
#
#       max(delta,0) - max(-delta,0) = delta
#
# and centered trajectories have approximately zero mean,
# mean skip and overlap should also be similar in aggregate.
#
# No agronomic acceptability threshold is imposed.
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
    / "coverage_geometry_convergence"
)

for directory in [
    TABLES_DIR,
    FIGURES_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ------------------------------------------------------------
# 2. Candidate integration spacings
# ------------------------------------------------------------

SPACINGS_M = [
    0.025,
    0.050,
    0.100,
    0.200,
    0.400
]

REFERENCE_SPACING_M = min(
    SPACINGS_M
)


# ------------------------------------------------------------
# 3. Utility functions
# ------------------------------------------------------------

def integrate_trapezoid(
    values,
    x
):

    if hasattr(
        np,
        "trapezoid"
    ):

        return np.trapezoid(
            values,
            x
        )

    return np.trapz(
        values,
        x
    )


def build_common_grid(
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
            "Two trajectories have no common along-track domain."
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
            "Common grid contains fewer than two points."
        )


    return grid


def coverage_metrics(
    x,
    error_1,
    error_2
):

    delta = (
        error_2
        - error_1
    )


    skip_width = np.maximum(
        delta,
        0.0
    )


    overlap_width = np.maximum(
        -delta,
        0.0
    )


    path_length = (
        x[-1]
        - x[0]
    )


    if path_length <= 0:

        raise RuntimeError(
            "Non-positive path length."
        )


    skip_area = integrate_trapezoid(
        skip_width,
        x
    )


    overlap_area = integrate_trapezoid(
        overlap_width,
        x
    )


    mean_skip_width = (
        skip_area
        / path_length
    )


    mean_overlap_width = (
        overlap_area
        / path_length
    )


    mean_absolute_spacing_error = (
        integrate_trapezoid(
            np.abs(
                delta
            ),
            x
        )
        / path_length
    )


    signed_mean_spacing_error = (
        integrate_trapezoid(
            delta,
            x
        )
        / path_length
    )


    return {
        "Path_Length_m":
            path_length,

        "Mean_Skip_Width_m":
            mean_skip_width,

        "Mean_Overlap_Width_m":
            mean_overlap_width,

        "Integrated_Skip_Area_m2":
            skip_area,

        "Integrated_Overlap_Area_m2":
            overlap_area,

        "Mean_Absolute_Spacing_Error_m":
            mean_absolute_spacing_error,

        "Signed_Mean_Spacing_Error_m":
            signed_mean_spacing_error
    }


def symmetric_relative_error(
    estimate,
    reference
):

    denominator = (
        abs(estimate)
        + abs(reference)
    ) / 2.0


    if denominator == 0:

        return 0.0


    return (
        abs(
            estimate
            - reference
        )
        / denominator
    )


# ------------------------------------------------------------
# 4. Load controlled trajectories
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
    "STAGE 2A — COVERAGE GEOMETRY / SPATIAL CONVERGENCE"
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
# 5. Select median empirical centered-RMS level
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
    len(
        rms_levels
    )
    // 2
)


representative_rms_record = rms_levels.iloc[
    median_index
]


REPRESENTATIVE_RMS_ID = (
    representative_rms_record[
        "Target_Centered_RMS_ID"
    ]
)


REPRESENTATIVE_RMS_M = float(
    representative_rms_record[
        "Target_Centered_RMS_m"
    ]
)


representative_df = df[
    df[
        "Target_Centered_RMS_ID"
    ]
    == REPRESENTATIVE_RMS_ID
].copy()


trajectory_ids = sorted(
    representative_df[
        "Structure_Trajectory_ID"
    ].unique()
)


print(
    f"\nRepresentative empirical centered RMS:"
)

print(
    f"  {REPRESENTATIVE_RMS_ID}"
    f" = "
    f"{REPRESENTATIVE_RMS_M:.9f} m"
)


print(
    f"\nNumber of empirical structures:"
    f"\n  {len(trajectory_ids)}"
)


expected_pairs = (
    len(
        trajectory_ids
    )
    * (
        len(
            trajectory_ids
        )
        - 1
    )
    // 2
)


print(
    f"\nExpected unordered structure pairs:"
    f"\n  {expected_pairs}"
)


# ------------------------------------------------------------
# 6. Store trajectory data
# ------------------------------------------------------------

trajectory_store = {}


for trajectory_id in trajectory_ids:

    group = representative_df[
        representative_df[
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


    error = error[
        unique_indices
    ]

    x = unique_x


    if len(x) < 2:

        raise RuntimeError(
            f"Too few data points for "
            f"{trajectory_id}"
        )


    trajectory_store[
        trajectory_id
    ] = {
        "X":
            x,

        "Error":
            error
    }


# ------------------------------------------------------------
# 7. Generate all unordered trajectory pairs
# ------------------------------------------------------------

pair_list = []


for i in range(
    len(
        trajectory_ids
    )
):

    for j in range(
        i + 1,
        len(
            trajectory_ids
        )
    ):

        pair_list.append(
            (
                trajectory_ids[i],
                trajectory_ids[j]
            )
        )


if len(
    pair_list
) != expected_pairs:

    raise RuntimeError(
        "Unexpected number of trajectory pairs."
    )


# ------------------------------------------------------------
# 8. Compute coverage metrics at every spacing
# ------------------------------------------------------------

result_rows = []


for pair_index, (
    trajectory_1,
    trajectory_2
) in enumerate(
    pair_list,
    start=1
):

    record_1 = trajectory_store[
        trajectory_1
    ]

    record_2 = trajectory_store[
        trajectory_2
    ]


    x1 = record_1[
        "X"
    ]

    e1 = record_1[
        "Error"
    ]


    x2 = record_2[
        "X"
    ]

    e2 = record_2[
        "Error"
    ]


    for spacing in SPACINGS_M:

        grid = build_common_grid(
            x1,
            x2,
            spacing
        )


        interp_1 = np.interp(
            grid,
            x1,
            e1
        )


        interp_2 = np.interp(
            grid,
            x2,
            e2
        )


        metrics = coverage_metrics(
            grid,
            interp_1,
            interp_2
        )


        result_rows.append(
            {
                "Pair_ID":
                    f"PAIR_{pair_index:03d}",

                "Trajectory_1":
                    trajectory_1,

                "Trajectory_2":
                    trajectory_2,

                "Target_Centered_RMS_ID":
                    REPRESENTATIVE_RMS_ID,

                "Target_Centered_RMS_m":
                    REPRESENTATIVE_RMS_M,

                "Integration_Spacing_m":
                    spacing,

                **metrics
            }
        )


results_df = pd.DataFrame(
    result_rows
)


# ------------------------------------------------------------
# 9. Reference metrics
# ------------------------------------------------------------

reference_df = (
    results_df[
        results_df[
            "Integration_Spacing_m"
        ]
        == REFERENCE_SPACING_M
    ]
    [
        [
            "Pair_ID",
            "Mean_Skip_Width_m",
            "Mean_Overlap_Width_m",
            "Integrated_Skip_Area_m2",
            "Integrated_Overlap_Area_m2",
            "Mean_Absolute_Spacing_Error_m"
        ]
    ]
    .copy()
)


reference_df = reference_df.rename(
    columns={
        "Mean_Skip_Width_m":
            "Reference_Mean_Skip_Width_m",

        "Mean_Overlap_Width_m":
            "Reference_Mean_Overlap_Width_m",

        "Integrated_Skip_Area_m2":
            "Reference_Integrated_Skip_Area_m2",

        "Integrated_Overlap_Area_m2":
            "Reference_Integrated_Overlap_Area_m2",

        "Mean_Absolute_Spacing_Error_m":
            "Reference_Mean_Absolute_Spacing_Error_m"
    }
)


comparison_df = results_df.merge(
    reference_df,
    on="Pair_ID",
    how="left"
)


# ------------------------------------------------------------
# 10. Absolute and relative numerical errors
# ------------------------------------------------------------

comparison_df[
    "Abs_Error_Mean_Skip_Width_m"
] = np.abs(
    comparison_df[
        "Mean_Skip_Width_m"
    ]
    -
    comparison_df[
        "Reference_Mean_Skip_Width_m"
    ]
)


comparison_df[
    "Abs_Error_Mean_Overlap_Width_m"
] = np.abs(
    comparison_df[
        "Mean_Overlap_Width_m"
    ]
    -
    comparison_df[
        "Reference_Mean_Overlap_Width_m"
    ]
)


comparison_df[
    "Abs_Error_Mean_Absolute_Spacing_Error_m"
] = np.abs(
    comparison_df[
        "Mean_Absolute_Spacing_Error_m"
    ]
    -
    comparison_df[
        "Reference_Mean_Absolute_Spacing_Error_m"
    ]
)


comparison_df[
    "Symmetric_Rel_Error_Mean_Skip"
] = comparison_df.apply(
    lambda row:
        symmetric_relative_error(
            row[
                "Mean_Skip_Width_m"
            ],
            row[
                "Reference_Mean_Skip_Width_m"
            ]
        ),
    axis=1
)


comparison_df[
    "Symmetric_Rel_Error_Mean_Overlap"
] = comparison_df.apply(
    lambda row:
        symmetric_relative_error(
            row[
                "Mean_Overlap_Width_m"
            ],
            row[
                "Reference_Mean_Overlap_Width_m"
            ]
        ),
    axis=1
)


comparison_df[
    "Symmetric_Rel_Error_Mean_Absolute_Spacing"
] = comparison_df.apply(
    lambda row:
        symmetric_relative_error(
            row[
                "Mean_Absolute_Spacing_Error_m"
            ],
            row[
                "Reference_Mean_Absolute_Spacing_Error_m"
            ]
        ),
    axis=1
)


# ------------------------------------------------------------
# 11. Convergence summary by spacing
# ------------------------------------------------------------

summary_rows = []


for spacing, group in comparison_df.groupby(
    "Integration_Spacing_m",
    sort=True
):

    summary_rows.append(
        {
            "Integration_Spacing_m":
                spacing,

            "Number_of_Pairs":
                len(
                    group
                ),

            "Median_Absolute_Error_Mean_Skip_m":
                group[
                    "Abs_Error_Mean_Skip_Width_m"
                ].median(),

            "Maximum_Absolute_Error_Mean_Skip_m":
                group[
                    "Abs_Error_Mean_Skip_Width_m"
                ].max(),

            "Median_Symmetric_Rel_Error_Mean_Skip":
                group[
                    "Symmetric_Rel_Error_Mean_Skip"
                ].median(),

            "Maximum_Symmetric_Rel_Error_Mean_Skip":
                group[
                    "Symmetric_Rel_Error_Mean_Skip"
                ].max(),

            "Median_Absolute_Error_Mean_Overlap_m":
                group[
                    "Abs_Error_Mean_Overlap_Width_m"
                ].median(),

            "Maximum_Absolute_Error_Mean_Overlap_m":
                group[
                    "Abs_Error_Mean_Overlap_Width_m"
                ].max(),

            "Median_Symmetric_Rel_Error_Mean_Overlap":
                group[
                    "Symmetric_Rel_Error_Mean_Overlap"
                ].median(),

            "Maximum_Symmetric_Rel_Error_Mean_Overlap":
                group[
                    "Symmetric_Rel_Error_Mean_Overlap"
                ].max(),

            "Median_Absolute_Error_Mean_Absolute_Spacing_m":
                group[
                    "Abs_Error_Mean_Absolute_Spacing_Error_m"
                ].median(),

            "Maximum_Absolute_Error_Mean_Absolute_Spacing_m":
                group[
                    "Abs_Error_Mean_Absolute_Spacing_Error_m"
                ].max(),

            "Median_Symmetric_Rel_Error_Mean_Absolute_Spacing":
                group[
                    "Symmetric_Rel_Error_Mean_Absolute_Spacing"
                ].median(),

            "Maximum_Symmetric_Rel_Error_Mean_Absolute_Spacing":
                group[
                    "Symmetric_Rel_Error_Mean_Absolute_Spacing"
                ].max()
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ------------------------------------------------------------
# 12. Algebraic identity verification
# ------------------------------------------------------------
#
# For every x:
#
#   skip + overlap = |e2 - e1|
#
# Therefore integrated/mean quantities must obey:
#
#   mean_skip + mean_overlap
#   = mean_absolute_spacing_error
#
# This is a useful independent implementation check.
#
# ------------------------------------------------------------

comparison_df[
    "Geometry_Identity_Error_m"
] = np.abs(
    (
        comparison_df[
            "Mean_Skip_Width_m"
        ]
        +
        comparison_df[
            "Mean_Overlap_Width_m"
        ]
    )
    -
    comparison_df[
        "Mean_Absolute_Spacing_Error_m"
    ]
)


max_geometry_identity_error = comparison_df[
    "Geometry_Identity_Error_m"
].max()


# ------------------------------------------------------------
# 13. Save tables
# ------------------------------------------------------------

detailed_file = (
    TABLES_DIR
    / "coverage_geometry_resolution_detailed.csv"
)

summary_file = (
    TABLES_DIR
    / "coverage_geometry_resolution_convergence.csv"
)


comparison_df.to_csv(
    detailed_file,
    index=False
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# 14. Figure — relative convergence
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    summary_df[
        "Integration_Spacing_m"
    ],
    100
    * summary_df[
        "Median_Symmetric_Rel_Error_Mean_Absolute_Spacing"
    ],
    marker="o",
    label="Median error"
)


plt.plot(
    summary_df[
        "Integration_Spacing_m"
    ],
    100
    * summary_df[
        "Maximum_Symmetric_Rel_Error_Mean_Absolute_Spacing"
    ],
    marker="o",
    label="Maximum error"
)


plt.xlabel(
    "Along-track integration spacing (m)"
)

plt.ylabel(
    "Symmetric relative numerical error (%)"
)

plt.title(
    "Numerical convergence of integrated spacing-error metric"
)

plt.legend()

plt.tight_layout()


relative_figure_file = (
    FIGURES_DIR
    / "coverage_resolution_relative_convergence.png"
)


plt.savefig(
    relative_figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 15. Figure — absolute convergence
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    summary_df[
        "Integration_Spacing_m"
    ],
    summary_df[
        "Median_Absolute_Error_Mean_Absolute_Spacing_m"
    ],
    marker="o",
    label="Median error"
)


plt.plot(
    summary_df[
        "Integration_Spacing_m"
    ],
    summary_df[
        "Maximum_Absolute_Error_Mean_Absolute_Spacing_m"
    ],
    marker="o",
    label="Maximum error"
)


plt.xlabel(
    "Along-track integration spacing (m)"
)

plt.ylabel(
    "Absolute numerical error (m)"
)

plt.title(
    "Absolute numerical convergence of spacing-error integration"
)

plt.legend()

plt.tight_layout()


absolute_figure_file = (
    FIGURES_DIR
    / "coverage_resolution_absolute_convergence.png"
)


plt.savefig(
    absolute_figure_file,
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
# 17. Print convergence summary
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "SPATIAL-INTEGRATION CONVERGENCE SUMMARY"
)

print(
    "=" * 90
)


display_columns = [
    "Integration_Spacing_m",
    "Number_of_Pairs",

    "Median_Absolute_Error_Mean_Skip_m",
    "Maximum_Absolute_Error_Mean_Skip_m",

    "Median_Symmetric_Rel_Error_Mean_Skip",
    "Maximum_Symmetric_Rel_Error_Mean_Skip",

    "Median_Absolute_Error_Mean_Overlap_m",
    "Maximum_Absolute_Error_Mean_Overlap_m",

    "Median_Symmetric_Rel_Error_Mean_Overlap",
    "Maximum_Symmetric_Rel_Error_Mean_Overlap",

    "Median_Absolute_Error_Mean_Absolute_Spacing_m",
    "Maximum_Absolute_Error_Mean_Absolute_Spacing_m",

    "Median_Symmetric_Rel_Error_Mean_Absolute_Spacing",
    "Maximum_Symmetric_Rel_Error_Mean_Absolute_Spacing"
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 18. Native-grid result
# ------------------------------------------------------------

native_row = summary_df[
    np.isclose(
        summary_df[
            "Integration_Spacing_m"
        ],
        0.200
    )
]


print(
    "\n"
    + "=" * 90
)

print(
    "NATIVE 0.20 m GRID COMPARED WITH 0.025 m NUMERICAL REFERENCE"
)

print(
    "=" * 90
)


if len(
    native_row
) == 1:

    native_row = native_row.iloc[
        0
    ]


    print(
        "\nMean absolute spacing-error metric:"
    )

    print(
        f"  median absolute numerical error = "
        f"{native_row['Median_Absolute_Error_Mean_Absolute_Spacing_m']:.12e} m"
    )

    print(
        f"  maximum absolute numerical error = "
        f"{native_row['Maximum_Absolute_Error_Mean_Absolute_Spacing_m']:.12e} m"
    )

    print(
        f"  median symmetric relative error = "
        f"{100 * native_row['Median_Symmetric_Rel_Error_Mean_Absolute_Spacing']:.6f} %"
    )

    print(
        f"  maximum symmetric relative error = "
        f"{100 * native_row['Maximum_Symmetric_Rel_Error_Mean_Absolute_Spacing']:.6f} %"
    )


# ------------------------------------------------------------
# 19. Implementation identity
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "GEOMETRY IMPLEMENTATION CHECK"
)

print(
    "=" * 90
)


print(
    "\nMaximum error in identity:"
)

print(
    "  mean_skip + mean_overlap "
    "= mean_absolute_spacing_error"
)

print(
    f"\n  {max_geometry_identity_error:.12e} m"
)


# ------------------------------------------------------------
# 20. Metric distribution at reference resolution
# ------------------------------------------------------------

reference_results = comparison_df[
    np.isclose(
        comparison_df[
            "Integration_Spacing_m"
        ],
        REFERENCE_SPACING_M
    )
]


print(
    "\n"
    + "=" * 90
)

print(
    "PAIRWISE COVERAGE-GEOMETRY RANGE AT REPRESENTATIVE RMS"
)

print(
    "=" * 90
)


for column, label in [
    (
        "Mean_Skip_Width_m",
        "Mean skip width"
    ),
    (
        "Mean_Overlap_Width_m",
        "Mean overlap width"
    ),
    (
        "Mean_Absolute_Spacing_Error_m",
        "Mean absolute spacing error"
    )
]:

    values = reference_results[
        column
    ]

    print(
        f"\n{label}:"
    )

    print(
        f"  min    = "
        f"{values.min():.9f} m"
    )

    print(
        f"  median = "
        f"{values.median():.9f} m"
    )

    print(
        f"  max    = "
        f"{values.max():.9f} m"
    )


# ------------------------------------------------------------
# 21. Outputs
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
    f"\nDetailed resolution results:\n"
    f"{detailed_file}"
)

print(
    f"\nConvergence summary:\n"
    f"{summary_file}"
)

print(
    f"\nRelative-convergence figure:\n"
    f"{relative_figure_file}"
)

print(
    f"\nAbsolute-convergence figure:\n"
    f"{absolute_figure_file}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 2A COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nNo implement width has been assumed."
)

print(
    "Coverage geometry is expressed through neighboring-pass "
    "spacing error and can later be normalized by any working width W."
)

print(
    "\nThe next stage will use the numerically validated spacing "
    "to build dimensionless skip/overlap response surfaces as "
    "functions of error magnitude relative to working width."
)