from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from shapely import contains_xy

except ImportError as exc:

    raise ImportError(
        "\nShapely >= 2.0 is required for Stage 4B.\n"
        "Install it with:\n\n"
        "    pip install shapely\n"
    ) from exc


# ============================================================
# STAGE 4B — DIRECT GEOMETRIC COVERAGE VALIDATION
# ============================================================
#
# PURPOSE
# -------
# Directly validate the analytical mapping from lateral guidance
# error to physical coverage-area discrepancy.
#
# This is the geometric closure test for the final operational
# tolerance framework.
#
#
# IMPORTANT MODEL DEFINITION
# --------------------------
#
# The analytical framework assumes a purely cross-track
# implement footprint:
#
#     [centerline(x) - W/2, centerline(x) + W/2]
#
# at every longitudinal position x.
#
# We therefore construct the footprint as an explicit polygon
# bounded by these curves.
#
# We DO NOT use Shapely.buffer(centerline), because a Euclidean
# line buffer follows the local normal direction and would
# introduce heading-dependent footprint geometry that is not
# part of the present model.
#
#
# TWO-PASS FIELD
# --------------
#
# Normalize implement width:
#
#     W = 1
#
# This is a dimensionless coordinate normalization, not a claim
# that the physical implement has a 1-m working width.
#
# Nominal centerlines:
#
#     pass 1: W/2
#     pass 2: 3W/2
#
# Actual centerlines:
#
#     y1(x) = W/2   + e1(x)
#     y2(x) = 3W/2  + e2(x)
#
# Ideal target field:
#
#     0 <= y <= 2W
#
#
# ANALYTICAL AREA COMPONENTS
# --------------------------
#
# Inside-field skip:
#
#     left inward gap
#         = max(e1, 0)
#
#     internal gap
#         = max(e2 - e1, 0)
#
#     right inward gap
#         = max(-e2, 0)
#
#
# Inside-field excess application / overlap:
#
#     = max(e1 - e2, 0)
#
#
# Outside-field application:
#
#     left:
#         max(-e1, 0)
#
#     right:
#         max(e2, 0)
#
#
# Total coverage-count discrepancy:
#
#     skip + excess overlap + outside application
#
#
# SHAPELY REFERENCE
# -----------------
#
# For actual footprint polygons:
#
#     unique_inside
#         = area(union of footprints clipped to target field)
#
#     skip_inside
#         = target_area - unique_inside
#
#     applied_inside
#         = sum(area(each footprint clipped to field))
#
#     excess_overlap_inside
#         = applied_inside - unique_inside
#
#     outside_application
#         = sum(total footprint areas)
#           - applied_inside
#
#
# Therefore:
#
#     total L1 coverage-count discrepancy
#
#         = skip_inside
#           + excess_overlap_inside
#           + outside_application
#
#
# RASTER VALIDATION
# -----------------
#
# After validating ALL 378 trajectory pairs using exact Shapely
# polygon areas, three automatically selected representative
# cases are rasterized:
#
#     minimum discrepancy
#     median discrepancy
#     maximum discrepancy
#
# Raster resolution is refined dyadically from the native
# 0.20-m longitudinal spacing:
#
#     0.20, 0.10, 0.05, 0.025
#
# No one raster resolution is declared "correct"; convergence
# toward the exact polygon solution is reported.
#
#
# MODEL VALIDITY CONDITION
# ------------------------
#
# The neighboring-pass derivation assumes pass ordering:
#
#     W + e2(x) - e1(x) > 0
#
# everywhere.
#
# This condition is checked explicitly for every pair.
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
    / "geometric_coverage_validation"
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
# 2. Numerical definitions
# ------------------------------------------------------------

DX_NATIVE_M = 0.20

W = 1.0

RASTER_RESOLUTIONS = [
    0.20,
    0.10,
    0.05,
    0.025
]


# ------------------------------------------------------------
# 3. Utility functions
# ------------------------------------------------------------

def exact_positive_part_integral(
    x,
    values
):

    """
    Exact integral of max(v(x), 0) when v is piecewise linear
    between the supplied x samples.

    Zero crossings are solved analytically inside each segment.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    if len(x) != len(values):

        raise ValueError(
            "x and values must have equal length."
        )

    if len(x) < 2:

        return 0.0


    total = 0.0


    for i in range(
        len(x) - 1
    ):

        x0 = x[i]

        x1 = x[i + 1]

        v0 = values[i]

        v1 = values[i + 1]

        dx = (
            x1
            - x0
        )


        if dx <= 0:

            raise ValueError(
                "x must be strictly increasing."
            )


        # Both non-positive.
        if (
            v0 <= 0
            and
            v1 <= 0
        ):

            continue


        # Both non-negative.
        if (
            v0 >= 0
            and
            v1 >= 0
        ):

            total += (
                0.5
                * (
                    v0
                    + v1
                )
                * dx
            )

            continue


        # Linear zero crossing.
        fraction = (
            -v0
            /
            (
                v1
                - v0
            )
        )


        x_cross = (
            x0
            + fraction
            * dx
        )


        if (
            v0 > 0
            and
            v1 < 0
        ):

            total += (
                0.5
                * v0
                * (
                    x_cross
                    - x0
                )
            )


        elif (
            v0 < 0
            and
            v1 > 0
        ):

            total += (
                0.5
                * v1
                * (
                    x1
                    - x_cross
                )
            )


    return total


def build_cross_track_footprint(
    x,
    center_y,
    width
):

    """
    Build the exact polygon corresponding to the cross-track
    strip assumed by the analytical model.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    center_y = np.asarray(
        center_y,
        dtype=float
    )


    lower = (
        center_y
        - width / 2.0
    )

    upper = (
        center_y
        + width / 2.0
    )


    lower_coordinates = list(
        zip(
            x,
            lower
        )
    )


    upper_coordinates = list(
        zip(
            x[::-1],
            upper[::-1]
        )
    )


    polygon = Polygon(
        lower_coordinates
        + upper_coordinates
    )


    if not polygon.is_valid:

        polygon = polygon.buffer(
            0
        )


    if not polygon.is_valid:

        raise RuntimeError(
            "Could not construct a valid footprint polygon."
        )


    return polygon


def common_grid(
    trajectory_store,
    spacing
):

    start = max(
        np.min(
            record[
                "X"
            ]
        )
        for record
        in trajectory_store.values()
    )


    end = min(
        np.max(
            record[
                "X"
            ]
        )
        for record
        in trajectory_store.values()
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


    return grid


def analytical_two_pass_metrics(
    x,
    e1,
    e2,
    width
):

    field_length = (
        x[-1]
        - x[0]
    )

    target_area = (
        2.0
        * width
        * field_length
    )


    left_inward = exact_positive_part_integral(
        x,
        e1
    )


    left_outward = exact_positive_part_integral(
        x,
        -e1
    )


    internal_gap = exact_positive_part_integral(
        x,
        e2 - e1
    )


    internal_overlap = exact_positive_part_integral(
        x,
        e1 - e2
    )


    right_inward = exact_positive_part_integral(
        x,
        -e2
    )


    right_outward = exact_positive_part_integral(
        x,
        e2
    )


    skip_area = (
        left_inward
        + internal_gap
        + right_inward
    )


    overlap_area = (
        internal_overlap
    )


    outside_area = (
        left_outward
        + right_outward
    )


    total_discrepancy_area = (
        skip_area
        + overlap_area
        + outside_area
    )


    return {
        "Analytic_Skip_Area":
            skip_area,

        "Analytic_Overlap_Area":
            overlap_area,

        "Analytic_Outside_Area":
            outside_area,

        "Analytic_Total_Discrepancy_Area":
            total_discrepancy_area,

        "Analytic_Skip_Fraction":
            skip_area
            / target_area,

        "Analytic_Overlap_Fraction":
            overlap_area
            / target_area,

        "Analytic_Outside_Fraction":
            outside_area
            / target_area,

        "Analytic_Total_Discrepancy_Fraction":
            total_discrepancy_area
            / target_area
    }


def shapely_two_pass_metrics(
    x,
    e1,
    e2,
    width
):

    field_length = (
        x[-1]
        - x[0]
    )


    target_field = box(
        x[0],
        0.0,
        x[-1],
        2.0 * width
    )


    target_area = (
        target_field.area
    )


    center_1 = (
        width / 2.0
        + e1
    )


    center_2 = (
        3.0
        * width / 2.0
        + e2
    )


    footprint_1 = build_cross_track_footprint(
        x,
        center_1,
        width
    )


    footprint_2 = build_cross_track_footprint(
        x,
        center_2,
        width
    )


    inside_1 = footprint_1.intersection(
        target_field
    )


    inside_2 = footprint_2.intersection(
        target_field
    )


    union_inside = unary_union(
        [
            inside_1,
            inside_2
        ]
    )


    unique_inside_area = (
        union_inside.area
    )


    applied_inside_area = (
        inside_1.area
        + inside_2.area
    )


    total_applied_area = (
        footprint_1.area
        + footprint_2.area
    )


    skip_area = (
        target_area
        - unique_inside_area
    )


    overlap_area = (
        applied_inside_area
        - unique_inside_area
    )


    outside_area = (
        total_applied_area
        - applied_inside_area
    )


    total_discrepancy_area = (
        skip_area
        + overlap_area
        + outside_area
    )


    return {
        "Target_Area":
            target_area,

        "Footprint_1_Area":
            footprint_1.area,

        "Footprint_2_Area":
            footprint_2.area,

        "Shapely_Skip_Area":
            skip_area,

        "Shapely_Overlap_Area":
            overlap_area,

        "Shapely_Outside_Area":
            outside_area,

        "Shapely_Total_Discrepancy_Area":
            total_discrepancy_area,

        "Shapely_Skip_Fraction":
            skip_area
            / target_area,

        "Shapely_Overlap_Fraction":
            overlap_area
            / target_area,

        "Shapely_Outside_Fraction":
            outside_area
            / target_area,

        "Shapely_Total_Discrepancy_Fraction":
            total_discrepancy_area
            / target_area,

        "Footprint_1":
            footprint_1,

        "Footprint_2":
            footprint_2,

        "Target_Field":
            target_field
    }


def raster_coverage_metrics(
    footprint_1,
    footprint_2,
    target_field,
    resolution
):

    min_x = min(
        footprint_1.bounds[0],
        footprint_2.bounds[0],
        target_field.bounds[0]
    )


    min_y = min(
        footprint_1.bounds[1],
        footprint_2.bounds[1],
        target_field.bounds[1]
    )


    max_x = max(
        footprint_1.bounds[2],
        footprint_2.bounds[2],
        target_field.bounds[2]
    )


    max_y = max(
        footprint_1.bounds[3],
        footprint_2.bounds[3],
        target_field.bounds[3]
    )


    x_centers = np.arange(
        min_x + resolution / 2.0,
        max_x,
        resolution
    )


    y_centers = np.arange(
        min_y + resolution / 2.0,
        max_y,
        resolution
    )


    cell_area = (
        resolution ** 2
    )


    skip_cells = 0

    overlap_excess_cells = 0

    outside_application_cells = 0


    # Chunk by y row to avoid building a very large 2-D mesh.
    for y_value in y_centers:

        y_array = np.full(
            len(x_centers),
            y_value,
            dtype=float
        )


        in_1 = contains_xy(
            footprint_1,
            x_centers,
            y_array
        )


        in_2 = contains_xy(
            footprint_2,
            x_centers,
            y_array
        )


        in_target = contains_xy(
            target_field,
            x_centers,
            y_array
        )


        coverage_count = (
            in_1.astype(int)
            + in_2.astype(int)
        )


        skip_cells += np.sum(
            in_target
            &
            (
                coverage_count == 0
            )
        )


        overlap_excess_cells += np.sum(
            np.where(
                in_target,
                np.maximum(
                    coverage_count - 1,
                    0
                ),
                0
            )
        )


        outside_application_cells += np.sum(
            np.where(
                ~in_target,
                coverage_count,
                0
            )
        )


    skip_area = (
        skip_cells
        * cell_area
    )


    overlap_area = (
        overlap_excess_cells
        * cell_area
    )


    outside_area = (
        outside_application_cells
        * cell_area
    )


    total_area = (
        skip_area
        + overlap_area
        + outside_area
    )


    target_area = (
        target_field.area
    )


    return {
        "Raster_Resolution":
            resolution,

        "Raster_Skip_Area":
            skip_area,

        "Raster_Overlap_Area":
            overlap_area,

        "Raster_Outside_Area":
            outside_area,

        "Raster_Total_Discrepancy_Area":
            total_area,

        "Raster_Skip_Fraction":
            skip_area
            / target_area,

        "Raster_Overlap_Fraction":
            overlap_area
            / target_area,

        "Raster_Outside_Fraction":
            outside_area
            / target_area,

        "Raster_Total_Discrepancy_Fraction":
            total_area
            / target_area
    }


# ------------------------------------------------------------
# 4. Load controlled trajectory matrix
# ------------------------------------------------------------

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing input file:\n{INPUT_FILE}"
    )


print(
    "=" * 90
)

print(
    "STAGE 4B — DIRECT GEOMETRIC COVERAGE VALIDATION"
)

print(
    "=" * 90
)


df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nLoaded:\n{INPUT_FILE}"
)

print(
    f"\nRows:"
    f"\n  {len(df):,}"
)


# ------------------------------------------------------------
# 5. Use the same representative centered-RMS level
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
    "\nValidation level:"
)

print(
    f"  {SELECTED_RMS_ID}"
    f" = "
    f"{SELECTED_RMS_M:.9f}"
)


print(
    "\nNormalized working width:"
)

print(
    f"  W = {W:.6f}"
)

print(
    "  This is a scale normalization, not a physical implement assumption."
)


print(
    "\nDimensionless validation magnitude:"
)

print(
    f"  eta = R/W = "
    f"{SELECTED_RMS_M / W:.9f}"
)


print(
    f"\nStructures:"
    f"\n  {len(trajectory_ids)}"
)


# ------------------------------------------------------------
# 6. Retrieve each empirical structure
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
# 7. Common native-resolution support
# ------------------------------------------------------------

grid = common_grid(
    raw_store,
    DX_NATIVE_M
)


FIELD_LENGTH_M = (
    grid[-1]
    - grid[0]
)


print(
    "\nCommon spatial support:"
)

print(
    f"  start  = "
    f"{grid[0]:.3f}"
)

print(
    f"  end    = "
    f"{grid[-1]:.3f}"
)

print(
    f"  length = "
    f"{FIELD_LENGTH_M:.3f}"
)

print(
    f"  points = "
    f"{len(grid):,}"
)


# ------------------------------------------------------------
# 8. Interpolate structures onto common grid
# ------------------------------------------------------------

structure_store = {}


for trajectory_id in trajectory_ids:

    record = raw_store[
        trajectory_id
    ]


    error = np.interp(
        grid,
        record[
            "X"
        ],
        record[
            "Error"
        ]
    )


    structure_store[
        trajectory_id
    ] = error


# ------------------------------------------------------------
# 9. Exact polygon validation for ALL 378 pairs
# ------------------------------------------------------------

pair_rows = []


pairs = list(
    itertools.combinations(
        trajectory_ids,
        2
    )
)


print(
    f"\nExact Shapely pair validations:"
    f"\n  {len(pairs):,}"
)


for pair_number, (
    trajectory_1,
    trajectory_2
) in enumerate(
    pairs,
    start=1
):

    e1 = structure_store[
        trajectory_1
    ]


    e2 = structure_store[
        trajectory_2
    ]


    spacing = (
        W
        + e2
        - e1
    )


    minimum_spacing = np.min(
        spacing
    )


    ordering_valid = bool(
        minimum_spacing > 0.0
    )


    analytic = analytical_two_pass_metrics(
        grid,
        e1,
        e2,
        W
    )


    geometric = shapely_two_pass_metrics(
        grid,
        e1,
        e2,
        W
    )


    pair_rows.append(
        {
            "Pair_ID":
                f"PAIR_{pair_number:03d}",

            "Trajectory_1":
                trajectory_1,

            "Trajectory_2":
                trajectory_2,

            "Minimum_Centerline_Spacing":
                minimum_spacing,

            "Pass_Ordering_Valid":
                ordering_valid,

            "Analytic_Skip_Fraction":
                analytic[
                    "Analytic_Skip_Fraction"
                ],

            "Shapely_Skip_Fraction":
                geometric[
                    "Shapely_Skip_Fraction"
                ],

            "Absolute_Skip_Error":
                abs(
                    analytic[
                        "Analytic_Skip_Fraction"
                    ]
                    -
                    geometric[
                        "Shapely_Skip_Fraction"
                    ]
                ),

            "Analytic_Overlap_Fraction":
                analytic[
                    "Analytic_Overlap_Fraction"
                ],

            "Shapely_Overlap_Fraction":
                geometric[
                    "Shapely_Overlap_Fraction"
                ],

            "Absolute_Overlap_Error":
                abs(
                    analytic[
                        "Analytic_Overlap_Fraction"
                    ]
                    -
                    geometric[
                        "Shapely_Overlap_Fraction"
                    ]
                ),

            "Analytic_Outside_Fraction":
                analytic[
                    "Analytic_Outside_Fraction"
                ],

            "Shapely_Outside_Fraction":
                geometric[
                    "Shapely_Outside_Fraction"
                ],

            "Absolute_Outside_Error":
                abs(
                    analytic[
                        "Analytic_Outside_Fraction"
                    ]
                    -
                    geometric[
                        "Shapely_Outside_Fraction"
                    ]
                ),

            "Analytic_Total_Discrepancy_Fraction":
                analytic[
                    "Analytic_Total_Discrepancy_Fraction"
                ],

            "Shapely_Total_Discrepancy_Fraction":
                geometric[
                    "Shapely_Total_Discrepancy_Fraction"
                ],

            "Absolute_Total_Error":
                abs(
                    analytic[
                        "Analytic_Total_Discrepancy_Fraction"
                    ]
                    -
                    geometric[
                        "Shapely_Total_Discrepancy_Fraction"
                    ]
                )
        }
    )


pair_df = pd.DataFrame(
    pair_rows
)


# ------------------------------------------------------------
# 10. Automatically select raster cases
# ------------------------------------------------------------

sorted_pairs = pair_df.sort_values(
    "Analytic_Total_Discrepancy_Fraction"
).reset_index(
    drop=True
)


minimum_case = sorted_pairs.iloc[
    0
]


median_case = sorted_pairs.iloc[
    len(sorted_pairs) // 2
]


maximum_case = sorted_pairs.iloc[
    -1
]


representative_cases = pd.DataFrame(
    [
        {
            "Case":
                "Minimum",

            **minimum_case.to_dict()
        },

        {
            "Case":
                "Median",

            **median_case.to_dict()
        },

        {
            "Case":
                "Maximum",

            **maximum_case.to_dict()
        }
    ]
)


# ------------------------------------------------------------
# 11. Raster convergence for representative cases
# ------------------------------------------------------------

raster_rows = []


for _, case_row in representative_cases.iterrows():

    case_name = (
        case_row[
            "Case"
        ]
    )


    trajectory_1 = (
        case_row[
            "Trajectory_1"
        ]
    )


    trajectory_2 = (
        case_row[
            "Trajectory_2"
        ]
    )


    e1 = structure_store[
        trajectory_1
    ]


    e2 = structure_store[
        trajectory_2
    ]


    exact_geometry = shapely_two_pass_metrics(
        grid,
        e1,
        e2,
        W
    )


    for resolution in RASTER_RESOLUTIONS:

        raster = raster_coverage_metrics(
            exact_geometry[
                "Footprint_1"
            ],
            exact_geometry[
                "Footprint_2"
            ],
            exact_geometry[
                "Target_Field"
            ],
            resolution
        )


        exact_total = (
            exact_geometry[
                "Shapely_Total_Discrepancy_Fraction"
            ]
        )


        raster_total = (
            raster[
                "Raster_Total_Discrepancy_Fraction"
            ]
        )


        if exact_total > 0:

            relative_error = (
                abs(
                    raster_total
                    - exact_total
                )
                / exact_total
            )

        else:

            relative_error = np.nan


        raster_rows.append(
            {
                "Case":
                    case_name,

                "Trajectory_1":
                    trajectory_1,

                "Trajectory_2":
                    trajectory_2,

                "Raster_Resolution":
                    resolution,

                "Exact_Shaped_Total_Fraction":
                    exact_total,

                "Raster_Total_Fraction":
                    raster_total,

                "Absolute_Raster_Error":
                    abs(
                        raster_total
                        - exact_total
                    ),

                "Relative_Raster_Error":
                    relative_error,

                "Raster_Skip_Fraction":
                    raster[
                        "Raster_Skip_Fraction"
                    ],

                "Raster_Overlap_Fraction":
                    raster[
                        "Raster_Overlap_Fraction"
                    ],

                "Raster_Outside_Fraction":
                    raster[
                        "Raster_Outside_Fraction"
                    ]
            }
        )


raster_df = pd.DataFrame(
    raster_rows
)


# ------------------------------------------------------------
# 12. Save tables
# ------------------------------------------------------------

pair_file = (
    TABLES_DIR
    / "geometric_coverage_validation_all_pairs.csv"
)


representative_file = (
    TABLES_DIR
    / "geometric_coverage_validation_representative_cases.csv"
)


raster_file = (
    TABLES_DIR
    / "geometric_coverage_raster_convergence.csv"
)


pair_df.to_csv(
    pair_file,
    index=False
)


representative_cases.to_csv(
    representative_file,
    index=False
)


raster_df.to_csv(
    raster_file,
    index=False
)


# ------------------------------------------------------------
# 13. Figure — analytic versus exact Shapely
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 8)
)


plt.scatter(
    pair_df[
        "Analytic_Total_Discrepancy_Fraction"
    ],
    pair_df[
        "Shapely_Total_Discrepancy_Fraction"
    ]
)


axis_min = min(
    pair_df[
        "Analytic_Total_Discrepancy_Fraction"
    ].min(),
    pair_df[
        "Shapely_Total_Discrepancy_Fraction"
    ].min()
)


axis_max = max(
    pair_df[
        "Analytic_Total_Discrepancy_Fraction"
    ].max(),
    pair_df[
        "Shapely_Total_Discrepancy_Fraction"
    ].max()
)


plt.plot(
    [
        axis_min,
        axis_max
    ],
    [
        axis_min,
        axis_max
    ],
    linestyle="--"
)


plt.xlabel(
    "Analytical coverage-count discrepancy fraction"
)


plt.ylabel(
    "Exact Shapely coverage-count discrepancy fraction"
)


plt.title(
    "Direct geometric validation of the analytical coverage model"
)


plt.tight_layout()


analytic_shapely_figure = (
    FIGURES_DIR
    / "analytic_vs_exact_shapely_coverage.png"
)


plt.savefig(
    analytic_shapely_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ------------------------------------------------------------
# 14. Figure — raster convergence
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


for case_name in [
    "Minimum",
    "Median",
    "Maximum"
]:

    subset = raster_df[
        raster_df[
            "Case"
        ]
        == case_name
    ].sort_values(
        "Raster_Resolution",
        ascending=False
    )


    plt.plot(
        subset[
            "Raster_Resolution"
        ],
        subset[
            "Relative_Raster_Error"
        ],
        marker="o",
        label=case_name
    )


plt.xlabel(
    "Square raster cell size"
)


plt.ylabel(
    "Relative error versus exact Shapely geometry"
)


plt.title(
    "Raster convergence toward exact footprint geometry"
)


plt.gca().invert_xaxis()


plt.legend()


plt.tight_layout()


raster_figure = (
    FIGURES_DIR
    / "coverage_raster_convergence.png"
)


plt.savefig(
    raster_figure,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ------------------------------------------------------------
# 15. Numerical summaries
# ------------------------------------------------------------

valid_order_count = int(
    pair_df[
        "Pass_Ordering_Valid"
    ].sum()
)


invalid_order_count = (
    len(pair_df)
    - valid_order_count
)


max_skip_error = (
    pair_df[
        "Absolute_Skip_Error"
    ].max()
)


max_overlap_error = (
    pair_df[
        "Absolute_Overlap_Error"
    ].max()
)


max_outside_error = (
    pair_df[
        "Absolute_Outside_Error"
    ].max()
)


max_total_error = (
    pair_df[
        "Absolute_Total_Error"
    ].max()
)


median_total_error = (
    pair_df[
        "Absolute_Total_Error"
    ].median()
)


min_centerline_spacing = (
    pair_df[
        "Minimum_Centerline_Spacing"
    ].min()
)


# ------------------------------------------------------------
# 16. Print results
# ------------------------------------------------------------

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
    + "=" * 90
)

print(
    "MODEL VALIDITY CONDITION"
)

print(
    "=" * 90
)


print(
    "\nPass ordering condition:"
)

print(
    "  W + e2(x) - e1(x) > 0"
)


print(
    f"\nValid pairs:"
    f"\n  {valid_order_count:,}"
    f" / "
    f"{len(pair_df):,}"
)


print(
    f"\nInvalid pairs:"
    f"\n  {invalid_order_count:,}"
)


print(
    "\nMinimum observed neighboring centerline spacing:"
)

print(
    f"  {min_centerline_spacing:.12f}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "ANALYTICAL VERSUS EXACT SHAPELY GEOMETRY"
)

print(
    "=" * 90
)


print(
    "\nMaximum absolute skip-fraction error:"
)

print(
    f"  {max_skip_error:.12e}"
)


print(
    "\nMaximum absolute overlap-fraction error:"
)

print(
    f"  {max_overlap_error:.12e}"
)


print(
    "\nMaximum absolute outside-application-fraction error:"
)

print(
    f"  {max_outside_error:.12e}"
)


print(
    "\nMedian absolute total-discrepancy error:"
)

print(
    f"  {median_total_error:.12e}"
)


print(
    "\nMaximum absolute total-discrepancy error:"
)

print(
    f"  {max_total_error:.12e}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "REPRESENTATIVE RASTER VALIDATION CASES"
)

print(
    "=" * 90
)


print(
    representative_cases[
        [
            "Case",
            "Trajectory_1",
            "Trajectory_2",
            "Analytic_Total_Discrepancy_Fraction",
            "Shapely_Total_Discrepancy_Fraction",
            "Absolute_Total_Error"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 90
)

print(
    "RASTER CONVERGENCE"
)

print(
    "=" * 90
)


print(
    raster_df[
        [
            "Case",
            "Raster_Resolution",
            "Exact_Shaped_Total_Fraction",
            "Raster_Total_Fraction",
            "Absolute_Raster_Error",
            "Relative_Raster_Error"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 90
)

print(
    "INTERPRETATION"
)

print(
    "=" * 90
)


if (
    invalid_order_count == 0
    and
    max_total_error < 1e-10
):

    print(
        "\nThe analytical coverage-area mapping is numerically "
        "identical to the independent exact Shapely footprint "
        "geometry for all tested empirical trajectory pairs."
    )


    print(
        "\nThis directly validates the mapping from lateral "
        "guidance-error signals to physical skip, overlap/excess "
        "application, boundary application, and total "
        "coverage-count discrepancy under the stated model "
        "assumptions."
    )

else:

    print(
        "\nThe analytical and exact geometric solutions are not "
        "identical for every tested case."
    )


    print(
        "Inspect the saved pair table before using the final "
        "operational tolerance equation."
    )


print(
    "\nThis validation applies specifically to:"
)

print(
    "  - straight parallel passes;"
)

print(
    "  - fixed implement working width;"
)

print(
    "  - purely cross-track lateral guidance error;"
)

print(
    "  - vertical/cross-track footprint edges;"
)

print(
    "  - preserved neighboring-pass ordering;"
)

print(
    "  - no turning/headland dynamics;"
)

print(
    "  - no heading-dependent implement rotation."
)


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
    f"\nAll-pair geometric validation:\n"
    f"{pair_file}"
)


print(
    f"\nRepresentative cases:\n"
    f"{representative_file}"
)


print(
    f"\nRaster convergence:\n"
    f"{raster_file}"
)


print(
    f"\nAnalytic-versus-Shapely figure:\n"
    f"{analytic_shapely_figure}"
)


print(
    f"\nRaster convergence figure:\n"
    f"{raster_figure}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 4B COMPLETE"
)

print(
    "=" * 90
)