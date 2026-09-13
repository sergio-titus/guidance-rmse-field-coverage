from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 4A — FINAL CONTINUOUS OPERATIONAL TOLERANCE FRAMEWORK
# ============================================================
#
# PURPOSE
# -------
# Convert the experimentally established dimensionless
# coefficients into a continuous operational framework linking:
#
#       guidance error magnitude
#       implement working width
#       field pass count
#       coverage discrepancy
#
#
# CENTRAL DIMENSIONLESS VARIABLE
# ------------------------------
#
#       eta = R / W
#
# where:
#
#       R = centered guidance-error RMS
#       W = implement working width
#
#
# INTERNAL COVERAGE DISCREPANCY
# -----------------------------
#
# For M parallel passes:
#
#       D_internal
#           = eta * (M - 1)/M * K
#
#
# OUTER-BOUNDARY CONTRIBUTION
# ---------------------------
#
#       D_boundary
#           = eta * 2B/M
#
#
# COMBINED MEAN DISCREPANCY
# -------------------------
#
#       D_total
#           = eta * C(M)
#
# where:
#
#       C(M)
#           = (M - 1)/M * K
#             + 2B/M
#
#
# TOLERANCE INVERSION
# -------------------
#
# If an application defines an allowable coverage discrepancy q:
#
#       D_total <= q
#
# then:
#
#       eta_max
#           = q / C(M)
#
# and:
#
#       R_max
#           = W * q / C(M)
#
#
# IMPORTANT
# ---------
#
# NO acceptable q is selected in this script.
#
# The study provides the continuous conversion function.
# The operational/agronomic decision-maker supplies q.
#
#
# TWO COEFFICIENT SETS
# --------------------
#
# 1. Common-marginal controlled experiment
# 2. Original empirical marginal sensitivity result
#
# Their difference quantifies robustness to marginal choice.
#
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

COMMON_STAGE_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "operational_coefficient_marginal_sensitivity.csv"
)

MULTIPASS_STAGE_FILE = (
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
    / "operational_tolerance_framework"
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
# 2. Load empirically derived coefficients
# ------------------------------------------------------------

for file_path in [
    COMMON_STAGE_FILE,
    MULTIPASS_STAGE_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{file_path}"
        )


print(
    "=" * 90
)

print(
    "STAGE 4A — FINAL CONTINUOUS OPERATIONAL TOLERANCE FRAMEWORK"
)

print(
    "=" * 90
)


sensitivity_df = pd.read_csv(
    COMMON_STAGE_FILE
)

multipass_df = pd.read_csv(
    MULTIPASS_STAGE_FILE
)


# ------------------------------------------------------------
# 3. Extract K and B
# ------------------------------------------------------------

internal_row = sensitivity_df[
    sensitivity_df[
        "Operational_Coefficient"
    ]
    == "All-phase internal K"
]


boundary_row = sensitivity_df[
    sensitivity_df[
        "Operational_Coefficient"
    ]
    == "One-edge boundary B"
]


if len(internal_row) != 1:

    raise RuntimeError(
        "Could not uniquely identify internal K."
    )


if len(boundary_row) != 1:

    raise RuntimeError(
        "Could not uniquely identify boundary B."
    )


internal_row = internal_row.iloc[
    0
]

boundary_row = boundary_row.iloc[
    0
]


K_COMMON = float(
    internal_row[
        "Common_Marginal_Value"
    ]
)


K_EMPIRICAL = float(
    internal_row[
        "Original_Marginal_Ensemble_Mean"
    ]
)


B_COMMON = float(
    boundary_row[
        "Common_Marginal_Value"
    ]
)


B_EMPIRICAL = float(
    boundary_row[
        "Original_Marginal_Ensemble_Mean"
    ]
)


# ------------------------------------------------------------
# 4. Extract Stage 3A uncertainty coefficient
# ------------------------------------------------------------

SIGMA_K = None


for m in [
    2
]:

    row = multipass_df[
        multipass_df[
            "Number_of_Passes_M"
        ]
        == m
    ]

    if len(row) == 1:

        row = row.iloc[
            0
        ]

        # For M = 2:
        #
        # independent SD / eta
        #     = sqrt(1)/2 * sigma_K
        #
        # therefore:
        #
        # sigma_K
        #     = 2 * SD
        #

        SIGMA_K = (
            2.0
            * float(
                row[
                    "Independent_SD_D_over_eta"
                ]
            )
        )


if SIGMA_K is None:

    raise RuntimeError(
        "Could not recover Stage 3A sigma_K."
    )


# ------------------------------------------------------------
# 5. Analytical functions
# ------------------------------------------------------------

def internal_coefficient(
    m,
    k
):

    m = np.asarray(
        m,
        dtype=float
    )

    return (
        (
            m - 1.0
        )
        / m
        * k
    )


def boundary_coefficient(
    m,
    b
):

    m = np.asarray(
        m,
        dtype=float
    )

    return (
        2.0
        * b
        / m
    )


def total_coefficient(
    m,
    k,
    b
):

    return (
        internal_coefficient(
            m,
            k
        )
        +
        boundary_coefficient(
            m,
            b
        )
    )


def discrepancy(
    eta,
    m,
    k,
    b
):

    return (
        eta
        * total_coefficient(
            m,
            k,
            b
        )
    )


def eta_tolerance(
    q,
    m,
    k,
    b
):

    return (
        q
        /
        total_coefficient(
            m,
            k,
            b
        )
    )


def rms_tolerance_over_width(
    q,
    m,
    k,
    b
):

    # Since:
    #
    #       R_max / W = eta_max
    #

    return eta_tolerance(
        q,
        m,
        k,
        b
    )


# ------------------------------------------------------------
# 6. Pass-count coefficient table
# ------------------------------------------------------------
#
# M = 2..100 is used ONLY to tabulate and visualize the
# analytical M-dependence.
#
# The equations are valid for any integer M >= 2.
#
# ------------------------------------------------------------

PASS_COUNTS = np.arange(
    2,
    101
)


rows = []


for m in PASS_COUNTS:

    common_internal = internal_coefficient(
        m,
        K_COMMON
    )

    common_boundary = boundary_coefficient(
        m,
        B_COMMON
    )

    common_total = (
        common_internal
        + common_boundary
    )


    empirical_internal = internal_coefficient(
        m,
        K_EMPIRICAL
    )

    empirical_boundary = boundary_coefficient(
        m,
        B_EMPIRICAL
    )

    empirical_total = (
        empirical_internal
        + empirical_boundary
    )


    independent_sd_slope = (
        np.sqrt(
            m - 1.0
        )
        / m
        * SIGMA_K
    )


    shared_sd_slope = (
        (
            m - 1.0
        )
        / m
        * SIGMA_K
    )


    rows.append(
        {
            "Number_of_Passes_M":
                m,

            "Common_Internal_Coefficient":
                common_internal,

            "Common_Boundary_Coefficient":
                common_boundary,

            "Common_Total_Coefficient":
                common_total,

            "Empirical_Internal_Coefficient":
                empirical_internal,

            "Empirical_Boundary_Coefficient":
                empirical_boundary,

            "Empirical_Total_Coefficient":
                empirical_total,

            "Relative_Total_Coefficient_Difference":
                (
                    empirical_total
                    - common_total
                )
                / common_total,

            "Independent_SD_Slope_per_eta":
                independent_sd_slope,

            "Perfectly_Shared_SD_Slope_per_eta":
                shared_sd_slope
        }
    )


coefficient_df = pd.DataFrame(
    rows
)


# ------------------------------------------------------------
# 7. Exact large-M limits
# ------------------------------------------------------------

COMMON_LARGE_M_LIMIT = K_COMMON

EMPIRICAL_LARGE_M_LIMIT = K_EMPIRICAL


# ------------------------------------------------------------
# 8. Normalized tolerance transfer function
# ------------------------------------------------------------
#
# Instead of selecting arbitrary allowable discrepancy values q,
# report:
#
#       eta_max / q = 1 / C(M)
#
# so any user-selected q can be inserted later.
#
#
# Since:
#
#       R_max / (W q)
#           = 1 / C(M)
#
# the same transfer factor converts an agronomic coverage
# tolerance into a maximum allowable guidance RMS.
#
# ------------------------------------------------------------

coefficient_df[
    "Common_eta_max_per_unit_q"
] = (
    1.0
    /
    coefficient_df[
        "Common_Total_Coefficient"
    ]
)


coefficient_df[
    "Empirical_eta_max_per_unit_q"
] = (
    1.0
    /
    coefficient_df[
        "Empirical_Total_Coefficient"
    ]
)


coefficient_df[
    "Common_Rmax_over_Wq"
] = (
    coefficient_df[
        "Common_eta_max_per_unit_q"
    ]
)


coefficient_df[
    "Empirical_Rmax_over_Wq"
] = (
    coefficient_df[
        "Empirical_eta_max_per_unit_q"
    ]
)


# ------------------------------------------------------------
# 9. Save master coefficient table
# ------------------------------------------------------------

coefficient_file = (
    TABLES_DIR
    / "final_operational_tolerance_coefficients.csv"
)


coefficient_df.to_csv(
    coefficient_file,
    index=False
)


# ------------------------------------------------------------
# 10. Equation summary table
# ------------------------------------------------------------

equation_df = pd.DataFrame(
    [
        {
            "Quantity":
                "Common marginal internal K",

            "Value":
                K_COMMON,

            "Meaning":
                "Phase-averaged internal neighboring-pass discrepancy coefficient"
        },

        {
            "Quantity":
                "Original empirical marginal mean K",

            "Value":
                K_EMPIRICAL,

            "Meaning":
                "Sensitivity-analysis ensemble mean internal coefficient"
        },

        {
            "Quantity":
                "Common marginal one-edge B",

            "Value":
                B_COMMON,

            "Meaning":
                "Mean one-sided outer-boundary excursion coefficient"
        },

        {
            "Quantity":
                "Original empirical marginal mean B",

            "Value":
                B_EMPIRICAL,

            "Meaning":
                "Sensitivity-analysis ensemble mean one-edge coefficient"
        },

        {
            "Quantity":
                "Common large-M coefficient",

            "Value":
                COMMON_LARGE_M_LIMIT,

            "Meaning":
                "Limit of D_total / eta as M approaches infinity"
        },

        {
            "Quantity":
                "Empirical large-M coefficient",

            "Value":
                EMPIRICAL_LARGE_M_LIMIT,

            "Meaning":
                "Sensitivity large-M limit"
        },

        {
            "Quantity":
                "Stage 2D single-interface SD K",

            "Value":
                SIGMA_K,

            "Meaning":
                "Used only for transparent inter-interface dependence benchmarks"
        }
    ]
)


equation_file = (
    TABLES_DIR
    / "final_operational_framework_equations.csv"
)


equation_df.to_csv(
    equation_file,
    index=False
)


# ------------------------------------------------------------
# 11. Figure — coefficient decomposition versus M
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    coefficient_df[
        "Number_of_Passes_M"
    ],
    coefficient_df[
        "Common_Internal_Coefficient"
    ],
    linewidth=2,
    label="Internal interfaces"
)


plt.plot(
    coefficient_df[
        "Number_of_Passes_M"
    ],
    coefficient_df[
        "Common_Boundary_Coefficient"
    ],
    linewidth=2,
    label="Outer boundaries"
)


plt.plot(
    coefficient_df[
        "Number_of_Passes_M"
    ],
    coefficient_df[
        "Common_Total_Coefficient"
    ],
    linewidth=2,
    label="Combined"
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "Dimensionless coefficient C(M)"
)

plt.title(
    "Decomposition of mean field discrepancy coefficient"
)

plt.legend()

plt.tight_layout()


decomposition_figure = (
    FIGURES_DIR
    / "final_coefficient_decomposition_vs_pass_count.png"
)


plt.savefig(
    decomposition_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 12. Figure — common versus empirical marginal robustness
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    coefficient_df[
        "Number_of_Passes_M"
    ],
    coefficient_df[
        "Common_Total_Coefficient"
    ],
    linewidth=2,
    label="Common-marginal control"
)


plt.plot(
    coefficient_df[
        "Number_of_Passes_M"
    ],
    coefficient_df[
        "Empirical_Total_Coefficient"
    ],
    linewidth=2,
    linestyle="--",
    label="Original empirical marginals"
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "Combined mean discrepancy coefficient C(M)"
)

plt.title(
    "Robustness of the operational coefficient to marginal-distribution choice"
)

plt.legend()

plt.tight_layout()


robustness_figure = (
    FIGURES_DIR
    / "final_common_vs_empirical_coefficient.png"
)


plt.savefig(
    robustness_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 13. Figure — normalized tolerance transfer function
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    coefficient_df[
        "Number_of_Passes_M"
    ],
    coefficient_df[
        "Common_eta_max_per_unit_q"
    ],
    linewidth=2,
    label="Common-marginal control"
)


plt.plot(
    coefficient_df[
        "Number_of_Passes_M"
    ],
    coefficient_df[
        "Empirical_eta_max_per_unit_q"
    ],
    linewidth=2,
    linestyle="--",
    label="Original empirical marginals"
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "eta_max / q = R_max / (W q)"
)

plt.title(
    "Continuous operational tolerance transfer function"
)

plt.legend()

plt.tight_layout()


tolerance_figure = (
    FIGURES_DIR
    / "final_operational_tolerance_transfer_function.png"
)


plt.savefig(
    tolerance_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 14. Figure — dependence benchmark slopes
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    coefficient_df[
        "Number_of_Passes_M"
    ],
    coefficient_df[
        "Independent_SD_Slope_per_eta"
    ],
    linewidth=2,
    label="Independent-interface benchmark"
)


plt.plot(
    coefficient_df[
        "Number_of_Passes_M"
    ],
    coefficient_df[
        "Perfectly_Shared_SD_Slope_per_eta"
    ],
    linewidth=2,
    label="Perfectly shared-interface benchmark"
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "SD[D] / eta"
)

plt.title(
    "Dependence-sensitive uncertainty around the mean operational curve"
)

plt.legend()

plt.tight_layout()


uncertainty_figure = (
    FIGURES_DIR
    / "final_uncertainty_benchmark_slopes.png"
)


plt.savefig(
    uncertainty_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 15. Selected M examples
# ------------------------------------------------------------

EXAMPLE_M = [
    2,
    3,
    5,
    10,
    20,
    50,
    100
]


example_df = coefficient_df[
    coefficient_df[
        "Number_of_Passes_M"
    ].isin(
        EXAMPLE_M
    )
][
    [
        "Number_of_Passes_M",
        "Common_Internal_Coefficient",
        "Common_Boundary_Coefficient",
        "Common_Total_Coefficient",
        "Empirical_Total_Coefficient",
        "Common_eta_max_per_unit_q",
        "Empirical_eta_max_per_unit_q"
    ]
]


# ------------------------------------------------------------
# 16. Numerical robustness checks
# ------------------------------------------------------------

relative_K_change = (
    K_EMPIRICAL
    - K_COMMON
) / K_COMMON


relative_B_change = (
    B_EMPIRICAL
    - B_COMMON
) / B_COMMON


max_relative_total_difference = np.max(
    np.abs(
        coefficient_df[
            "Relative_Total_Coefficient_Difference"
        ]
    )
)


# ------------------------------------------------------------
# 17. Terminal formatting
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
# 18. Print derived coefficients
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "FINAL EMPIRICALLY DERIVED COEFFICIENTS"
)

print(
    "=" * 90
)


print(
    "\nCommon-marginal controlled coefficients:"
)

print(
    f"  K = "
    f"{K_COMMON:.12f}"
)

print(
    f"  B = "
    f"{B_COMMON:.12f}"
)


print(
    "\nOriginal empirical marginal sensitivity:"
)

print(
    f"  K = "
    f"{K_EMPIRICAL:.12f}"
)

print(
    f"  B = "
    f"{B_EMPIRICAL:.12f}"
)


print(
    "\nRelative coefficient changes:"
)

print(
    f"  K change = "
    f"{100.0 * relative_K_change:.6f}%"
)

print(
    f"  B change = "
    f"{100.0 * relative_B_change:.6f}%"
)


# ------------------------------------------------------------
# 19. Print final analytical equations
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "FINAL CONTINUOUS COVERAGE EQUATIONS"
)

print(
    "=" * 90
)


print(
    "\nLet:"
)

print(
    "  eta = R / W"
)

print(
    "  R   = centered guidance-error RMS"
)

print(
    "  W   = implement working width"
)

print(
    "  M   = number of parallel passes"
)


print(
    "\nCommon-marginal controlled result:"
)

print(
    "\n  D_total"
)

print(
    "    = eta * ["
)

print(
    f"        (M - 1)/M * "
    f"{K_COMMON:.12f}"
)

print(
    f"        + "
    f"{2.0 * B_COMMON:.12f}"
    f"/M"
)

print(
    "      ]"
)


print(
    "\nOriginal empirical marginal sensitivity result:"
)

print(
    "\n  D_total_emp"
)

print(
    "    = eta * ["
)

print(
    f"        (M - 1)/M * "
    f"{K_EMPIRICAL:.12f}"
)

print(
    f"        + "
    f"{2.0 * B_EMPIRICAL:.12f}"
    f"/M"
)

print(
    "      ]"
)


# ------------------------------------------------------------
# 20. Print tolerance inversion
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "OPERATIONAL TOLERANCE INVERSION"
)

print(
    "=" * 90
)


print(
    "\nFor any externally specified allowable discrepancy q:"
)


print(
    "\n  eta_max"
)

print(
    "    = q / C(M)"
)


print(
    "\nwhere:"
)

print(
    "\n  C(M)"
)

print(
    f"    = (M - 1)/M * "
    f"{K_COMMON:.12f}"
)

print(
    f"      + "
    f"{2.0 * B_COMMON:.12f}"
    f"/M"
)


print(
    "\nTherefore:"
)

print(
    "\n  R_max"
)

print(
    "    = W * q / C(M)"
)


print(
    "\nNo numerical q is imposed by this study."
)


# ------------------------------------------------------------
# 21. Pass-count examples
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "PASS-COUNT DEPENDENCE"
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
# 22. Large-M behavior
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "LARGE-FIELD LIMIT"
)

print(
    "=" * 90
)


print(
    "\nAs M -> infinity:"
)


print(
    "\nCommon-marginal control:"
)

print(
    f"  D_total / eta"
    f" -> "
    f"{COMMON_LARGE_M_LIMIT:.12f}"
)


print(
    "\nOriginal empirical marginals:"
)

print(
    f"  D_total / eta"
    f" -> "
    f"{EMPIRICAL_LARGE_M_LIMIT:.12f}"
)


print(
    "\nOuter-boundary contribution -> 0."
)

print(
    "Internal neighboring-pass discrepancy dominates the "
    "large-field mean."
)


# ------------------------------------------------------------
# 23. Marginal robustness
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "MARGINAL-DISTRIBUTION ROBUSTNESS"
)

print(
    "=" * 90
)


print(
    "\nMaximum relative difference in the combined coefficient "
    "over M = 2..100:"
)

print(
    f"  "
    f"{100.0 * max_relative_total_difference:.6f}%"
)


print(
    "\nThis quantifies the sensitivity of the final mean "
    "operational curve to the common-marginal transformation."
)


# ------------------------------------------------------------
# 24. Uncertainty benchmark reminder
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "UNCERTAINTY AROUND THE MEAN CURVE"
)

print(
    "=" * 90
)


print(
    f"\nStage 2D single-interface SD:"
    f"\n  sigma_K = "
    f"{SIGMA_K:.12f}"
)


print(
    "\nIndependent-interface benchmark:"
)

print(
    "  SD[D] / eta"
)

print(
    f"    = sqrt(M - 1)/M * "
    f"{SIGMA_K:.12f}"
)


print(
    "\nPerfectly shared-interface benchmark:"
)

print(
    "  SD[D] / eta"
)

print(
    f"    = (M - 1)/M * "
    f"{SIGMA_K:.12f}"
)


print(
    "\nThese are dependence benchmarks, not claims about the "
    "true inter-pass covariance."
)


# ------------------------------------------------------------
# 25. Output files
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
    f"\nMaster tolerance coefficient table:\n"
    f"{coefficient_file}"
)

print(
    f"\nEquation/coefficient summary:\n"
    f"{equation_file}"
)

print(
    f"\nCoefficient decomposition figure:\n"
    f"{decomposition_figure}"
)

print(
    f"\nMarginal robustness figure:\n"
    f"{robustness_figure}"
)

print(
    f"\nTolerance transfer function:\n"
    f"{tolerance_figure}"
)

print(
    f"\nUncertainty benchmark figure:\n"
    f"{uncertainty_figure}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 4A COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nThe simulation/control stages are now complete."
)

print(
    "The final operational result is a continuous dimensionless "
    "mapping rather than an arbitrarily chosen agronomic "
    "acceptability threshold."
)

print(
    "\nFor any application-specific allowable discrepancy q and "
    "implement width W, the maximum admissible centered guidance "
    "RMS follows directly from:"
)

print(
    "\n  R_max = W * q / C(M)"
)