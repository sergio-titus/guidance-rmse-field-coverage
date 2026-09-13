from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 3A — ANALYTICAL MULTI-PASS UNCERTAINTY PROPAGATION
# ============================================================
#
# PURPOSE
# -------
# Extend the single-interface results from Stage 2D to a field
# with M parallel passes WITHOUT inventing an empirical
# inter-pass dependence model.
#
#
# FIELD-LEVEL RELATION
# --------------------
#
# For M passes there are M-1 internal neighboring interfaces.
#
# Let K_j denote the dimensionless discrepancy factor at
# interface j.
#
# Then:
#
#       D_M = eta/M * sum_{j=1}^{M-1} K_j
#
# where:
#
#       eta = centered RMS / implement working width
#
#
# EXPECTATION
# -----------
#
# Regardless of dependence among interfaces:
#
#       E[D_M]
#           = eta * (M-1)/M * mu_K
#
#
# VARIABILITY
# -----------
#
# The empirical dataset does not identify the joint dependence
# among all neighboring operational passes.
#
# Therefore we DO NOT impose one as physical truth.
#
# Instead two clearly labelled analytical benchmarks are used.
#
#
# BENCHMARK A — INDEPENDENT INTERFACES
#
# If K_j are independent draws from the empirical interface
# distribution:
#
#       Var(D_M)
#           = eta^2 * (M-1)/M^2 * sigma_K^2
#
#       SD(D_M)
#           = eta * sqrt(M-1)/M * sigma_K
#
#
# BENCHMARK B — PERFECTLY SHARED INTERFACE STATE
#
# If all interfaces experience exactly the same K realization:
#
#       D_M
#           = eta * (M-1)/M * K
#
# therefore:
#
#       SD(D_M)
#           = eta * (M-1)/M * sigma_K
#
#
# These are mathematical dependence benchmarks.
# Neither is claimed to represent the true agricultural field.
#
#
# NORMALIZED COEFFICIENTS
# -----------------------
#
# Since eta is only a multiplicative factor, this script works
# with:
#
#       D_M / eta
#
# so no implement width and no physical RMS value are imposed.
#
#
# PASS COUNT
# ----------
#
# Instead of choosing one arbitrary field size, the script
# reports the analytical dependence on M.
#
# A tabulation from M=2 to M=100 is produced only for convenient
# visualization of the asymptotic behavior.
#
# The equations themselves remain valid for any integer M >= 2.
#
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

STAGE_2D_SUMMARY_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "exhaustive_phase_clustering_dataset_summary.csv"
)

STAGE_2D_SCENARIO_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "exhaustive_phase_clustering_scenarios.csv"
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
    / "multipass_uncertainty"
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
# 2. Load Stage 2D results
# ------------------------------------------------------------

for file_path in [
    STAGE_2D_SUMMARY_FILE,
    STAGE_2D_SCENARIO_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing Stage 2D file:\n"
            f"{file_path}"
        )


print(
    "=" * 90
)

print(
    "STAGE 3A — ANALYTICAL MULTI-PASS UNCERTAINTY PROPAGATION"
)

print(
    "=" * 90
)


summary_df = pd.read_csv(
    STAGE_2D_SUMMARY_FILE
)

scenario_df = pd.read_csv(
    STAGE_2D_SCENARIO_FILE
)


print(
    f"\nLoaded Stage 2D summary:"
    f"\n{STAGE_2D_SUMMARY_FILE}"
)

print(
    f"\nLoaded exhaustive scenarios:"
    f"\n{STAGE_2D_SCENARIO_FILE}"
)

print(
    f"\nNumber of interface scenarios:"
    f"\n  {len(scenario_df):,}"
)


# ------------------------------------------------------------
# 3. Empirical single-interface K distribution
# ------------------------------------------------------------

k_values = scenario_df[
    "K_L1"
].to_numpy(
    dtype=float
)


finite = np.isfinite(
    k_values
)


k_values = k_values[
    finite
]


MU_K = np.mean(
    k_values
)


SIGMA_K = np.std(
    k_values,
    ddof=0
)


K_MIN = np.min(
    k_values
)

K_Q05 = np.quantile(
    k_values,
    0.05
)

K_Q25 = np.quantile(
    k_values,
    0.25
)

K_MEDIAN = np.median(
    k_values
)

K_Q75 = np.quantile(
    k_values,
    0.75
)

K_Q95 = np.quantile(
    k_values,
    0.95
)

K_MAX = np.max(
    k_values
)


# ------------------------------------------------------------
# 4. Verify against saved Stage 2D summary
# ------------------------------------------------------------

saved_k_row = summary_df[
    summary_df[
        "Metric"
    ]
    == "K_L1"
]


if len(
    saved_k_row
) != 1:

    raise RuntimeError(
        "Could not uniquely identify K_L1 in Stage 2D summary."
    )


saved_k_row = saved_k_row.iloc[
    0
]


mean_difference = abs(
    MU_K
    - float(
        saved_k_row[
            "Mean"
        ]
    )
)


sd_difference = abs(
    SIGMA_K
    - float(
        saved_k_row[
            "SD"
        ]
    )
)


# ------------------------------------------------------------
# 5. Analytical functions
# ------------------------------------------------------------

def mean_factor(
    number_of_passes
):

    m = float(
        number_of_passes
    )

    return (
        (
            m - 1.0
        )
        / m
        * MU_K
    )


def independent_sd_factor(
    number_of_passes
):

    m = float(
        number_of_passes
    )

    return (
        np.sqrt(
            m - 1.0
        )
        / m
        * SIGMA_K
    )


def shared_sd_factor(
    number_of_passes
):

    m = float(
        number_of_passes
    )

    return (
        (
            m - 1.0
        )
        / m
        * SIGMA_K
    )


def shared_quantile_factor(
    number_of_passes,
    k_quantile
):

    m = float(
        number_of_passes
    )

    return (
        (
            m - 1.0
        )
        / m
        * k_quantile
    )


# ------------------------------------------------------------
# 6. Tabulation across pass counts
# ------------------------------------------------------------
#
# The 2..100 range is ONLY for displaying convergence toward
# the analytical large-M limit.
#
# No claim is made that 100 passes represents a particular
# agricultural field.
#
# ------------------------------------------------------------

PASS_COUNTS = np.arange(
    2,
    101
)


rows = []


for m in PASS_COUNTS:

    expected_factor = mean_factor(
        m
    )


    independent_sd = independent_sd_factor(
        m
    )


    shared_sd = shared_sd_factor(
        m
    )


    rows.append(
        {
            "Number_of_Passes_M":
                m,

            "Number_of_Internal_Interfaces":
                m - 1,

            "Mean_D_over_eta":
                expected_factor,

            "Independent_SD_D_over_eta":
                independent_sd,

            "Perfectly_Shared_SD_D_over_eta":
                shared_sd,

            "Independent_CV":
                (
                    independent_sd
                    / expected_factor
                ),

            "Perfectly_Shared_CV":
                (
                    shared_sd
                    / expected_factor
                ),

            "Perfectly_Shared_Min_D_over_eta":
                shared_quantile_factor(
                    m,
                    K_MIN
                ),

            "Perfectly_Shared_Q05_D_over_eta":
                shared_quantile_factor(
                    m,
                    K_Q05
                ),

            "Perfectly_Shared_Q25_D_over_eta":
                shared_quantile_factor(
                    m,
                    K_Q25
                ),

            "Perfectly_Shared_Median_D_over_eta":
                shared_quantile_factor(
                    m,
                    K_MEDIAN
                ),

            "Perfectly_Shared_Q75_D_over_eta":
                shared_quantile_factor(
                    m,
                    K_Q75
                ),

            "Perfectly_Shared_Q95_D_over_eta":
                shared_quantile_factor(
                    m,
                    K_Q95
                ),

            "Perfectly_Shared_Max_D_over_eta":
                shared_quantile_factor(
                    m,
                    K_MAX
                )
        }
    )


multipass_df = pd.DataFrame(
    rows
)


# ------------------------------------------------------------
# 7. Analytical limiting behavior
# ------------------------------------------------------------

large_m_mean_limit = MU_K

large_m_independent_sd_limit = 0.0

large_m_shared_sd_limit = SIGMA_K


# ------------------------------------------------------------
# 8. Variance-reduction ratio
# ------------------------------------------------------------
#
# Ratio:
#
#     shared SD / independent SD
#
#       = sqrt(M - 1)
#
# exactly.
#
# This quantifies how strongly the inter-interface dependence
# assumption changes field-level uncertainty.
#
# ------------------------------------------------------------

multipass_df[
    "Shared_to_Independent_SD_Ratio"
] = (
    multipass_df[
        "Perfectly_Shared_SD_D_over_eta"
    ]
    /
    multipass_df[
        "Independent_SD_D_over_eta"
    ]
)


multipass_df[
    "Analytical_SD_Ratio_Check"
] = np.sqrt(
    multipass_df[
        "Number_of_Passes_M"
    ]
    - 1
)


max_ratio_identity_error = np.max(
    np.abs(
        multipass_df[
            "Shared_to_Independent_SD_Ratio"
        ]
        -
        multipass_df[
            "Analytical_SD_Ratio_Check"
        ]
    )
)


# ------------------------------------------------------------
# 9. Effective averaging factor
# ------------------------------------------------------------
#
# Under independent interfaces:
#
#       CV decreases as 1/sqrt(M-1)
#
# Under perfectly shared interfaces:
#
#       CV is invariant with M:
#
#       sigma_K / mu_K
#
# ------------------------------------------------------------

single_interface_cv = (
    SIGMA_K
    / MU_K
)


# ------------------------------------------------------------
# 10. Save analytical table
# ------------------------------------------------------------

multipass_file = (
    TABLES_DIR
    / "analytical_multipass_uncertainty_propagation.csv"
)


multipass_df.to_csv(
    multipass_file,
    index=False
)


# ------------------------------------------------------------
# 11. Save analytical coefficient summary
# ------------------------------------------------------------

coefficient_df = pd.DataFrame(
    [
        {
            "Quantity":
                "Single-interface empirical mean K",

            "Value":
                MU_K
        },

        {
            "Quantity":
                "Single-interface empirical SD K",

            "Value":
                SIGMA_K
        },

        {
            "Quantity":
                "Single-interface empirical CV",

            "Value":
                single_interface_cv
        },

        {
            "Quantity":
                "Single-interface minimum K",

            "Value":
                K_MIN
        },

        {
            "Quantity":
                "Single-interface Q05 K",

            "Value":
                K_Q05
        },

        {
            "Quantity":
                "Single-interface median K",

            "Value":
                K_MEDIAN
        },

        {
            "Quantity":
                "Single-interface Q95 K",

            "Value":
                K_Q95
        },

        {
            "Quantity":
                "Single-interface maximum K",

            "Value":
                K_MAX
        },

        {
            "Quantity":
                "Large-M mean D/eta limit",

            "Value":
                large_m_mean_limit
        },

        {
            "Quantity":
                "Large-M independent SD D/eta limit",

            "Value":
                large_m_independent_sd_limit
        },

        {
            "Quantity":
                "Large-M perfectly shared SD D/eta limit",

            "Value":
                large_m_shared_sd_limit
        }
    ]
)


coefficient_file = (
    TABLES_DIR
    / "multipass_analytical_coefficients.csv"
)


coefficient_df.to_csv(
    coefficient_file,
    index=False
)


# ------------------------------------------------------------
# 12. Figure — mean field discrepancy factor
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    multipass_df[
        "Number_of_Passes_M"
    ],
    multipass_df[
        "Mean_D_over_eta"
    ],
    linewidth=2
)


plt.axhline(
    MU_K,
    linestyle="--",
    label=(
        f"Large-M limit = {MU_K:.3f}"
    )
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "Expected normalized field discrepancy / eta"
)

plt.title(
    "Accumulation of expected internal coverage discrepancy"
)

plt.legend()

plt.tight_layout()


mean_figure = (
    FIGURES_DIR
    / "multipass_expected_discrepancy.png"
)


plt.savefig(
    mean_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 13. Figure — uncertainty benchmark comparison
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    multipass_df[
        "Number_of_Passes_M"
    ],
    multipass_df[
        "Independent_SD_D_over_eta"
    ],
    linewidth=2,
    label="Independent-interface benchmark"
)


plt.plot(
    multipass_df[
        "Number_of_Passes_M"
    ],
    multipass_df[
        "Perfectly_Shared_SD_D_over_eta"
    ],
    linewidth=2,
    label="Perfectly shared-interface benchmark"
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "SD of normalized field discrepancy / eta"
)

plt.title(
    "Effect of inter-interface dependence on field-level uncertainty"
)

plt.legend()

plt.tight_layout()


uncertainty_figure = (
    FIGURES_DIR
    / "multipass_dependence_uncertainty_benchmarks.png"
)


plt.savefig(
    uncertainty_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 14. Figure — coefficient of variation
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    multipass_df[
        "Number_of_Passes_M"
    ],
    multipass_df[
        "Independent_CV"
    ],
    linewidth=2,
    label="Independent-interface benchmark"
)


plt.plot(
    multipass_df[
        "Number_of_Passes_M"
    ],
    multipass_df[
        "Perfectly_Shared_CV"
    ],
    linewidth=2,
    label="Perfectly shared-interface benchmark"
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "Coefficient of variation"
)

plt.title(
    "Does multi-pass averaging reduce operational uncertainty?"
)

plt.legend()

plt.tight_layout()


cv_figure = (
    FIGURES_DIR
    / "multipass_coefficient_of_variation.png"
)


plt.savefig(
    cv_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 15. Figure — perfectly shared quantile envelope
# ------------------------------------------------------------
#
# Under perfectly shared dependence, field quantiles scale
# directly from the empirical single-interface distribution.
#
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.fill_between(
    multipass_df[
        "Number_of_Passes_M"
    ],
    multipass_df[
        "Perfectly_Shared_Q05_D_over_eta"
    ],
    multipass_df[
        "Perfectly_Shared_Q95_D_over_eta"
    ],
    alpha=0.3,
    label="5th–95th percentile"
)


plt.plot(
    multipass_df[
        "Number_of_Passes_M"
    ],
    multipass_df[
        "Perfectly_Shared_Median_D_over_eta"
    ],
    linewidth=2,
    label="Median"
)


plt.plot(
    multipass_df[
        "Number_of_Passes_M"
    ],
    multipass_df[
        "Mean_D_over_eta"
    ],
    linewidth=2,
    linestyle="--",
    label="Mean"
)


plt.xlabel(
    "Number of parallel passes M"
)

plt.ylabel(
    "Normalized field discrepancy / eta"
)

plt.title(
    "Perfectly shared-interface empirical uncertainty envelope"
)

plt.legend()

plt.tight_layout()


shared_envelope_figure = (
    FIGURES_DIR
    / "multipass_shared_dependence_quantile_envelope.png"
)


plt.savefig(
    shared_envelope_figure,
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
    280
)

pd.set_option(
    "display.max_rows",
    120
)


# ------------------------------------------------------------
# 17. Print empirical coefficients
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "SINGLE-INTERFACE EMPIRICAL DISTRIBUTION"
)

print(
    "=" * 90
)


print(
    f"\nMean K:"
    f"\n  {MU_K:.12f}"
)


print(
    f"\nSD K:"
    f"\n  {SIGMA_K:.12f}"
)


print(
    f"\nCoefficient of variation:"
    f"\n  {single_interface_cv:.12f}"
)


print(
    "\nEmpirical K quantiles:"
)

print(
    f"  min    = "
    f"{K_MIN:.9f}"
)

print(
    f"  Q05    = "
    f"{K_Q05:.9f}"
)

print(
    f"  Q25    = "
    f"{K_Q25:.9f}"
)

print(
    f"  median = "
    f"{K_MEDIAN:.9f}"
)

print(
    f"  Q75    = "
    f"{K_Q75:.9f}"
)

print(
    f"  Q95    = "
    f"{K_Q95:.9f}"
)

print(
    f"  max    = "
    f"{K_MAX:.9f}"
)


# ------------------------------------------------------------
# 18. Verification against Stage 2D saved summary
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 2D CONSISTENCY CHECK"
)

print(
    "=" * 90
)


print(
    f"\nAbsolute mean difference:"
    f"\n  {mean_difference:.12e}"
)


print(
    f"\nAbsolute SD difference:"
    f"\n  {sd_difference:.12e}"
)


# ------------------------------------------------------------
# 19. Exact field-level equations
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "ANALYTICAL MULTI-PASS RELATIONSHIPS"
)

print(
    "=" * 90
)


print(
    "\nFor M passes:"
)


print(
    "\n  E[D_M] / eta"
)

print(
    f"    = (M - 1)/M * "
    f"{MU_K:.12f}"
)


print(
    "\nIndependent-interface benchmark:"
)

print(
    "  SD[D_M] / eta"
)

print(
    f"    = sqrt(M - 1)/M * "
    f"{SIGMA_K:.12f}"
)


print(
    "\nPerfectly shared-interface benchmark:"
)

print(
    "  SD[D_M] / eta"
)

print(
    f"    = (M - 1)/M * "
    f"{SIGMA_K:.12f}"
)


print(
    "\nwhere:"
)

print(
    "  eta = centered RMS / implement working width"
)


# ------------------------------------------------------------
# 20. Selected pass-count examples
# ------------------------------------------------------------
#
# These are display examples only.
#
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


example_df = multipass_df[
    multipass_df[
        "Number_of_Passes_M"
    ].isin(
        example_pass_counts
    )
][
    [
        "Number_of_Passes_M",
        "Number_of_Internal_Interfaces",
        "Mean_D_over_eta",
        "Independent_SD_D_over_eta",
        "Perfectly_Shared_SD_D_over_eta",
        "Independent_CV",
        "Perfectly_Shared_CV",
        "Shared_to_Independent_SD_Ratio"
    ]
]


print(
    "\n"
    + "=" * 90
)

print(
    "ILLUSTRATIVE PASS-COUNT SCALING"
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
# 21. Dependence importance
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DEPENDENCE-SENSITIVITY RESULT"
)

print(
    "=" * 90
)


print(
    "\nPerfectly-shared / independent SD ratio:"
)

print(
    "  sqrt(M - 1)"
)


print(
    f"\nMaximum numerical identity error:"
    f"\n  {max_ratio_identity_error:.12e}"
)


print(
    "\nTherefore increasing the number of passes reduces "
    "relative uncertainty under the independent-interface "
    "benchmark, but does NOT remove relative uncertainty when "
    "interface states remain perfectly shared."
)


# ------------------------------------------------------------
# 22. Large-M behavior
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "LARGE-M ANALYTICAL LIMIT"
)

print(
    "=" * 90
)


print(
    "\nAs M -> infinity:"
)

print(
    f"\n  E[D_M] / eta"
    f" -> "
    f"{large_m_mean_limit:.12f}"
)

print(
    f"\n  independent SD[D_M] / eta"
    f" -> "
    f"{large_m_independent_sd_limit:.12f}"
)

print(
    f"\n  perfectly shared SD[D_M] / eta"
    f" -> "
    f"{large_m_shared_sd_limit:.12f}"
)


# ------------------------------------------------------------
# 23. Output files
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
    f"\nMulti-pass analytical table:\n"
    f"{multipass_file}"
)

print(
    f"\nAnalytical coefficients:\n"
    f"{coefficient_file}"
)

print(
    f"\nExpected-discrepancy figure:\n"
    f"{mean_figure}"
)

print(
    f"\nDependence-benchmark figure:\n"
    f"{uncertainty_figure}"
)

print(
    f"\nCoefficient-of-variation figure:\n"
    f"{cv_figure}"
)

print(
    f"\nShared-dependence quantile envelope:\n"
    f"{shared_envelope_figure}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 3A COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nNo inter-pass dependence model has been claimed as "
    "empirical truth."
)

print(
    "Independent and perfectly shared interfaces are reported "
    "only as transparent mathematical benchmarks."
)

print(
    "\nThe expected multi-pass discrepancy is identified exactly "
    "from the Stage 2D empirical interface distribution."
)

print(
    "What remains unidentified from the current dataset is the "
    "true covariance among multiple neighboring field interfaces."
)

print(
    "\nThe next stage should address OUTER FIELD BOUNDARIES "
    "separately from internal skip/overlap."
)