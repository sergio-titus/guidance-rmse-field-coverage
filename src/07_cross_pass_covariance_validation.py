from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import kv, gamma


# ============================================================
# STAGE 1D — CROSS-PASS COVARIANCE VALIDATION
# ============================================================
#
# PURPOSE
# -------
# Determine whether covariance structures estimated from one
# primary guidance pass generalize to the other primary pass
# of the SAME GNSS / guidance configuration.
#
# Validation directions:
#
#       Pass 1 model  -> Pass 3 empirical residual ACF
#       Pass 3 model  -> Pass 1 empirical residual ACF
#
# WHY THIS MATTERS
# ----------------
# In-sample ACF fitting showed:
#
#       Matérn      : most frequent winner
#       Gaussian    : several passes
#       Exponential : several passes
#
# But a Monte Carlo simulator should not assume that parameters
# estimated from one realization are stable unless the second
# realization supports them.
#
# THIS SCRIPT:
#
#   1. Loads the best AICc covariance model for each primary pass.
#   2. Reconstructs the fitted covariance function.
#   3. Applies Pass 1 parameters to Pass 3 empirical ACF.
#   4. Applies Pass 3 parameters to Pass 1 empirical ACF.
#   5. Computes cross-pass ACF RMSE.
#   6. Compares cross-pass RMSE with own-pass fitted RMSE.
#   7. Quantifies parameter disagreement.
#   8. Reports whether the same model family was selected for
#      both primary passes.
#   9. Does NOT impose an arbitrary acceptance threshold.
#
# No final stochastic generator is selected here.
#
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

BEST_MODEL_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "residual_covariance_best_models_aicc.csv"
)

ACF_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "original_vs_residual_spatial_acf.csv"
)

DECOMP_FILE = (
    PROJECT_DIR
    / "results"
    / "tables"
    / "trend_residual_decomposition_summary.csv"
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
    / "cross_pass_covariance_validation"
)

for directory in [
    TABLES_DIR,
    FIGURES_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


PRIMARY_PASSES = [
    1,
    3
]


# ------------------------------------------------------------
# 2. Covariance functions
# ------------------------------------------------------------

def exponential_correlation(
    h,
    ell
):

    h = np.asarray(
        h,
        dtype=float
    )

    return np.exp(
        -h / ell
    )


def gaussian_correlation(
    h,
    ell
):

    h = np.asarray(
        h,
        dtype=float
    )

    return np.exp(
        -(h / ell) ** 2
    )


def matern_correlation(
    h,
    ell,
    nu
):

    h = np.asarray(
        h,
        dtype=float
    )

    result = np.ones_like(
        h,
        dtype=float
    )

    positive = (
        h > 0
    )

    if not np.any(
        positive
    ):
        return result

    z = (
        np.sqrt(
            2.0 * nu
        )
        * h[
            positive
        ]
        / ell
    )

    coefficient = (
        2.0 ** (
            1.0 - nu
        )
        / gamma(nu)
    )

    with np.errstate(
        all="ignore"
    ):

        values = (
            coefficient
            * (
                z ** nu
            )
            * kv(
                nu,
                z
            )
        )

    result[
        positive
    ] = values

    return result


def evaluate_model(
    model_name,
    lag,
    ell,
    nu
):

    if model_name == "Exponential":

        return exponential_correlation(
            lag,
            ell
        )

    if model_name == "Gaussian":

        return gaussian_correlation(
            lag,
            ell
        )

    if model_name == "Matern":

        return matern_correlation(
            lag,
            ell,
            nu
        )

    raise ValueError(
        f"Unknown covariance model: "
        f"{model_name}"
    )


# ------------------------------------------------------------
# 3. Utility functions
# ------------------------------------------------------------

def rmse(
    observed,
    predicted
):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return np.sqrt(
        np.mean(
            (
                observed
                - predicted
            ) ** 2
        )
    )


def first_positive_lobe(
    lag,
    acf
):
    """
    Return empirical ACF from zero lag to the last positive
    point immediately before the first zero/negative crossing.
    """

    lag = np.asarray(
        lag,
        dtype=float
    )

    acf = np.asarray(
        acf,
        dtype=float
    )

    finite = (
        np.isfinite(
            lag
        )
        & np.isfinite(
            acf
        )
    )

    lag = lag[
        finite
    ]

    acf = acf[
        finite
    ]

    if len(
        acf
    ) < 2:

        return (
            lag,
            acf
        )

    crossing = np.where(
        acf[
            1:
        ]
        <= 0
    )[0]

    if len(
        crossing
    ) > 0:

        end_index = (
            crossing[
                0
            ]
            + 1
        )

    else:

        end_index = len(
            acf
        )

    end_index = max(
        2,
        end_index
    )

    return (
        lag[
            :end_index
        ],
        acf[
            :end_index
        ]
    )


def safe_relative_difference(
    a,
    b
):
    """
    Symmetric relative difference:

        |a-b| / ((|a|+|b|)/2)

    Avoids choosing one pass as the denominator.
    """

    denominator = (
        (
            abs(a)
            + abs(b)
        )
        / 2.0
    )

    if denominator == 0:

        return 0.0

    return (
        abs(
            a - b
        )
        / denominator
    )


# ------------------------------------------------------------
# 4. Load files
# ------------------------------------------------------------

for file_path in [
    BEST_MODEL_FILE,
    ACF_FILE,
    DECOMP_FILE
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
    "STAGE 1D — CROSS-PASS COVARIANCE VALIDATION"
)

print(
    "=" * 90
)


best_df = pd.read_csv(
    BEST_MODEL_FILE
)

acf_df = pd.read_csv(
    ACF_FILE
)

decomp_df = pd.read_csv(
    DECOMP_FILE
)


best_df = best_df[
    best_df[
        "Pass"
    ].isin(
        PRIMARY_PASSES
    )
].copy()


acf_df = acf_df[
    acf_df[
        "Pass"
    ].isin(
        PRIMARY_PASSES
    )
].copy()


decomp_df = decomp_df[
    decomp_df[
        "Pass"
    ].isin(
        PRIMARY_PASSES
    )
].copy()


print(
    f"\nBest-model records: "
    f"{len(best_df)}"
)

print(
    f"Configurations: "
    f"{best_df['Configuration'].nunique()}"
)

print(
    f"Primary passes: "
    f"{sorted(best_df['Pass'].unique())}"
)


# ------------------------------------------------------------
# 5. Validate expected structure
# ------------------------------------------------------------

expected_records = (
    best_df[
        "Configuration"
    ].nunique()
    * 2
)

if len(
    best_df
) != expected_records:

    raise RuntimeError(
        "Expected exactly one best-model record "
        "for Pass 1 and Pass 3 of every configuration."
    )


# ------------------------------------------------------------
# 6. Containers
# ------------------------------------------------------------

cross_rows = []

configuration_rows = []

curve_rows = []


# ------------------------------------------------------------
# 7. Configuration-by-configuration analysis
# ------------------------------------------------------------

configuration_names = (
    best_df[
        [
            "Configuration",
            "Configuration_Name"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "Configuration"
    )
)


for _, config_record in configuration_names.iterrows():

    config = config_record[
        "Configuration"
    ]

    config_name = config_record[
        "Configuration_Name"
    ]


    # --------------------------------------------------------
    # Retrieve Pass 1 and Pass 3 fitted models
    # --------------------------------------------------------

    pass_models = {}

    for pass_number in PRIMARY_PASSES:

        row = best_df[
            (
                best_df[
                    "Configuration"
                ]
                == config
            )
            &
            (
                best_df[
                    "Pass"
                ]
                == pass_number
            )
        ]

        if len(
            row
        ) != 1:

            raise RuntimeError(
                f"Expected one best model for "
                f"{config} Pass {pass_number}"
            )

        row = row.iloc[
            0
        ]

        pass_models[
            pass_number
        ] = {
            "Model":
                row[
                    "Model"
                ],

            "Ell_m":
                float(
                    row[
                        "Ell_m"
                    ]
                ),

            "Nu":
                (
                    float(
                        row[
                            "Nu"
                        ]
                    )
                    if pd.notna(
                        row[
                            "Nu"
                        ]
                    )
                    else np.nan
                ),

            "Own_Fit_RMSE":
                float(
                    row[
                        "Fit_RMSE"
                    ]
                ),

            "AICc":
                float(
                    row[
                        "AICc"
                    ]
                )
        }


    # --------------------------------------------------------
    # Retrieve empirical residual ACFs
    # --------------------------------------------------------

    empirical_acfs = {}

    for pass_number in PRIMARY_PASSES:

        group = acf_df[
            (
                acf_df[
                    "Configuration"
                ]
                == config
            )
            &
            (
                acf_df[
                    "Pass"
                ]
                == pass_number
            )
        ].sort_values(
            "Lag_m"
        )

        lag = group[
            "Lag_m"
        ].to_numpy(
            dtype=float
        )

        empirical = group[
            "Linear_Detrended_Residual_ACF"
        ].to_numpy(
            dtype=float
        )

        (
            positive_lag,
            positive_acf
        ) = first_positive_lobe(
            lag,
            empirical
        )

        empirical_acfs[
            pass_number
        ] = {
            "Lag":
                positive_lag,

            "ACF":
                positive_acf
        }


    # --------------------------------------------------------
    # Pass 1 -> Pass 3 and Pass 3 -> Pass 1
    # --------------------------------------------------------

    directions = [
        (
            1,
            3
        ),
        (
            3,
            1
        )
    ]


    directional_results = {}


    for (
        source_pass,
        target_pass
    ) in directions:

        source_model = pass_models[
            source_pass
        ]

        target_empirical = empirical_acfs[
            target_pass
        ]

        target_lag = target_empirical[
            "Lag"
        ]

        target_acf = target_empirical[
            "ACF"
        ]


        # ----------------------------------------------------
        # Cross-pass prediction using SOURCE parameters
        # ----------------------------------------------------

        cross_prediction = evaluate_model(
            source_model[
                "Model"
            ],
            target_lag,
            source_model[
                "Ell_m"
            ],
            source_model[
                "Nu"
            ]
        )


        cross_rmse = rmse(
            target_acf,
            cross_prediction
        )


        # ----------------------------------------------------
        # Target pass own fitted model over SAME target domain
        # ----------------------------------------------------

        target_model = pass_models[
            target_pass
        ]

        target_own_prediction = evaluate_model(
            target_model[
                "Model"
            ],
            target_lag,
            target_model[
                "Ell_m"
            ],
            target_model[
                "Nu"
            ]
        )


        target_own_rmse_same_domain = rmse(
            target_acf,
            target_own_prediction
        )


        # ----------------------------------------------------
        # Cross / own RMSE ratio
        # ----------------------------------------------------

        if target_own_rmse_same_domain > 0:

            cross_to_own_ratio = (
                cross_rmse
                / target_own_rmse_same_domain
            )

        else:

            cross_to_own_ratio = np.nan


        directional_results[
            (
                source_pass,
                target_pass
            )
        ] = {
            "Cross_RMSE":
                cross_rmse,

            "Target_Own_RMSE":
                target_own_rmse_same_domain,

            "Ratio":
                cross_to_own_ratio
        }


        cross_rows.append(
            {
                "Configuration":
                    config,

                "Configuration_Name":
                    config_name,

                "Source_Pass":
                    source_pass,

                "Target_Pass":
                    target_pass,

                "Source_Model":
                    source_model[
                        "Model"
                    ],

                "Source_Ell_m":
                    source_model[
                        "Ell_m"
                    ],

                "Source_Nu":
                    source_model[
                        "Nu"
                    ],

                "Target_Model":
                    target_model[
                        "Model"
                    ],

                "Target_Ell_m":
                    target_model[
                        "Ell_m"
                    ],

                "Target_Nu":
                    target_model[
                        "Nu"
                    ],

                "Target_Positive_Lobe_Length_m":
                    target_lag[
                        -1
                    ],

                "Target_ACF_Points":
                    len(
                        target_lag
                    ),

                "Cross_Pass_RMSE":
                    cross_rmse,

                "Target_Own_Model_RMSE_Same_Domain":
                    target_own_rmse_same_domain,

                "Cross_to_Own_RMSE_Ratio":
                    cross_to_own_ratio
            }
        )


        for (
            lag_value,
            empirical_value,
            cross_value,
            own_value
        ) in zip(
            target_lag,
            target_acf,
            cross_prediction,
            target_own_prediction
        ):

            curve_rows.append(
                {
                    "Configuration":
                        config,

                    "Configuration_Name":
                        config_name,

                    "Source_Pass":
                        source_pass,

                    "Target_Pass":
                        target_pass,

                    "Lag_m":
                        lag_value,

                    "Target_Empirical_ACF":
                        empirical_value,

                    "Cross_Pass_Prediction":
                        cross_value,

                    "Target_Own_Model_Prediction":
                        own_value
                }
            )


    # --------------------------------------------------------
    # Parameter consistency
    # --------------------------------------------------------

    model1 = pass_models[
        1
    ]

    model3 = pass_models[
        3
    ]


    same_family = (
        model1[
            "Model"
        ]
        == model3[
            "Model"
        ]
    )


    ell_symmetric_difference = safe_relative_difference(
        model1[
            "Ell_m"
        ],
        model3[
            "Ell_m"
        ]
    )


    if (
        model1[
            "Model"
        ]
        == "Matern"
        and model3[
            "Model"
        ]
        == "Matern"
        and np.isfinite(
            model1[
                "Nu"
            ]
        )
        and np.isfinite(
            model3[
                "Nu"
            ]
        )
    ):

        nu_symmetric_difference = safe_relative_difference(
            model1[
                "Nu"
            ],
            model3[
                "Nu"
            ]
        )

    else:

        nu_symmetric_difference = np.nan


    # --------------------------------------------------------
    # Residual SD pass-to-pass variation
    # --------------------------------------------------------

    sd_values = {}

    for pass_number in PRIMARY_PASSES:

        sd_row = decomp_df[
            (
                decomp_df[
                    "Configuration"
                ]
                == config
            )
            &
            (
                decomp_df[
                    "Pass"
                ]
                == pass_number
            )
        ]

        if len(
            sd_row
        ) != 1:

            raise RuntimeError(
                f"Missing residual SD for "
                f"{config} Pass {pass_number}"
            )

        sd_values[
            pass_number
        ] = float(
            sd_row.iloc[
                0
            ][
                "Residual_SD_m"
            ]
        )


    sd_symmetric_difference = safe_relative_difference(
        sd_values[
            1
        ],
        sd_values[
            3
        ]
    )


    # --------------------------------------------------------
    # Configuration summary
    # --------------------------------------------------------

    ratio_1_to_3 = directional_results[
        (
            1,
            3
        )
    ][
        "Ratio"
    ]

    ratio_3_to_1 = directional_results[
        (
            3,
            1
        )
    ][
        "Ratio"
    ]


    configuration_rows.append(
        {
            "Configuration":
                config,

            "Configuration_Name":
                config_name,

            "Pass1_Model":
                model1[
                    "Model"
                ],

            "Pass3_Model":
                model3[
                    "Model"
                ],

            "Same_Model_Family":
                same_family,

            "Pass1_Ell_m":
                model1[
                    "Ell_m"
                ],

            "Pass3_Ell_m":
                model3[
                    "Ell_m"
                ],

            "Ell_Symmetric_Relative_Difference":
                ell_symmetric_difference,

            "Pass1_Nu":
                model1[
                    "Nu"
                ],

            "Pass3_Nu":
                model3[
                    "Nu"
                ],

            "Nu_Symmetric_Relative_Difference":
                nu_symmetric_difference,

            "Pass1_Residual_SD_m":
                sd_values[
                    1
                ],

            "Pass3_Residual_SD_m":
                sd_values[
                    3
                ],

            "Residual_SD_Symmetric_Relative_Difference":
                sd_symmetric_difference,

            "Pass1_to_Pass3_Cross_RMSE":
                directional_results[
                    (
                        1,
                        3
                    )
                ][
                    "Cross_RMSE"
                ],

            "Pass1_to_Pass3_RMSE_Ratio":
                ratio_1_to_3,

            "Pass3_to_Pass1_Cross_RMSE":
                directional_results[
                    (
                        3,
                        1
                    )
                ][
                    "Cross_RMSE"
                ],

            "Pass3_to_Pass1_RMSE_Ratio":
                ratio_3_to_1,

            "Mean_Cross_to_Own_RMSE_Ratio":
                np.mean(
                    [
                        ratio_1_to_3,
                        ratio_3_to_1
                    ]
                )
        }
    )


# ------------------------------------------------------------
# 8. DataFrames
# ------------------------------------------------------------

cross_df = pd.DataFrame(
    cross_rows
)

config_df = pd.DataFrame(
    configuration_rows
)

curve_df = pd.DataFrame(
    curve_rows
)


# ------------------------------------------------------------
# 9. Save tables
# ------------------------------------------------------------

cross_file = (
    TABLES_DIR
    / "cross_pass_covariance_validation.csv"
)

config_file = (
    TABLES_DIR
    / "cross_pass_covariance_configuration_summary.csv"
)

curve_file = (
    TABLES_DIR
    / "cross_pass_covariance_validation_curves.csv"
)


cross_df.to_csv(
    cross_file,
    index=False
)

config_df.to_csv(
    config_file,
    index=False
)

curve_df.to_csv(
    curve_file,
    index=False
)


# ------------------------------------------------------------
# 10. Figures
# ------------------------------------------------------------

for (
    config,
    config_name,
    source_pass,
    target_pass
), group in curve_df.groupby(
    [
        "Configuration",
        "Configuration_Name",
        "Source_Pass",
        "Target_Pass"
    ],
    sort=True
):

    group = group.sort_values(
        "Lag_m"
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        group[
            "Lag_m"
        ],
        group[
            "Target_Empirical_ACF"
        ],
        linewidth=2,
        label=(
            f"Pass {target_pass} empirical ACF"
        )
    )

    plt.plot(
        group[
            "Lag_m"
        ],
        group[
            "Cross_Pass_Prediction"
        ],
        linewidth=1.4,
        label=(
            f"Pass {source_pass} model "
            f"applied to Pass {target_pass}"
        )
    )

    plt.plot(
        group[
            "Lag_m"
        ],
        group[
            "Target_Own_Model_Prediction"
        ],
        linewidth=1.4,
        linestyle="--",
        label=(
            f"Pass {target_pass} own fitted model"
        )
    )

    plt.axhline(
        0,
        linestyle=":",
        linewidth=1
    )

    plt.xlabel(
        "Spatial lag (m)"
    )

    plt.ylabel(
        "Residual autocorrelation"
    )

    plt.title(
        f"{config}: Pass {source_pass} → Pass {target_pass}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / (
            f"{config}_"
            f"P{source_pass}_to_P{target_pass}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 11. Same-family counts
# ------------------------------------------------------------

same_family_count = int(
    config_df[
        "Same_Model_Family"
    ].sum()
)

different_family_count = (
    len(
        config_df
    )
    - same_family_count
)


# ------------------------------------------------------------
# 12. Family transition table
# ------------------------------------------------------------

family_transition = pd.crosstab(
    config_df[
        "Pass1_Model"
    ],
    config_df[
        "Pass3_Model"
    ],
    margins=True
)


family_transition_file = (
    TABLES_DIR
    / "cross_pass_model_family_transition.csv"
)


family_transition.to_csv(
    family_transition_file
)


# ------------------------------------------------------------
# 13. Ranking by cross-pass instability
# ------------------------------------------------------------

instability_df = (
    config_df
    .sort_values(
        "Mean_Cross_to_Own_RMSE_Ratio",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


instability_file = (
    TABLES_DIR
    / "cross_pass_covariance_instability_ranking.csv"
)


instability_df.to_csv(
    instability_file,
    index=False
)


# ------------------------------------------------------------
# 14. Terminal formatting
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
    100
)


# ------------------------------------------------------------
# 15. Directional results
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DIRECTIONAL CROSS-PASS VALIDATION"
)

print(
    "=" * 90
)


print(
    cross_df[
        [
            "Configuration",
            "Source_Pass",
            "Target_Pass",
            "Source_Model",
            "Target_Model",
            "Cross_Pass_RMSE",
            "Target_Own_Model_RMSE_Same_Domain",
            "Cross_to_Own_RMSE_Ratio"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 16. Configuration consistency
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "CONFIGURATION-LEVEL CROSS-PASS CONSISTENCY"
)

print(
    "=" * 90
)


print(
    config_df[
        [
            "Configuration",

            "Pass1_Model",
            "Pass3_Model",

            "Same_Model_Family",

            "Pass1_Ell_m",
            "Pass3_Ell_m",

            "Ell_Symmetric_Relative_Difference",

            "Pass1_Residual_SD_m",
            "Pass3_Residual_SD_m",

            "Residual_SD_Symmetric_Relative_Difference",

            "Pass1_to_Pass3_RMSE_Ratio",
            "Pass3_to_Pass1_RMSE_Ratio",

            "Mean_Cross_to_Own_RMSE_Ratio"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 17. Dataset-wide summary
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "DATASET-WIDE CROSS-PASS SUMMARY"
)

print(
    "=" * 90
)


print(
    f"\nSame best model family in Pass 1 and Pass 3:"
)

print(
    f"  {same_family_count} / "
    f"{len(config_df)} configurations"
)

print(
    f"\nDifferent best model family:"
)

print(
    f"  {different_family_count} / "
    f"{len(config_df)} configurations"
)


cross_ratios = cross_df[
    "Cross_to_Own_RMSE_Ratio"
].replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
).dropna()


print(
    "\nCross-pass / target-own ACF-fit RMSE ratio:"
)

print(
    f"  min    = "
    f"{cross_ratios.min():.4f}"
)

print(
    f"  median = "
    f"{cross_ratios.median():.4f}"
)

print(
    f"  max    = "
    f"{cross_ratios.max():.4f}"
)


ell_difference = config_df[
    "Ell_Symmetric_Relative_Difference"
].dropna()


print(
    "\nPass-to-pass symmetric relative difference in ell:"
)

print(
    f"  min    = "
    f"{ell_difference.min():.4f}"
)

print(
    f"  median = "
    f"{ell_difference.median():.4f}"
)

print(
    f"  max    = "
    f"{ell_difference.max():.4f}"
)


sd_difference = config_df[
    "Residual_SD_Symmetric_Relative_Difference"
].dropna()


print(
    "\nPass-to-pass symmetric relative difference in residual SD:"
)

print(
    f"  min    = "
    f"{sd_difference.min():.4f}"
)

print(
    f"  median = "
    f"{sd_difference.median():.4f}"
)

print(
    f"  max    = "
    f"{sd_difference.max():.4f}"
)


# ------------------------------------------------------------
# 18. Model transition table
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "PASS 1 → PASS 3 MODEL-FAMILY TRANSITIONS"
)

print(
    "=" * 90
)

print(
    family_transition.to_string()
)


# ------------------------------------------------------------
# 19. Instability ranking
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 90
)

print(
    "CONFIGURATIONS RANKED BY CROSS-PASS MODEL INSTABILITY"
)

print(
    "=" * 90
)


print(
    instability_df[
        [
            "Configuration",
            "Pass1_Model",
            "Pass3_Model",
            "Mean_Cross_to_Own_RMSE_Ratio",
            "Ell_Symmetric_Relative_Difference",
            "Residual_SD_Symmetric_Relative_Difference"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 20. Output paths
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
    f"\nDirectional validation:\n"
    f"{cross_file}"
)

print(
    f"\nConfiguration summary:\n"
    f"{config_file}"
)

print(
    f"\nValidation curves:\n"
    f"{curve_file}"
)

print(
    f"\nModel-family transition table:\n"
    f"{family_transition_file}"
)

print(
    f"\nInstability ranking:\n"
    f"{instability_file}"
)

print(
    f"\nFigures:\n"
    f"{FIGURES_DIR}"
)


print(
    "\n"
    + "=" * 90
)

print(
    "STAGE 1D COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nNo arbitrary cross-pass acceptance threshold was used."
)

print(
    "The results quantify how much covariance structure varies "
    "between independent primary passes of the same configuration."
)

print(
    "\nThe next decision will be whether the Monte Carlo generator "
    "should use fixed configuration-specific covariance parameters, "
    "a distribution/ensemble of empirically observed structures, "
    "or a nonparametric structure-preserving benchmark."
)