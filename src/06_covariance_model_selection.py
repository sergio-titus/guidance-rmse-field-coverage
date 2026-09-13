from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.special import kv, gamma


# ============================================================
# STAGE 1C — RESIDUAL COVARIANCE MODEL-FAMILY SELECTION
# ============================================================
#
# PURPOSE
# -------
# Fit candidate stationary covariance/correlation families to
# the LINEAR-DETRENDED residual spatial ACF.
#
# Primary calibration:
#       Pass 1 and Pass 3 only
#
# Candidate normalized correlation families:
#
#   1. Exponential
#          rho(h) = exp(-h / ell)
#
#   2. Gaussian
#          rho(h) = exp(-(h / ell)^2)
#
#   3. Matérn
#          rho(h) =
#          2^(1-nu)/Gamma(nu)
#          * (sqrt(2nu) h / ell)^nu
#          * K_nu(sqrt(2nu) h / ell)
#
# FITTING DOMAIN
# --------------
# The empirical residual ACF is fitted from lag zero through the
# first zero crossing.
#
# This avoids forcing a strictly positive covariance family
# through later empirical negative/oscillatory lobes.
#
# MODEL COMPARISON
# ----------------
# Models are compared using:
#
#   - RMSE of ACF fit
#   - SSE
#   - AICc
#   - BIC
#
# AICc/BIC here are comparative criteria based on Gaussian
# residuals of the ACF-fit errors. They are used for candidate
# ranking, not as a claim that empirical ACF ordinates are
# statistically independent observations.
#
# IMPORTANT
# ---------
# No model is selected globally in advance.
# Each primary pass is fitted independently.
#
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

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
    / "covariance_model_selection"
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
# 2. Correlation functions
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
        * h[positive]
        / ell
    )

    with np.errstate(
        all="ignore"
    ):

        coefficient = (
            2.0 ** (
                1.0 - nu
            )
            / gamma(nu)
        )

        values = (
            coefficient
            * (z ** nu)
            * kv(
                nu,
                z
            )
        )

    result[
        positive
    ] = values

    return result


# ------------------------------------------------------------
# 3. Information criteria
# ------------------------------------------------------------

def information_criteria(
    residuals,
    k
):

    residuals = np.asarray(
        residuals,
        dtype=float
    )

    n = len(
        residuals
    )

    sse = np.sum(
        residuals ** 2
    )

    sse_safe = max(
        sse,
        np.finfo(float).tiny
    )

    aic = (
        n
        * np.log(
            sse_safe / n
        )
        + 2 * k
    )

    if (
        n - k - 1
    ) > 0:

        aicc = (
            aic
            + (
                2
                * k
                * (
                    k + 1
                )
                / (
                    n
                    - k
                    - 1
                )
            )
        )

    else:

        aicc = np.inf

    bic = (
        n
        * np.log(
            sse_safe / n
        )
        + k
        * np.log(n)
    )

    rmse = np.sqrt(
        sse / n
    )

    return (
        sse,
        rmse,
        aic,
        aicc,
        bic
    )


# ------------------------------------------------------------
# 4. Fit functions
# ------------------------------------------------------------

def fit_exponential(
    h,
    empirical
):

    positive_h = h[
        h > 0
    ]

    initial_ell = (
        np.median(
            positive_h
        )
        if len(
            positive_h
        ) > 0
        else 1.0
    )

    initial_ell = max(
        initial_ell,
        0.2
    )

    def residual_function(
        parameters
    ):

        log_ell = parameters[
            0
        ]

        ell = np.exp(
            log_ell
        )

        predicted = exponential_correlation(
            h,
            ell
        )

        return (
            predicted
            - empirical
        )

    result = least_squares(
        residual_function,
        x0=[
            np.log(
                initial_ell
            )
        ],
        max_nfev=10000
    )

    ell = np.exp(
        result.x[
            0
        ]
    )

    predicted = exponential_correlation(
        h,
        ell
    )

    residuals = (
        empirical
        - predicted
    )

    return (
        ell,
        np.nan,
        predicted,
        residuals,
        result.success
    )


def fit_gaussian(
    h,
    empirical
):

    positive_h = h[
        h > 0
    ]

    initial_ell = (
        np.median(
            positive_h
        )
        if len(
            positive_h
        ) > 0
        else 1.0
    )

    initial_ell = max(
        initial_ell,
        0.2
    )

    def residual_function(
        parameters
    ):

        log_ell = parameters[
            0
        ]

        ell = np.exp(
            log_ell
        )

        predicted = gaussian_correlation(
            h,
            ell
        )

        return (
            predicted
            - empirical
        )

    result = least_squares(
        residual_function,
        x0=[
            np.log(
                initial_ell
            )
        ],
        max_nfev=10000
    )

    ell = np.exp(
        result.x[
            0
        ]
    )

    predicted = gaussian_correlation(
        h,
        ell
    )

    residuals = (
        empirical
        - predicted
    )

    return (
        ell,
        np.nan,
        predicted,
        residuals,
        result.success
    )


def fit_matern(
    h,
    empirical
):

    positive_h = h[
        h > 0
    ]

    initial_ell = (
        np.median(
            positive_h
        )
        if len(
            positive_h
        ) > 0
        else 1.0
    )

    initial_ell = max(
        initial_ell,
        0.2
    )

    # Multiple starting values are used only to reduce
    # sensitivity to local optimization.
    #
    # These are NOT fixed model assumptions.
    initial_nu_values = [
        0.5,
        1.5,
        2.5,
        5.0
    ]

    best_result = None
    best_sse = np.inf

    def residual_function(
        parameters
    ):

        log_ell = parameters[
            0
        ]

        log_nu = parameters[
            1
        ]

        ell = np.exp(
            log_ell
        )

        nu = np.exp(
            log_nu
        )

        predicted = matern_correlation(
            h,
            ell,
            nu
        )

        if not np.all(
            np.isfinite(
                predicted
            )
        ):

            return np.full_like(
                empirical,
                1e6
            )

        return (
            predicted
            - empirical
        )

    for initial_nu in initial_nu_values:

        result = least_squares(
            residual_function,
            x0=[
                np.log(
                    initial_ell
                ),
                np.log(
                    initial_nu
                )
            ],
            bounds=(
                [
                    np.log(
                        1e-3
                    ),
                    np.log(
                        0.05
                    )
                ],
                [
                    np.log(
                        1e4
                    ),
                    np.log(
                        50.0
                    )
                ]
            ),
            max_nfev=20000
        )

        current_residual = residual_function(
            result.x
        )

        current_sse = np.sum(
            current_residual ** 2
        )

        if (
            np.isfinite(
                current_sse
            )
            and current_sse
            < best_sse
        ):

            best_sse = current_sse
            best_result = result

    if best_result is None:

        return (
            np.nan,
            np.nan,
            np.full_like(
                empirical,
                np.nan
            ),
            np.full_like(
                empirical,
                np.nan
            ),
            False
        )

    ell = np.exp(
        best_result.x[
            0
        ]
    )

    nu = np.exp(
        best_result.x[
            1
        ]
    )

    predicted = matern_correlation(
        h,
        ell,
        nu
    )

    residuals = (
        empirical
        - predicted
    )

    return (
        ell,
        nu,
        predicted,
        residuals,
        best_result.success
    )


# ------------------------------------------------------------
# 5. Load inputs
# ------------------------------------------------------------

if not ACF_FILE.exists():

    raise FileNotFoundError(
        f"ACF file not found:\n"
        f"{ACF_FILE}"
    )


if not DECOMP_FILE.exists():

    raise FileNotFoundError(
        f"Decomposition file not found:\n"
        f"{DECOMP_FILE}"
    )


print(
    "=" * 90
)

print(
    "STAGE 1C — RESIDUAL COVARIANCE MODEL SELECTION"
)

print(
    "=" * 90
)


acf_df = pd.read_csv(
    ACF_FILE
)

decomp_df = pd.read_csv(
    DECOMP_FILE
)


print(
    f"\nLoaded residual ACF:"
    f"\n{ACF_FILE}"
)

print(
    f"\nRows: "
    f"{len(acf_df):,}"
)


# ------------------------------------------------------------
# 6. Primary passes only
# ------------------------------------------------------------

acf_primary = acf_df[
    acf_df[
        "Pass"
    ].isin(
        PRIMARY_PASSES
    )
].copy()


decomp_primary = decomp_df[
    decomp_df[
        "Pass"
    ].isin(
        PRIMARY_PASSES
    )
].copy()


# ------------------------------------------------------------
# 7. Containers
# ------------------------------------------------------------

fit_rows = []

curve_rows = []


# ------------------------------------------------------------
# 8. Fit every primary trajectory
# ------------------------------------------------------------

group_columns = [
    "Configuration",
    "Configuration_Name",
    "Pass"
]


for (
    config,
    config_name,
    pass_number
), group in acf_primary.groupby(
    group_columns,
    sort=True
):

    group = group.sort_values(
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


    # --------------------------------------------------------
    # Find first empirical zero crossing
    # --------------------------------------------------------

    negative_indices = np.where(
        empirical[
            1:
        ]
        <= 0
    )[0]

    if len(
        negative_indices
    ) > 0:

        first_zero_index = (
            negative_indices[
                0
            ]
            + 1
        )

    else:

        first_zero_index = (
            len(
                empirical
            )
            - 1
        )


    # Include lag zero and the last positive ACF point.
    #
    # The zero/negative crossing itself is excluded from fitting
    # because the candidate families are strictly positive.
    fit_end_index = max(
        2,
        first_zero_index
    )

    h_fit = lag[
        :fit_end_index
    ]

    empirical_fit = empirical[
        :fit_end_index
    ]


    finite = (
        np.isfinite(
            h_fit
        )
        & np.isfinite(
            empirical_fit
        )
    )

    h_fit = h_fit[
        finite
    ]

    empirical_fit = empirical_fit[
        finite
    ]


    if len(
        h_fit
    ) < 5:

        warnings.warn(
            f"Too few ACF points for "
            f"{config} Pass {pass_number}"
        )

        continue


    fit_length_m = h_fit[
        -1
    ]


    # --------------------------------------------------------
    # Residual SD from decomposition table
    # --------------------------------------------------------

    match = decomp_primary[
        (
            decomp_primary[
                "Configuration"
            ]
            == config
        )
        &
        (
            decomp_primary[
                "Pass"
            ]
            == pass_number
        )
    ]

    if len(
        match
    ) != 1:

        raise RuntimeError(
            f"Could not uniquely match decomposition "
            f"record for {config} Pass {pass_number}"
        )

    residual_sd = float(
        match.iloc[
            0
        ][
            "Residual_SD_m"
        ]
    )


    # --------------------------------------------------------
    # Candidate fits
    # --------------------------------------------------------

    candidate_results = []


    # Exponential
    (
        ell,
        nu,
        predicted,
        residuals,
        success
    ) = fit_exponential(
        h_fit,
        empirical_fit
    )

    (
        sse,
        fit_rmse,
        aic,
        aicc,
        bic
    ) = information_criteria(
        residuals,
        k=1
    )

    candidate_results.append(
        {
            "Model":
                "Exponential",

            "Ell_m":
                ell,

            "Nu":
                nu,

            "Predicted":
                predicted,

            "Residuals":
                residuals,

            "Success":
                success,

            "SSE":
                sse,

            "Fit_RMSE":
                fit_rmse,

            "AIC":
                aic,

            "AICc":
                aicc,

            "BIC":
                bic,

            "K":
                1
        }
    )


    # Gaussian
    (
        ell,
        nu,
        predicted,
        residuals,
        success
    ) = fit_gaussian(
        h_fit,
        empirical_fit
    )

    (
        sse,
        fit_rmse,
        aic,
        aicc,
        bic
    ) = information_criteria(
        residuals,
        k=1
    )

    candidate_results.append(
        {
            "Model":
                "Gaussian",

            "Ell_m":
                ell,

            "Nu":
                nu,

            "Predicted":
                predicted,

            "Residuals":
                residuals,

            "Success":
                success,

            "SSE":
                sse,

            "Fit_RMSE":
                fit_rmse,

            "AIC":
                aic,

            "AICc":
                aicc,

            "BIC":
                bic,

            "K":
                1
        }
    )


    # Matérn
    (
        ell,
        nu,
        predicted,
        residuals,
        success
    ) = fit_matern(
        h_fit,
        empirical_fit
    )

    (
        sse,
        fit_rmse,
        aic,
        aicc,
        bic
    ) = information_criteria(
        residuals,
        k=2
    )

    candidate_results.append(
        {
            "Model":
                "Matern",

            "Ell_m":
                ell,

            "Nu":
                nu,

            "Predicted":
                predicted,

            "Residuals":
                residuals,

            "Success":
                success,

            "SSE":
                sse,

            "Fit_RMSE":
                fit_rmse,

            "AIC":
                aic,

            "AICc":
                aicc,

            "BIC":
                bic,

            "K":
                2
        }
    )


    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    valid_aicc = [
        item[
            "AICc"
        ]
        for item
        in candidate_results
        if np.isfinite(
            item[
                "AICc"
            ]
        )
    ]

    best_aicc = min(
        valid_aicc
    )


    valid_bic = [
        item[
            "BIC"
        ]
        for item
        in candidate_results
        if np.isfinite(
            item[
                "BIC"
            ]
        )
    ]

    best_bic = min(
        valid_bic
    )


    for item in candidate_results:

        delta_aicc = (
            item[
                "AICc"
            ]
            - best_aicc
        )

        delta_bic = (
            item[
                "BIC"
            ]
            - best_bic
        )


        fit_rows.append(
            {
                "Configuration":
                    config,

                "Configuration_Name":
                    config_name,

                "Pass":
                    pass_number,

                "Residual_SD_m":
                    residual_sd,

                "Fit_Length_m":
                    fit_length_m,

                "N_ACF_Fit_Points":
                    len(
                        h_fit
                    ),

                "Model":
                    item[
                        "Model"
                    ],

                "Ell_m":
                    item[
                        "Ell_m"
                    ],

                "Nu":
                    item[
                        "Nu"
                    ],

                "Optimization_Success":
                    item[
                        "Success"
                    ],

                "SSE":
                    item[
                        "SSE"
                    ],

                "Fit_RMSE":
                    item[
                        "Fit_RMSE"
                    ],

                "AIC":
                    item[
                        "AIC"
                    ],

                "AICc":
                    item[
                        "AICc"
                    ],

                "Delta_AICc":
                    delta_aicc,

                "BIC":
                    item[
                        "BIC"
                    ],

                "Delta_BIC":
                    delta_bic
            }
        )


        for (
            h_value,
            empirical_value,
            predicted_value
        ) in zip(
            h_fit,
            empirical_fit,
            item[
                "Predicted"
            ]
        ):

            curve_rows.append(
                {
                    "Configuration":
                        config,

                    "Configuration_Name":
                        config_name,

                    "Pass":
                        pass_number,

                    "Lag_m":
                        h_value,

                    "Empirical_Residual_ACF":
                        empirical_value,

                    "Model":
                        item[
                            "Model"
                        ],

                    "Fitted_ACF":
                        predicted_value
                }
            )


# ------------------------------------------------------------
# 9. DataFrames
# ------------------------------------------------------------

fit_df = pd.DataFrame(
    fit_rows
)

curve_df = pd.DataFrame(
    curve_rows
)


# ------------------------------------------------------------
# 10. Winner flags
# ------------------------------------------------------------

fit_df[
    "Best_AICc"
] = False

fit_df[
    "Best_BIC"
] = False


for (
    config,
    pass_number
), group in fit_df.groupby(
    [
        "Configuration",
        "Pass"
    ]
):

    best_aicc_index = group[
        "AICc"
    ].idxmin()

    best_bic_index = group[
        "BIC"
    ].idxmin()

    fit_df.loc[
        best_aicc_index,
        "Best_AICc"
    ] = True

    fit_df.loc[
        best_bic_index,
        "Best_BIC"
    ] = True


# ------------------------------------------------------------
# 11. Save detailed fits
# ------------------------------------------------------------

fit_file = (
    TABLES_DIR
    / "residual_covariance_candidate_fits.csv"
)

curve_file = (
    TABLES_DIR
    / "residual_covariance_fitted_curves.csv"
)


fit_df.to_csv(
    fit_file,
    index=False
)

curve_df.to_csv(
    curve_file,
    index=False
)


# ------------------------------------------------------------
# 12. Best model table
# ------------------------------------------------------------

best_aicc_df = (
    fit_df[
        fit_df[
            "Best_AICc"
        ]
    ]
    .copy()
    .sort_values(
        [
            "Configuration",
            "Pass"
        ]
    )
)


best_file = (
    TABLES_DIR
    / "residual_covariance_best_models_aicc.csv"
)


best_aicc_df.to_csv(
    best_file,
    index=False
)


# ------------------------------------------------------------
# 13. Model winner counts
# ------------------------------------------------------------

winner_counts = (
    best_aicc_df[
        "Model"
    ]
    .value_counts()
    .rename_axis(
        "Model"
    )
    .reset_index(
        name="AICc_Wins"
    )
)


bic_winners = (
    fit_df[
        fit_df[
            "Best_BIC"
        ]
    ][
        "Model"
    ]
    .value_counts()
)


winner_counts[
    "BIC_Wins"
] = (
    winner_counts[
        "Model"
    ]
    .map(
        bic_winners
    )
    .fillna(0)
    .astype(int)
)


winner_file = (
    TABLES_DIR
    / "residual_covariance_model_winner_counts.csv"
)


winner_counts.to_csv(
    winner_file,
    index=False
)


# ------------------------------------------------------------
# 14. Evidence-strength table
# ------------------------------------------------------------

evidence_rows = []


for (
    config,
    config_name,
    pass_number
), group in fit_df.groupby(
    [
        "Configuration",
        "Configuration_Name",
        "Pass"
    ],
    sort=True
):

    ordered = group.sort_values(
        "AICc"
    )

    first = ordered.iloc[
        0
    ]

    second = ordered.iloc[
        1
    ]

    delta_second = (
        second[
            "AICc"
        ]
        - first[
            "AICc"
        ]
    )


    if delta_second < 2:

        evidence = (
            "Indistinguishable"
        )

    elif delta_second < 6:

        evidence = (
            "Moderate"
        )

    elif delta_second < 10:

        evidence = (
            "Strong"
        )

    else:

        evidence = (
            "Very strong"
        )


    evidence_rows.append(
        {
            "Configuration":
                config,

            "Configuration_Name":
                config_name,

            "Pass":
                pass_number,

            "Best_Model":
                first[
                    "Model"
                ],

            "Second_Model":
                second[
                    "Model"
                ],

            "Delta_AICc_Second_vs_Best":
                delta_second,

            "Evidence_Category":
                evidence,

            "Best_Ell_m":
                first[
                    "Ell_m"
                ],

            "Best_Nu":
                first[
                    "Nu"
                ],

            "Best_Fit_RMSE":
                first[
                    "Fit_RMSE"
                ]
        }
    )


evidence_df = pd.DataFrame(
    evidence_rows
)


evidence_file = (
    TABLES_DIR
    / "residual_covariance_model_evidence.csv"
)


evidence_df.to_csv(
    evidence_file,
    index=False
)


# ------------------------------------------------------------
# 15. Figures
# ------------------------------------------------------------

for (
    config,
    config_name,
    pass_number
), group in curve_df.groupby(
    [
        "Configuration",
        "Configuration_Name",
        "Pass"
    ],
    sort=True
):

    plt.figure(
        figsize=(10, 5)
    )


    empirical_curve = (
        group[
            [
                "Lag_m",
                "Empirical_Residual_ACF"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "Lag_m"
        )
    )


    plt.plot(
        empirical_curve[
            "Lag_m"
        ],
        empirical_curve[
            "Empirical_Residual_ACF"
        ],
        linewidth=2,
        label="Empirical residual ACF"
    )


    for model_name in [
        "Exponential",
        "Gaussian",
        "Matern"
    ]:

        model_curve = group[
            group[
                "Model"
            ]
            == model_name
        ].sort_values(
            "Lag_m"
        )

        plt.plot(
            model_curve[
                "Lag_m"
            ],
            model_curve[
                "Fitted_ACF"
            ],
            linewidth=1.3,
            label=model_name
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
        "Correlation"
    )

    plt.title(
        f"{config} Pass {pass_number} — "
        f"residual covariance-family fit"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / (
            f"{config}_Pass{pass_number}_"
            f"covariance_fit.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 16. Terminal output
# ------------------------------------------------------------

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    260
)

pd.set_option(
    "display.max_rows",
    100
)


print(
    "\n"
    + "=" * 90
)

print(
    "BEST MODEL FOR EACH PRIMARY PASS — AICc"
)

print(
    "=" * 90
)


display_columns = [
    "Configuration",
    "Pass",
    "Model",
    "Ell_m",
    "Nu",
    "Fit_RMSE",
    "AICc",
    "BIC"
]


print(
    best_aicc_df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 90
)

print(
    "MODEL WINNER COUNTS"
)

print(
    "=" * 90
)


print(
    winner_counts.to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 90
)

print(
    "MODEL-SELECTION EVIDENCE"
)

print(
    "=" * 90
)


print(
    evidence_df[
        [
            "Configuration",
            "Pass",
            "Best_Model",
            "Second_Model",
            "Delta_AICc_Second_vs_Best",
            "Evidence_Category",
            "Best_Ell_m",
            "Best_Nu",
            "Best_Fit_RMSE"
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 17. Matérn parameter diagnostics
# ------------------------------------------------------------

matern_winners = best_aicc_df[
    best_aicc_df[
        "Model"
    ]
    == "Matern"
]


print(
    "\n"
    + "=" * 90
)

print(
    "MATERN-WINNER PARAMETER SUMMARY"
)

print(
    "=" * 90
)


if len(
    matern_winners
) > 0:

    print(
        f"\nNumber of Matérn AICc wins: "
        f"{len(matern_winners)}"
    )

    print(
        "\nFitted Matérn nu:"
    )

    print(
        f"  min    = "
        f"{matern_winners['Nu'].min():.4f}"
    )

    print(
        f"  median = "
        f"{matern_winners['Nu'].median():.4f}"
    )

    print(
        f"  max    = "
        f"{matern_winners['Nu'].max():.4f}"
    )

    print(
        "\nFitted Matérn ell:"
    )

    print(
        f"  min    = "
        f"{matern_winners['Ell_m'].min():.4f} m"
    )

    print(
        f"  median = "
        f"{matern_winners['Ell_m'].median():.4f} m"
    )

    print(
        f"  max    = "
        f"{matern_winners['Ell_m'].max():.4f} m"
    )

else:

    print(
        "\nNo Matérn winner."
    )


# ------------------------------------------------------------
# 18. Optimization check
# ------------------------------------------------------------

failed = fit_df[
    ~fit_df[
        "Optimization_Success"
    ]
]


print(
    "\n"
    + "=" * 90
)

print(
    "OPTIMIZATION CHECK"
)

print(
    "=" * 90
)


print(
    f"\nTotal candidate fits: "
    f"{len(fit_df)}"
)

print(
    f"Failed optimizer reports: "
    f"{len(failed)}"
)


# ------------------------------------------------------------
# 19. Output files
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
    f"\nDetailed candidate fits:\n"
    f"{fit_file}"
)

print(
    f"\nFitted ACF curves:\n"
    f"{curve_file}"
)

print(
    f"\nBest models:\n"
    f"{best_file}"
)

print(
    f"\nWinner counts:\n"
    f"{winner_file}"
)

print(
    f"\nEvidence table:\n"
    f"{evidence_file}"
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
    "STAGE 1C COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nNo final simulation model has been imposed."
)

print(
    "The next stage will validate the selected covariance "
    "families against held empirical properties before Monte "
    "Carlo coverage simulation."
)