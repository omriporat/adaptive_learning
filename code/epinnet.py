from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


SINGLETS_RATIO_THRESHOLD = 0.5


def base_mlp(oh_funclib_df, y):
    mlp_reg_oh = MLPRegressor(
        hidden_layer_sizes=(128),
        activation="relu",
        learning_rate="invscaling",
        solver="lbfgs",
        random_state=42,
        verbose=True,
        max_iter=20000,
    )
    mlp_reg_oh.fit(oh_funclib_df, y)

    pred_weights = pd.DataFrame(
        np.eye(len(oh_funclib_df.columns)), columns=oh_funclib_df.columns
    )
    predictions = mlp_reg_oh.predict(pred_weights)
    results = pd.DataFrame(
        [i.split("_") for i in oh_funclib_df.columns], columns=["position", "AA"]
    )
    results["prediction"] = predictions
    results_sorted = results.sort_values("prediction", ascending=False, inplace=False)

    return mlp_reg_oh, results_sorted


def mutation_predictability(oh_df: pd.DataFrame, mut: str, y: pd.Series):
    mut_df = oh_df[oh_df[mut] == 1]
    non_mut_df = oh_df[oh_df[mut] == 0]

    mlp_reg_oh = MLPRegressor(
        hidden_layer_sizes=(128),
        activation="relu",
        learning_rate="invscaling",
        solver="lbfgs",
        random_state=42,
        verbose=True,
        max_iter=20000,
    )
    mlp_reg_oh.fit(non_mut_df, y[non_mut_df.index])
    mut_pred = mlp_reg_oh.predict(mut_df)

    # compute loss
    mut_mse = mean_squared_error(y[mut_df.index], mut_pred)
    spearman_corr, _ = spearmanr(y[mut_df.index], mut_pred)

    return mut_mse, spearman_corr


def position_pair_epistasis(
    epinnet_results_df: pd.DataFrame,
    pos1: str,
    pos2: str,
    mlp: MLPRegressor,
    empty_oh_series: pd.Series,
):
    mut1_df = epinnet_results_df[epinnet_results_df["position"] == pos1]
    mut2_df = epinnet_results_df[epinnet_results_df["position"] == pos2]

    pos1_aas = []
    pos2_aas = []
    pairs_list = []

    for i, row1 in mut1_df.iterrows():
        for j, row2 in mut2_df.iterrows():
            combo_series = empty_oh_series.copy()
            combo_series[f"{pos1}_{row1['AA']}"] = 1
            pos1_aas.append(row1["AA"])
            combo_series[f"{pos2}_{row2['AA']}"] = 1
            pos2_aas.append(row2["AA"])
            pairs_list.append(combo_series)

    pairs_df = pd.DataFrame(pairs_list)
    pair_predictions = mlp.predict(pairs_df.values)
    pair_results = pd.DataFrame(
        {f"{pos1}_AA": pos1_aas, f"{pos2}_AA": pos2_aas, "prediction": pair_predictions}
    )
    pair_results.sort_values("prediction", ascending=False, inplace=True)

    wt_pair_series = empty_oh_series.copy()
    wt_pair_series[f"{pos1}_{pos1[0]}"] = 1
    wt_pair_series[f"{pos2}_{pos2[0]}"] = 1
    wt_pair_prediction = mlp.predict(wt_pair_series.values.reshape(1, -1))[0]

    pair_results["normalized_prediction"] = pair_results.apply(
        lambda row: (
            row["prediction"]
            - epinnet_results_df[
                (epinnet_results_df["position"] == pos1)
                & (epinnet_results_df["AA"] == row[f"{pos1}_AA"])
            ]["prediction"].values[0]
            - epinnet_results_df[
                (epinnet_results_df["position"] == pos2)
                & (epinnet_results_df["AA"] == row[f"{pos2}_AA"])
            ]["prediction"].values[0]
            - wt_pair_prediction
        ),
        axis=1,
    )

    return pair_results


def mutation_pair_epistasis(
    pos1: str,
    mut1: str,
    pos2: str,
    mut2: str,
    mlp: MLPRegressor,
    empty_oh_series: pd.Series,
):
    both_series = empty_oh_series.copy()
    both_series[f"{pos1}_{mut1}"] = 1
    both_series[f"{pos2}_{mut2}"] = 1

    pos1_wt_aa = pos1[0]
    pos2_wt_aa = pos2[0]

    only1_series = empty_oh_series.copy()
    only1_series[f"{pos1}_{mut1}"] = 1
    only1_series[f"{pos2}_{pos2_wt_aa}"] = 1
    only2_series = empty_oh_series.copy()
    only2_series[f"{pos1}_{pos1_wt_aa}"] = 1
    only2_series[f"{pos2}_{mut2}"] = 1
    wt_series = empty_oh_series.copy()
    wt_series[f"{pos1}_{pos1_wt_aa}"] = 1
    wt_series[f"{pos2}_{pos2_wt_aa}"] = 1

    both_prediction = mlp.predict(both_series.values.reshape(1, -1))[0]
    only1_prediction = mlp.predict(only1_series.values.reshape(1, -1))[0]
    only2_prediction = mlp.predict(only2_series.values.reshape(1, -1))[0]
    wt_pair_prediction = mlp.predict(wt_series.values.reshape(1, -1))[0]

    # create series with the results
    pair_result = pd.Series(
        {
            "pos1": pos1,
            "mut1": mut1,
            "pos2": pos2,
            "mut2": mut2,
            "both_prediction": both_prediction,
            "only1_prediction": only1_prediction,
            "only2_prediction": only2_prediction,
            "wt_pair_prediction": wt_pair_prediction,
            "normalized_prediction": both_prediction - only1_prediction - only2_prediction - wt_pair_prediction,
        }
    )

    return pair_result


def all_pairs_epinnet_analysis(epinnet_results_df: pd.DataFrame):
    results_list = []
    for position in epinnet_results_df["position"].unique():
        positon_df = epinnet_results_df[epinnet_results_df["position"] == position]
        # get all possible distinct AA combinations
        AA_combinations = list(combinations(positon_df["AA"], 2))

        # iter over combinations
        for AA_combination in AA_combinations:
            # get the two rows
            row1 = positon_df[positon_df["AA"] == AA_combination[0]]
            row2 = positon_df[positon_df["AA"] == AA_combination[1]]
            # get the difference
            diff = row1["prediction"].values[0] - row2["prediction"].values[0]
            results_list.append(
                {
                    "position": position,
                    "AA1": AA_combination[0],
                    "AA2": AA_combination[1],
                    "diff": diff,
                }
            )
            results_list.append(
                {
                    "position": position,
                    "AA1": AA_combination[1],
                    "AA2": AA_combination[0],
                    "diff": -diff,
                }
            )
        results_df = pd.DataFrame(results_list)
        results_df.sort_values("diff", ascending=False, inplace=True)

    return results_df


def all_pairs_epistasis(
    oh_funclib_df: pd.DataFrame, sequences_df: pd.DataFrame, mlp: MLPRegressor, empty_oh_series: pd.Series
):
    pairs = list(combinations(oh_funclib_df.columns, 2))
    series_list = []
    for col1, col2 in pairs:
        pos1, mut1 = col1.split("_")
        pos2, mut2 = col2.split("_")
        if pos1 == pos2:
            continue
        summary_series = mutation_pair_epistasis(
            pos1, mut1, pos2, mut2, mlp, empty_oh_series
        )

        # add position and mutation support from the sequences_df
        pos1_value_counts = sequences_df[sequences_df[pos1] == mut1][pos2].value_counts()
        summary_series["u1"] = len(pos1_value_counts)
        summary_series["n1"] = pos1_value_counts.sum()
        summary_series["s1"] = np.sqrt(summary_series["u1"] * summary_series["n1"])
        pos2_value_counts = sequences_df[sequences_df[pos2] == mut2][pos1].value_counts()
        summary_series["u2"] = len(pos2_value_counts)
        summary_series["n2"] = pos2_value_counts.sum()
        summary_series["s2"] = np.sqrt(summary_series["u2"] * summary_series["n2"])
        summary_series["support"] = (2 * summary_series["s1"] * summary_series["s2"]) / (summary_series["s1"] + summary_series["s2"])

        series_list.append(summary_series)
    pair_comparison_df = pd.DataFrame(series_list)

    return pair_comparison_df


def extract_epistatic_pairs(
    pair_comparison_df: pd.DataFrame,
    threshold: float,
    singlets_ratio_threshold: float = SINGLETS_RATIO_THRESHOLD,
):
    both_to_neither_ratio = pair_comparison_df["both_prediction"] / pair_comparison_df["wt_pair_prediction"]
    singlets_ratio = pair_comparison_df["only1_prediction"] / pair_comparison_df["only2_prediction"]
    one_to_neither_ratio = pair_comparison_df["only1_prediction"] / pair_comparison_df["wt_pair_prediction"]
    two_to_neither_ratio = pair_comparison_df["only2_prediction"] / pair_comparison_df["wt_pair_prediction"]

    # TODO: I miss cases where there is strong epistasis but one of the single mutations seems to have no effect
    positive_epistasis_df = pair_comparison_df[
        (both_to_neither_ratio >= threshold)
        # & (
        #     (singlets_ratio > singlets_ratio_threshold)
        #     & (singlets_ratio < 1 / singlets_ratio_threshold)
        # )
        & (one_to_neither_ratio > 1) & (two_to_neither_ratio > 1)
        & (pair_comparison_df["normalized_prediction"] > 0)
    ]

    # TODO: add ratio, for cases where the singlet are very slightly deleterious
    sign_epistasis_df = pair_comparison_df[
        (both_to_neither_ratio >= threshold)
        & (
            (singlets_ratio > singlets_ratio_threshold)
            & (singlets_ratio < 1 / singlets_ratio_threshold)
        )
        & (one_to_neither_ratio < 1) & (two_to_neither_ratio < 1)
        & (pair_comparison_df["normalized_prediction"] > 0)
    ]

    return positive_epistasis_df, sign_epistasis_df


def main():
    pass


if __name__ == "__main__":
    main()
