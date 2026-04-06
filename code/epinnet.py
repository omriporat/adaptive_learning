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


def extract_support_values(sequences_df: pd.DataFrame, pos1: str, mut1: str, pos2: str, mut2: str):
    pos1_value_counts = sequences_df[sequences_df[pos1] == mut1][pos2].value_counts()
    n1 = pos1_value_counts.sum()
    pos1_frequencies = pos1_value_counts / pos1_value_counts.sum()
    h1_scaled = -np.sum(pos1_frequencies * np.log(pos1_frequencies)) * n1
    u1 = len(pos1_value_counts)
    s1 = np.sqrt(u1 * n1)

    pos2_value_counts = sequences_df[sequences_df[pos2] == mut2][pos1].value_counts()
    n2 = pos2_value_counts.sum()
    pos2_frequencies = pos2_value_counts / pos2_value_counts.sum()
    h2_scaled = -np.sum(pos2_frequencies * np.log(pos2_frequencies)) * n2
    u2 = len(pos2_value_counts)
    s2 = np.sqrt(u2 * n2)

    support = (2 * s1 * s2) / (s1 + s2)
    h_scaled_product = h1_scaled * h2_scaled
    h_scaled_min = min(h1_scaled, h2_scaled)

    return support, h_scaled_product, h_scaled_min



def _add_support_columns(pair_comparison_df: pd.DataFrame):
    df_copy = pair_comparison_df.copy()
    df_copy["support"] = 2 * pair_comparison_df["s1"] * pair_comparison_df["s2"] / (pair_comparison_df["s1"] + pair_comparison_df["s2"])
    df_copy["h_scaled_product"] = pair_comparison_df["h1_scaled"] * pair_comparison_df["h2_scaled"]
    df_copy["h_scaled_min"] = pair_comparison_df[["h1_scaled", "h2_scaled"]].min(axis=1)
    return df_copy


def _empty_pair_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "pos1",
            "mut1",
            "pos2",
            "mut2",
            "both_prediction",
            "only1_prediction",
            "only2_prediction",
            "wt_pair_prediction",
            "normalized_prediction",
            "h1_scaled",
            "u1",
            "n1",
            "s1",
            "h2_scaled",
            "u2",
            "n2",
            "s2",
        ]
    )


def _split_feature_columns(columns_index: pd.Index):
    parsed_columns = [col.split("_", 1) for col in columns_index]
    positions = np.array([p[0] for p in parsed_columns], dtype=object)
    mutations = np.array([p[1] for p in parsed_columns], dtype=object)
    return positions, mutations


def _build_valid_pair_indices(positions: np.ndarray):
    # take upper triangle of the position array to get all unique pairs
    i_idx, j_idx = np.triu_indices(len(positions), k=1)
    # filter out pairs where the positions are the same, because positions are repeated for different mutations
    valid_mask = positions[i_idx] != positions[j_idx]
    return i_idx[valid_mask], j_idx[valid_mask]


def _get_wt_feature_indices(columns_index: pd.Index, pos1: np.ndarray, pos2: np.ndarray):
    # construct the corresponding WT column names for each position in the pairs
    wt_col_pos1 = [f"{p}_{p[0]}" for p in pos1]
    wt_col_pos2 = [f"{p}_{p[0]}" for p in pos2]
    # get the indices of the WT columns in the one-hot encoding
    wt_idx_pos1 = columns_index.get_indexer(wt_col_pos1)
    wt_idx_pos2 = columns_index.get_indexer(wt_col_pos2)
    if (wt_idx_pos1 < 0).any() or (wt_idx_pos2 < 0).any():
        raise ValueError("Could not find one or more WT one-hot columns for pair construction.")
    return wt_idx_pos1, wt_idx_pos2


def _predict_pair_states_in_batches(
    mlp: MLPRegressor,
    base_vector: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    wt_idx_pos1: np.ndarray,
    wt_idx_pos2: np.ndarray,
    batch_size: int = 10000,
):
    """
    Predicts the four states for each pair of mutations in batches to avoid memory issues. The four states are:
    1. Both mutations present (i and j)
    2. Only mutation i present (i and wt at j)
    3. Only mutation j present (wt at i and j)
    4. Neither mutation present (wt at i and wt at j)
    """
    pair_count = len(i_idx)
    both_prediction = np.empty(pair_count, dtype=float)
    only1_prediction = np.empty(pair_count, dtype=float)
    only2_prediction = np.empty(pair_count, dtype=float)
    wt_pair_prediction = np.empty(pair_count, dtype=float)

    for start in range(0, pair_count, batch_size):
        end = min(start + batch_size, pair_count)
        chunk_size = end - start
        chunk_rows = np.arange(chunk_size)

        # create a batch of input vectors for the four states of the current chunk of pairs
        X = np.tile(base_vector, (4 * chunk_size, 1))

        # start with the double mutant state (both mutations present)
        X[chunk_rows, i_idx[start:end]] = 1
        X[chunk_rows, j_idx[start:end]] = 1

        # then the single mutant states (only mutation i present and only mutation j present)
        # this chunk follows the double mutant chunk, so we add the chunk size to the row indices to get the correct position in the batch
        only1_rows = chunk_rows + chunk_size
        X[only1_rows, i_idx[start:end]] = 1
        X[only1_rows, wt_idx_pos2[start:end]] = 1

        # the other single mutant state
        # this time, we add 2 times the chunk size to get the correct position in the batch
        only2_rows = chunk_rows + (2 * chunk_size)
        X[only2_rows, wt_idx_pos1[start:end]] = 1
        X[only2_rows, j_idx[start:end]] = 1

        # then the double WT state (neither mutation present)
        wt_rows = chunk_rows + (3 * chunk_size)
        X[wt_rows, wt_idx_pos1[start:end]] = 1
        X[wt_rows, wt_idx_pos2[start:end]] = 1

        preds = mlp.predict(X)

        # store the predictions in the corresponding arrays using the correct offsets for each state
        both_prediction[start:end] = preds[:chunk_size]
        only1_prediction[start:end] = preds[chunk_size : 2 * chunk_size]
        only2_prediction[start:end] = preds[2 * chunk_size : 3 * chunk_size]
        wt_pair_prediction[start:end] = preds[3 * chunk_size :]

    return both_prediction, only1_prediction, only2_prediction, wt_pair_prediction


def _entropy_stats_from_counts(counts: np.ndarray):
    """Compute entropy-derived support statistics for each contingency table.

    Parameters
    ----------
    counts:
        A 2D array where each row is a contingency table for one mutation pair.
        Within a row, the values are counts of the partner mutation/allele states
        observed in the background of the focal mutation.

    Returns
    -------
    h_vals, u_vals, n_vals, s_vals:
        - ``h_vals``: entropy scaled by the total count in each row.
        - ``u_vals``: number of non-zero bins in each row.
        - ``n_vals``: total observations per row.
        - ``s_vals``: support score, defined as ``sqrt(u * n)``.

    Notes
    -----
    The ``np.errstate`` block suppresses divide-by-zero and invalid-value warnings
    when a row is empty or when a probability is zero. Those cases are handled
    explicitly by the ``where=`` masks in ``np.divide`` and ``np.log``.
    """
    n_vals = counts.sum(axis=1)
    u_vals = (counts > 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Convert raw counts to row-wise probabilities.
        # Empty rows stay at zero because the division is masked off for n == 0.
        probs = np.divide(
            counts,
            n_vals[:, None],
            out=np.zeros_like(counts),
            where=n_vals[:, None] != 0,
        )
        # Take log only where the probability is positive; log(0) would be undefined.
        log_probs = np.zeros_like(probs)
        np.log(probs, out=log_probs, where=probs > 0)
        # Standard entropy, but scaled by n_vals so the statistic reflects both
        # diversity and how much data supports that diversity.
        h_vals = -(probs * log_probs).sum(axis=1) * n_vals
    s_vals = np.sqrt(u_vals * n_vals)
    return h_vals, u_vals, n_vals, s_vals


def _compute_support_arrays(
    sequences_df: pd.DataFrame,
    pos1: np.ndarray,
    mut1: np.ndarray,
    pos2: np.ndarray,
    mut2: np.ndarray,
):
    """Vectorized support-statistic extraction for all mutation pairs.

    For each pair of mutations, this function estimates how much empirical
    support exists for the pair by looking at the cross-tabulation of the two
    positions across the full sequence table.

    The key idea is to avoid recomputing the same contingency table for every
    mutation pair at the same position pair. Instead, we build one cross table
    per unique position pair and then slice the relevant rows/columns for all
    mutation combinations that belong to that position pair.

    Example
    -------
    If a single position pair produces four mutation combinations such as
    ``(A, X)``, ``(A, Y)``, ``(C, X)``, and ``(C, Y)``, then ``subset_mut1`` is
    ``[A, A, C, C]`` and ``subset_mut2`` is ``[X, Y, X, Y]``. Reindexing the
    contingency table with those arrays repeats the ``A`` row twice and the ``C``
    row twice so that each repeated row still lines up with one mutation pair in
    the original order.

    Parameters
    ----------
    sequences_df:
        DataFrame of observed sequences. Each column corresponds to a position,
        and each cell contains the amino acid observed at that position.
    pos1, mut1, pos2, mut2:
        Parallel arrays describing the mutation pairs to analyze.

    Returns
    -------
    Tuple of arrays matching the input order:
    ``h1_scaled, u1, n1, s1, h2_scaled, u2, n2, s2``.
    """
    pair_count = len(pos1)
    h1_scaled = np.zeros(pair_count, dtype=float)
    u1 = np.zeros(pair_count, dtype=int)
    n1 = np.zeros(pair_count, dtype=float)
    s1 = np.zeros(pair_count, dtype=float)
    h2_scaled = np.zeros(pair_count, dtype=float)
    u2 = np.zeros(pair_count, dtype=int)
    n2 = np.zeros(pair_count, dtype=float)
    s2 = np.zeros(pair_count, dtype=float)

    unique_position_pairs = pd.DataFrame({"pos1": pos1, "pos2": pos2}).drop_duplicates()
    for _, pair in unique_position_pairs.iterrows():
        p1 = pair["pos1"]
        p2 = pair["pos2"]
        pair_mask = (pos1 == p1) & (pos2 == p2)
        subset_mut1 = mut1[pair_mask]
        subset_mut2 = mut2[pair_mask]

        # Build the full contingency table for this pair of positions.
        # Rows correspond to amino acids at p1, columns correspond to amino acids at p2.
        # Example: cross_counts.loc['A', 'V'] is the number of sequences with A at p1 and V at p2.
        cross_counts = pd.crosstab(sequences_df[p1], sequences_df[p2])

        # For all mutation choices that share the same position pair, extract the
        # relevant row and column slices from the same cross table.
        # These slices can contain repeated rows/columns because the same amino
        # acid may appear in multiple pair combinations, and we need one aligned
        # row per mutation pair.
        row_counts = cross_counts.reindex(index=subset_mut1, fill_value=0).to_numpy(dtype=float)
        col_counts = cross_counts.reindex(columns=subset_mut2, fill_value=0).to_numpy(dtype=float).T

        # The row slice measures how the partner-position distribution looks when
        # the focal mutation at p1 is present. The column slice does the symmetric
        # calculation with the positions swapped.
        h1_vals, u1_vals, n1_vals, s1_vals = _entropy_stats_from_counts(row_counts)
        h2_vals, u2_vals, n2_vals, s2_vals = _entropy_stats_from_counts(col_counts)

        h1_scaled[pair_mask] = h1_vals
        u1[pair_mask] = u1_vals
        n1[pair_mask] = n1_vals
        s1[pair_mask] = s1_vals
        h2_scaled[pair_mask] = h2_vals
        u2[pair_mask] = u2_vals
        n2[pair_mask] = n2_vals
        s2[pair_mask] = s2_vals

    return h1_scaled, u1, n1, s1, h2_scaled, u2, n2, s2


def all_pairs_epistasis_legacy(
    oh_funclib_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    mlp: MLPRegressor,
    empty_oh_series: pd.Series,
):
    """Original loop-based implementation kept for baseline comparison."""
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

        pos1_value_counts = sequences_df[sequences_df[pos1] == mut1][pos2].value_counts()
        n1 = pos1_value_counts.sum()
        pos1_frequencies = pos1_value_counts / pos1_value_counts.sum()
        summary_series["h1_scaled"] = -np.sum(pos1_frequencies * np.log(pos1_frequencies)) * n1
        summary_series["u1"] = len(pos1_value_counts)
        summary_series["n1"] = n1
        summary_series["s1"] = np.sqrt(summary_series["u1"] * summary_series["n1"])

        pos2_value_counts = sequences_df[sequences_df[pos2] == mut2][pos1].value_counts()
        n2 = pos2_value_counts.sum()
        pos2_frequencies = pos2_value_counts / pos2_value_counts.sum()
        summary_series["h2_scaled"] = -np.sum(pos2_frequencies * np.log(pos2_frequencies)) * n2
        summary_series["u2"] = len(pos2_value_counts)
        summary_series["n2"] = n2
        summary_series["s2"] = np.sqrt(summary_series["u2"] * summary_series["n2"])

        series_list.append(summary_series)

    pair_comparison_df = pd.DataFrame(series_list)
    # round all numeric columns to 4 decimal places
    numeric_cols = pair_comparison_df.select_dtypes(include=[np.number]).columns
    pair_comparison_df[numeric_cols] = pair_comparison_df[numeric_cols].round(4)
    pair_comparison_df = _add_support_columns(pair_comparison_df)

    return pair_comparison_df


def all_pairs_epistasis(
    oh_funclib_df: pd.DataFrame, sequences_df: pd.DataFrame, mlp: MLPRegressor, empty_oh_series: pd.Series
):
    columns_index = pd.Index(oh_funclib_df.columns)
    positions, mutations = _split_feature_columns(columns_index)

    i_idx, j_idx = _build_valid_pair_indices(positions)

    if len(i_idx) == 0:
        return _empty_pair_comparison_df()

    pos1 = positions[i_idx]
    mut1 = mutations[i_idx]
    pos2 = positions[j_idx]
    mut2 = mutations[j_idx]

    wt_idx_pos1, wt_idx_pos2 = _get_wt_feature_indices(columns_index, pos1, pos2)
    base_vector = empty_oh_series.reindex(columns_index).to_numpy(dtype=float)
    (
        both_prediction,
        only1_prediction,
        only2_prediction,
        wt_pair_prediction,
    ) = _predict_pair_states_in_batches(
        mlp=mlp,
        base_vector=base_vector,
        i_idx=i_idx,
        j_idx=j_idx,
        wt_idx_pos1=wt_idx_pos1,
        wt_idx_pos2=wt_idx_pos2,
    )

    normalized_prediction = (
        both_prediction - only1_prediction - only2_prediction - wt_pair_prediction
    )

    h1_scaled, u1, n1, s1, h2_scaled, u2, n2, s2 = _compute_support_arrays(
        sequences_df=sequences_df,
        pos1=pos1,
        mut1=mut1,
        pos2=pos2,
        mut2=mut2,
    )

    pair_comparison_df = pd.DataFrame(
        {
            "pos1": pos1,
            "mut1": mut1,
            "pos2": pos2,
            "mut2": mut2,
            "both_prediction": both_prediction,
            "only1_prediction": only1_prediction,
            "only2_prediction": only2_prediction,
            "wt_pair_prediction": wt_pair_prediction,
            "normalized_prediction": normalized_prediction,
            "h1_scaled": h1_scaled,
            "u1": u1,
            "n1": n1,
            "s1": s1,
            "h2_scaled": h2_scaled,
            "u2": u2,
            "n2": n2,
            "s2": s2,
        }
    )
        # round all numeric columns to 4 decimal places
    numeric_cols = pair_comparison_df.select_dtypes(include=[np.number]).columns
    pair_comparison_df[numeric_cols] = pair_comparison_df[numeric_cols].round(4)
    pair_comparison_df = _add_support_columns(pair_comparison_df)
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


def create_singlets_df(oh_funclib_df: pd.DataFrame, sequences_df: pd.DataFrame, positions: list):
    # get vocab per position
    vocab_per_position = {}
    for pos in positions:
        vocab_per_position[pos] = set(sequences_df[pos].unique())
        vocab_per_position[pos].remove(pos[0])  # remove wt amino acid from vocab

    singlets = []
    mutations = []
    for pos in positions:
        other_positions = [p for p in positions if p != pos]
        # pos is the current position to mutate
        wt_template = pd.Series(0, index=oh_funclib_df.columns)
        wt_template[[f"{p}_{p[0]}" for p in other_positions]] = 1
        for aa in vocab_per_position[pos]:
            mutant = f"{pos}_{aa}"
            mutations.append(mutant)
            wt_template[mutant] = 1
            singlets.append(wt_template.copy())
            wt_template[mutant] = 0  # reset for next mutant

    return pd.DataFrame(singlets)


def print_summary(singlets_df, epistasis_df, top_percentage=20, support_threshold=10, return_text=False):
    top_n = int(len(singlets_df) * top_percentage / 100)
    top_singlets = singlets_df.sort_values("prediction", ascending=False).head(top_n)
    top_singlets_positions = set(top_singlets["position"])

    top_pairs = epistasis_df[epistasis_df["support"] >= support_threshold].sort_values("normalized_prediction", ascending=False).head(top_n)
    positions_in_top_pairs = set(top_pairs["pos1"]).union(set(top_pairs["pos2"]))
    mutations_in_top_pairs = set(top_pairs["pos1"].astype(str) + top_pairs["mut1"].astype(str)).union(set(top_pairs["pos2"].astype(str) + top_pairs["mut2"].astype(str)))

    lines = []
    lines.append(f"Top {top_percentage}% singlets:")
    lines.append(top_singlets[["position", "AA", "prediction"]].to_string(index=False))
    lines.append(f"\nTop {top_percentage}% epistatic pairs (out of those with support >= {support_threshold}):")
    lines.append(top_pairs[["pos1", "mut1", "pos2", "mut2", "normalized_prediction", "support"]].to_string(index=False))
    lines.append("-------------")
    lines.append(f"Positions in top {top_percentage}% singlets: {sorted(top_singlets_positions)}")
    lines.append(f"Positions in top {top_percentage}% epistatic pairs: {sorted(positions_in_top_pairs)}")
    lines.append(f"Mutations in top {top_percentage}% epistatic pairs: {sorted(mutations_in_top_pairs)}")
    lines.append(f"Overlap in positions: {sorted(top_singlets_positions.intersection(positions_in_top_pairs))}")

    summary_text = "\n".join(lines)
    if return_text:
        return summary_text
    print(summary_text)


def main():
    config = load_config("configs/config_hammer.yaml")
    sequences_df = pd.read_csv(config["dataset_path"])
    oh_funclib_df, y, design_numbers = get_oh_table(
        config["dataset_path"],
        first_col=config["first_column_name"],
        last_col=config["last_column_name"],
        y_col="fold_improvement"
    )
    mlp_reg_oh, results_sorted = base_mlp(oh_funclib_df, y)
    empty_series = pd.Series(0, index=oh_funclib_df.columns)
    pair_comparison_df = all_pairs_epistasis(oh_funclib_df, sequences_df, mlp_reg_oh, empty_series)


if __name__ == "__main__":
    from config import load_config
    from bootstrap import get_oh_table
    main()
