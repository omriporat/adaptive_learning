# %%
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.stats import spearmanr

from config import load_config


def get_oh_table(dataset_path: str, first_col: str, last_col: str) -> pd.DataFrame:
    experimental_results_df = pd.read_csv(dataset_path)
    experimental_results_df = experimental_results_df[
        experimental_results_df["design"] != -1
    ]
    cols = []
    should_append = False
    for col in experimental_results_df.columns:
        if col == first_col:
            should_append = True
        if should_append:
            cols.append(col)
        if col == last_col:
            break
    funclib_df = experimental_results_df[cols]
    oh_funclib_df = pd.get_dummies(funclib_df)

    activity = experimental_results_df["fold_improvement"]
    design_numbers = experimental_results_df["design"]

    return oh_funclib_df, activity, design_numbers


def get_metrics(prediction_probs: np.array, true_labels: pd.Series):
    true_labels = true_labels.to_numpy()
    accuracy = accuracy_score(true_labels, prediction_probs > 0.5)
    precision = precision_score(true_labels, prediction_probs > 0.5)
    recall = recall_score(true_labels, prediction_probs > 0.5)
    f1 = f1_score(true_labels, prediction_probs > 0.5)
    # get top 5 predictions and compare to true labels
    top_5_predictions_indices = np.argsort(prediction_probs)[-5:]
    top_5_predictions = prediction_probs[top_5_predictions_indices]
    top_5_true_labels = true_labels[top_5_predictions_indices]
    top_5_precision = precision_score(top_5_true_labels, top_5_predictions > 0.5)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "top_5_precision": top_5_precision,
        "recall": recall,
    }


def bootstrap(
    X: np.ndarray,
    y: pd.Series,
    cluster_labels: pd.Series,
    results_path: str,
    test_fraction: float = 0.7,
    n_iterations: int = 1000,
):
    np.random.seed(42)
    n_clusters = cluster_labels.nunique()
    n_train_samples = int(len(y) * (1 - test_fraction))
    n_test_samples = len(y) - n_train_samples
    always_active = np.ones(n_test_samples, dtype=bool)
    always_inactive = np.zeros(n_test_samples, dtype=bool)

    results = []
    always_active_results = []
    always_inactive_results = []

    for i in range(n_iterations):
        train_indices = np.random.choice(
            y.shape[0], size=n_train_samples, replace=False
        )
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)
        X_train = X[train_indices]
        X_test = X[test_indices]
        train_activity = y[train_indices]
        # fit the model
        mlp = MLPClassifier(
            hidden_layer_sizes=(16),
            activation="logistic",
            learning_rate="invscaling",
            random_state=42,
            verbose=False,
            max_iter=20000,
        )
        svc = SVC(probability=True)
        gnb = GaussianNB()
        mlp.fit(X_train, train_activity)
        svc.fit(X_train, train_activity)
        gnb.fit(X_train, train_activity)
        predictions = mlp.predict_proba(X_test)
        svc_predictions = svc.predict_proba(X_test)
        gnb_predictions = gnb.predict_proba(X_test)
        # find cluster coverage of train set
        train_clusters = cluster_labels[train_indices]
        train_covered_clusters = set(train_clusters.unique())
        train_coverage = len(train_covered_clusters) / n_clusters
        results_dict = get_metrics(predictions[:, 1], y[test_indices])
        svc_results_dict = get_metrics(svc_predictions[:, 1], y[test_indices])
        gnb_results_dict = get_metrics(gnb_predictions[:, 1], y[test_indices])
        results_dict["train_coverage"] = train_coverage
        results_dict["model"] = "MLP"
        svc_results_dict["train_coverage"] = train_coverage
        svc_results_dict["model"] = "SVC"
        gnb_results_dict["train_coverage"] = train_coverage
        gnb_results_dict["model"] = "GNB"

        results.append(results_dict)
        results.append(svc_results_dict)
        results.append(gnb_results_dict)

        always_active_results.append(get_metrics(always_active, y[test_indices]))
        always_inactive_results.append(get_metrics(always_inactive, y[test_indices]))

        print(
            f"Iteration {i + 1}/{n_iterations}: Embedding accuracy: {results[-1]['accuracy']:.4f}, precision: {results[-1]['precision']:.4f}"
        )
        print(
            f"Iteration {i + 1}/{n_iterations}: SVC accuracy: {svc_results_dict['accuracy']:.4f}, precision: {svc_results_dict['precision']:.4f}"
        )
        print(
            f"Iteration {i + 1}/{n_iterations}: GNB accuracy: {gnb_results_dict['accuracy']:.4f}, precision: {gnb_results_dict['precision']:.4f}"
        )

    embedding_results_df = pd.DataFrame(results)
    always_active_df = pd.DataFrame(always_active_results)
    always_inactive_df = pd.DataFrame(always_inactive_results)

    # save results to csv
    embedding_results_df.to_csv(results_path, index=False)

    return embedding_results_df, always_active_df, always_inactive_df


def plot_bootstrap_results(
    enzyme: str,
    substrate: str,
    mode: str,
    test_frac: float,
    embedding_results_df: pd.DataFrame,
    oh_results_df: pd.DataFrame,
    always_active_df: pd.DataFrame,
    always_inactive_df: pd.DataFrame,
):
    # boxplot for each metric. put the embeddings box and the oh box side by side
    for metric in embedding_results_df.columns:
        plt.figure(figsize=(12, 6))
        # Combine data for side-by-side boxplots
        plot_df = pd.DataFrame(
            {
                metric: np.concatenate(
                    [
                        embedding_results_df[metric],
                        oh_results_df[metric],
                        always_active_df[metric],
                        always_inactive_df[metric],
                    ]
                ),
                "method": ["Embedding"] * len(embedding_results_df)
                + ["One-Hot"] * len(oh_results_df)
                + ["Always Active"] * len(always_active_df)
                + ["Always Inactive"] * len(always_inactive_df),
            }
        )
        sns.boxplot(
            x="method",
            y=metric,
            data=plot_df,
            palette=["blue", "orange", "green", "red"],
        )
        plt.title(f"Bootstrap Results - {metric.capitalize()}")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Method")
        plt.savefig(
            f"results/{enzyme}/{substrate}/{mode}/{test_frac}/bootstrap_{metric}.png"
        )
        plt.close()


def plot_coverage(
    enzyme: str,
    substrate: str,
    mode: str,
    test_frac: float,
    finetune_tag: str,
    metric: str,
    results_path: str,
    embedding_results_df: pd.DataFrame,
):
    # bin coverages into 5 bins
    embedding_results_df["train_coverage"] = pd.cut(
        embedding_results_df["train_coverage"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )
    # boxplot for each metric. split by train coverage
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="train_coverage",
        y=metric,
        data=embedding_results_df,
        palette="viridis",
    )
    plt.title(f"Bootstrap Results - {metric.capitalize()} vs Train Coverage")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Train Coverage")
    plt.savefig(
        f"{results_path}/bootstrap_{metric}_vs_coverage_{finetune_tag}_{mode}.png"
    )


def bootstrap_regressor(
    oh_df,
    embeddings,
    y,
    test_fraction: float = 0.7,
    n_iterations: int = 1000,
    hidden_layer_size: int = 16,
):
    np.random.seed(42)
    n_train_samples = int(len(y) * (1 - test_fraction))

    embeddings_r2_results = []
    embeddings_spearman_results = []
    oh_r2_results = []
    oh_spearman_results = []

    for i in range(n_iterations):
        train_indices_within = np.random.choice(
            y.shape[0], size=n_train_samples, replace=False
        )
        train_indices_without = y.index[train_indices_within].tolist()
        test_indices_within = np.setdiff1d(np.arange(y.shape[0]), train_indices_within)
        test_indices_without = y.index[test_indices_within].tolist()

        X_train_emb = embeddings[train_indices_without, :]
        X_test_emb = embeddings[test_indices_without, :]
        X_train_oh = oh_df.loc[train_indices_without].to_numpy()
        X_test_oh = oh_df.loc[test_indices_without].to_numpy()
        y_train = y[train_indices_without]
        y_test = y[test_indices_without]
        # fit the model
        mlp_reg_emb = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_size),
            activation="relu",
            learning_rate="invscaling",
            solver="lbfgs",
            random_state=42,
            verbose=False,
            max_iter=20000,
        )
        mlp_reg_oh = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_size),
            activation="relu",
            learning_rate="invscaling",
            solver="lbfgs",
            random_state=42,
            verbose=False,
            max_iter=20000,
        )
        mlp_reg_emb.fit(X_train_emb, y_train)
        mlp_reg_oh.fit(X_train_oh, y_train)
        emb_predictions = mlp_reg_emb.predict(X_test_emb)
        if np.var(emb_predictions) < 0.01:
            print(f"Low variance detected in embedding-based predictions: {np.var(emb_predictions)}")
            # print indices in list format
            print(f"Indices used for training: {train_indices_without}")
            print(f"Indices used for testing: {test_indices_without}")
            print("retraining with decreased initial learning rate")
            mlp_reg_emb = MLPRegressor(
                hidden_layer_sizes=(hidden_layer_size),
                activation="relu",
                learning_rate="invscaling",
                solver="lbfgs",
                random_state=42,
                verbose=False,
                max_iter=20000,
                learning_rate_init=0.0001,
            )
            mlp_reg_emb.fit(X_train_emb, y_train)
            emb_predictions = mlp_reg_emb.predict(X_test_emb)
            print(f"New variance: {np.var(emb_predictions)}")
        oh_predictions = mlp_reg_oh.predict(X_test_oh)
        # get metrics
        emb_mse = mean_squared_error(y_test, emb_predictions)
        emb_r2 = r2_score(y_test, emb_predictions)
        emb_spearman = spearmanr(y_test, emb_predictions).correlation
        oh_mse = mean_squared_error(y_test, oh_predictions)
        oh_r2 = r2_score(y_test, oh_predictions)
        oh_spearman = spearmanr(y_test, oh_predictions).correlation

        embeddings_r2_results.append(emb_r2)
        embeddings_spearman_results.append(emb_spearman)
        oh_r2_results.append(oh_r2)
        oh_spearman_results.append(oh_spearman)

        print(
            f"Iteration {i + 1}/{n_iterations}: Embedding Spearman: {emb_spearman:.4f}, One-Hot Spearman: {oh_spearman:.4f}",
            flush=True
        )

    # return as dataframe
    results_df = pd.DataFrame(
        {
            "embedding_r2": embeddings_r2_results,
            "embedding_spearman": embeddings_spearman_results,
            "oh_r2": oh_r2_results,
            "oh_spearman": oh_spearman_results,
        }
    )
    return results_df


def boxplot_regressor_results(
    embedding_corr,
    oh_corr,
    enzyme: str,
    substrate: str,
    corr_type: str,
    finetune_tag: str,
    mode: str,
    test_frac: float,
    results_path: str,
):
    plt.figure(figsize=(8, 6))
    plot_df = pd.DataFrame(
        {
            corr_type: embedding_corr.to_list() + oh_corr.to_list(),
            "Method": ["Embedding"] * len(embedding_corr) + ["One-Hot"] * len(oh_corr),
        }
    )
    sns.boxplot(x="Method", y=corr_type, data=plot_df, palette=["blue", "orange"])
    plt.title(
        f"Bootstrap Regressor - {corr_type} correlations\n{enzyme} - {substrate}\n{finetune_tag} - {mode} - Test Fraction: {test_frac}"
    )
    plt.ylabel(corr_type)
    plt.xlabel("Method")
    print(f"Saving regressor bootstrap plot in {results_path}")
    plt.savefig(
        f"{results_path}/regressor_bootstrap_{finetune_tag}_{mode}_{test_frac}_{corr_type}.png"
    )
    plt.close()


def scatter_regressor_results(
    embedding_corr,
    oh_corr,
    enzyme: str,
    substrate: str,
    corr_type: str,
    finetune_tag: str,
    mode: str,
    test_frac: float,
    results_path: str,
):
    # like a boxplot, but connect paired points with lines
    plt.figure(figsize=(8, 6))
    for i in range(len(embedding_corr)):
        if embedding_corr.iloc[i] < oh_corr.iloc[i]:
            color = "teal"
        else:
            color = "sandybrown"
        plt.plot(
            ["Embedding", "One-Hot"],
            [embedding_corr.iloc[i], oh_corr.iloc[i]],
            marker="o",
            color=color,
        )
    plt.title(
        f"Bootstrap Regressor - {corr_type} correlations\n{enzyme} - {substrate}\n{finetune_tag} - {mode} - Test Fraction: {test_frac}"
    )
    plt.ylabel(corr_type)
    plt.xlabel("Method")
    print(f"Saving regressor bootstrap scatter plot in {results_path}")
    plt.savefig(
        f"{results_path}/regressor_bootstrap_scatter_{finetune_tag}_{mode}_{test_frac}_{corr_type}.png"
    )
    plt.close()


def main():
    config = load_config(sys.argv[1:])
    embeddings = np.load(config["embeddings_path"], allow_pickle=True)
    oh_funclib_df, y, design_numbers = get_oh_table(
        config["substrate_specific_dataset_path"],
        first_col=config["first_column_name"],
        last_col=config["last_column_name"],
    )

    y = np.log(y + 1e-6)

    results_df = bootstrap_regressor(
        oh_funclib_df,
        embeddings,
        y,
        hidden_layer_size=config["bootstrap_hidden_layer_size"],
        n_iterations=config["n_bootstrap"],
        test_fraction=config["test_fraction"],
    )

    results_df.to_csv(
        f"{config['substrate_specific_results_path']}/regressor_bootstrap_results_{config['test_fraction']}.csv",
        index=False,
    )

    boxplot_regressor_results(
        results_df["embedding_spearman"],
        results_df["oh_spearman"],
        enzyme=config["enzyme"],
        substrate=config["substrate"],
        corr_type="Spearman",
        finetune_tag=config["finetune_tag"],
        mode=config["opmode"],
        test_frac=config["test_fraction"],
        results_path=config["substrate_specific_results_path"],
    )
    scatter_regressor_results(
        results_df["embedding_spearman"],
        results_df["oh_spearman"],
        enzyme=config["enzyme"],
        substrate=config["substrate"],
        corr_type="Spearman",
        finetune_tag=config["finetune_tag"],
        mode=config["opmode"],
        test_frac=config["test_fraction"],
        results_path=config["substrate_specific_results_path"],
    )


if __name__ == "__main__":
    main()

# %%
