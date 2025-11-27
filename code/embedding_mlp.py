# %%
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
import torch
import umap
import yaml
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from train_epinnet import EpiNNetActivityTrainTest, EpiNNetDataset, plmTrunkModel

config_path = "config.yaml"
EXPERIMENTAL_RESULTS_FILENAME = "experimental_results.csv"
FUNCLIB_TABLE_FILENAME = "funclib_table.csv"


# Load configuration from YAML file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


def get_oh_table_by_substrate(
    enzyme: str, substrate: str, add_wt: bool = False
) -> pd.DataFrame:
    experimental_results_df = pd.read_csv(
        os.path.join("data", enzyme, EXPERIMENTAL_RESULTS_FILENAME), index_col=0
    )
    activity = experimental_results_df[substrate]
    if add_wt:
        # add activity of 1 for design 0
        activity[0] = 1
    # sort by index
    activity = activity.sort_index()
    activity = activity.reset_index(drop=True)
    funclib_df = pd.read_csv(
        os.path.join("data", enzyme, FUNCLIB_TABLE_FILENAME), index_col=0
    )
    if not add_wt:
        funclib_df = funclib_df[
            funclib_df.index != 0
        ]  # remove the wild type design if not adding it
        # reindex to start with 0
        funclib_df = funclib_df.reset_index(drop=True)

    n_pos = funclib_df.shape[1]
    oh_funclib_df = pd.get_dummies(funclib_df)
    oh_funclib_df = oh_funclib_df.loc[:, ~oh_funclib_df.columns.duplicated()]
    return oh_funclib_df, activity, n_pos


def get_oh_funclib_table(enzyme: str):
    funclib_df = pd.read_csv(
        os.path.join("data", enzyme, FUNCLIB_TABLE_FILENAME), index_col=0
    )
    oh_funclib_df = pd.get_dummies(funclib_df)
    oh_funclib_df = oh_funclib_df.loc[:, ~oh_funclib_df.columns.duplicated()]
    return oh_funclib_df


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
    X_embeddings: np.ndarray,
    X_oh: pd.DataFrame,
    y,
    test_fraction: float = 0.25,
    n_iterations: int = 1000,
):
    np.random.seed(42)
    n_train_samples = int(len(y) * (1 - test_fraction))
    n_test_samples = len(y) - n_train_samples
    always_active = np.ones(n_test_samples, dtype=bool)
    always_inactive = np.zeros(n_test_samples, dtype=bool)

    embedding_mlp_results = []
    oh_mlp_results = []
    always_active_results = []
    always_inactive_results = []

    for i in range(n_iterations):
        train_indices = np.random.choice(
            y.shape[0], size=n_train_samples, replace=False
        )
        test_indices = np.setdiff1d(np.arange(y.shape[0]), train_indices)
        X_embeddings_train = X_embeddings[train_indices]
        X_oh_train = X_oh.iloc[train_indices]
        X_embeddings_test = X_embeddings[test_indices]
        X_oh_test = X_oh.iloc[test_indices]
        train_activity = y[train_indices]
        # fit the model
        embedding_mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="logistic",
            learning_rate="invscaling",
            random_state=42,
            verbose=False,
            max_iter=20000,
        )
        oh_mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="logistic",
            learning_rate="invscaling",
            random_state=42,
            verbose=False,
            max_iter=20000,
        )
        embedding_mlp.fit(X_embeddings_train, train_activity)
        oh_mlp.fit(X_oh_train, train_activity)
        embedding_predictions = embedding_mlp.predict_proba(X_embeddings_test)
        oh_predictions = oh_mlp.predict_proba(X_oh_test)
        embedding_mlp_results.append(
            get_metrics(embedding_predictions[:, 1], y[test_indices])
        )
        oh_mlp_results.append(get_metrics(oh_predictions[:, 1], y[test_indices]))
        always_active_results.append(get_metrics(always_active, y[test_indices]))
        always_inactive_results.append(get_metrics(always_inactive, y[test_indices]))

        print(
            f"Iteration {i + 1}/{n_iterations}: Embedding accuracy: {embedding_mlp_results[-1]['accuracy']:.4f}, precision: {embedding_mlp_results[-1]['precision']:.4f}"
        )
        print(
            f"Iteration {i + 1}/{n_iterations}: OH accuracy: {oh_mlp_results[-1]['accuracy']:.4f}, precision: {oh_mlp_results[-1]['precision']:.4f}"
        )

    embedding_results_df = pd.DataFrame(embedding_mlp_results)
    oh_results_df = pd.DataFrame(oh_mlp_results)
    always_active_df = pd.DataFrame(always_active_results)
    always_inactive_df = pd.DataFrame(always_inactive_results)

    return embedding_results_df, oh_results_df, always_active_df, always_inactive_df


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


def get_embeddings(model_type: str = "original", mode: str = "mean"):
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_to_use = config["pos_to_use"]
    plm_name = config["plm_name"]

    if model_type == "original":
        model = plmTrunkModel(
            plm_name=plm_name,
            opmode="mean",
            emb_only=True,
            hidden_layers=[256, 128],
            activation="relu",
            layer_norm=False,
            activation_on_last_layer=False,
            specific_pos=config["pos_to_use"],
            device=device,
        ).to(device)
    elif model_type == "finetuned":
        model = plmTrunkModel(
            plm_name=plm_name,
            opmode="mean",
            emb_only=True,
            logits_only=False,
            specific_pos=pos_to_use,
            hidden_layers=[516, 256],
            activation="relu",
            layer_norm=False,
            activation_on_last_layer=False,
            device=device,
        ).to(device)
        model_path = os.path.join(config["save_path"], "final_model.pt")
        model.load_state_dict(torch.load(model_path))
    else:
        raise ValueError("model_type must be 'original' or 'finetuned'")

    dataset = EpiNNetDataset(
        config["dataset_path"],
        indices=None,
        cache=True,
        encoding_function=model.encode,
        encoding_identifier=plm_name,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    emb = []
    ys = []

    for step, batch in enumerate(loader):
        x = batch[0].to(device)
        y = batch[1].to(device)
        hh = model(x)
        print(f"Processed batch {step + 1}/{len(loader)}")
        if mode == "mean":
            batch_emb = torch.nn.functional.normalize(
                hh[:, torch.tensor(pos_to_use), :], dim=1
            ).mean(dim=1)
        elif mode == "flat":
            batch_emb = hh[:, torch.tensor(pos_to_use), :].reshape(hh.shape[0], -1)
        else:
            raise ValueError("mode must be 'mean' or 'flat'")
        emb.append(batch_emb.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

    emb = np.concatenate(emb, axis=0)
    ys = np.concatenate(ys, axis=0)

    # x, y = dataset[:]
    # x = x.to(device)
    # y = y.to(device)
    # hh = model(x)
    # if mode == "mean":
    #     emb = torch.nn.functional.normalize(hh[:, torch.tensor(pos_to_use), :], dim=1).mean(
    #         dim=1
    #     )
    # elif mode == "flat":
    #     emb = hh[:, torch.tensor(pos_to_use), :].reshape(hh.shape[0], -1)
    # else:
    #     raise ValueError("mode must be 'mean' or 'flat'")
    # emb = torch.nn.functional.normalize(emb, dim=1)

    return emb, ys


def plot_embeddings(
    embeddings,
    cluster_labels=None,
    activity=None,
    enzyme="PTE",
    substrate="p-nitrophenyl_acetate",
):
    "Create a UMAP plot of the embeddings"
    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(embeddings)
    plt.figure(figsize=(6, 6))
    # create scatter plot with binary color map and a legend
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=cluster_labels if cluster_labels is not None else activity,
        cmap="Spectral",
        s=5,
    )
    plt.title("UMAP projection of the embeddings")
    plt.savefig(f"data/{enzyme}/{substrate}/umap_projection.png")


def cluster_embeddings(embeddings, n_clusters=15):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = model.fit_predict(embeddings)
    representative_indices = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_center = model.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        representative_index = cluster_indices[np.argmin(distances)]
        representative_indices.append(representative_index)
    return cluster_labels, representative_indices


def pairwise_distance_matrix(
    oh_funclib_df: pd.DataFrame, enzyme="PTE", substrate="p-nitrophenyl_acetate"
):
    distance_matrix = pdist(oh_funclib_df.values, metric="hamming")
    distance_square = squareform(distance_matrix)
    linkage_matrix = linkage(distance_matrix, method="average")
    ordered_leaves = leaves_list(linkage_matrix)
    distance_reordered = distance_square[ordered_leaves][:, ordered_leaves]
    sns.clustermap(
        distance_square,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="viridis",
        figsize=(10, 10),
    )
    plt.savefig(f"results/{enzyme}/{substrate}/oh_clustermap.png")


def generate_and_save_bootstrap_indices(test_frac=0.2):
    df = pd.read_csv(config["dataset_path"], index_col=0)
    train_set_size = int((1 - test_frac) * df.shape[0])
    np.random.seed(42)

    # Save bootstrap indices
    for i in range(config["n_bootstrap"]):
        train_indices = np.random.choice(
            df.shape[0], size=train_set_size, replace=False
        )
        test_indices = np.setdiff1d(np.arange(df.shape[0]), train_indices)
        bootstrap_indices = {
            "train": train_indices.tolist(),
            "test": test_indices.tolist(),
        }
        with open(
            f"{config['bootstrap_path']}{config['bootstrap_indices_prefix']}_{i}.json",
            "w",
        ) as f:
            json.dump(bootstrap_indices, f)


def load_bootstrap_dfs(enzyme: str, substrate: str, test_frac: float):
    embedding_results_df = pd.read_csv(
        f"results/{enzyme}/{substrate}/embedding_results_test-frac-{test_frac}.csv",
        index_col=0,
    )
    oh_results_df = pd.read_csv(
        f"results/{enzyme}/{substrate}/oh_results_test-frac-{test_frac}.csv",
        index_col=0,
    )
    always_active_df = pd.read_csv(
        f"results/{enzyme}/{substrate}/always_active_test-frac-{test_frac}.csv",
        index_col=0,
    )
    always_inactive_df = pd.read_csv(
        f"results/{enzyme}/{substrate}/always_inactive_test-frac-{test_frac}.csv",
        index_col=0,
    )
    return embedding_results_df, oh_results_df, always_active_df, always_inactive_df


def main():
    enzyme = "PTE"
    substrate = "clustering"
    mode = "flat"
    fracs = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    # fracs = [0.8]
    # pairwise_distance_matrix(oh_funclib_df)
    load_embeddings = False
    oh_funclib_df = get_oh_funclib_table(enzyme)
    if load_embeddings:
        embeddings = np.load(
            f"data/{enzyme}/{substrate}/embeddings_{mode}_original.npy"
        )
    else:
        embeddings, binary_inactivities = get_embeddings("original", mode=mode)
        # save embeddings
        np.save(f"data/{enzyme}/{substrate}/embeddings_{mode}_original.npy", embeddings)
    cluster_labels, representative_indices = cluster_embeddings(
        embeddings, n_clusters=16
    )
    print("Representative indices for each cluster:", representative_indices)
    # binary_activities = activity >= activity_threshold
    plot_embeddings(
        embeddings, cluster_labels=cluster_labels, enzyme=enzyme, substrate=substrate
    )

    return
    activity_threshold = 5
    oh_funclib_df, activity, n_pos = get_oh_table_by_substrate(enzyme, substrate)
    binary_activities = binary_inactivities >= activity_threshold
    for test_frac in fracs:
        # create results directories if needed
        os.makedirs(f"results/{enzyme}/{substrate}/{mode}/{test_frac}", exist_ok=True)

        embedding_results_df, oh_results_df, always_active_df, always_inactive_df = (
            bootstrap(
                embeddings,
                oh_funclib_df,
                binary_activities,
                test_fraction=test_frac,
                n_iterations=config["n_bootstrap"],
            )
        )
        # save dfs
        embedding_results_df.to_csv(
            f"results/{enzyme}/{substrate}/{mode}/{test_frac}/embedding_results_test-frac-{test_frac}.csv"
        )
        oh_results_df.to_csv(
            f"results/{enzyme}/{substrate}/{mode}/{test_frac}/oh_results_test-frac-{test_frac}.csv"
        )
        always_active_df.to_csv(
            f"results/{enzyme}/{substrate}/{mode}/{test_frac}/always_active_test-frac-{test_frac}.csv"
        )
        always_inactive_df.to_csv(
            f"results/{enzyme}/{substrate}/{mode}/{test_frac}/always_inactive_test-frac-{test_frac}.csv"
        )
        # embedding_results_df, oh_results_df, always_active_df, always_inactive_df = load_bootstrap_dfs(enzyme, test_frac)
        plot_bootstrap_results(
            enzyme,
            substrate,
            mode,
            test_frac,
            embedding_results_df,
            oh_results_df,
            always_active_df,
            always_inactive_df,
        )


def create_bootstrap_jobs():
    pass


if __name__ == "__main__":
    main()

# %%
# -----DRINTVRGPITISEAGFTLTHEHICGSSAGFLRAWPEFFGSRKALAEKAVRGLRRARAAGVRTIVDVSTFDIGRDVSLLAEVSRAADVHIVAATGLWFDPPLSMRLRSVEELTQFFLREIQYGIEDTGIRAGIIKVATTGKATPFQELVLKAAARASLATGVPVTTHTAASQRDGEQQAAIFESEGLSPSRVCIGHSDDTDDLSYLTALAARGYLIGLDHIPHSAIGLEDNASASALLGIRSWQTRALLIKALIDQGYMKQILVSNDWLFGFSSYVTNIMDVMDRVNPDGMAFIPLRVIPFLREKGVPQETLAGITVTNPARFLSPTLRAS
# ITNSGDRINTVRGPITISEAGFTLMHEHICGSSAGFLRAWPEFFGSRDALAEKAVRGLRRARAAGVRTIVDVSTFDIGRDVELLAEVSEAADVHIVAATGLWFDPPLSMRLRSVEELTQFFLREIQYGIEDTGIRAGIIKVATTGKATPFQERVLRAAARASLATGVPVTTHTDASQRDGEQQADIFESEGLDPSRVCIGHSDDTDDLDYLTALAARGYLIGLDHIPHSAIGLEDNASAAALLGLRSWQTRALLIKALIDQGYADQILVSNDWLFGFSSYVTNIMDVMDRVNPDGMAFIPLRVIPFLREKGVPDETLETIMVDNPARFLSPTLRAS
# ITNSGDRINTVRGPITISEAGFTLMHEHICGSSAGFLRAWPEFFGSRDALAEKAVRGLRRARAAGVRTIVDVSTFDCGRDVELLAEVSEAADVHIVAATGLWFDPPLSMRLRSVEELTQFFLREIQYGIEDTGIRAGIIKVATTGKATPFQERVLRAAARASLATGVPVTTHTDASQRDGEQQADIFESEGLDPSRVCIGHSDDTDDLDYLTALAARGYLIGLDHIPHSAIGLEDNASAAARLGLRSWQTRALLIKALIDQGYADQILVSNDWLFGFSSYVTNIMDVLDRVNPDGMAFIPLRVIPFLREKGVPDETLETIMVDNPARFLSPTLRAS
