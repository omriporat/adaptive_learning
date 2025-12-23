import logomaker as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform

from config import load_config


def plot_embeddings(
    embeddings,
    save_dir: str,
    opmode: str,
    positive_indices=None,
    negative_indices=None,
    cluster_labels=None,
    activity=None,
    finetune_tag: str = "naive",
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

    # if positive_indices is not None and negative_indices is not None
    # plot using the cluster colormap, but change shapes of designed points
    if positive_indices is not None and negative_indices is not None:
        plt.scatter(
            reduced_embeddings[negative_indices, 0],
            reduced_embeddings[negative_indices, 1],
            c="none",
            edgecolor="black",
            marker="X",
            s=50,
            label="Negative Design",
        )
        plt.scatter(
            reduced_embeddings[positive_indices, 0],
            reduced_embeddings[positive_indices, 1],
            c="none",
            edgecolor="red",
            marker="^",
            s=50,
            label="Positive Design",
        )
        plt.legend()

    plt.title("UMAP projection of the embeddings")
    plt.savefig(f"{save_dir}/umap_projection_{opmode}_{finetune_tag}.png")


def plot_logos(
    sequences,
    cluster_labels,
    results_dir: str,
    opmode: str,
    finetune_tag: str,
    enzyme="PTE",
    clustering_type="kmeans_embedding",
):
    "Create sequence logos for each cluster and plot them in the same figure"
    n_clusters = len(np.unique(cluster_labels))
    # create one figure with n_clusters subplots. Each row should have 4 subplots
    n_cols = 4
    n_rows = int(np.ceil(n_clusters / n_cols))
    plt.figure(figsize=(n_cols * 4, n_rows * 4))
    for cluster_id in range(n_clusters):
        cluster_seqs = sequences[np.where(cluster_labels == cluster_id)[0]]
        # create a count matrix
        count_matrix = lm.alignment_to_matrix(sequences=cluster_seqs, to_type="counts")
        ax = plt.subplot(n_rows, n_cols, cluster_id + 1)
        # create a logo
        logo = lm.Logo(count_matrix, color_scheme="chemistry", ax=ax)
        logo.style_spines(visible=False)
        logo.style_spines(spines=["left", "bottom"], visible=True)
        logo.style_xticks(rotation=90, fmt="%d", anchor=0)
        plt.title(f"Cluster {cluster_id} (n={len(cluster_seqs)})")
    plt.suptitle(f"Sequence Logos for {enzyme} Clusters", fontsize=16)
    plt.savefig(
        f"{results_dir}/cluster_logos_{clustering_type}_{opmode}_{finetune_tag}.png"
    )


def cluster_embeddings(
    embeddings,
    n_clusters: int,
    save_path: str,
):
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

    np.save(
        save_path,
        cluster_labels,
    )

    return cluster_labels, representative_indices


def cluster_sequences(sequences, n_clusters: int, save_path: str):
    "Cluster sequences with Agglomerative Clustering based on Hamming distance"
    # convert sequences to numpy array of shape (n_sequences, n_pos)
    sequences = np.array([list(seq) for seq in sequences])
    distance_matrix = pairwise_distances(
        sequences, metric=lambda x, y: sum(c1 != c2 for c1, c2 in zip(x, y))
    )
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", metric="precomputed")
    cluster_labels = model.fit_predict(distance_matrix)
    representative_indices = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_seqs = sequences[cluster_indices]
        # compute the consensus sequence
        consensus_seq = []
        for pos in range(sequences.shape[1]):
            unique_aa, counts = np.unique(cluster_seqs[:, pos], return_counts=True)
            consensus_aa = unique_aa[np.argmax(counts)]
            consensus_seq.append(consensus_aa)
        consensus_seq = np.array(consensus_seq)
        # find the sequence closest to the consensus
        distances = np.sum(cluster_seqs != consensus_seq, axis=1)
        representative_index = cluster_indices[np.argmin(distances)]
        representative_indices.append(representative_index)

    np.save(
        save_path,
        cluster_labels,
    )
    return cluster_labels, representative_indices


def get_designed_indices(dataset_path: str, positive_threshold: float = 15, negative_threshold: float = 0.5):
    "Get indices of designed sequences based on activity threshold"
    df = pd.read_csv(dataset_path)
    df = df[df["design"] != -1]
    positive_indices = df.index[df["fold_improvement"] >= positive_threshold].tolist()
    negative_indices = df.index[df["fold_improvement"] < negative_threshold].tolist()
    return positive_indices, negative_indices


def plot_cluster_coverage(
    cluster_indices,
    positive_indices,
    negative_indices,
    save_path,
    opmode: str,
    finetune_tag: str,
    clustering_type: str,
):
    "Plot the coverage of designed sequences in each cluster"
    n_clusters = len(np.unique(cluster_indices))
    coverage_data = []
    for cluster_id in range(n_clusters):
        cluster_seq_indices = np.where(cluster_indices == cluster_id)[0]

        n_positive = len(set(cluster_seq_indices) & set(positive_indices))
        n_negative = len(set(cluster_seq_indices) & set(negative_indices))
        n_total = n_positive + n_negative
        coverage_data.append(
            {
                "cluster_id": cluster_id,
                "n_total": n_total,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "positive_fraction": n_positive / n_total if n_total > 0 else 0,
                "negative_fraction": n_negative / n_total if n_total > 0 else 0,
            }
        )
    coverage_df = pd.DataFrame(coverage_data)

    # plot bar chart of positive and negative fractions per cluster
    coverage_df.set_index("cluster_id")[
        ["positive_fraction", "negative_fraction"]
    ].plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.ylabel("Fraction of Designed Sequences")
    plt.title("Coverage of Designed Sequences in Clusters")
    plt.savefig(f"{save_path}/cluster_coverage_{opmode}_{finetune_tag}_{clustering_type}.png")


def main():
    n_clusters = 15
    config = load_config()
    embeddings = np.load(config["embeddings_path"])
    positive_indices, negative_indices = get_designed_indices(
        dataset_path=config["substrate_specific_dataset_path"], threshold=10
    )
    cluster_labels, representative_indices = cluster_embeddings(
        embeddings,
        n_clusters=n_clusters,
        save_path=config["embeddings_clusters_path"],
    )
    sequences_df = pd.read_csv(config["dataset_path"])
    # use only the specific positions
    sequences = sequences_df["full_seq"]
    sequences = sequences.apply(
        lambda x: "".join([x[i - 1] for i in config["pos_to_use"]])
    )


    plot_cluster_coverage(
        cluster_labels,
        positive_indices,
        negative_indices,
        save_path=config["results_path"],
        clustering_type="kmeans_embedding",
        opmode=config["opmode"],
        finetune_tag=config["finetune_tag"],
    )
    # sequence_cluster_labels, seq_representative_indices = cluster_sequences(
    #     sequences.values, n_clusters=n_clusters, save_path=config["sequence_clusters_path"]
    # )

    # plot_cluster_coverage(
    #     sequence_cluster_labels,
    #     positive_indices,
    #     negative_indices,
    #     save_path=config["results_path"],
    #     clustering_type="agglomerative_sequence",
    #     opmode=config["opmode"],
    #     finetune_tag=config["finetune_tag"],
    # )
    return

    plot_logos(
        sequences,
        results_dir=config["results_path"],
        cluster_labels=cluster_labels,
        finetune_tag=config["finetune_tag"],
        opmode=config["opmode"],
    )

    plot_embeddings(
        embeddings,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        save_dir=config["results_path"],
        cluster_labels=cluster_labels,
        finetune_tag=config["finetune_tag"],
        opmode=config.get("opmode", "mean"),
    )


if __name__ == "__main__":
    main()
