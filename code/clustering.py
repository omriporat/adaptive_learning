import logomaker as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from config import load_config
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def plot_embeddings(
    embeddings,
    save_dir: str,
    opmode: str,
    enzyme: str,
    substrate: str,
    positive_threshold: float = 15,
    negative_threshold: float = 0.5,
    cluster_labels=None,
    fold_improvements=None,
    finetune_tag: str = "naive",
):
    "Create a UMAP plot of the embeddings"
    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    # create scatter plot with binary color map and a legend
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=cluster_labels if cluster_labels is not None else fold_improvements,
        cmap="Spectral",
        s=5,
    )

    # if positive_indices is not None and negative_indices is not None
    # plot using the cluster colormap, but change shapes of designed points
    positive_indices = (
        fold_improvements.index[fold_improvements >= positive_threshold].tolist()
        if fold_improvements is not None
        else None
    )
    negative_indices = (
        fold_improvements.index[fold_improvements < negative_threshold].tolist()
        if fold_improvements is not None
        else None
    )
    neutral_indices = (
        fold_improvements.index[
            (fold_improvements >= negative_threshold)
            & (fold_improvements < positive_threshold)
        ].tolist()
        if fold_improvements is not None
        else None
    )
    if fold_improvements is not None:
        plt.scatter(
            reduced_embeddings[negative_indices, 0],
            reduced_embeddings[negative_indices, 1],
            c="none",
            edgecolor="black",
            marker="X",
            s=50,
            label=f"Negative Design (< {negative_threshold})",
        )
        plt.scatter(
            reduced_embeddings[positive_indices, 0],
            reduced_embeddings[positive_indices, 1],
            c="none",
            edgecolor="red",
            marker="^",
            s=50,
            label=f"Positive Design (>= {positive_threshold})",
        )
        plt.scatter(
            reduced_embeddings[neutral_indices, 0],
            reduced_embeddings[neutral_indices, 1],
            c="none",
            edgecolor="gray",
            marker="o",
            s=50,
            label="Neutral Design (between)",
        )
        plt.legend()

    plt.title(
        f"UMAP projection\n{enzyme} - {substrate}\nMode: {opmode}, {finetune_tag}"
    )
    plt.savefig(f"{save_dir}/umap_projection_{opmode}_{finetune_tag}.png")


def plot_logos(
    sequences,
    cluster_labels,
    results_dir: str,
    opmode: str,
    finetune_tag: str,
    cluster_type: str = "kmeans_embedding",
    enzyme="PTE",
    clustering_type="kmeans_embedding",
):
    "Create sequence logos for each cluster and plot them in the same figure"
    n_clusters = len(np.unique(cluster_labels))
    sorted_cluster_labels = np.sort(np.unique(cluster_labels))
    # create one figure with n_clusters subplots. Each row should have 4 subplots
    n_cols = 4
    n_rows = int(np.ceil(n_clusters / n_cols))
    plt.figure(figsize=(n_cols * 4, n_rows * 4))
    for i, cluster_id in enumerate(sorted_cluster_labels):
        cluster_seqs = sequences[np.where(cluster_labels == cluster_id)[0]]
        # create a count matrix
        count_matrix = lm.alignment_to_matrix(sequences=cluster_seqs, to_type="counts")
        ax = plt.subplot(n_rows, n_cols, i + 1)
        # create a logo
        logo = lm.Logo(count_matrix, color_scheme="chemistry", ax=ax)
        logo.style_spines(visible=False)
        logo.style_spines(spines=["left", "bottom"], visible=True)
        logo.style_xticks(rotation=90, fmt="%d", anchor=0)
        plt.title(f"Cluster {cluster_id} (n={len(cluster_seqs)})")
    plt.suptitle(f"Sequence Logos for {enzyme} Clusters", fontsize=16)
    plt.savefig(
        f"{results_dir}/cluster_logos_{clustering_type}_{opmode}_{finetune_tag}_{cluster_type}.png"
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


def cluster_embeddings_dbscan(
    embeddings,
    eps: float,
    min_samples: int,
    save_path: str,
):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cluster_labels = model.fit_predict(embeddings)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"DBSCAN found {n_clusters} clusters.")
    representative_indices = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_center = np.mean(cluster_embeddings, axis=0)
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
    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="average", metric="precomputed"
    )
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


def get_designed_indices(
    dataset_path: str, positive_threshold: float = 15, negative_threshold: float = 0.5
):
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
    save_path: str,
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
    plt.savefig(
        f"{save_path}/cluster_coverage_{opmode}_{finetune_tag}_{clustering_type}.png"
    )


def plot_sequence_clusters_dendrogram(
    sequences,
    n_clusters: int,
    enzyme: str,
    substrate: str,
    save_path: str,
):
    "Plot dendrogram of sequence clusters"

    # convert sequences to numpy array of shape (n_sequences, n_pos)
    sequences = np.array([list(seq) for seq in sequences])
    distance_matrix = pairwise_distances(
        sequences, metric=lambda x, y: sum(c1 != c2 for c1, c2 in zip(x, y))
    )
    condensed_distance = squareform(distance_matrix)
    linked = linkage(condensed_distance, method="average")

    plt.figure(figsize=(10, 7))
    dendrogram(
        linked, orientation="top", distance_sort="descending", show_leaf_counts=True
    )
    plt.title(f"Dendrogram of Sequence Clusters: {enzyme} - {substrate}")
    plt.savefig(f"{save_path}/sequence_clusters_dendrogram.png")


def pca_variance_plot(
    embeddings,
    n_components: int,
    save_path,
    opmode: str,
    finetune_tag: str,
):
    "Plot explained variance ratio of PCA components"
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.cumsum(explained_variance),
        marker="o",
    )
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid()
    plt.savefig(f"{save_path}/pca_explained_variance_{opmode}_{finetune_tag}.png")

    # plot heatmap of first 10 components
    plt.figure(figsize=(10, 6))
    plt.imshow(pca.components_, aspect="auto", cmap="viridis")
    plt.colorbar(label="Component Weight")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("PCA Component")
    plt.title("PCA Component Weights Heatmap")
    plt.savefig(f"{save_path}/pca_components_heatmap_{opmode}_{finetune_tag}.png")

    # save components as csv
    components_df = pd.DataFrame(pca.components_)
    components_df.to_csv(
        f"{save_path}/pca_components_{opmode}_{finetune_tag}.csv", index=False
    )   


def distance_histogram(embeddings, save_path: str, opmode: str, finetune_tag: str):
    "Plot histogram of pairwise distances between embeddings"
    distance_matrix = pairwise_distances(embeddings, metric="euclidean")
    # get upper triangle distances without diagonal
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    plt.figure(figsize=(8, 6))
    plt.hist(distances, bins=50, color="blue", alpha=0.7)
    plt.xlabel("Pairwise Distance")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pairwise Distances Between Embeddings")
    plt.grid()
    plt.savefig(f"{save_path}/pairwise_distance_histogram_{opmode}_{finetune_tag}.png")


def within_cluster_distances(embeddings, cluster_labels, save_path: str):
    "Compute within-cluster distances for each cluster"
    "plot boxplots of within-cluster distances - one subplot per cluster,"
    " all in one figure"  
    n_clusters = len(np.unique(cluster_labels))
    sorted_labels = np.sort(np.unique(cluster_labels))
    within_distances = []
    for cluster_id in sorted_labels:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        distance_matrix = pairwise_distances(cluster_embeddings, metric="euclidean")
        # get upper triangle distances without diagonal
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        within_distances.append(distances)
    plt.figure(figsize=(12, 6))
    plt.boxplot(within_distances, positions=sorted_labels)
    plt.xlabel("Cluster ID")
    plt.ylabel("Within-Cluster Pairwise Distances")
    plt.title("Within-Cluster Pairwise Distances")
    plt.grid()
    plt.savefig(save_path)


def position_specific_vocabulary(sequences, cluster_labels, position, save_path: str):
    "plot a breakdown of the vocabulary at a specific position for each cluster"
    "plot bar charts of amino acid frequencies at the position for each cluster"
    n_clusters = len(np.unique(cluster_labels))
    sorted_labels = np.sort(np.unique(cluster_labels))
    vocab_data = []
    for cluster_id in sorted_labels:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_seqs = sequences[cluster_indices]
        aa, counts = np.unique(
            [seq[position] for seq in cluster_seqs], return_counts=True
        )
        total = sum(counts)
        for a, c in zip(aa, counts):
            vocab_data.append(
                {
                    "cluster_id": cluster_id,
                    "amino_acid": a,
                    "frequency": c / total,
                }
            )
    vocab_df = pd.DataFrame(vocab_data)
    # plot bar chart
    plt.figure(figsize=(12, 6))
    for cluster_id in sorted_labels:
        cluster_data = vocab_df[vocab_df["cluster_id"] == cluster_id]
        plt.bar(
            cluster_data["amino_acid"] + f"_c{cluster_id}",
            cluster_data["frequency"],
            label=f"Cluster {cluster_id}",
        )
    plt.xlabel("Amino Acid")
    plt.ylabel("Frequency at Position {position}")
    plt.title(f"Amino Acid Vocabulary at Position {position} Across Clusters")
    plt.legend()
    plt.savefig(f"{save_path}/position_{position}_vocabulary_across_clusters.png")





def main():
    n_clusters = 25
    config = load_config("config_PTE.yaml")
    embeddings = np.load(config["embeddings_path"])
    # distance_histogram(
    #     embeddings,
    #     save_path=config["results_path"],
    #     opmode=config["opmode"],
    #     finetune_tag=config["finetune_tag"],
    # )
    
    # pca_variance_plot(
    #     embeddings,
    #     n_components=20,
    #     save_path=config["results_path"],
    #     opmode=config["opmode"],
    #     finetune_tag=config["finetune_tag"],
    # )
    cluster_labels, representative_indices = cluster_embeddings(
        embeddings,
        n_clusters=n_clusters,
        save_path=config["embeddings_clusters_path"],
    )

    # plot_embeddings(
    #     embeddings,
    #     enzyme=config["enzyme"],
    #     substrate=config["substrate"],
    #     save_dir=config["results_path"],
    #     cluster_labels=cluster_labels,
    #     finetune_tag=config["finetune_tag"],
    #     opmode=config["opmode"],
    # )

    # within_cluster_distances(
    #     embeddings,
    #     cluster_labels,
    #     save_path=f"{config['results_path']}/within_cluster_distances_{config['opmode']}_{config['finetune_tag']}_kmeans.png",
    # )

    # dbscan_cluster_labels, dbscan_representative_indices = cluster_embeddings_dbscan(
    #     embeddings,
    #     eps=0.52,
    #     min_samples=10,
    #     save_path=f"{config['results_path']}/embeddings_dbscan_clusters_{config['opmode']}_{config['finetune_tag']}.npy",
    # )

    # within_cluster_distances(
    #     embeddings,
    #     dbscan_cluster_labels,
    #     save_path=f"{config['results_path']}/within_cluster_distances_{config['opmode']}_{config['finetune_tag']}_dbscan.png",
    # )
    

    sequences_df = pd.read_csv(config["dataset_path"])
    # use only the specific positions
    sequences = sequences_df["full_seq"]
    sequences = sequences.apply(
        lambda x: "".join([x[i - 1] for i in config["pos_to_use"]])
    )

    position_specific_vocabulary(
        sequences,
        cluster_labels,
        position=2,
        save_path=config["results_path"],
    )

    # plot_logos(
    #     sequences,
    #     results_dir=config["results_path"],
    #     cluster_labels=dbscan_cluster_labels,
    #     finetune_tag=config["finetune_tag"],
    #     opmode=config["opmode"],
    #     cluster_type="dbscan",
    # )

    # print representative sequences
    # print("Representative sequences for each cluster:")
    # for cluster_id, rep_idx in enumerate(representative_indices):
    #     print(f"Cluster {cluster_id}: Sequence Index {rep_idx}, Sequence: {sequences.iloc[rep_idx]}")
    #     print(sequences_df.iloc[rep_idx])

    
    # save representatives as df

    return
    representatives_df = sequences_df.iloc[representative_indices]
    representatives_df.to_csv(
        f"{config['results_path']}/cluster_representative_sequences_{config['opmode']}_{config['finetune_tag']}.csv",
        index=False,
    )

    dbscan_representatives_df = sequences_df.iloc[dbscan_representative_indices]
    dbscan_representatives_df.to_csv(
        f"{config['results_path']}/dbscan_cluster_representative_sequences_{config['opmode']}_{config['finetune_tag']}.csv",
        index=False,
    )

    plot_logos(
        sequences,
        results_dir=config["results_path"],
        cluster_labels=cluster_labels,
        finetune_tag=config["finetune_tag"],
        opmode=config["opmode"],
    )

    sequence_clusters_labels, sequence_clusters_representatives = cluster_sequences(
        sequences.values,
        n_clusters=n_clusters,
        save_path=config["sequence_clusters_path"],
    )

    plot_logos(
        sequences,
        results_dir=config["results_path"],
        cluster_labels=sequence_clusters_labels,
        finetune_tag=config["finetune_tag"],
        opmode=config["opmode"],
        clustering_type="agglomerative_sequence",
    )   

    # plot_cluster_coverage(
    #     cluster_labels,
    #     positive_indices,
    #     negative_indices,
    #     save_path=config["results_path"],
    #     clustering_type="kmeans_embedding",
    #     opmode=config["opmode"],
    #     finetune_tag=config["finetune_tag"],
    # )
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

    # plot_sequence_clusters_dendrogram(
    #     sequences,
    #     n_clusters=n_clusters,
    #     enzyme=config["enzyme"],
    #     substrate=config["substrate"],
    #     save_path=config["substrate_specific_results_path"],
    # )

    fold_improvements = sequences_df[sequences_df["design"] != -1]["fold_improvement"]


if __name__ == "__main__":
    main()
