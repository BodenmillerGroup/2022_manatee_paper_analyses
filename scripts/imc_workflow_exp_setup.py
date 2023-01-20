import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn import metrics
from sklearn.cluster import KMeans

# positive_coexpression_pairs = [
#     ["MPO", "CD15"],
#     ["CD16", "HLADR"],
#     ["CD16", "CD163"],
#     ["CD16", "CD68"],
#     ["CD16", "CD11c"],
#     ["CD16", "CD40"],
#     ["CD16", "CD40"],
#     ["CD16", "CD14"],
#     ["CD16", "CD206"],
#     ["CD38", "CD27"],
#     ["HLADR", "CD163"],
#     ["HLADR", "B2M"],
#     ["HLADR", "CD68"],
#     ["HLADR", "CD11c"],
#     ["HLADR", "CD40"],
#     ["HLADR", "CD4"],
#     ["HLADR", "CD14"],
#     ["HLADR", "CD206"],
#     ["CD27", "CD45RA"],
#     ["CD27", "B2M"],
#     ["CD27", "CD3"],
#     ["CD27", "TCF7"],
#     ["CD27", "CD45RO"],
#     ["CD27", "CD4"],
#     ["CD45RA", "CD20"],
#     ["CD163", "CD68"],
#     ["CD163", "CD11c"],
#     ["CD163", "CD14"],
#     ["CD163", "CD206"],
#     ["B2M", "PDL1"],
#     ["B2M", "CD40"],
#     ["B2M", "CD4"],
#     ["B2M", "CD14"],
#     ["CD68", "CD11c"],
#     ["CD68", "PDL1"],
#     ["CD68", "CD40"],
#     ["CD68", "CD4"],
#     ["CD68", "CD14"],
#     ["CD68", "CD206"],
#     ["CD3", "PD1"],
#     ["CD3", "CD7"],
#     ["CD3", "CD45RO"],
#     ["CD3", "ICOS"],
#     ["CD3", "CD8a"],
#     ["CD3", "CD4"],
#     ["LAG3 / LAG33", "PD1"],
#     ["LAG3 / LAG33", "GrzB"],
#     ["LAG3 / LAG33", "ICOS"],
#     ["CD11c", "VISTA"],
#     ["CD11c", "CD40"],
#     ["CD11c", "CD4"],
#     ["CD11c", "CD14"],
#     ["CD11c", "CD206"],
#     ["PD1", "GrzB"],
#     ["PD1", "ICOS"],
#     ["PD1", "CD40"],
#     ["PD1", "CD4"],
#     ["CD7", "CD45RO"],
#     ["CD7", "ICOS"],
#     ["CD7", "CD8a"],
#     ["CD7", "CD4"],
#     ["CD45RO", "CD8a"],
#     ["CD45RO", "CD4"],
#     ["Ecad", "CarbonicAnhydrase"],
#     ["VISTA", "CD40"],
#     ["VISTA", "CD4"],
#     ["CD14", "CD206"],
# ]

positive_coexpression_pairs = [
    ["MPO", "CD15"],
    ["CD45RA", "CD20"],
    ["CD163", "CD68"],
    ["CD163", "CD206"],
    ["CD68", "CD11c"],
    ["CD68", "CD14"],
    ["CD3", "CD8a"],
    ["CD3", "CD4"],
    ["CD11c", "CD206"],
    ["CD45RO", "CD8a"],
    ["CD45RO", "CD4"],
    ["Ecad", "CarbonicAnhydrase"],
    ["CD14", "CD206"],
]


def load_data():
    # This function loads IMC Workflow dataset.

    # Load data
    df = pd.concat(
        {
            f.name: pd.read_csv(f, index_col="Object")
            for f in sorted(Path("../data/imc_workflow/intensities").glob("*.csv"))
        },
        names=["Image", "Cell"],
    )
    df.index = (
        df.index.get_level_values("Image").str[:-4]
        + "_"
        + df.index.get_level_values("Cell").astype(str)
    ).to_numpy()
    df.drop(columns=["DNA1", "DNA2", "HistoneH3"], inplace=True)
    clusters = pd.read_csv("../data/imc_workflow/cell_metadata.csv", index_col=0)

    df = df.loc[np.isin(df.index, clusters.index), :]
    clusters = clusters.loc[df.index, :]

    adata = AnnData(df, dtype=np.float32)
    adata.obs["target"] = clusters["pg_clusters_corrected"].tolist()

    # Subsample
    seed = 10
    random.seed(a=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    adata = adata[np.random.choice(adata.shape[0], size=5000, replace=False), :]
    print(f"max in subsampled adata: {adata.X.max()}")
    print(f"shape of adata: {adata.shape}")

    # Load clusters
    Y = adata.X.copy()
    Y = np.arcsinh(Y)
    return adata, Y


def split_data(adata, Y):
    # Split subsampled data to train and test
    train = np.random.choice(
        adata.shape[0], size=int(adata.shape[0] * 0.7), replace=False
    )
    adata_train = adata[train, :]
    Y_train = Y[train, :]

    test = set(range(adata.shape[0])).difference(set(train))
    adata_test = adata[list(test), :]
    Y_test = Y[list(test), :]
    return adata_train, Y_train, adata_test, Y_test


# This function computes correlation between cluster
# mean expression.
def get_coexpression(cluster_means, p1, p2, var_names):
    df = pd.DataFrame(cluster_means, columns=var_names)
    return df.corr()[p1][p2]


# This function arcisnh-normalised the data with
# the given cofactor and clusters it with k-means.
def norm_adata(adata, cofactor=5.0, n_clusters=10):
    # Normalise data with cofactor
    adata.X = np.arcsinh(adata.X / cofactor)
    km = KMeans(n_clusters, random_state=int(np.random.choice(1000, 1)), n_init=10)
    # Cluster data
    km.fit(adata.X)
    adata.obs["leiden"] = km.labels_
    return adata


# This function computes the clustering resulting from a given cofactor
# and computes objective values for it (supervised objectives are
# w.r.t. true labels).
def true_f(cofactors, adata, Y):
    results = {}

    from collections import defaultdict

    results_list = defaultdict(list)
    ari = []
    nmi = []

    for cofactor in cofactors:
        results[cofactor] = {}
        adata_norm = norm_adata(adata.copy(), cofactor)

        unique_clusters = np.unique(adata_norm.obs.leiden)
        cluster_means = np.concatenate(
            [
                Y[adata_norm.obs.leiden == cl, :].mean(0).reshape(1, -1)
                for cl in unique_clusters
            ],
            axis=0,
        )

        for pair in positive_coexpression_pairs:
            pair_str = pair[0] + "_" + pair[1] + "_+"
            results[cofactor][pair_str] = get_coexpression(
                cluster_means, pair[0], pair[1], adata_norm.var_names
            ).astype(np.float32)

        ari.append(
            metrics.adjusted_rand_score(
                adata_norm.obs.target.astype(str), adata_norm.obs.leiden
            )
        )
        nmi.append(
            metrics.normalized_mutual_info_score(
                adata_norm.obs.target.astype(str), adata_norm.obs.leiden
            )
        )

        for k, v in results[cofactor].items():
            results_list[k].append(v)

    pos_pairs = []
    for pair in positive_coexpression_pairs:
        pair_str = pair[0] + "_" + pair[1] + "_+"
        obj = np.array(results_list[pair_str])
        pos_pairs.append(torch.reshape(torch.tensor([obj]), (obj.shape[0], 1)))

    ari = np.array(ari)
    nmi = np.array(nmi)
    ari = torch.reshape(torch.tensor([ari]), (ari.shape[0], 1))
    nmi = torch.reshape(torch.tensor([nmi]), (nmi.shape[0], 1))

    y = torch.cat(pos_pairs, axis=1)

    return y, ari, nmi


def get_labels():
    labels = []
    for pair in positive_coexpression_pairs:
        labels.append(pair[0] + "_" + pair[1] + "_+")

    return labels
