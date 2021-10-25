# author: steeve laquitaine

import os
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance
from src.intent.nodes import similarity
from typing import Dict, Any


def cluster_queries(
    text: pd.Series,
    dist_thresh: float,
    verbose: bool = False,
    hcl_method=None,
    params: Dict[str, Any] = dict(),
) -> pd.DataFrame:
    """Label queries using semantic hierarchical clustering

    Args:
        text (pd.Series): series of text queries with their raw indices
        DIST_THRES (float):
            hierarchical clustering thresholding parameter
            distance threshold t - the maximum inter-cluster distance allowed
        verbose (bool, optional): print or not. Defaults to False.
        hcl_method (str):

    Returns:
        pd.DataFrame: dataframe of queries with their cluster label

    Usage:
        text = (
        "want to drink a cup of coffee",
        "want to drink a cup of tea",
        "would like a bottle of water",
        "want to track my credit card"
        )
        df = label_queries(text, 1.8)
    """
    CRITERION = params["CRITERION"]
    MAX_CLUST = params["MAX_CLUST"]

    t0 = time()

    # convert pandas series to tuple
    text_tuple = tuple(text)

    # compute query similarity matrix
    sim_mtx = similarity.get_semantic_similarity_matrix(
        text_tuple
    )
    sim_mtx = pd.DataFrame(sim_mtx)

    # patch weird values with -1
    sim_mtx[np.logical_or(sim_mtx < 0, sim_mtx > 1)] = -0.1
    sim_mtx[sim_mtx.isnull()] = -1

    # apply hierarchical clustering
    row_linkage = hierarchy.linkage(
        distance.pdist(sim_mtx), method=hcl_method
    )
    if verbose:
        sns.clustermap(
            sim_mtx,
            row_linkage=row_linkage,
            method="average",
            figsize=(13, 13),
            cmap="vlag",
        )
    label = fcluster(
        row_linkage, t=MAX_CLUST, criterion=CRITERION,
    )
    if verbose:
        print(f"{round(time() - t0, 2)} secs")

    # convert to dataframe
    labelled = pd.DataFrame([text_tuple, label]).T.rename(
        columns={0: "query", 1: "cluster_labels"}
    )

    # keep corpus indices
    labelled.index = text.index

    # sort by label
    labelled_sorted = labelled.sort_values(
        by=["cluster_labels"]
    )

    return labelled_sorted
