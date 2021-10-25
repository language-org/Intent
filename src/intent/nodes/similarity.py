# author: steeve laquitaine

from difflib import SequenceMatcher
from itertools import chain, repeat
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from nltk.corpus import wordnet as wn
from pywsd.lesk import simple_lesk
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance
import logging

logger = logging.getLogger()

nlp = spacy.load("en_core_web_sm")

# graph edit distance's cost functions
def node_subst_cost(node1, node2):
    if node1 == node2:
        return 0
    return 1


def node_del_cost(node):
    return 1


def node_ins_cost(node):
    return 1


def edge_subst_cost(edge1, edge2):
    if edge1 == edge2:
        return 0
    return 1


def edge_del_cost(node):
    return 1  # here you apply the cost for edge deletion


def edge_ins_cost(node):
    return 1  # here you apply the cost for edge insertion


def calc_ged(graphs_of_VPs: list):
    """Calculate graph edit distance

    Args:
        graphs_of_VPs (list): list of networkx graphs

    Returns:
        [np.array]: matrix of pairwise graph edit distances
    """
    n_graphs = len(graphs_of_VPs)

    ged_sim = np.zeros((n_graphs, n_graphs))
    for ix in range(n_graphs):
        for jx in range(n_graphs):
            ged_sim[ix, jx] = nx.graph_edit_distance(
                graphs_of_VPs[ix],
                graphs_of_VPs[jx],
                node_subst_cost=node_subst_cost,
                node_del_cost=node_del_cost,
                node_ins_cost=node_ins_cost,
                edge_subst_cost=edge_subst_cost,
                edge_del_cost=edge_del_cost,
                edge_ins_cost=edge_ins_cost,
            )
    return ged_sim


def calc_lcs(str1: str, str2: str) -> float:
    """Calculate the length ratio of the longest common subsequence between two strings

    Args:
        str1 (str): string to match
        str2 (str): string to match

    Returns:
        float: length ratio of the longest common subsequence b/w str1 and str2
    """
    s = SequenceMatcher(None, str1, str2)
    match = s.find_longest_match(0, len(str1), 0, len(str2))
    match_content = str1[match.a : match.size]
    lcs_similarity = s.ratio()
    return lcs_similarity


def print_ranked_VPs(
    cfg: pd.DataFrame,
    posting_list,
    sorted_series: pd.Series,
) -> pd.Series:
    """Rank verb phrases by syntactic similarity score

    Args:
        cfg (pd.DataFrame): context free grammar production rules (VP -> VB NP)
        posting_list (defauldict): list of position indices for the production right side (e.g., VB NP)
        sorted_series (pd.Series): [description]

    Returns:
        pd.Series: syntactic similarity score b/w seed query and other queries
    """

    index = get_posting_index(posting_list, sorted_series)

    score = list(
        chain.from_iterable(
            [
                list(
                    repeat(
                        sorted_series[ix],
                        len(
                            posting_list[
                                sorted_series.index[ix]
                            ]
                        ),
                    )
                )
                for ix in range(len(sorted_series))
            ]
        )
    )

    ranked_vps = cfg["VP"].iloc[index]
    df = pd.DataFrame(ranked_vps, columns=["VP"])
    df["score"] = score
    return df


def print_ranked_VPs_on_raw_ix(
    cfg: pd.DataFrame,
    posting_list,
    sorted_series: pd.Series,
) -> pd.Series:
    """Rank verb phrases by syntactic similarity score

    Args:
        cfg (pd.DataFrame): 
            context free grammar production rules (VP -> VB NP)
        posting_list (defauldict): 
            list of production's right side indices (e.g., VB NP)
        sorted_series (pd.Series): [description]

    Returns:
        pd.Series: 
            syntactic similarity b/w seed & queries
    """
    # flatten posting list into a list of queries of raw
    # indices in original corpus
    index = get_posting_index_on_raw_ix(
        posting_list, sorted_series
    )

    # list production's right side score of similarity
    # with seed
    score = list(
        chain.from_iterable(
            [
                list(
                    repeat(
                        sorted_series[ix],
                        len(
                            posting_list[
                                sorted_series.index[ix]
                            ]
                        ),
                    )
                )
                for ix in range(len(sorted_series))
            ]
        )
    )
    # get verb phrases via raw index
    cfg.index = cfg["index"]
    ranked_vps = cfg["VP"].loc[index]
    df = pd.DataFrame(ranked_vps, columns=["VP"])
    df["score"] = score
    return df


def get_posting_index(
    posting_list: dict, sorted_series: pd.Series
) -> list:
    """Get indices from constituent posting list

    Args:
        posting_list (dict): [description]
        sorted_series (pd.Series): [description]

    Returns:
        list: [description]
    """
    index = list(
        chain.from_iterable(
            [
                posting_list[sorted_series.index[ix]]
                for ix in range(len(sorted_series))
            ]
        )
    )
    return index


def get_posting_index_on_raw_ix(
    posting_list: dict, sorted_series: pd.Series
) -> list:
    """Get indices from constituent posting list

    Args:
        posting_list (dict): [description]
        sorted_series (pd.Series): [description]

    Returns:
        list: [description]
    """
    # flatten posting list into a list of queries of raw
    # indices in original corpus
    index = list(
        chain.from_iterable(
            [
                posting_list[sorted_series.index[ix]]
                for ix in range(len(sorted_series))
            ]
        )
    )
    return index


def rank_nearest_to_seed(
    sim_matx: pd.DataFrame, seed: str, verbose: bool = False
) -> pd.Series:
    """Rank verb phrase syntaxes by similarity to a 'seed' syntax

    Args:
        sim_matx (pd.DataFrame): syntax similarity matrix
        seed (str): syntax seed
            e.g., 'VB NP'

    Returns:
        pd.Series: queries' syntax ranked in descending order of similarity to seed
    """
    constt_set = set(sim_matx.index)
    dedup = sim_matx[sim_matx.index == seed][constt_set].T
    if verbose:
        print(
            f"{len(sim_matx) - len(dedup)} duplicated cfgs were dropped."
        )
    sim_ranked = dedup.squeeze().sort_index(ascending=False)
    return sim_ranked


def filter_by_similarity(
    ranked: pd.DataFrame, thresh: float
) -> pd.DataFrame:
    """Filter queries dissimilar to intent seed syntax

    Args:
        ranked (pd.DataFrame): 
             "score" column contains query similarity scores (floats)
        thresh (float): 
            threshold. We keep queries with scores > threshold

    Returns:
        pd.DataFrame: [description]
    """
    ranked = ranked[ranked["score"] >= thresh]
    logger.info(
        f"{len(ranked)} querie(s) left after filtering."
    )
    return ranked


class SentenceSimilarity:
    """
    Sentence similarity
    """

    def __init__(self):
        self.word_order = False

    def identifyWordsForComparison(self, sentence):
        # Taking out Noun and Verb for comparison word based
        # tokens = nltk.word_tokenize(sentence)
        # pos = nltk.pos_tag(tokens)
        # pos = [p for p in pos if p[1].startswith("N") or p[1].startswith("V")]
        # [TODO]: speed up
        doc = nlp(sentence)
        pos = [(token.text, token.tag_) for token in doc]
        pos = [
            p
            for p in pos
            if p[1].startswith("N") or p[1].startswith("V")
        ]
        return pos

    def wordSenseDisambiguation(self, sentence):
        # removing the disambiguity by getting the context
        pos = self.identifyWordsForComparison(sentence)
        sense = []
        for p in pos:
            sense.append(
                simple_lesk(
                    sentence, p[0], pos=p[1][0].lower()
                )
            )
        return set(sense)

    def getSimilarity(self, arr1, arr2, vector_len):
        # cross multilping all domains
        vector = [0.0] * vector_len
        count = 0
        for i, a1 in enumerate(arr1):
            all_similarityIndex = []
            for a2 in arr2:
                similarity = wn.synset(
                    a1.name()
                ).wup_similarity(wn.synset(a2.name()))
                if similarity != None:
                    all_similarityIndex.append(similarity)
                else:
                    all_similarityIndex.append(0.0)
            all_similarityIndex = sorted(
                all_similarityIndex, reverse=True
            )
            vector[i] = all_similarityIndex[0]
            if vector[i] >= 0.804:
                count += 1
        return vector, count

    def shortestPathDistance(self, sense1, sense2):
        # getting the shortest path to get the similarity
        if len(sense1) >= len(sense2):
            grt_Sense = len(sense1)
            v1, c1 = self.getSimilarity(
                sense1, sense2, grt_Sense
            )
            v2, c2 = self.getSimilarity(
                sense2, sense1, grt_Sense
            )
        if len(sense2) > len(sense1):
            grt_Sense = len(sense2)
            v1, c1 = self.getSimilarity(
                sense2, sense1, grt_Sense
            )
            v2, c2 = self.getSimilarity(
                sense1, sense2, grt_Sense
            )
        return np.array(v1), np.array(v2), c1, c2

    def main(self, sentence1, sentence2):
        sense1 = self.wordSenseDisambiguation(sentence1)
        sense2 = self.wordSenseDisambiguation(sentence2)
        v1, v2, c1, c2 = self.shortestPathDistance(
            sense1, sense2
        )
        dot = np.dot(v1, v2)
        tow = (c1 + c2) / 1.8
        final_similarity = dot / tow
        return final_similarity


def get_semantic_similarity_matrix(text: tuple):
    """[summary]

    Args:
        text (tuple): [description]

    Returns:
        [type]: [description]
    """
    matx_all = []
    for t_i in text:
        matx = []
        for t_j in text:
            out = SentenceSimilarity().main(t_i, t_j)
            matx.append(out)
        matx_all.append(matx)
    return matx_all


def get_semantic_clusters(
    text: list, DIST_THRES: float, verbose: bool
) -> pd.DataFrame:
    """get semantic clusters

    Args:
        text (list): list of queries
        DIST_THRES (float): distance threshold
        verbose (bool): [description]

    Returns:
        pd.DataFrame: queries "text" with their cluster labels "cluster"
    """
    t0 = time()
    sem_sim_matx = get_semantic_similarity_matrix(text)
    if verbose:
        print(f"{round(time() - t0, 2)} secs")
    sem_sim_matx = pd.DataFrame(sem_sim_matx)

    # patch weird values with -1
    sem_sim_matx[
        np.logical_or(sem_sim_matx < 0, sem_sim_matx > 1)
    ] = -0.1
    sem_sim_matx[sem_sim_matx.isnull()] = -1

    row_linkage = hierarchy.linkage(
        distance.pdist(sem_sim_matx), method="average"
    )
    sns.clustermap(
        sem_sim_matx,
        row_linkage=row_linkage,
        method="average",
        figsize=(13, 13),
        cmap="vlag",
    )
    label = fcluster(
        row_linkage, t=DIST_THRES, criterion="distance"
    )
    clust = pd.DataFrame([text, label]).T
    clust.columns = ["text", "cluster"]
    return clust
