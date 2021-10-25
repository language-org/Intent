from pathlib import Path

import pandas as pd

from src.intent.nodes import parsing, similarity

import pytest


@pytest.fixture
def project_context():
    return ProjectContext(str(Path.cwd()))


class TestProjectContext:
    def test_project_name(self, project_context):
        assert project_context.project_name == "intent"

    def test_project_version(self, project_context):
        assert project_context.project_version == "0.16.2"


def test_extract_all_VPs(VPs, data, prm):
    assert len(VPs) == len(data) or len(VPs) == prm["sample"]
    , '''VP's length does not match "data"'''


def test_extract_VP(al_prdctor):
    assert (
        len(parsing.extract_VP(al_prdctor, "I want coffee"))
        > 0
    ), "VP is Empty"


def test_annots_df(annots_df: pd.DataFrame):
    assert set(annots_df.columns).issuperset(
        {"index", "VP", "annots"}
    ), """ "index", "VP", "annots" columns are missing" """


def simil_matx(simil_matx):
    assert (
        simil_matx.shape[0] + 1 == simil_matx.shape[1]
    ), "similarity matrix shape should be (n, n+1)"
    assert (
        not len(simil_matx) == 0
    ), "similarity matrix is empty"


def test_len_similarity_matx(
    cfg: pd.DataFrame, sim_matx: pd.DataFrame
):
    tag = parsing.chunk_cfg(cfg["cfg"])

    assert tag.nunique() == len(
        sim_matx
    ), f""" Number of unique constituents in 'cfg' {tag.nunique()} must match 'len(sim_matx)' {len(
        sim_matx)} """


def test_rank_nearest_to_seed(
    sim_matx: pd.DataFrame, seed: str
):
    l_ranked = len(
        similarity.rank_nearest_to_seed(sim_matx, seed=seed)
    )
    assert l_ranked == len(
        sim_matx
    ), """ The length of 'rank_nearest_to_seed()''s output should match len(sim_matx) """


def test_posting_list(
    posting_list: dict, sim_matx: pd.DataFrame, seed: str
):

    ranked = similarity.rank_nearest_to_seed(
        sim_matx, seed=seed, verbose=False
    )
    assert (
        len(
            set(posting_list.keys()).difference(
                set(ranked.index)
            )
        )
        == 0
    ), """ posting_list and 'rank_nearest_to_seed''s output should have the 
    same set of constituents"""


def test_get_posting_index(
    cfg: pd.DataFrame,
    posting_list: dict,
    sorted_series: pd.Series,
) -> list:
    l_index = len(
        similarity.get_posting_index(
            posting_list, sorted_series
        )
    )
    assert l_index == len(
        cfg
    ), f""" 'index' length {l_index} must be same as 'cfg' length {len(cfg)} """
