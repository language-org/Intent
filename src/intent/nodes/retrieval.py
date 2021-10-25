from collections import defaultdict

import pandas as pd


def create_posting_list(constituents: pd.Series) -> dict:
    """Create a posting list with key:value made of constituents:index in dataframe

    Args:
        cfg (pd.DataFrame): [description]

    Returns:
        dict: posting list (a dictionary of listed position indices)
    """
    posting_list = defaultdict(list)
    for ix, cfg in enumerate(constituents):
        posting_list[cfg].append(ix)
    return posting_list


def create_posting_list_from_raw_ix(
    constituents: tuple, raw_indices: tuple
) -> dict:
    """Create a posting list with key:value made of constituents:index in dataframe

    Args:
        cfg (pd.DataFrame): [description]

    Returns:
        dict: posting list (a dictionary of listed position indices)
    """
    posting_list = defaultdict(list)
    for raw_ix, cfg in zip(raw_indices, constituents):
        posting_list[cfg].append(raw_ix)
    return posting_list
