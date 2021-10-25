# author: Steeve LAQUITAINE

import re

import pandas as pd

# either ? or ! or .
SENT_TYPE_PATTN = re.compile(r"[\?\!\.]")


def classify_mood(sentences: pd.Series) -> list:
    """Classify sentence type: ask, state, wish-or-excl,..

    Args:
        sentences (pd.Series): series of queries: a query can contain several sentences  

    Returns:
        [type]: list of list of mood for each query sentence.
    """
    sent_type = []
    for sent in sentences:
        out = SENT_TYPE_PATTN.findall(sent)
        sent_type.append(
            [
                "ask"
                if ix == "?"
                else "wish-or-excl"
                if ix == "!"
                else "state"
                for ix in out
            ]
        )
    return sent_type


def detect_sentence_type(df: pd.DataFrame, sent_type: str):
    """
    Detect sentence types

    parameters
    ----------
    sent_type: str
        'state', 'ask', 'wish-excl' 
    """
    return sent_type in df


def count(query: pd.DataFrame) -> list:
    """Count number of sentences in query

    Args:
        query (pd.DataFrame): a query per row

    Returns:
        list: each query's sentence count
    """
    return [len(SENT_TYPE_PATTN.findall(sent)) for sent in query]
