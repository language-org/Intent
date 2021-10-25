import logging
import os

import pandas as pd
import yaml
from nltk.corpus import wordnet as wn

from . import features

PROJ_PATH = os.getenv("PROJ_PATH")

PARAMS_PATH = os.path.join(
    PROJ_PATH, "conf/base/parameters.yml"
)
with open(PARAMS_PATH) as file:
    prms = yaml.load(file)

# configurate logging
logging_path = os.path.join(
    PROJ_PATH + "/conf/base/logging.yml"
)
with open(logging_path, "r") as f:
    LOG_CONF = yaml.load(f, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger(__name__)


def sample(tr_data: pd.DataFrame) -> pd.DataFrame:
    """Sample queries

    Args:
        tr_data (pd.DataFrame): corpus of queries

    Returns:
        pd.DataFrame: [description]
    """

    # data = tr_data[tr_data["category"].eq(prms["intent_class"])]
    data = tr_data[
        tr_data["category"].isin(prms["intent_class"])
    ]
    if len(data) > prms["sampling"]["sample"]:
        sample = data.sample(
            prms["sampling"]["sample"],
            random_state=prms["sampling"]["random_state"],
        )
    else:
        sample = data

    # set index as a tracking column
    sample.reset_index(level=0, inplace=True)
    return sample


def filter_by_sent_count(
    query: pd.DataFrame, thresh: int, verbose: bool = False
) -> pd.Series:
    """Filter queries with a number of sentences above a threshold

    Args:
        query (pd.DataFrame): dataset of queries in column 'text' each made
            of 1 to N sentences
        threshold (int): max number of sentences per query allowed in filtered dataset
        verbose: (bool)

    Returns:
        pd.Series: filtered queries
    """
    count = pd.Series(features.count(query["text"]))
    if verbose:
        logger.info(
            f"There are {len(count)} original queries."
        )
        logger.info(
            f"{len(query[count <= thresh])} after filtering queries with > {thresh} sentence."
        )
    return query[count <= thresh]


def filter_n_sent_eq(
    query: pd.DataFrame, n_sent: int, verbose: bool = False
) -> pd.Series:
    """Filter queries dataset, keep query rows w/ N = n_sent sentences

    Args:
        query (pd.DataFrame): dataset of queries in column 'text' each made
            of 1 to N sentences
        threshold (int): number of sentences per query allowed in filtered dataset
        verbose: (bool)

    Returns:
        pd.Series: filtered queries
    """
    count = pd.Series(features.count(query["text"]))
    if verbose:
        logger.info(
            f"There are {len(count)} input queries."
        )
        logger.info(
            f"{len(query[count == n_sent])} after filtering queries with {n_sent} sentence"
        )
    return query[count == n_sent]


def filter_mood(
    cfg: pd.DataFrame, FILT_MOOD: str
) -> pd.Series:
    """Filter queries by their mood

    Args:
        cfg (pd.DataFrame): queries' context-free-grammar
        FILT_MOOD (str or tuple): selected mood(s)

    Returns:
        pd.Series: [description]
    """

    mood_set = ("ask", "state", "wish-or-excl")

    # convert FILT_MOOD to set
    if isinstance(FILT_MOOD, str):
        FILT_MOOD = {FILT_MOOD}
    elif isinstance(FILT_MOOD, tuple):
        FILT_MOOD = set(FILT_MOOD)

    # detect the mood(s) to drop
    try:
        to_drop = set(mood_set).difference(FILT_MOOD)
    except:
        raise (
            ValueError(
                """(filter_mood) Please set FILT_MOOD. 
                """
            )
        )

    # classify sentence types (state, ask, ..)
    query_moods = features.classify_mood(cfg["text"])

    # filter indices
    ix = [
        ix
        for ix, mood in enumerate(query_moods)
        if not set(mood).isdisjoint(FILT_MOOD)
        and set(mood).isdisjoint(set(to_drop))
    ]

    # add moods to data
    mood_filt = (
        pd.DataFrame(query_moods).iloc[ix].reset_index()
    )
    nb_moods = len(mood_filt.columns) - 1
    mood_filt.columns = ["ix"] + [
        f"mood_{i}" for i in range(nb_moods)
    ]
    cfg_filt = cfg.iloc[ix].reset_index()
    cfg = pd.concat(
        [cfg_filt, mood_filt], ignore_index=False, axis=1,
    )
    return cfg


def filter_words_not_in_wordnet(corpus: tuple) -> tuple:
    """Filter mispelled words (absent from wordnet)

    [TO BE DEPRECATED]

    Args:
        corpus (tuple): tuple of queries

    Returns:
        [tuple]: tuple of queries from which mispelled words have been filtered
    """
    # find mispelled words
    misspelled = []
    for query in corpus:
        if query:
            query = query.split()
        for word in query:
            if not wn.synsets(word):
                misspelled.append(word)

    # filter them from corpus
    queries = []
    for query in corpus:
        if query:
            query = query.split()
        filtered = []
        for word in query:
            if not word in misspelled:
                filtered.append(word)
        queries.append(" ".join(filtered))
    return tuple(queries)


def filter_words(corpus: pd.Series, how: str) -> tuple:
    """Filter mispelled words (absent from wordnet)

    [TO BE DEPRECATED]

    Args:
        corpus (tuple): tuple of queries
        how (str):
            "not_in_wordnet": remove words not in wordnet
    Returns:
        [tuple]: tuple of queries from which mispelled words have been filtered
    """
    if how == "not_in_wordnet":

        # find mispelled words
        misspelled = []
        for query in corpus:
            if query:
                query = query.split()
            for word in query:
                if not wn.synsets(word):
                    misspelled.append(word)

        # filter them from corpus
        queries = []
        for query in corpus:
            if query:
                query = query.split()
            filtered = []
            for word in query:
                if not word in misspelled:
                    filtered.append(word)
            queries.append(" ".join(filtered))
    return pd.Series(queries, index=corpus.index)


def filter_empty_queries(corpus: tuple) -> tuple:
    """Filter out empty queries

    [TO BE DEPRECATED]

    Args:
        corpus (tuple): corpus of string queries

    Returns:
        tuple: corpus of string queries without empty queries
    """
    return tuple(filter(None, corpus))


def drop_empty_queries(corpus: pd.Series) -> tuple:
    """Filter out empty queries

    Args:
        corpus (pd.Series): corpus of string queries

    Returns:
        pd.Series: corpus of string queries without empty queries
    """
    not_empty = ~corpus.isin(["", None])
    return corpus[not_empty], not_empty
