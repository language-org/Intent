# author: laquitainesteeve@gmail.com

import logging
import os
from collections import defaultdict
from datetime import datetime
from time import time

import allennlp_models.structured_prediction
import nltk
import nltk_tgrep
import numpy as np
import pandas as pd
import spacy
import stanza
import yaml
from allennlp.predictors.predictor import Predictor
from nltk.tree import ParentedTree
from stanza.server import CoreNLPClient

PROJ_PATH = os.getenv("PROJ_PATH")

from src.intent.nodes import parsing, preprocess, setup

# configurate logging
logging_path = os.path.join(
    PROJ_PATH + "/conf/base/logging.yml"
)
with open(logging_path, "r") as f:
    LOG_CONF = yaml.load(f, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger(__name__)

params = setup.get_params()
paths = setup.get_paths()

# load paths
catalog_path = os.path.join(
    PROJ_PATH, "conf/base/catalog.yml"
)
with open(catalog_path) as file:
    catalog = yaml.load(file)

# load parser
nlp = spacy.load("en_core_web_sm")


def init_allen_parser():
    """
    output[trees]
      '(SQ (SBAR (IN If) (S (NP (PRP I))
      (VP (VBP bring) (NP (CD 10) (NNS dollars))
      (NP (NN tomorrow))))) (, ,) (MD can)
      (NP (PRP you)) (VP (VB buy) (NP (PRP me)) (NP (NN lunch))) (. ?))'
    - takes: ~0.4 s per sentence (slow)
    """
    tic = time()
    out = Predictor.from_path(
        paths["path_constituency_parser"]
    )
    logger.info(
        f"(Instantiation) took {round(time()-tic,2)} secs"
    )

    return out


def init_Stanza_constituency_parsing(sample):
    """[summary]

    Args:
        sample ([type]): [description]

    Returns:
        [type]: [description]
    """
    tic = time()
    with CoreNLPClient(
        annotators=[
            "tokenize",
            "ssplit",
            "pos",
            "lemma",
            "parse",
            "depparse",
            "coref",
        ],
        timeout=30000,
        memory="16G",
        endpoint="http://localhost:8888",
        be_quiet=True,
    ) as client:
        parser = client.annotate(sample)
    logger.info(
        f"(init_Stanza_constituency_parsing) took {round(time()-tic,2)} secs"
    )
    return parser


def setup_stanza():
    """[summary]"""
    # install corenlp
    if not os.path.exists(paths["coreNLP_path"]):
        logger.info("(setup_stanza) Installing CoreNLP ...")
        stanza.install_corenlp(dir=paths["coreNLP_path"])
    else:
        logger.info(
            "(setup_stanza) CoreNLP already exists."
        )

    (status, model_file) = is_exist_model()
    if status:
        logger.info(
            "(setup_stanza) CoreNLP model already exists. The model is: {model_file}"
        )
    else:
        logger.info(
            "(setup_stanza) Downloading CoreNLP model ..."
        )
        stanza.download_corenlp_models(
            model=params["corenlp_model"]["lang"],
            version=params["corenlp_model"]["version"],
            dir=paths["coreNLP_path"],
        )

    logger.info(
        "(setup_stanza) Setting up Environment variables ..."
    )
    os.environ["CORENLP_HOME"] = paths["coreNLP_path"]


def is_exist_model():
    """Check if model exists in path

    Returns:
        bool: True or False
    """
    model_file = [
        file
        for file in os.listdir(paths["coreNLP_path"])
        if file.endswith(".jar")
        and params["corenlp_model"]["lang"] in file
        and params["corenlp_model"]["version"] in file
    ]
    if len(model_file) == 0:
        return False, model_file
    else:
        return True, model_file


def extract_VP(al_prdctor, query):
    """
    Parse and return the first verb phrase of a query

    Args:
        al_prdctor ([type]): [description]
        query ([type]): [description]

    Returns:
        [type]: [description]
    """
    output = al_prdctor.predict(sentence=query)
    parsed_txt = output["trees"]
    tree = ParentedTree.fromstring(parsed_txt)
    verb_p = nltk_tgrep.tgrep_nodes(tree, "VP")
    out = dict()

    if not len(verb_p) == 0:
        out["terminals"] = verb_p[0].leaves()
        out["tree"] = verb_p[0]
        out["message"] = "chunks found"
    else:
        out["terminals"] = out["tree"] = None
        out["message"] = "chunking failed"
    return out


def extract_all_VPs(data: pd.DataFrame, predictor):
    """parse all verb phrases from data

    Args:
        data (pd.DataFrame):
            rows: each contains a string
            columns: include name: 'text'
        predictor ([type]):
            allennlp's predictor for constituency parsing

    Returns:
        [type]: [description]
    """

    tic = time()
    VPs = []
    dur = []
    for ix in range(len(data)):
        t0 = time()
        VP = parsing.extract_VP(
            predictor, data["text"].iloc[ix]
        )
        if VP["terminals"] is not None:
            VPs.append(VP)
        else:
            VPs.append([])
        if ix <= 10:
            dur.append(time() - t0)
            estTime = round(len(data) * np.mean(dur), 2)
            logger.info(f"Time to completion: {estTime}")
    logger.info(f"{round(time()-tic,2)}")
    return VPs


def make_VPs_readable(VPs) -> list:
    """Convert verb phrases to list of strings of terminals

    Args:
        VPs ([type]): [description]

    Returns:
        [type]: [description]
    """
    VP_list = []
    for VP in VPs:
        if not len(VP) == 0:
            VP_list.append(" ".join(VP["terminals"]))
        else:
            VP_list.append([])
    return VP_list


def get_CFG(VP):
    """Get CFG productions of a query from its parsed tree

    Args:
        VP ([type]): [description]
    """
    return VP["tree"].productions()[0]


def get_CFGs(VP_info: list) -> list:
    """[summary]

    Args:
        VP_info (list): [description]

    Returns:
        list: [description]
    """
    for ix in range(len(VP_info)):
        if not len(VP_info[ix]) == 0:
            VP_info[ix]["cfg"] = parsing.get_CFG(
                VP_info[ix]
            )
    return VP_info


def from_text_to_cfg(
    data: pd.DataFrame,
    al_prdctor: allennlp_models.structured_prediction.predictors.constituency_parser.ConstituencyParserPredictor,
):
    """Extract the context free grammars from the verb phrases of a sample of texts

    Args:
        data (pd.DataFrame): dataframe of texts
        al_prdctor ([type]): [description]

    Returns:
        [type]: [description]
    """
    # preprocess, parse verb phrases and get production rules
    data = preprocess.sample(data)
    VP_info = parsing.extract_all_VPs(data, al_prdctor)
    VP_info = parsing.get_CFGs(VP_info)
    VPs = parsing.make_VPs_readable(VP_info)
    data["VP"] = np.asarray(VPs)
    data["cfg"] = np.asarray(
        [
            VP["cfg"] if not len(VP) == 0 else None
            for VP in VP_info
        ]
    )
    return data


def chunk_cfg(cfg: pd.Series) -> pd.Series:
    """Convert a series of string VP production rules ('VP -> V NP') to
    a series of string constituents ('V NP')

    Args:
        cfg (pd.Series): pandas series of string VP production rules ('VP -> V NP')

    Returns:
        pd.Series: a series of string constituents ('V NP')
    """

    # case not empty
    if not cfg.empty:

        # set null entries to empty string
        cfg[cfg.isnull()] = ""

        # case values are string, drop irrelevant component
        if isinstance(cfg.iloc[0], str):
            constt = cfg.apply(
                lambda x: x.replace("VP ->", "")
            )

        # case values are nltk production objects, drop irrelevant component
        elif isinstance(
            cfg.iloc[0], nltk.grammar.Production
        ):
            constt = cfg.apply(
                lambda x: str(x).replace("VP ->", "")
            )

        # else request a known value type
        else:
            raise TypeError(
                f"""(chunk_cfg) cfg entries 
                are of type {type(cfg.iloc[0])}. 
                Please use type 'str' 
                or 'nltk.grammar.Production'
                """
            )
    # else
    else:
        raise ValueError(
            """"(chunk_cfg) cfg is empty. 
            Please relax your parameters"""
        )

    return constt


def from_text_to_constituents(
    data: pd.DataFrame,
    al_prdctor: allennlp_models.structured_prediction.predictors.constituency_parser.ConstituencyParserPredictor,
) -> pd.DataFrame:
    """Convert text to constituents (e.g., 'V NP')

    Args:
        data (pd.DataFrame): dataframe of
            rows (str): speech sentence texts
            columns: 'text', 'category'
        al_prdctor (allennlp_models.structured_prediction.predictors.constituency_parser.ConstituencyParserPredictor): [description]

    Returns:
        pd.DataFrame: [description]
    """

    data = parsing.from_text_to_cfg(data, al_prdctor)
    constt = parsing.chunk_cfg(data["cfg"])
    return constt


def run_parsing_pipe(
    data: pd.DataFrame,
    predictor: object,
    prm: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """[summary]

    Parse the texts contained in rows of a dataframe

    Args:
        prm (dict): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        pd.DataFrame: [description]
    """

    t0 = time()

    # select data for an input class
    data_c = data[
        data["category"].isin(prm["intent_class"])
    ]
    sample = data_c["text"].iloc[
        0
    ]  # get first VPs, others often variants

    # PARSING
    tic = time()
    output = predictor.predict(sentence=sample)
    parsed_txt = output["trees"]
    if verbose:
        logger.info(
            f"(run_parsing_pipe)(Inference) took {round(time()-tic,2)} secs"
        )
        logger.info(f"Parsed sample:\n{parsed_txt}\n")

    # VP EXTRACTION
    tree = ParentedTree.fromstring(parsed_txt)

    # test
    assert (
        len(parsing.extract_VP(predictor, "I want coffee"))
        > 0
    ), "VP is Empty\n"

    # Speed up (1 hour / 10K queries)
    VPs = parsing.extract_all_VPs(data_c, predictor)

    # test
    assert (
        len(VPs) == len(data_c) or len(VPs) == prm["sample"]
    ), """(run_parsing_pipe) VP's length does not match "data_c"\n"""

    # convert to list of strings
    list_of_VPs = make_VPs_readable(VPs)

    # add to data, show
    data_c["VP"] = pd.DataFrame(list_of_VPs)

    # test
    assert data_c.category.nunique() == len(
        prm["intent_class"]
    ), """The intent classes in the parsed_data does not match input data's"""
    logger.info(
        f"(run_parsing_pipe) took {round(time()-tic,2)} secs\n"
    )
    return data_c


def write_cfg(cfg: pd.DataFrame) -> pd.DataFrame:
    """Write dataset augmented with context free grammar productions

    Args:
        cfg (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    # add current time to filename
    now = (
        datetime.now()
        .strftime("%d/%m/%Y %H:%M:%S")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("/", "_")
    )
    filepath = os.path.splitext(catalog["cfg"])
    myfile, myext = filepath[0], filepath[1]
    cfg.to_excel(f"{myfile}_{now}{myext}")


def to_dict_list(doc: spacy.tokens.doc.Doc):
    """Extract 'ROOT', 'dobj' dependencies and 'Entities' from a query

    Args:
        doc (spacy.tokens.doc.Doc): a spacy Doc object
            e.g., generated by doc = nlp("track my credit card")

    Returns:
        defauldict: [description]
            e.g.,
                [
                    defaultdict(list, {'intent': ['track'], 'intendeed': ['card', 'me']}),
                    defaultdict(list, {'intent': ['expecting'], 'intendeed': ['card']})
                ]
    """
    d = defaultdict(list)
    for token in doc:
        if token.dep_ == "ROOT":
            d["intent"].append(token.text)
        if token.dep_ == "dobj":
            d["intendeed"].append(token.text)
    for ent in doc.ents:
        if ent is not None:
            d[str(ent.label_)] = ent.text
    return dict(d)


def parse_intent(cfg: pd.DataFrame) -> list:
    """parse intents from dataframe of string queries

    Args:
        cfg (pd.DataFrame): dataframe of string queries

    Returns:
        list: list of intent dictionaries with two keys: intent and intendeed
        e.g.,
            [
                {'intent': ['track'], 'intendeed': ['card', 'me']},
                {'intent': ['expecting'], 'intendeed': ['card']}
            ]
    """
    return [to_dict_list(nlp(vp)) for vp in cfg["VP"]]
