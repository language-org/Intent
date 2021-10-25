import os

import pandas as pd

PROJ_PATH = os.getenv("PROJ_PATH")

import logging
from time import time
import mlflow
from src.intent.nodes import (
    config,
    parsing,
    preprocess,
    retrieval,
    similarity,
)
from src.intent.pipelines.parsing import Cfg
from src.intent.pipelines.similarity import Lcs
from typing import Dict, Any

# prep logging
logger = logging.getLogger()

# load parameters
PARAMS = config.load_parameters(PROJ_PATH)


class Processing:
    """Intent Processing class"""

    def __init__(
        self,
        params: dict,
        num_sent: int,
        filt_mood: tuple,
        intent_score: float,
        seed: str,
        denoising: str,
        inspect: bool,
    ):
        """Instantiate processing class

        Args:
            params (dict): [description]
            num_sent (int, optional): [description]. Defaults to None.
            filt_mood (str, optional): [description]. Defaults to None.
            INTENT_SCORE (float, optional): [description]. Defaults to None.
            seed (str, optional): [description]. Defaults to None.
            inspect (bool, optional): [description]. Defaults to None.
        Returns:
            Instance of Processing class

        """
        self.params = params
        self.NUM_SENT = num_sent
        self.FILT_MOOD = filt_mood
        self.INTENT_SCORE = intent_score
        self.DENOISING = denoising
        self.SEED = seed
        self.inspect = inspect

        # print and log processing pipeline parameters
        self._print_params()

    def _print_params(self):
        """Print and log pipeline parameters"""

        # print
        logger.info("-------- PROCESSING ------------")
        logger.info("Parameters:")
        logger.info(f"- Sentences/query: {self.NUM_SENT}")
        logger.info(f"- Mood: {self.FILT_MOOD}")
        logger.info(
            f" Threshold similarity score:  {self.INTENT_SCORE}"
        )
        logger.info(f"- Seed: {self.SEED}")

        # log
        mlflow.log_param("nb_sentences", self.NUM_SENT)
        mlflow.log_param("mood", self.FILT_MOOD)
        mlflow.log_param("denoising", self.DENOISING)
        mlflow.log_param("seed", self.SEED)
        mlflow.log_param(
            "similarity threshold", self.INTENT_SCORE
        )

    def run(self, corpus) -> pd.DataFrame:
        """Run pipeline

        Args:
            corpus (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """
        # tape queries
        introsp = corpus["text"].to_frame()

        # parse structure (bottleneck)
        data = self.parse_struct(corpus, introsp)

        # choose complexity
        data = self.filter_cplx(data)

        # choose mood
        data = self.filter_mood(data)

        # parse intents
        (
            introsp,
            similarity_matrix,
            intents_df,
        ) = self.parse_intents(data)

        # filter unknown words
        data = self.filter_unknown(data, intents_df)

        # write representations
        self.write_internal_rep(data["introsp"])
        self.write_syntx_sim(similarity_matrix)
        return (data["data"], data["intents"])

    def filter_unknown(self, data, intents_df):
        # get data
        introsp = data["introsp"]

        # check that verb phrase words are known
        data["data"] = data["data"].drop(columns="index")
        vp = data["data"].merge(
            intents_df, left_on="index", right_index=True
        )["VP"]
        wordnet_filtered = preprocess.filter_words(
            vp, "not_in_wordnet"
        )
        wordnet_filtered.index = intents_df.index

        # filter empty queries
        (
            processed,
            not_empty,
        ) = preprocess.drop_empty_queries(wordnet_filtered)
        raw_ix = wordnet_filtered.index[not_empty]

        # inspect
        if self.inspect:
            introsp["known_words"] = None
            introsp["known_words"].loc[
                wordnet_filtered.index
            ] = 0
            introsp["known_words"].loc[raw_ix] = 1
        intents = intents_df.loc[raw_ix]
        return {
            "data": processed,
            "intents": intents,
            "introsp": introsp,
        }

    def parse_intents(self, data):

        # filter not-intent syntax
        data = self.filter_syntax(data)

        # get data
        introsp = data["introsp"]
        sim_mx = data["sim_mx"]
        data = data["data"]

        # Inference & slot filling
        intents = parsing.parse_intent(data)
        intents = pd.DataFrame(intents, index=data.index)

        # inspect
        if self.inspect:
            introsp = self._inspect_intent(introsp, intents)
        return introsp, sim_mx, intents

    def filter_syntax(self, data: Dict[str, Any]):

        # get data
        introsp = data["introsp"]
        data = data["data"]

        # get components
        tag = parsing.chunk_cfg(data["cfg"])
        tag.index = data["index"]

        # filter syntax
        t_sx = time()
        filtered = self._filter_syntax(data, tag)
        self._log_syntax(t_sx, filtered["data"])

        # inspect
        if self.inspect:
            introsp = self._inspect_syntax(
                introsp, filtered
            )
        return {
            "data": filtered["data"],
            "introsp": introsp,
            "sim_mx": filtered["sim_mx"],
            "raw_ix": data.index,
        }

    def filter_mood(self, data):
        """Choose mood

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        t_mood = time()
        introsp = data["introsp"]
        data = data["data"]
        cfg_mood = preprocess.filter_mood(
            data, self.FILT_MOOD
        )
        self._log_mood(t_mood, cfg_mood)

        # inspect
        if self.inspect:
            introsp["good_mood"] = None
            introsp["good_mood"].loc[cfg_mood["index"]] = 1
        return {"data": cfg_mood, "introsp": introsp}

    def write_syntx_sim(self, similarity_matrix):
        file_path = os.path.join(
            PROJ_PATH,
            "data/08_introspection/syntax_similarity.csv",
        )
        similarity_matrix.to_csv(file_path)

    def write_internal_rep(self, introsp):
        file_path = os.path.join(
            PROJ_PATH,
            "data/08_introspection/representations.csv",
        )
        introsp.to_csv(file_path)

    def filter_cplx(self, data):
        """Choose level of complexity

        Args:
            data (Dict[str, Any]): 
                "data":
                    data 
                "introsp":
                    taped internal representations

        Returns:
            Dict[str, Any]: [description]
                "data": 
                    filtered data
                "introsp" (pd.DataFrame):
                    taped internal representations
        """
        t_cx = time()
        introsp = data["introsp"]
        data = data["data"]
        cfg_cx = preprocess.filter_n_sent_eq(
            data, self.NUM_SENT, verbose=True
        )
        self._log_cpx(t_cx, cfg_cx)

        # inspect
        if self.inspect:
            introsp = self._inspect_cplx(
                introsp, data, cfg_cx
            )
        return {"data": data, "introsp": introsp}

    def parse_struct(self, corpus, introsp):
        """parse queries structure

        Args:
            corpus ([type]): [description]
            introsp ([type]): [description]

        Returns:
            [type]: [description]
        """
        t_cfg = time()
        # slow [bottleneck]
        cfg = Cfg(corpus, self.params).do()
        self._log_cfg(t_cfg, cfg)

        # inspect
        if self.inspect:
            introsp = self._inspect_cfg(introsp, cfg)
        cfg = cfg.reset_index(drop=True)
        return {"data": cfg, "introsp": introsp}

    def _filter_syntax(self, data, tag):
        """Process syntax

        Args:
            cfg_mood ([type]): [description]
            tag ([type]): [description]

        Returns:
            [Dict]: [description]
        """

        # [TODO]: tag and data["text"] should sync their
        # indices
        sim_mx = Lcs().do(data)
        sim_ranked = similarity.rank_nearest_to_seed(
            sim_mx, seed=self.SEED, verbose=True
        )

        # posting_list = retrieval.create_posting_list(tag)
        posting_list = retrieval.create_posting_list_from_raw_ix(
            tag, data["index"]
        )
        # ranked = similarity.print_ranked_VPs(
        # data, posting_list, sim_ranked
        # )
        ranked = similarity.print_ranked_VPs_on_raw_ix(
            data, posting_list, sim_ranked
        )
        data = similarity.filter_by_similarity(
            ranked, self.INTENT_SCORE
        )
        return {
            "data": data,
            "sim_mx": sim_mx,
            "score": ranked["score"],
        }

    def _log_syntax(self, t_sx, filtered):
        """Log syntax processing

        Args:
            t_sx ([type]): [description]
            filtered ([type]): [description]
        """
        logger.info("Filtering 'not-intent' syntax")
        logger.info(f"N={len(filtered)} queries left")
        logger.info(f"took {time()-t_sx} secs")

    def _log_mood(self, t_mood, cfg_mood):
        """Log mood processing

        Args:
            t_mood ([type]): [description]
            cfg_mood ([type]): [description]
        """
        logger.info("Filtering moods")
        logger.info(f"N={len(cfg_mood)} queries left")
        logger.info(f"took {time()-t_mood} secs")

    def _log_cpx(self, t_cx, cfg_cx):
        """Log complexity processing

        Args:
            t_cx ([type]): [description]
            cfg_cx ([type]): [description]
        """
        logger.info("Filtering complex queries")
        logger.info(f"N={len(cfg_cx)} queries left")
        logger.info(f"took {time()-t_cx} secs")

    def _log_cfg(self, t_cfg, cfg):
        """Log context free grammar parsing

        Args:
            t_cfg ([type]): [description]
            cfg ([type]): [description]
        """
        logger.info("Parsing constituents")
        logger.info(f"N={len(cfg)} queries left")
        logger.info(f"took {time()-t_cfg} secs")

    def _inspect_intent(self, introsp, intents_df):
        """Inspect processed intent

        Args:
            introsp ([type]): [description]
            intents_df ([type]): [description]

        Returns:
            [type]: [description]
        """
        kept = introsp["good_intent_syntx"].notnull()
        for col in intents_df.columns:
            introsp[col] = None
            introsp[col].loc[kept] = 0
            introsp[col].loc[intents_df.index] = intents_df[
                col
            ]
        return introsp

    def _inspect_syntax(self, introsp, data):
        """Inspect processed syntax 

        Args:
            introsp ([type]): [description]
            data ([type]): [description]

        Returns:
            [type]: [description]
        """

        # tape score
        introsp["syntx_score"] = None
        introsp["syntx_score"].loc[
            data["score"].index
        ] = data["score"]

        # tape filtered syntax
        introsp["good_intent_syntx"] = None
        kept = introsp["good_cplx"].notnull()
        introsp["good_intent_syntx"].loc[kept] = 0
        introsp["good_intent_syntx"].loc[
            data["data"].index
        ] = 1
        return introsp

    def _inspect_cplx(self, introsp, cfg, cfg_cx):
        """Inspect processed complexity

        Args:
            introsp ([type]): [description]
            cfg ([type]): [description]
            cfg_cx ([type]): [description]

        Returns:
            [type]: [description]
        """
        introsp["good_cplx"] = None
        introsp["good_cplx"].loc[cfg["index"]] = 0
        introsp["good_cplx"].loc[cfg_cx["index"]] = 1
        return introsp

    def _inspect_cfg(self, introsp, cfg):
        """Inspect processed context free grammar

        Args:
            introsp ([type]): [description]
            cfg ([type]): [description]

        Returns:
            [type]: [description]
        """
        introsp["VP"] = None
        introsp["cfg"] = None
        cfg.index = cfg["index"]
        introsp["VP"].loc[cfg["index"]] = cfg["VP"]
        introsp["cfg"].loc[cfg["index"]] = cfg["cfg"]
        return introsp
