import os

import joblib
import numpy as np
import pandas as pd
import yaml
from pigeon import annotate

# set project path
PROJ_PATH = os.getenv("PROJ_PATH")

# import custom nodes
from src.intent.nodes import annotation, parsing, preprocess

# display
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 1000)

# shortcuts
to_df = pd.DataFrame

# catalog
catalog_path = os.path.join(PROJ_PATH, "conf/base/catalog.yml")
with open(catalog_path) as file:
    catalog = yaml.load(file)


class Cfg:
    """Context free grammar processing class"""

    def __init__(self, corpus, prms):
        """Instantiate Cfg
        Args:
            corpus:
            prms: model parameters
        """
        self.corpus = corpus
        self.predictor = parsing.init_allen_parser()
        self.prms = prms

    def do(self, verbose: bool = False):
        """Apply cfg parsing to self.corpus

        Args:
            verbose (bool, optional): Whether to print and plot. Defaults to False.

        Returns:
            pd.DataFrame: [description]
        """

        # sample corpus
        sample = preprocess.sample(self.corpus)

        # parse
        out = self.predictor.predict(sentence=sample["text"].iloc[0])
        parsed_txt = out["trees"]
        if verbose:
            print(f"Parsed sample:\n{parsed_txt}")

        # retrieve verb phrases
        VP_info = parsing.extract_all_VPs(sample, self.predictor)

        # test
        VPs = parsing.make_VPs_readable(VP_info)

        # retrieve cfg
        VP_info = parsing.get_CFGs(VP_info)
        sample["VP"] = np.asarray(VPs)
        sample["cfg"] = np.asarray(
            [VP["cfg"] if not len(VP) == 0 else None for VP in VP_info]
        )

        # write cfg-augmented corpus
        sample.to_excel(catalog["parsed"])
        joblib.dump(sample, catalog["parsed"])

        # annotate well-formed intent
        if self.prms["annotation"] == "do":
            annots = annotate(sample["VP"], options=["yes", "no"])
        elif self.prms["annotation"] == "load":
            (annots, myfile, myext,) = annotation.get_annotation(
                catalog, self.prms
            )
        else:
            annots = pd.concat(
                [sample, to_df({"annots": [None] * len(sample)}),], axis=1,
            )
        annots_df = annotation.index_annots(self.prms, sample, annots)

        # write annotation-augmented corpus
        if self.prms["annotation"] in ["do", "load"]:
            annotation.write_annotation(
                catalog, self.prms, annots_df, myfile, myext
            )

        # replace failed verb phrase parsing with empty string
        annots_df["annots"][annots_df["VP"].isnull()] = np.nan

        # write augmented corpus
        annots_df.insert(loc=1, column="text", value=sample["text"])
        annots_df["cfg"] = sample["cfg"]
        annots_df["category"] = sample["category"]
        parsing.write_cfg(annots_df)
        return annots_df
