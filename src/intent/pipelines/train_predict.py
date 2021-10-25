import os
import yaml
import logging
from src.intent.nodes import config
from time import time
import pandas as pd
from src.intent.nodes.model import IntentModel
import logging.config
import pandas as pd
import yaml
from src.intent.nodes import config
from src.intent.nodes.inference import (
    write_preds,
    write_metrics,
)
from src.intent.nodes import evaluation as evaln

proj_path = os.getenv("PROJ_PATH")

# configurate logging
logging_path = os.path.join(
    proj_path + "/conf/base/logging.yml"
)
with open(logging_path, "r") as f:
    LOG_CONF = yaml.load(f, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger(__name__)


def run():

    # load parameters
    logger.info("Loading pipeline parameters...")
    prms = config.load_parameters(proj_path)
    DIST_THRES = prms["DIST_THRES"]
    HCL_METHOD = prms["HCL_METHOD"]

    # Loading data
    t0 = time()
    t_read = time()
    corpus_path = (
        proj_path + "/data/01_raw/banking77/train.csv"
    )
    corpus = pd.read_csv(corpus_path)
    logger.info(
        f"Reading dataset took {time()-t_read} secs"
    )

    # train model
    logger.info("Training model...")
    t_train = time()
    model = IntentModel(DIST_THRES, HCL_METHOD, prms)
    fitted, intents = model.fit(corpus)
    logger.info(f"Training took {time()-t_train} secs")

    # infer intent
    logger.info("Calculating preds...")
    t_infer = time()
    pred = model.predict(corpus, fitted)
    logger.info(f"Inference took {time()-t_infer} secs")

    # evaluate model
    logger.info("Evaluating model...")
    metrics = evaln.Metrics(
        ("rand_index", "mutual_info"),
        pred["cluster_labels"],
        pred["true_labels"],
    ).run()
    logger.info(f"Metrics: {metrics}")

    contingency = evaln.Description(
        ("contingency_table",),
        pred["cluster_labels"],
        pred["true_labels"],
    ).run()
    true_labels = pred["true_labels"].unique()
    contingency_df = pd.DataFrame(
        contingency, index=true_labels.tolist()
    )

    # write predictions and metrics
    write_preds(corpus, intents, pred)
    write_metrics(metrics, contingency_df)
    logger.info("Precictions & metrics have been written")
    logger.info(f"Pipeline took {time()-t0} secs")
