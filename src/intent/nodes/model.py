# author: steeve laquitaine


from src.intent.nodes.inference import Prediction
from src.intent.nodes.layers import cluster_queries
from src.intent.nodes.processing import Processing


class IntentModel(object):
    """Unsupervised Intent Inference model class

    Args:
        object ([type]): [description]
    """

    def __init__(
        self, dist_thresh, hcl_method, prms,
    ):
        self.dist_thresh = dist_thresh
        self.hcl_method = hcl_method
        self.prms = prms

    def fit(self, corpus):

        # processing layer
        X, intents = Processing(
            params=self.prms,
            num_sent=self.prms["NUM_SENT"],
            filt_mood=self.prms["FILT_MOOD"],
            intent_score=self.prms["INTENT_SCORE"],
            seed=self.prms["DENOISING"]["SEED"],
            denoising=self.prms["DENOISING"][
                "FILTERING_METHOD"
            ],
            inspect=self.prms["INSPECT"],
        ).run(corpus)

        # clustering layer
        X = cluster_queries(
            X,
            dist_thresh=self.dist_thresh,
            hcl_method=self.hcl_method,
            params=self.prms,
        )
        return X, intents

    def predict(self, corpus, fitted):
        return Prediction(
            method=self.prms["PREDICT_METHOD"]
        ).run(corpus, fitted)
