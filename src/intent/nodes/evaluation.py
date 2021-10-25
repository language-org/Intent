# author: steeve laquitaine

from sklearn.metrics.cluster import (
    rand_score,
    contingency_matrix,
    mutual_info_score,
)
import pandas as pd


class Metrics:
    """Metrics class"""

    def __init__(
        self,
        metrics: tuple,
        predictions: pd.DataFrame,
        true_labels: pd.DataFrame,
    ):
        self.metrics = metrics
        self.predictions = predictions
        self.true_labels = true_labels

        print("(calculate_accuracy) task info:")
        print(
            "(calculate_accuracy) - number of classes:",
            self.true_labels.nunique(),
        )

    def _accuracy(self):
        """Calculate accuracy

        Returns:
            [type]: [description]
        """
        return sum(
            self.predictions == self.true_labels
        ) / len(self.predictions)

    def _rand_index(self):
        """Calculate Rand index

        Returns:
            [type]: [description]
        """
        # rand index (0 - 1: perfect)
        ri = rand_score(
            self.true_labels.tolist(),
            self.predictions.tolist(),
        )
        return ri

    def _mutual_info(self):
        """Calculate mutual information

        Returns:
            [type]: [description]
        """
        mi = mutual_info_score(
            self.true_labels.tolist(),
            self.predictions.tolist(),
        )
        return mi

    def run(self):
        """Run instantiated Metrics class

        Returns:
            [type]: [description]
        """
        out = dict()
        for metric in self.metrics:
            out[metric] = eval(f"self._{metric}()")
        return out


class Description:
    """Metrics description class"""

    def __init__(
        self,
        graphics: tuple,
        predictions: pd.DataFrame,
        true_labels: pd.DataFrame,
    ):
        """Instantiate class

        Args:
            graphics (tuple): [description]
            predictions (pd.DataFrame): [description]
            true_labels (pd.DataFrame): [description]
        """
        self.predictions = predictions
        self.true_labels = true_labels
        self.graphics = graphics

    def _contingency_table(self):
        """Calculate contingency table

        Returns:
            [type]: [description]
        """
        return contingency_matrix(
            self.true_labels, self.predictions
        )

    def run(self):
        """Run instantiated Description class"""
        for graphic in self.graphics:
            out = eval(f"self._{graphic}()")
        return out
