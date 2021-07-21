import itertools
from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression

from .ConfigReader import ConfigReader


def _triangular_number(n: int) -> int:
    return int(n * (n + 1) / 2)


class PredictionBuilder:
    """
    Constructs Linear Regression Model for the wilson coefficient of each operator for some observable
    """

    def __init__(
        self,
        config: ConfigReader,
    ):
        """
        Constructor for PredictionBuilder

        :param config:
            Configuration for analysis.
        :type config: ConfigReader.
        """
        self.nOps = len(config.prior_limits)
        self.nSamples = int((self.nOps + 1) * (self.nOps + 2) / 2)
        scaled_predictions = config.predictions * config.inclusive_k_factor
        self.model = self._build_regression_model(config.samples, scaled_predictions)

    def _build_regression_model(
        self, samples: List[float], preds: List[List[float]]
    ) -> LinearRegression:
        """Initialise morphing model using samples with predicted values"""
        if len(preds) < self.nSamples:
            raise TypeError(
                "morphing with "
                + str(self.nOps)
                + " coefficients requires at least "
                + str(self.nSamples)
                + " samples,  but only "
                + str(len(preds))
                + " are provided"
            )

        # convert to vector of coefficient factors to linearise the morphing
        cInputAr = self._make_coefficients(samples)

        # define model
        model = LinearRegression()

        # fit model
        model.fit(cInputAr, preds)

        return model

    def _make_coefficients(self, ci: List[float]) -> np.ndarray:
        X = np.array([])
        num_rows = np.shape(ci)[0]
        for row in ci:
            # Account for quadratic self term
            X = np.append(X, row ** 2)

            # Account for quadratic cross term
            combs = itertools.combinations(list(row), 2)
            for comb in combs:
                X = np.append(X, comb[0] * comb[1])
        X = X.reshape(num_rows, _triangular_number(len(ci[0])))
        return X

    def _make_coeff_point(self, ci: np.ndarray) -> np.ndarray:
        X = np.array([])
        X = np.append(X, ci ** 2)
        combs = itertools.combinations(list(ci), 2)
        for comb in combs:
            X = np.append(X, comb[0] * comb[1])

        X = X.reshape(1, _triangular_number(len(ci)))
        return X

    def make_prediction(self, c: np.ndarray) -> np.ndarray:
        """
        Produce the predicted observable for a set of coefficients (excluding SM coefficient)

        :param c:
            A set of coefficients to be used by the model in order to make a prediction. Should have a coefficient for each operator in the model.
        :type c: numpy.ndarray
        """

        if len(c) != self.nOps:
            raise Exception(
                f"An incorrect number of coefficients were supplied. Model requires coeffiecient for {self.nOps} operators but {len(c)} were supplied"
            )

        c = np.append(1.0, c)
        cInputAr = self._make_coeff_point(c)
        pred = self.model.predict(cInputAr)
        return pred[0]
