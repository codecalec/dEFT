import numpy as np
import itertools
from sklearn.linear_model import LinearRegression

from typing import List


def triangular_number(n: int) -> int:
    return int(n * (n + 1) / 2)


class PredictionBuilder:
    """
    Constructs Linear Regression Model for the wilson coefficient of each operator for some observable
    """

    def __init__(self, nOps: int, samples: List[float], preds: List[List[float]]):
        """ Constructor for PredictionBuilder """

        self.nOps = nOps
        self.nSamples = int((nOps + 1) * (nOps + 2) / 2)
        self.model = self.build_regression_model(nOps, samples, preds)

    def build_regression_model(
        self, nOps: int, samples: List[float], preds: List[List[float]]
    ) -> LinearRegression:
        """ Initialise morphing model using samples with predicted values """
        if len(preds) < self.nSamples:
            raise TypeError(
                "morphing with "
                + str(nOps)
                + " coefficients requires at least "
                + str(self.nSamples)
                + " samples,  but only "
                + str(len(preds))
                + " are provided"
            )

        # convert to vector of coefficient factors to linearise the morphing
        cInputAr = self.make_coefficients(samples)

        # define model
        model = LinearRegression()

        # fit model
        model.fit(cInputAr, preds)

        return model

    def make_coefficients(self, ci: List[float]) -> np.ndarray:
        X = np.array([])
        num_rows = np.shape(ci)[0]
        for row in ci:
            # Account for quadratic self term
            X = np.append(X, row ** 2)

            # Account for quadratic cross term
            combs = itertools.combinations(list(row), 2)
            for comb in combs:
                X = np.append(X, comb[0] * comb[1])
        X = X.reshape(num_rows, triangular_number(len(ci[0])))
        return X

    def makeCoeffPoint(self, ci: np.ndarray) -> np.ndarray:
        X = np.array([])
        X = np.append(X, ci ** 2)
        combs = itertools.combinations(list(ci), 2)
        for comb in combs:
            X = np.append(X, comb[0] * comb[1])

        X = X.reshape(1, triangular_number(len(ci)))
        return X

    def make_prediction(self, c: np.ndarray) -> np.ndarray:
        """ Produce the predicted observable for a set of coefficients (excluding SM coefficient) """
        c = np.append(1.0, c)
        cInputAr = self.makeCoeffPoint(c)
        pred = self.model.predict(cInputAr)
        return pred[0]

    # TODO: Not in use. Figure out if it can be deleted
    # def init(self, nOps, samples, preds):
    #     # 'exact' morphing, soon to be defunct

    #     if len(preds) <= self.nSamples:
    #         raise TypeError(
    #             "morphing with "
    #             + str(nOps)
    #             + " coefficients requires at least "
    #             + str(self.nSamples)
    #             + " samples,  but only "
    #             + str(len(preds))
    #             + " are provided"
    #         )

    #     self.nOps = nOps
    #     cInputAr = np.array([])

    #     for sample in samples:
    #         couplings = np.array([])
    #         combs = list(itertools.combinations(sample, 2))
    #         print("sample = " + str(sample))

    #         # SM term
    #         couplings = np.append(couplings, sample[0] ** 2)
    #         #'SM--Oi' interference terms
    #         for comb in range(0, nOps):
    #             couplings = np.append(couplings, combs[comb][0] * combs[comb][1])
    #         #'squared' terms
    #         for ci in range(1, nOps + 1):
    #             couplings = np.append(couplings, sample[ci] ** 2)
    #         #'cross' terms
    #         for comb in range(nOps, len(combs)):
    #             couplings = np.append(couplings, combs[comb][0] * combs[comb][1])

    #         cInputAr = np.append(cInputAr, couplings, axis=0)

    #     cInputAr = np.reshape(cInputAr, (len(samples), len(samples)))
    #     cInputAr = np.transpose(cInputAr)
    #     cInputAr.tofile("cInputAr.txt", sep=" ", format="%s")

    #     inWeightsAr = np.linalg.inv(cInputAr)

    #     self.inWeightsAr = inWeightsAr
    #     inPreds = preds.transpose()
    #     self.inPreds = inPreds
    #     inPreds.tofile("inPreds.txt", sep=" ", format="%s")
