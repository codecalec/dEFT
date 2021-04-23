import numpy as np
import matplotlib.pyplot as pl

from .ConfigReader import ConfigReader
from .PredictionBuilder import PredictionBuilder


class ModelValidator:
    """
    Allows for a model to be validated against dataset with known operator paramters (like MC signal). The model must be defined using `PredictionBuilder` class.
    """
    def __init__(self, pb: PredictionBuilder):
        self.nOps = pb.nOps
        self.pb = pb

    def validate(self, config_test: ConfigReader):
        test_samples = config_test.params["config"]["model"]["samples"]
        test_preds = np.asarray(config_test.params["config"]["model"]["predictions"])
        assert (
            self.nOps != len(config_test.params["config"]["model"]["prior_limits"]),
            "Operator mismatch. Make sure the supplied ConfigReader has the correct number of operators for the model",
        )

        print("##############################################################")
        print("###############  VALIDATION OF MORPHING MODEL  ###############")
        print("##############################################################")

        model_preds = np.fromiter(
            (self.pb.make_prediction(sample[1:]) for sample in test_samples),
            float,
            len(test_samples),
        )
        assert(len(model_preds[0]) == len(test_preds[0]))

        diff_preds = np.square(model_preds - test_preds).sum(axis=1)
