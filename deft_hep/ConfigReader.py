import sys
import json
import numpy as np
from numpy.linalg import inv


class ConfigReader:
    """ Reads JSON configuration file"""

    def __init__(self, filename: str):
        self.filename = filename
        coefficients = []
        predictions = {}
        with open(filename, "r") as f:
            config = json.load(f)

        try:
            self.params = config
            self.run_name = self.params["config"]["run_name"]
            self.observable = self.params["config"]["data"]["observable"]
            self.bins = self.params["config"]["data"]["bins"]
            self.data = self.params["config"]["data"]["central_values"]
            self.prior_limits = self.params["config"]["model"]["prior_limits"]
            self.coefficients = list(
                self.params["config"]["model"]["prior_limits"].keys()
            )
            self.n_walkers = self.params["config"]["fit"]["n_walkers"]
            self.n_burnin = self.params["config"]["fit"]["n_burnin"]
            self.n_total = self.params["config"]["fit"]["n_total"]
            self.cov = self.params["config"]["data"]["covariance_matrix"]
            self.icov = inv(self.cov)
            self.x_vals = self.params["config"]["data"]["bins"]
            self.samples = np.asarray(self.params["config"]["model"]["samples"])
            if self.params["config"]["model"]["input"] == "numpy":
                self.predictions = np.asarray(
                    self.params["config"]["model"]["predictions"]
                )
                if len(self.predictions) != len(self.samples):
                    raise TypeError(
                        "number of samples must equal number of predictions"
                    )
        except KeyError as err:
            print(
                f"Error reading file {filename}: Could not find option {err}",
                file=sys.stderr,
            )
            exit(0)

        self.tex_labels = ["$" + c + "$" for c in self.coefficients]
        x_vals = np.zeros(len(self.x_vals) - 1)
        for x_val in range(0, len(self.bins) - 1):
            x_vals[x_val] = (self.bins[x_val] + self.bins[x_val + 1]) / 2.0
        self.x_vals = x_vals
