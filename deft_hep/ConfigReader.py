import sys
import json
import numpy as np
from numpy.linalg import inv


class ConfigReader:
    """
    Reads JSON configuration file and acts as a full configuration of the analysis

    :param run_name:
        Label for analysis
    :type run_name: str

    :param observable:
        Label for observables
    :type observable: str, list[str]

    :param bins:
        Bin edges
    :type bins: list[float], list[list[float]]

    :param data:
        Central value of bins
    :type data: list[float], list[list[float]]

    :param coefficients:
        Label for each coefficient

    :param prior_limits:
        Bounds for each coefficient.
    :type prior_limits: dict[str,list[float]]

    :param samples:
        Samples of coefficient values
    :type samples: np.ndarray

    :param predictions:
        Predicted Monte Carlo signal for the corresponding set of coefficients
    :type predictions: np.ndarray

    :param inclusive_k_factor:
        K factor used to scale predictions depending on order of Monte Carlo generation. This can be given as a histogram wide scaling or provided per bin.
    :type inclusive_k_factor: float or np.ndarray

    :param n_walkers:
        Number of walkers to use for MCMC fitting
    :type n_walkers: int

    :param n_burnin:
        Number of burn in runs to perform before beginning MCMC fitting
    :type n_burnin: int

    :param n_total:
        Number of total iterations for MCMC fitting
    :type n_total: int
    """

    # TODO: Sort out parameters and improve usability

    def __init__(self, filename: str):
        """
        Constructor for ConfigReader

        :param filename: Configuration file
        :type filename: str
        """
        self.filename = filename
        with open(filename, "r") as f:
            config = json.load(f)

        try:
            self.params = config
            self.run_name = self.params["config"]["run_name"]
            self.observable = self.params["config"]["data"]["observable"]
            self.bins = self.params["config"]["data"]["bins"]
            self.data = self.params["config"]["data"]["central_values"]
            self.cov = self.params["config"]["data"]["covariance_matrix"]

            if len(self.cov) != len(self.data):
                raise Exception(
                    "Covariance matrix does not have suitable dimensions. Expected ({len(self.data), len(self.data)})"
                )

            for row in self.cov:
                if len(row) != len(self.data):
                    raise Exception(
                        "Covariance matrix does not have suitable dimensions. Expected ({len(self.data), len(self.data)})"
                    )

            self.prior_limits = self.params["config"]["model"]["prior_limits"]
            for label, limits in self.prior_limits.items():
                if len(limits) != 2:
                    raise Exception(f"Coefficient {label} does not have 2 values.")
                if limits[0] >= limits[1]:
                    raise Exception(
                        f"Prior limit for coefficient {label} are invalid: {limit}"
                    )

            self.coefficients = list(
                self.params["config"]["model"]["prior_limits"].keys()
            )
            self.n_walkers = self.params["config"]["fit"]["n_walkers"]
            self.n_burnin = self.params["config"]["fit"]["n_burnin"]
            self.n_total = self.params["config"]["fit"]["n_total"]
            self.samples = np.array(self.params["config"]["model"]["samples"])

            k_factor = self.params["config"]["model"]["inclusive_k_factor"]
            self.inclusive_k_factor = (
                k_factor if not isinstance(k_factor, list) else np.array(k_factor)
            )

            if (
                "input" not in self.params["config"]["model"]
                or self.params["config"]["model"]["input"] == "numpy"
            ):
                self.predictions = np.array(
                    self.params["config"]["model"]["predictions"]
                )
                if len(self.predictions) != len(self.samples):
                    raise Exception(
                        "Number of samples must equal number of predictions"
                    )
        except KeyError as err:
            print(
                f"Error reading file {filename}: Could not find option {err}",
                file=sys.stderr,
            )
            exit(0)

        self.tex_labels = ["$" + c + "$" for c in self.coefficients]

        x_vals = np.zeros(len(self.bins) - 1)
        for x_val in range(0, len(self.bins) - 1):
            x_vals[x_val] = (self.bins[x_val] + self.bins[x_val + 1]) / 2.0
        self.x_vals = x_vals
