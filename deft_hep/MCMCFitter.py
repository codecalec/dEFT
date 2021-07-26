from multiprocessing import Pool
from typing import Optional, List, Union

import numpy as np
import emcee

from .ConfigReader import ConfigReader
from .PredictionBuilder import PredictionBuilder

_PB: Optional[PredictionBuilder] = None
_DATA: Optional[np.ndarray] = None
_ICOV: Optional[np.ndarray] = None
_BOUNDS: Optional[np.ndarray] = None


def ln_prior(c: np.ndarray) -> float:
    lnp = 0.0

    if (c < _BOUNDS[0]).any() or (_BOUNDS[1] < c).any():
        lnp = -np.inf
    return lnp


def ln_prob(
    c: np.ndarray,
) -> float:
    pred = _PB.make_prediction(c)
    diff = pred - _DATA
    ll = (-np.dot(diff, np.dot(_ICOV, diff))) + ln_prior(c)
    return ll


class MCMCFitter:
    """ Use Markov chain Monte Carlo method to fit the model from the PredictionBuilder to some data specified in the configuration file """

    def __init__(
        self,
        config: ConfigReader,
        pb: PredictionBuilder,
        initial_pos: Union[float, np.ndarray] = 0,
        initial_deviation: Union[float, np.ndarray] = 1e-4,
        use_multiprocessing: bool = False,
    ):
        """
        :param config:
            Configuration for analysis
        :type config: ConfigReader

        :param pb:
            Model on which the fitting will be performed
        :type pb: PredictionBuilder

        :param initial_pos:
            Initial location on which to place walkers around. Must be a single number which will apply to all coefficients or a NumPy array of the same dimension as number of coefficients.
        :type initial_pos: float or numpy.ndarray

        :param initial_deviation:
            Standard deviation of normal distribution centred at ``initial_pos`` around which the walkers will be placed.

            **Warning**: Walkers are currently uniformly placed with a width of the deviation. This is planned to change.

        :param use_multiprocessing:
            Toggles whether to use `emcee's <https://emcee.readthedocs.io/en/stable/>_` multiprocessing capabilities.
        :type user_multiprocessing: bool
        """

        n_walkers = config.n_walkers
        n_dim = len(config.prior_limits)
        n_burnin = config.n_burnin
        n_total = config.n_total

        p0 = [
            initial_pos + initial_deviation * np.random.rand(n_dim)
            for _ in range(n_walkers)
        ]  # Initial position of walkers

        global _DATA
        global _ICOV
        global _BOUNDS
        global _PB
        _DATA = config.data
        _ICOV = np.linalg.inv(config.cov)
        _BOUNDS = np.array(list(config.prior_limits.values())).T
        _PB = pb

        if use_multiprocessing:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(
                    n_walkers,
                    n_dim,
                    ln_prob,
                    pool=pool,
                )
                # Run burn in runs
                pos, _, _ = sampler.run_mcmc(p0, n_burnin, progress=True)
                sampler.reset()

                # Perform proper run
                sampler.run_mcmc(pos, n_total, progress=True)
        else:
            sampler = emcee.EnsembleSampler(
                n_walkers,
                n_dim,
                ln_prob,
            )
            # Run burn in runs
            pos, _, _ = sampler.run_mcmc(p0, n_burnin, progress=True)
            sampler.reset()

            # Perform proper run
            sampler.run_mcmc(pos, n_total, progress=True)

        self.sampler = sampler
