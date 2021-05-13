from multiprocessing import Pool

import numpy as np
import emcee

from .ConfigReader import ConfigReader
from .PredictionBuilder import PredictionBuilder

_PB = None
_DATA = None
_ICOV = None
_CONFIG = None

def ln_prior(c: np.ndarray, config: ConfigReader) -> float:
    lnp = 0.0
    for scan_ind in range(0, len(c)):
        if (c[scan_ind] < config.prior_limits[config.coefficients[scan_ind]][0]) | (
            c[scan_ind] > config.prior_limits[config.coefficients[scan_ind]][1]
        ):
            lnp = -np.inf
    return lnp


def ln_prob(
    c: np.ndarray,
) -> float:
    pred = _PB.make_prediction(c)
    diff = pred - _DATA
    ll = (-np.dot(diff, np.dot(_ICOV, diff))) + (ln_prior(c, _CONFIG))
    return ll


class MCMCFitter:
    """ Use Markov chain Monte Carlo method to fit the model from the PredictionBuilder to some data specified in the configuration file """

    def __init__(
        self,
        config: ConfigReader,
        pb: PredictionBuilder,
        use_multiprocessing: bool = True,
    ):
        n_walkers = config.n_walkers

        n_dim = int(len(config.prior_limits))
        n_burnin = config.n_burnin
        n_total = config.n_total
        p0 = [np.zeros(n_dim) + 1e-4 * np.random.randn(n_dim) for i in range(n_walkers)]

        global _DATA
        global _ICOV
        global _CONFIG
        global _PB
        _DATA = config.params["config"]["data"]["central_values"]
        _ICOV = config.icov
        _CONFIG = config
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
                pos, prob, state = sampler.run_mcmc(p0, n_burnin, progress=True)
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
            pos, prob, state = sampler.run_mcmc(p0, n_burnin, progress=True)
            sampler.reset()

            # Perform proper run
            sampler.run_mcmc(pos, n_total, progress=True)

        self.sampler = sampler
