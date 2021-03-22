import time
import json
import sys
from multiprocessing import Pool

import emcee
import numpy as np
import matplotlib.pyplot as pl

from tools.ConfigReader import ConfigReader
from tools.PredictionBuilder import PredictionBuilder
from tools.ModelValidator import ModelValidator
from tools.summaryPlotter import summaryPlotter

start = time.time()

###################################################
####### READ CONFIG, BUILD MORPHING MODEL #########
###################################################
filename = sys.argv[1]
config = ConfigReader(filename)

# if config.params["config"]["model"]["input"] == "numpy":

pb = PredictionBuilder(len(config.prior_limits), config.samples, config.predictions)

##########################################################
########  VALIDATE OF MORPHING MODEL (OPTIONAL)  #########
##########################################################

if len(sys.argv) > 2:
    filename_test = sys.argv[2]
    config_test = ConfigReader(filename_test)

    ModelValidator(config_test, pb)

#######################################
######### ESTIMATE POSTERIOR ##########
#######################################


def lnprior(c):
    lnp = 0.0
    for scan_ind in range(0, len(c)):
        if (c[scan_ind] < config.prior_limits[config.coefficients[scan_ind]][0]) | (
            c[scan_ind] > config.prior_limits[config.coefficients[scan_ind]][1]
        ):
            lnp = -np.inf
    return lnp


def lnprob(c, data, icov):
    pred = pb.makeRMPred(c)
    diff = pred - data
    ll = (-np.dot(diff, np.dot(icov, diff))) + (lnprior(c))
    return ll


if len(sys.argv) <= 2:
    nWalkers = config.n_walkers
    ndim = int(len(config.params["config"]["model"]["prior_limits"]))
    nBurnIn = config.n_burnin
    nTotal = config.n_total
    p0 = [np.zeros(ndim) + 1e-4 * np.random.randn(ndim) for i in range(nWalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nWalkers,
            ndim,
            lnprob,
            pool=pool,
            args=[config.params["config"]["data"]["central_values"], config.icov],
        )
        # sampler = emcee.EnsembleSampler(nWalkers, ndim, lnprob, args=[config.params["config"]["data"]["central_values"], config.icov])
        pos, prob, state = sampler.run_mcmc(p0, nBurnIn, progress=True)
        sampler.reset()
        sampler.run_mcmc(pos, nTotal, progress=True)
        samples = sampler.chain.reshape((-1, ndim))

        sp = summaryPlotter()
        sp.summarise(config, pb, sampler, samples)
end = time.time()
print("Total elapsed wall time  = " + str(int(end - start)) + " seconds")
