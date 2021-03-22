import time
import json
import sys

import emcee
import numpy as np
import matplotlib.pyplot as pl

from dEFT import (
    ConfigReader,
    PredictionBuilder,
    ModelValidator,
    MCMCFitter,
    SummaryPlotter,
)

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


if len(sys.argv) <= 2:

    fitter = MCMCFitter(config, pb)
    sampler = fitter.sampler

    print(
        "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    )
    print(
        "Mean autocorrelation time: {0:.3f} steps".format(
            np.mean(sampler.get_autocorr_time())
        )
    )

    SummaryPlotter(config, pb, sampler)

end = time.time()
print("Total elapsed wall time  = " + str(int(end - start)) + " seconds")
