import time
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from deft_hep import (
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

pb = PredictionBuilder(config)

##########################################################
########  VALIDATE OF MORPHING MODEL (OPTIONAL)  #########
##########################################################

if len(sys.argv) > 2:
    filename_test = sys.argv[2]
    config_test = ConfigReader(filename_test)

    mv = ModelValidator(pb)
    samples, predictions = mv.validate(config_test)
    mv.comparison_plot(config_test, predictions, use_pdf=False)

#######################################
######### ESTIMATE POSTERIOR ##########
#######################################


if len(sys.argv) <= 2:

    fitter = MCMCFitter(config, pb, use_multiprocessing=True)
    sampler = fitter.sampler

    print(
        "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    )
    print(
        "Mean autocorrelation time: {0:.3f} steps".format(
            np.mean(sampler.get_autocorr_time())
        )
    )

    # samples = sampler.chain.reshape(-1, len(config.prior_limits))

    # pt_bins = [0, 90, 180, 270, 800]
    # mttbar_small_bins = [300, 360, 430, 500, 580, 680, 800, 1000, 2000]
    # mttbar_large_bins = [300, 430, 500, 580, 680, 800, 1000, 1200, 2000]

    # from matplotlib.colors import to_rgba

    # mc_colour_dict = {
    # 0: to_rgba("r"),
    # 1: to_rgba("g"),
    # 2: to_rgba("b"),
    # 3: to_rgba("m"),
    # }
    # data_colour_dict = {
    # 0: to_rgba("r", 0.5),
    # 1: to_rgba("g", 0.5),
    # 2: to_rgba("b", 0.5),
    # 3: to_rgba("m", 0.5),
    # }

    # make directory to hold results of this run
    # run_name = config.params["config"]["run_name"]
    # results_path = Path("results") / run_name
    # results_path.mkdir(parents=True, exist_ok=True)

    mcmc_params = np.mean(sampler.flatchain, axis=0)
    mcmc_params_cov = np.cov(np.transpose(sampler.flatchain))
    print("Fit Results")
    print(mcmc_params)
    print(mcmc_params_cov)

    # data_label = f"Data ({run_name})"
    # max_val = (1.5) * (max(config.params["config"]["data"]["central_values"]))
    # min_val = (0.0) * (min(config.params["config"]["data"]["central_values"]))

    # make plots of prediction at max point of posterior versus data

    # coefficients = list(config.coefficients)
    # label_string_bestfit = "best-fit: ("
    # for c in range(0, len(config.coefficients)):
    # if c == (len(config.coefficients) - 1):
    # label_string_bestfit = (
    # label_string_bestfit
    # + coefficients[c]
    # + " = "
    # + "%.1f" % mcmc_params[c]
    # + ")"
    # )
    # else:
    # label_string_bestfit = (
    # label_string_bestfit
    # + coefficients[c]
    # + " = "
    # + "%.1f" % mcmc_params[c]
    # + ", "
    # )

    # print(config.x_vals, len(config.x_vals))
    # print(
    # pb.make_prediction(mcmc_params),
    # len(pb.make_prediction(mcmc_params)),
    # )
    # print(
    # config.params["config"]["data"]["central_values"],
    # len(config.params["config"]["data"]["central_values"]),
    # )
    # print(
    # "data",
    # config.params["config"]["data"]["central_values"],
    # )
    # print("predicted", pb.make_prediction(mcmc_params))

    # predictions = pb.make_prediction(mcmc_params)
    # data_values = config.params["config"]["data"]["central_values"]
    # data_errors = np.sqrt(
    # np.array(config.params["config"]["data"]["covariance_matrix"]).diagonal()
    # )
    # for i in range(4):
    # start_index = i * 8
    # end_index = (i + 1) * 8
    # mttbar_bins = mttbar_small_bins if i < 2 else mttbar_large_bins
    # bin_widths = [
    # mttbar_bins[i + 1] - mttbar_bins[i] for i in range(len(mttbar_bins) - 1)
    # ]
    # x_values = [mttbar_bins[i] + width / 2 for i, width in enumerate(bin_widths)]
    # plt.errorbar(
    # x_values,
    # predictions[start_index:end_index],
    # fmt="x",
    # color=mc_colour_dict[i],
    # xerr=[width / 2 for width in bin_widths],
    # yerr=0.0,
    # label=label_string_bestfit,
    # )
    # plt.errorbar(
    # x_values,
    # data_values[start_index:end_index],
    # fmt=".",
    # color=data_colour_dict[i],
    # xerr=0,
    # yerr=data_errors[start_index:end_index],
    # label=data_label,
    # )
    # plt.plot(
    # [],
    # [],
    # label="$p_{t}$" + f"=[{pt_bins[i]}-{pt_bins[i + 1]}] ",
    # ls="None",
    # )
    # plt.xlabel(r"$m_{t\bar{t}}$ [GeV]")
    # plt.ylabel(r"$d^{2} \sigma / dm_{t\bar{t}} \, dp_{t}$ [pb GeV$^{-2}$]")
    # plt.xlim(left=mttbar_bins[0], right=mttbar_bins[-1])
    # plt.ylim(bottom=0)
    # plt.legend(loc=2)
    # plt.savefig(
    # results_path
    # / f"{run_name}_bestfit_predictions_{pt_bins[i]}_{pt_bins[i+1]}.png"
    # )
    # plt.close()

    sp = SummaryPlotter(config, pb, fitter)
    sp.fit_result(r"$\frac{d m_{t\bar{t}}}{d\sigma}$")
    sp.corner()

end = time.time()
print("Total elapsed wall time  = " + str(int(end - start)) + " seconds")
