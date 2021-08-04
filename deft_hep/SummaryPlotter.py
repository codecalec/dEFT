from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import emcee
import corner
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text")

from .ConfigReader import ConfigReader
from .PredictionBuilder import PredictionBuilder
from .MCMCFitter import MCMCFitter


class SummaryPlotter:
    def __init__(
        self,
        config: ConfigReader,
        pb: PredictionBuilder,
        fitter: MCMCFitter,
        **kwargs,
    ):
        """
        Create basic plots using results from MCMCFitter

        :param config: Config for the analysis
        :type config: ConfigReader

        :param pb: Model for predicting coefficient values
        :type pb: PredictionBuilder

        :param fitter: Fitter for the analysis
        :type fitter: MCMCFitter

        :param result_path: Path where plots should be saved
        :type result_path: str or pathlib.Path
        """

        if "result_path" not in kwargs:
            # make results directory based on run name
            run_name = config.params["config"]["run_name"]
            results_path = Path("results") / run_name
        else:
            results_path = kwargs["results_path"]
        results_path.mkdir(parents=True, exist_ok=True)

        self.path = results_path
        self.mcmc_params = np.mean(fitter.sampler.flatchain, axis=0)
        self.mcmc_params_cov = np.cov(np.transpose(fitter.sampler.flatchain))
        self.samples = fitter.sampler.chain.reshape(-1, len(config.prior_limits))
        self.config = config
        self.pb = pb

    def fit_result(
        self, ylabel: str, show_plot: bool = True, log_scale: bool = False, **kwargs
    ):
        """
        Generate plot comparing the model with optimised coefficients with data present in the configuration within ConfigBuilder

        :param ylabel: Label for y-axis of plot
        :type ylabel: str

        :param show_plot: Determines whether `plt.show()` is called after a plot is created
        :type show_plot: bool

        :param filename: Name of output file
        :type filename: str

        :param log_scale: Use a log scale for plot
        :type log_scale: bool
        """

        if "filename" in kwargs:
            filename = kwargs["filename"]
        else:
            filename = f"{self.config.run_name}_bestfit_predictions.png"

        data_label = f"Data ({self.config.run_name})"
        xlabel = self.config.observable if "xlabel" not in kwargs else kwargs["xlabel"]

        if len(self.config.coefficients) == 1:
            label_bestfit = f"Best Fit: {self.config.tex_labels[0]}={self.mcmc_params[0]:.3f}$\pm${np.sqrt(self.mcmc_params_cov):.3f}"
        else:
            label_bestfit = "Best Fit:"
            for c_name, c_value, c_error in zip(
                self.config.tex_labels,
                self.mcmc_params,
                np.sqrt(np.diagonal(self.mcmc_params_cov)),
            ):
                label_bestfit += f" {c_name}={c_value:.3f}$\pm${c_error:.3f}\n"

        data_err = np.sqrt(np.diagonal(self.config.cov))

        half_bin_width = [
            (self.config.bins[i + 1] - self.config.bins[i]) / 2
            for i in range(len(self.config.bins) - 1)
        ]

        plt.errorbar(
            self.config.x_vals,
            self.config.data,
            fmt=".",
            xerr=half_bin_width,
            yerr=data_err,
            label=data_label,
        )

        plt.errorbar(
            self.config.x_vals,
            self.pb.make_prediction(self.mcmc_params),
            fmt="mo",
            ls="None",
            xerr=0.0,
            yerr=0.0,
            label=label_bestfit,
        )

        if log_scale:
            plt.yscale("log")

        ax = plt.gca()
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig(self.path / filename)

        if show_plot:
            plt.show()

        plt.close()

    def corner(self, show_plot: bool = True, filename: str = None, **corner_kwargs):
        """
        Generate corner plot using `corner <https://corner.readthedocs.io/en/latest/>`_.

        :param show_plot: Determines whether `plt.show()` is called after a plot is created.
        :type show_plot: bool

        :param filename: Provide filename for output file
        :type filename: Optional[str]
        """
        if filename is None:
            filename = f"{self.config.run_name}.png"

        #  make corner plot
        ranges = [limit for limit in self.config.prior_limits.values()]

        fig = corner.corner(
            self.samples,
            labels=self.config.tex_labels,
            label_kwargs={"fontsize": 18},
            # range=ranges, todo: Reimplement
            quantiles=[0.16, 0.84],
            # levels=(1 - np.exp(-0.5),), Causes ValueError: Contour levels must be increasing
            truths=np.zeros(len(self.config.tex_labels)),
            show_titles=True,
            title_kwargs={"fontsize": 18},
            **corner_kwargs,
        )

        ax0 = fig.add_subplot(999)
        ax0.axis("off")

        fig.savefig(self.path / filename)
        if show_plot:
            plt.show()
        plt.close()


# class SummaryPlotter:
# def __init__(
# self,
# config: ConfigReader,
# pb: PredictionBuilder,
# sampler: emcee.EnsembleSampler,
# ):

# samples = sampler.chain.reshape(-1, len(config.prior_limits))

# # make directory to hold results of this run
# run_name = config.params["config"]["run_name"]
# results_path = Path("results") / run_name
# results_path.mkdir(parents=True, exist_ok=True)

# mcmc_params = np.mean(sampler.flatchain, axis=0)
# mcmc_params_cov = np.cov(np.transpose(sampler.flatchain))

# data_label = f"Data ({run_name})"
# max_val = (1.5) * (max(config.params["config"]["data"]["central_values"]))
# min_val = (0.0) * (min(config.params["config"]["data"]["central_values"]))
# xlabel = config.observable
# ylabel = "d$\sigma_{tt}$/dX"

# # make plots of prediction at max point of posterior versus data
# coefficients = list(config.coefficients)
# pl.figure()
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

# pl.errorbar(
# config.x_vals,
# pb.make_prediction(mcmc_params),
# fmt="m",
# xerr=0.0,
# yerr=0.0,
# label=label_string_bestfit,
# )
# pl.errorbar(
# config.x_vals,
# config.params["config"]["data"]["central_values"],
# fmt="o",
# xerr=0.25,
# yerr=0.05,
# label=data_label,
# )
# pl.axis(
# [
# config.x_vals[0] - 0.25,
# config.x_vals[len(config.x_vals) - 1] + 0.25,
# min_val,
# max_val,
# ]
# )
# ax = pl.gca()
# labely = ax.set_xlabel(xlabel, fontsize=18)
# labely = ax.set_ylabel(ylabel, fontsize=18)
# ax.xaxis.set_label_coords(0.85, -0.065)
# ax.yaxis.set_label_coords(-0.037, 0.83)
# pl.legend(loc=2)
# pl.savefig(results_path / f"{run_name}_bestfit_predictions.png")
# pl.close()

# #  make "corner" plot
# labels = []
# ranges = []
# for c in config.params["config"]["model"]["prior_limits"].keys():
# label = "$" + c + "$"
# labels.append(label)
# ranges.append(config.params["config"]["model"]["prior_limits"][c])

# fig = corner.corner(
# samples,
# labels=config.tex_labels,
# label_kwargs={"fontsize": 18},
# range=ranges,
# quantiles=[0.16, 0.84],
# levels=(1 - np.exp(-0.5),),
# truths=np.zeros(len(labels)),
# show_titles=True,
# title_kwargs={"fontsize": 18},
# )

# plotfilename = results_path / f"{run_name}.png"

# ax0 = fig.add_subplot(999)
# ax0.axis("off")

# fig.savefig(plotfilename)
# pl.close()

# bestFits = []
# marginUncsUp = []
# marginUncsDown = []
# x = []

# for i in range(len(labels)):
# mcmc = np.percentile(samples[:, i], [16, 50, 84])
# q = np.diff(mcmc)
# txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
# txt = txt.format(mcmc[1], q[0], q[1], labels[i])
# x.append(i)
# bestFits.append(mcmc[1])
# marginUncsUp.append(q[0])
# marginUncsDown.append(q[1])

# pl.figure()
# pl.tight_layout()
# pl.gcf().subplots_adjust(bottom=0.15)

# fig = pl.errorbar(
# x,
# bestFits,
# yerr=[marginUncsDown, marginUncsUp],
# fmt="o",
# label=r"posterior median and CI",
# )
# pl.xticks(range(len(labels)), config.tex_labels, rotation="45", fontsize=21)
# ax = pl.gca()
# ax.yaxis.set_label_coords(-0.075, 0.83)
# labely = ax.set_ylabel(r"$c_{i}$ [$GeV^{-2}$]", fontsize=18)
# pl.legend(loc=1, fontsize=14)
# pl.savefig(results_path / f"{run_name}_bestFits.png")
# pl.close()

# ############################################
# ######## Corner with 1-d CL overlaid  ######
# ############################################

# fig_overlay = corner.corner(
# samples,
# labels=config.tex_labels,
# label_kwargs={"fontsize": 21},
# range=ranges,
# color="k",
# quantiles=[0.16, 0.84],
# levels=(1 - np.exp(-0.5),),
# truths=np.zeros(len(labels)),
# show_titles=True,
# title_kwargs={"fontsize": 19},
# hist2d_kwargs={"fill_contours": True, "plot_density": True},
# )

# resplot = image.imread(results_path / f"{run_name}_bestFits.png")

# ax0 = fig_overlay.add_subplot(322)
# ax0.axis("off")
# img = ax0.imshow(resplot)

# ax0 = fig_overlay.add_subplot(5, 5, 15)
# ax0.axis("off")

# plotfilename = results_path / f"{run_name}_overlay.png"

# fig_overlay.savefig(plotfilename)
# pl.close()

# # make plots of best fit prediction versus data
# pl.figure()
# label_string_bestfit = "best-fit prediction"

# pl.errorbar(
# config.x_vals,
# pb.make_prediction(bestFits),
# fmt="m",
# xerr=0.0,
# yerr=0.0,
# label=label_string_bestfit,
# )
# pl.errorbar(
# config.x_vals,
# config.params["config"]["data"]["central_values"],
# fmt="o",
# xerr=0.25,
# yerr=np.sqrt(
# np.diagonal(config.params["config"]["data"]["covariance_matrix"])
# ),
# label=data_label,
# )
# pl.axis(
# [
# config.x_vals[0] - 0.25,
# config.x_vals[len(config.x_vals) - 1] + 0.25,
# min_val,
# max_val,
# ]
# )
# ax = pl.gca()
# labely = ax.set_xlabel(xlabel, fontsize=18)
# labely = ax.set_ylabel(ylabel, fontsize=18)
# ax.xaxis.set_label_coords(0.85, -0.065)
# ax.yaxis.set_label_coords(-0.037, 0.83)
# pl.legend(loc=1)
# pl.savefig(results_path / f"{run_name}_bestfit_predictions.png")
# pl.close()

# # make fit summary json
# fitSummary = {}
# fitSummary["bestFit"] = bestFits
# fitSummary["bestFit"] = bestFits
# fitSummary["UncsUp"] = marginUncsUp
# fitSummary["UncsDown"] = marginUncsDown
# fitSummary["labels"] = labels
# fitSummary["x"] = x

# with open(run_name + ".json", "w") as fs:
# json.dump(fitSummary, fs)

# pl.figure()

# # best-fit prediction and random samples compared to data
# uncX = np.absolute((np.diff(config.bins))) / 2.0
# dataUncY = np.sqrt(np.diag(config.cov))
# pl.errorbar(
# config.x_vals,
# config.params["config"]["data"]["central_values"],
# fmt="o",
# xerr=uncX,
# yerr=dataUncY,
# label=data_label,
# )

# inds = np.random.randint(len(samples), size=499)
# for ind in inds:
# sample = samples[ind]
# pl.plot(config.x_vals, pb.make_prediction(sample), "C1", alpha=0.02)

# pl.plot(
# config.x_vals,
# pb.make_prediction(samples[42]),
# "C1",
# label="500 random samples",
# alpha=0.1,
# )

# ax = pl.gca()
# labelx = ax.set_xlabel(xlabel, fontsize=18)
# labely = ax.set_ylabel(ylabel, fontsize=18)
# ax.xaxis.set_label_coords(0.85, -0.065)
# ax.yaxis.set_label_coords(-0.037, 0.83)
# pl.legend(loc=1)
# pl.savefig(results_path / f"{run_name}_postfit_predictions.png")
# pl.close()

# pl.figure()
# fig, axes = pl.subplots(len(labels), figsize=(10, 7), sharex=True)
# samples = sampler.get_chain()
# for i in range(len(labels)):
# ax = axes[i]
# ax.plot(samples[:, :, i], "k", alpha=0.3)
# ax.set_xlim(0, len(samples))
# ax.set_ylabel(labels[i])
# ax.yaxis.set_label_coords(-0.1, 0.5)

# axes[-1].set_xlabel("step number")

# pl.savefig(results_path / f"{run_name}_walkersPaths.png")
# pl.close()

# # covariance matrix of coefficicents
# fig, ax = pl.subplots()
# im = ax.imshow(mcmc_params_cov, cmap=pl.cm.Blues)

# ax.set_xticks(np.arange(len(config.tex_labels)))
# ax.set_yticks(np.arange(len(config.tex_labels)))
# # ... and label them with the respective list entries
# ax.set_xticklabels(config.tex_labels)
# ax.set_yticklabels(config.tex_labels)

# # fixing yticks with matplotlib.ticker "FixedLocator"
# # ticks_loc = ax.get_yticks().tolist()
# # ticks_loc = config.tex_labels

# # ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
# # label_format = '{:,.0f}'

# # pl.imshow(mcmc_params_cov, cmap=pl.cm.Blues)
# # ax.set_xticks(np.arange(len(config.tex_labels)))

# # ax.set_yticklabels(ticks_loc)

# pl.savefig(results_path / f"{run_name}_mcmc_params_cov.png")
# pl.close()

# # Write SM pred to text file for validation
# sm_file_name = results_path / f"{run_name}_postFit_sm_pred.txt"
# f = open(sm_file_name, "w")
# sm_pred = str(
# repr(
# pb.make_prediction(
# np.zeros(len(config.params["config"]["model"]["prior_limits"]))
# )
# )
# )
# f.write(sm_pred)
# f.close()
