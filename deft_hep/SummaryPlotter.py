import matplotlib.pyplot as pl
import matplotlib.image as image
import matplotlib.ticker as mticker

import numpy as np
import os
import json
from pathlib import Path

import numpy as np
import emcee
import corner
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text")

from .ConfigReader import ConfigReader
from .PredictionBuilder import PredictionBuilder


class SummaryPlotter:
    def __init__(
        self,
        config: ConfigReader,
        pb: PredictionBuilder,
        sampler: emcee.EnsembleSampler,
    ):

        samples = sampler.chain.reshape(-1, len(config.prior_limits))

        # make directory to hold results of this run
        run_name = config.params["config"]["run_name"]
        results_path = Path("results") / run_name
        results_path.mkdir(parents=True, exist_ok=True)

        mcmc_params = np.mean(sampler.flatchain, axis=0)
        mcmc_params_cov = np.cov(np.transpose(sampler.flatchain))

        data_label = f"Data ({run_name})"
        max_val = (1.5) * (max(config.params["config"]["data"]["central_values"]))
        min_val = (0.0) * (min(config.params["config"]["data"]["central_values"]))
        xlabel = config.observable
        ylabel = "d$\sigma_{tt}$/dX"

        # make plots of prediction at max point of posterior versus data
        coefficients = list(config.coefficients)
        pl.figure()
        label_string_bestfit = "best-fit: ("
        for c in range(0, len(config.coefficients)):
            if c == (len(config.coefficients) - 1):
                label_string_bestfit = (
                    label_string_bestfit
                    + coefficients[c]
                    + " = "
                    + "%.1f" % mcmc_params[c]
                    + ")"
                )
            else:
                label_string_bestfit = (
                    label_string_bestfit
                    + coefficients[c]
                    + " = "
                    + "%.1f" % mcmc_params[c]
                    + ", "
                )

        pl.errorbar(
            config.x_vals,
            pb.make_prediction(mcmc_params),
            fmt="m",
            xerr=0.0,
            yerr=0.0,
            label=label_string_bestfit,
        )
        pl.errorbar(
            config.x_vals,
            config.params["config"]["data"]["central_values"],
            fmt="o",
            xerr=0.25,
            yerr=0.05,
            label=data_label,
        )
        pl.axis(
            [
                config.x_vals[0] - 0.25,
                config.x_vals[len(config.x_vals) - 1] + 0.25,
                min_val,
                max_val,
            ]
        )
        ax = pl.gca()
        labely = ax.set_xlabel(xlabel, fontsize=18)
        labely = ax.set_ylabel(ylabel, fontsize=18)
        ax.xaxis.set_label_coords(0.85, -0.065)
        ax.yaxis.set_label_coords(-0.037, 0.83)
        pl.legend(loc=2)
        pl.savefig(results_path / f"{run_name}_bestfit_predictions.png")
        pl.close()

        #  make "corner" plot
        labels = []
        ranges = []
        for c in config.params["config"]["model"]["prior_limits"].keys():
            label = "$" + c + "$"
            labels.append(label)
            ranges.append(config.params["config"]["model"]["prior_limits"][c])

        fig = corner.corner(
            samples,
            labels=config.tex_labels,
            label_kwargs={"fontsize": 18},
            range=ranges,
            quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5),),
            truths=np.zeros(len(labels)),
            show_titles=True,
            title_kwargs={"fontsize": 18},
        )

        plotfilename = results_path / f"{run_name}.png"

        ax0 = fig.add_subplot(999)
        ax0.axis("off")

        fig.savefig(plotfilename)
        pl.close()

        bestFits = []
        marginUncsUp = []
        marginUncsDown = []
        x = []

        for i in range(len(labels)):
            mcmc = np.percentile(samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            x.append(i)
            bestFits.append(mcmc[1])
            marginUncsUp.append(q[0])
            marginUncsDown.append(q[1])

        pl.figure()
        pl.tight_layout()
        pl.gcf().subplots_adjust(bottom=0.15)

        fig = pl.errorbar(
            x,
            bestFits,
            yerr=[marginUncsDown, marginUncsUp],
            fmt="o",
            label=r"posterior median and CI",
        )
        pl.xticks(range(len(labels)), config.tex_labels, rotation="45", fontsize=21)
        ax = pl.gca()
        ax.yaxis.set_label_coords(-0.075, 0.83)
        labely = ax.set_ylabel(r"$c_{i}$ [$GeV^{-2}$]", fontsize=18)
        pl.legend(loc=1, fontsize=14)
        pl.savefig(results_path / f"{run_name}_bestFits.png")
        pl.close()

        ############################################
        ######## Corner with 1-d CL overlaid  ######
        ############################################

        fig_overlay = corner.corner(
            samples,
            labels=config.tex_labels,
            label_kwargs={"fontsize": 21},
            range=ranges,
            color="k",
            quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5),),
            truths=np.zeros(len(labels)),
            show_titles=True,
            title_kwargs={"fontsize": 19},
            hist2d_kwargs={"fill_contours": True, "plot_density": True},
        )

        resplot = image.imread(results_path / f"{run_name}_bestFits.png")

        ax0 = fig_overlay.add_subplot(322)
        ax0.axis("off")
        img = ax0.imshow(resplot)

        ax0 = fig_overlay.add_subplot(5, 5, 15)
        ax0.axis("off")

        plotfilename = results_path / f"{run_name}_overlay.png"

        fig_overlay.savefig(plotfilename)
        pl.close()

        # make plots of best fit prediction versus data
        pl.figure()
        label_string_bestfit = "best-fit prediction"

        pl.errorbar(
            config.x_vals,
            pb.make_prediction(bestFits),
            fmt="m",
            xerr=0.0,
            yerr=0.0,
            label=label_string_bestfit,
        )
        pl.errorbar(
            config.x_vals,
            config.params["config"]["data"]["central_values"],
            fmt="o",
            xerr=0.25,
            yerr=np.sqrt(
                np.diagonal(config.params["config"]["data"]["covariance_matrix"])
            ),
            label=data_label,
        )
        pl.axis(
            [
                config.x_vals[0] - 0.25,
                config.x_vals[len(config.x_vals) - 1] + 0.25,
                min_val,
                max_val,
            ]
        )
        ax = pl.gca()
        labely = ax.set_xlabel(xlabel, fontsize=18)
        labely = ax.set_ylabel(ylabel, fontsize=18)
        ax.xaxis.set_label_coords(0.85, -0.065)
        ax.yaxis.set_label_coords(-0.037, 0.83)
        pl.legend(loc=1)
        pl.savefig(results_path / f"{run_name}_bestfit_predictions.png")
        pl.close()

        # make fit summary json
        fitSummary = {}
        fitSummary["bestFit"] = bestFits
        fitSummary["bestFit"] = bestFits
        fitSummary["UncsUp"] = marginUncsUp
        fitSummary["UncsDown"] = marginUncsDown
        fitSummary["labels"] = labels
        fitSummary["x"] = x

        with open(run_name + ".json", "w") as fs:
            json.dump(fitSummary, fs)

        pl.figure()

        # best-fit prediction and random samples compared to data
        uncX = np.absolute((np.diff(config.bins))) / 2.0
        dataUncY = np.sqrt(np.diag(config.cov))
        pl.errorbar(
            config.x_vals,
            config.params["config"]["data"]["central_values"],
            fmt="o",
            xerr=uncX,
            yerr=dataUncY,
            label=data_label,
        )

        inds = np.random.randint(len(samples), size=499)
        for ind in inds:
            sample = samples[ind]
            pl.plot(config.x_vals, pb.make_prediction(sample), "C1", alpha=0.02)

        pl.plot(
            config.x_vals,
            pb.make_prediction(samples[42]),
            "C1",
            label="500 random samples",
            alpha=0.1,
        )

        ax = pl.gca()
        labelx = ax.set_xlabel(xlabel, fontsize=18)
        labely = ax.set_ylabel(ylabel, fontsize=18)
        ax.xaxis.set_label_coords(0.85, -0.065)
        ax.yaxis.set_label_coords(-0.037, 0.83)
        pl.legend(loc=1)
        pl.savefig(results_path / f"{run_name}_postfit_predictions.png")
        pl.close()

        pl.figure()
        fig, axes = pl.subplots(len(labels), figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        for i in range(len(labels)):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        pl.savefig(results_path / f"{run_name}_walkersPaths.png")
        pl.close()

        # covariance matrix of coefficicents
        fig, ax = pl.subplots()
        im = ax.imshow(mcmc_params_cov, cmap=pl.cm.Blues)

        ax.set_xticks(np.arange(len(config.tex_labels)))
        ax.set_yticks(np.arange(len(config.tex_labels)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(config.tex_labels)
        ax.set_yticklabels(config.tex_labels)

        # fixing yticks with matplotlib.ticker "FixedLocator"
        # ticks_loc = ax.get_yticks().tolist()
        # ticks_loc = config.tex_labels

        # ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        # label_format = '{:,.0f}'

        # pl.imshow(mcmc_params_cov, cmap=pl.cm.Blues)
        # ax.set_xticks(np.arange(len(config.tex_labels)))

        # ax.set_yticklabels(ticks_loc)

        pl.savefig(results_path / f"{run_name}_mcmc_params_cov.png")
        pl.close()

        # Write SM pred to text file for validation
        sm_file_name = results_path / f"{run_name}_postFit_sm_pred.txt"
        f = open(sm_file_name, "w")
        sm_pred = str(
            repr(
                pb.make_prediction(
                    np.zeros(len(config.params["config"]["model"]["prior_limits"]))
                )
            )
        )
        f.write(sm_pred)
        f.close()
