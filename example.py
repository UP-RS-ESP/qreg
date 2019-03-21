#! /usr/bin/env python3
"""
Example script to demonstrate the use of the QREG module
--------------------------------------------------------

"""
# Created: Wed Mar 20, 2019  10:25am
# Last modified: Thu Mar 21, 2019  12:19pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

import qreg


def run_example():
    """
    Runs the example
    """
    print("load Engel's expenditure data set ...")
    engel = np.genfromtxt("engel.csv", delimiter=",",
                          names=True, skip_header=4)
    # print(engel.dtype.names)
    tau = [0.10, 0.25, 0.50, 0.75, 0.90]
    beta = qreg.linear(engel["income_bef"], engel["foodexp_bef"], tau)


    # estimate the conditional mean: Ls. Sq. slope and intercept of the line
    t, x = engel["income_bef"], engel["foodexp_bef"]
    m = np.corrcoef(t, x)[0, 1] * (np.std(x) / np.std(t))
    c = np.mean(x) - m * np.mean(t)

    # plot
    fig = pl.figure(figsize=[8., 6.])
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    ax.plot(engel["income_bef"], engel["foodexp_bef"], "o", alpha=0.5)
    for i in range(len(beta)):
        ax.plot(engel["income_bef"],
                beta[i][0] + beta[i][1] * engel["income_bef"],
                "-", label="%d-th percentile" % int(tau[i] * 100.),
                alpha=0.5)
    ax.plot(t, m * t + c, ":", label="Least Squares Fit")
    ax.set_xlabel("Income (BEF)", fontsize=axlabfs)
    ax.set_ylabel("Food Expenditure (BEF)", fontsize=axlabfs)
    ax.set_title("Engel Food Expenditure Dataset",
                 fontsize=axlabfs+2, fontweight="bold", pad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=31))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=31))
    ax.tick_params(which="major", size=10)
    ax.tick_params(which="minor", size=5)
    ax.grid(which="both")
    ax.legend()
    FN = "qreg_engel.png"
    fig.savefig(FN)
    print("figure saved to: %s" % FN)

    return None


if __name__ == "__main__":
    axlabfs, tiklabfs = 14., 12.
    run_example()
