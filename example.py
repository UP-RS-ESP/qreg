#! /usr/bin/env python3
"""
Example script to demonstrate the use of the QREG module
--------------------------------------------------------

"""
# Created: Wed Mar 20, 2019  10:25am
# Last modified: Wed Mar 20, 2019  10:59am
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

    x, y = qreg.quantile_regression(engel["income_bef"], engel["foodexp_bef"])

    # plot
    fig = pl.figure(figsize=[12., 8.])
    ax = fig.add_axes([0.15, 0.15, 0.70, 0.70])
    ax.plot(engel["income_bef"], engel["foodexp_bef"], "o", alpha=0.5)
    ax.plot(x, y, "x", alpha=0.5)
    ax.set_xlabel("Income (BEF)", fontsize=axlabfs)
    ax.set_ylabel("Food Expenditure (BEF)", fontsize=axlabfs)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=51))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=51))
    ax.tick_params(which="major", size=10)
    ax.tick_params(which="minor", size=5)
    ax.grid(which="both")
    pl.show()

    return None


if __name__ == "__main__":
    axlabfs, tiklabfs = 14., 12.
    run_example()
