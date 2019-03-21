"""
Python module to carry out quantile regression

This module can be used to estimate linear regression coefficients for
different quantiles for a give data set.

"""
# Created: Fri Feb 22, 2019  11:52pm
# Last modified: Thu Mar 21, 2019  12:19pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import numpy as np
import sys
from scipy.optimize import linprog


def linear(x, y, tau=[]):
    """
    Estimates quantile regression coefficients for given data and quantiles.

    Parameters
    ----------
    t : 1D numpy.ndarray
        array of the time points of measurements
    x : 1D numpy.ndarray
        array containing the measurements corresponding to entries of 't'
    eps : scalar, float, greater than zero
        least count error of measurements which help determine ties in the data
    alpha : scalar, float, greater than zero
        significance level of the statistical test (Type I error)
    Ha : string, options include 'up', 'down', 'upordown'
        type of test: one-sided ('up' or 'down') or two-sided ('updown')

    Returns
    -------
    MK : string
        result of the statistical test indicating whether or not to accept the
        alternative hypothesis 'Ha'
    m : scalar, float
        slope of the linear fit to the data
    c : scalar, float
        intercept of the linear fit to the data
    p : scalar, float, greater than zero
        p-value of the obtained Z-score statistic for the Mann-Kendall test

    Raises
    ------
    AssertionError : error
                    least count error of measurements 'eps' is not given
    AssertionError : error
                    significance level of test 'alpha' is not given
    AssertionError : error
                    alternative hypothesis 'Ha' is not given

    """
    print("quantile regression...")
    assert (len(tau) > 0), "Please give a list of quantiles"

    ntau = len(tau)
    N = y.shape[0]
    X = np.c_[np.ones(N), x]
    K = X.shape[1]
    i_N = np.ones(N)
    I_N = np.diagflat(i_N)
    A = np.c_[X, -X, I_N, -I_N]
    b = y
    beta = np.zeros((ntau, 2))
    for i in range(ntau):
        # first attempt with linear programming
        c = np.r_[np.zeros(2 * K), tau[i] * i_N, (1. - tau[i]) * i_N]
        res = linprog(c=c,
                      A_eq=A, b_eq=b,
                      method="interior-point",
                      bounds=(0, None),
                      )
        z = res.x
        u = z[2 * K:2 * K + N]
        v = z[2 * K + N:2 * K + 2 * N]
        beta[i] = z[0:K] - z[K:(K + K)]

    return beta
