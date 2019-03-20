"""
Python module to carry out quantile regression

This module can be used to estimate linear regression coefficients for
different quantiles for a give data set.

"""
# Created: Fri Feb 22, 2019  11:52pm
# Last modified: Wed Mar 20, 2019  01:36pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import numpy as np
from scipy.special import ndtri, ndtr
import sys


def quantile_regression(x, y):
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
        result of the statistical test indicating whether or not to accept hte
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

    return x, y
