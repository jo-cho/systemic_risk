import math
from math import log, sqrt, exp
from statistics import NormalDist
import numpy as np
from numpy.random import RandomState
from typing import Tuple
from scipy.optimize import fsolve
from scipy.stats import norm


def cca_func(x, e, vol, rf, d, t):
    init_e, init_vol = x
    d1 = (log(pow(init_e, 2) / d) + (rf + (pow(init_vol, 4)) / 2) * t) / (
            pow(init_vol, 2) * sqrt(t)
    )
    d2 = d1 - pow(init_vol, 2) * sqrt(t)

    eqty = e - init_e ** 2 * norm.cdf(d1) + d * exp(-rf * t) * norm.cdf(d2)
    sigm = e * vol - init_e ** 2 * init_vol ** 2 * norm.cdf(d1)
    return eqty, sigm


def cca(
    equity: float,
    volatility: float,
    risk_free_rate: float,
    default_barrier: float,
    time_to_maturity: float,
    cds_spread: float,
) -> Tuple[float, float]:
    """
    Systemic risk based on contingent claims analysis (CCA).
    The difference between put price and CDS price as a measure of firm's contribution
    to systemic risk based on Gray and Jobst (2010)

    :param equity: (float): the market value of the equity of the firm.
    :param volatility: (float): the volatility of equity.
    :param risk_free_rate: (float): the risk-free rate in annualized terms.
    :param default_barrier: (float): the face value of the outstandind debt at maturity.
    :param time_to_maturity: (float): the time to maturity of the debt.
    :param cds_spread: (float): the CDS spread for the firm.

    :return: Tuple[float, float]:
    A tuple of put price and the firm's contribution to the systemic risk indicator (put price - CDS put price).
    """

    # We need to solve a system of non-linear equations for asset price and asset volatility
    # x = [equity, volatility]
    x = fsolve(
        cca_func,
        (equity, volatility),  # initial values set to equity and its volatility
        args=(equity, volatility, risk_free_rate, default_barrier, time_to_maturity),
    )

    # We solved for (asset price)^1/2 and (asset volatility)^1/2 to ensure the
    # values are positive. We recover asset price and asset volatility here.
    x = x ** 2

    #  Solve for implied price of put
    d1 = (
        log(x[0] / default_barrier)
        + (risk_free_rate + (x[1] ** 2) / 2) * time_to_maturity
    ) / (x[1] * sqrt(time_to_maturity))
    d2 = d1 - x[1] * sqrt(time_to_maturity)

    # The price of the put
    put_price = default_barrier * exp(-risk_free_rate * time_to_maturity) * norm.cdf(
        -d2
    ) - x[0] * norm.cdf(-d1)

    # Solve for price of CDS implied put
    # Risky debt
    debt = default_barrier * exp(-risk_free_rate * time_to_maturity) - put_price

    # The price of the CDS put option
    cds_put = (
        (1 - exp(-(cds_spread / 10000) * (default_barrier / debt - 1) * time_to_maturity))
        * default_barrier
        * exp(-risk_free_rate * time_to_maturity)
    )
    out = (put_price, put_price - cds_put)
    return out


def distress_insurance_premium(
    default_prob: np.ndarray,
    correlations: np.ndarray,
    default_threshold: float = 0.15,
    random_seed: int = 0,
    n_simulated_returns: int = 500_000,
    n_simulations: int = 1_000,
) -> float:
    """
    Distress Insurance Preimum (DIP)
    A systemic risk metric by [Huang, Zhou, and Zhu (2009)](https://doi.org/10.1016/j.jbankfin.2009.05.017)
    which represents a hypothetical insurance premium against a systemic financial distress, defined as total losses that
    exceed a given threshold, say 15%, of total bank liabilities.

    :param default_prob: (np.ndarray): (n_banks,) array of the bank risk-neutral default probabilities.

    :param correlations: (np.ndarray): (n_banks, n_banks) array of the correlation matrix of the banks' asset returns.

    :param default_threshold: (float, optional): the threshold used to calculate the total losses to total liabilities. Defaults to 0.15.

    :param random_seed: (int, optional): the random seed used in Monte Carlo simulation for reproducibility. Defaults to 0.

    :param n_simulated_returns: (int, optional): the number of simulations to compute the distrituion of joint defaults. Defaults to 500,000.

    :param n_simulations: (int, optional): the number of simulations to compute the probability of losses. Defaults to 1,000.

    :return: float:
    The distress insurance premium against a systemic financial distress.
    """

    # Use the class to avoid impacting the global numpy state
    rng = RandomState(random_seed)
    n_banks = len(default_prob)
    # Simulate correlated normal distributions
    norm = NormalDist()
    default_thresholds = np.fromiter(
        (norm.inv_cdf(i) for i in default_prob),
        default_prob.dtype,
        count=n_banks,
    )
    R = np.linalg.cholesky(correlations).T
    z = np.dot(rng.normal(0, 1, size=(n_simulated_returns, n_banks)), R)

    default_dist = np.sum(z < default_thresholds, axis=1)

    # an array where the i-th element is the frequency of i banks jointly default
    # where len(frequency_of_join_defaults) is n_banks+1
    frequency_of_join_defaults = np.bincount(default_dist, minlength=n_banks + 1)
    dist_joint_defaults = frequency_of_join_defaults / n_simulated_returns

    loss_given_default = np.empty(shape=(n_banks, n_simulations))
    for i in range(n_banks):
        lgd = np.sum(rng.triangular(0.1, 0.55, 1, size=(i + 1, n_simulations)), axis=0)
        loss_given_default[i:] = lgd

    # Maximum losses are N. Divide this into N*100 intervals.
    # Find the probability distribution of total losses in the default case
    intervals = 100
    loss_given_default *= intervals

    prob_losses = np.zeros(n_banks * intervals)
    for i in range(n_banks):
        for j in range(n_simulations):
            # Multiply losses_given_default(i,j) by intervals to find the right slot
            # in the prob_losses. Then we increment this by probability of i defaults
            idx = math.ceil(loss_given_default[i, j])
            prob_losses[idx] += dist_joint_defaults[i + 1]

    # Convert to probabilities
    prob_losses = prob_losses / n_simulations
    pct_threshold = int(default_threshold * 100)

    # Find the probability that the losses are great than 0.15 the total liabilities i.e. > 0.15*N
    prob_great_losses = np.sum(prob_losses[pct_threshold * n_banks :])

    exp_losses = (
        np.dot(
            np.array(range(pct_threshold * n_banks, intervals * n_banks)),
            prob_losses[pct_threshold * n_banks :],
        )
        / (100 * prob_great_losses)
    )
    out = exp_losses * prob_great_losses
    return out
