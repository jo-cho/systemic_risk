import numpy as np


def marginal_expected_shortfall(
        firm_returns: np.ndarray, market_returns: np.ndarray, q: float = 0.05
) -> float:
    """Marginal Expected Shortfall (MES).
    The firm's average return during the 5% worst days for the market.
    MES measures how exposed a firm is to aggregate tail shocks and, interestingly, together with leverage,
    it has a significant explanatory power for which firms contribute to a potential crisis

    Args:
        firm_returns (np.ndarray): (n_days,) array of the returns (equity or CDS) for the firm.
        market_returns (np.ndarray): (n_days,) array of the returns (equity or CDS) for the market as a whole.
        q (float, optional): The percentile. Range is [0, 1]. Deaults to 0.05.
    Returns:
        float: The marginal expected shortfall of firm $i$ at time $t$.
    """
    assert 0 <= q <= 1
    assert firm_returns.shape == market_returns.shape
    low_threshold = np.percentile(market_returns, q * 100)
    worst_days = np.argwhere(market_returns < low_threshold)
    out = np.mean(firm_returns[worst_days])
    return out


def systemic_expected_shortfall(
    mes_training_sample: np.ndarray,
    lvg_training_sample: np.ndarray,
    ses_training_sample: np.ndarray,
    mes_firm: float,
    lvg_firm: float,
) -> float:
    r"""Systemic Expected Shortfall (SES)
    A measure of a financial institution's contribution to a systemic crisis, which equals to
    the expected amount a bank is undercapitalized in a future systemic event in which the overall financial system is undercapitalized.
    Args:
        mes_training_sample (np.ndarray): (n_firms,) array of firm ex ante MES.
        lvg_training_sample (np.ndarray): (n_firms,) array of firm ex ante LVG (say, on the last day of the period of training data)
        ses_training_sample (np.ndarray): (n_firms,) array of firm ex post cumulative return for date range after `lvg_training_sample`.
        mes_firm (float): The current firm MES used to calculate the firm (fitted) SES value.
        lvg_firm (float): The current firm leverage used to calculate the firm (fitted) SES value.
    Returns:
        float: The systemic risk that firm $i$ poses to the system at a future time.
    """
    assert mes_training_sample.shape == lvg_training_sample.shape
    assert mes_training_sample.shape == ses_training_sample.shape

    n_firms = mes_training_sample.shape

    data = np.vstack([np.ones(n_firms), mes_training_sample, lvg_training_sample]).T
    betas = np.linalg.lstsq(data, ses_training_sample, rcond=None)[0]
    _, b, c = betas
    ses = (b * mes_firm + c * lvg_firm) / (b + c)
    return ses
