from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd


def absorption_ratio(
        asset_returns, fraction_eigenvectors=0.2
) -> float:
    """
    A measure of systemic risk defined as the fraction of the total variance of a set of asset returns
    explained or absorbed by a fixed number of eigenvectors.

    :param asset_returns: (np.ndarray or pd.DataFrame):
    (n_days, n_assets) arrays of asset returns
    :param fraction_eigenvectors: (float, optional):
    The fraction of eigenvectors used to calculate the absorption ratio. Defaults to 0.2 as in the paper

    :return: (float):
    Absorption ratio for the market
    """
    if isinstance(asset_returns, pd.DataFrame):
        asset_returns = asset_returns.to_numpy

    asset_returns = asset_returns.T
    n_assets, _ = asset_returns.shape
    cov = np.cov(asset_returns)
    eig = np.linalg.eigvals(cov)
    eig_sorted = sorted(eig)
    num_eigenvalues = int(
        Decimal(fraction_eigenvectors * n_assets).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )
    return sum(eig_sorted[len(eig_sorted) - num_eigenvalues:]) / np.trace(cov)
