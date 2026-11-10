"""Portfolio statistics and Markowitz mean-variance optimization."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


def portfolio_stats(weights: np.ndarray, mean_returns: pd.Series, cov: pd.DataFrame,
                    risk_free_rate: float = 0.0) -> dict:
    """Annualized return, volatility and Sharpe ratio for a weight vector."""
    ret = float(np.dot(weights, mean_returns)) * TRADING_DAYS
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov * TRADING_DAYS, weights))))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


def random_portfolios(mean_returns: pd.Series, cov: pd.DataFrame, n: int = 20000,
                      seed: int = 42) -> pd.DataFrame:
    """Generate random long-only portfolios for the efficient frontier cloud."""
    rng = np.random.default_rng(seed)
    n_assets = len(mean_returns)
    rows = []
    for _ in range(n):
        w = rng.random(n_assets)
        w /= w.sum()
        rows.append(portfolio_stats(w, mean_returns, cov))
    return pd.DataFrame(rows)


def _optimize(mean_returns: pd.Series, cov: pd.DataFrame, objective: str,
              target_return: float | None = None) -> dict:
    """Shared SLSQP optimizer for max-Sharpe, min-volatility or target-return portfolios."""
    n_assets = len(mean_returns)
    bounds = [(0.0, 1.0)] * n_assets  # long-only
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({"type": "eq",
                            "fun": lambda w: np.dot(w, mean_returns) * TRADING_DAYS - target_return})

    if objective == "sharpe":
        def neg_sharpe(w):
            s = portfolio_stats(w, mean_returns, cov)
            return -s["sharpe"]
        result = minimize(neg_sharpe, np.full(n_assets, 1.0 / n_assets), method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-9})
    elif objective == "vol":
        def vol_fn(w):
            return portfolio_stats(w, mean_returns, cov)["volatility"]
        result = minimize(vol_fn, np.full(n_assets, 1.0 / n_assets), method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-9})
    else:
        raise ValueError(f"unknown objective: {objective}")

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return {"weights": result.x, **portfolio_stats(result.x, mean_returns, cov)}


def max_sharpe_portfolio(mean_returns: pd.Series, cov: pd.DataFrame) -> dict:
    return _optimize(mean_returns, cov, "sharpe")


def min_volatility_portfolio(mean_returns: pd.Series, cov: pd.DataFrame) -> dict:
    return _optimize(mean_returns, cov, "vol")


def efficient_frontier(mean_returns: pd.Series, cov: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    """Trace the efficient frontier between the min-vol return and the best single-asset return."""
    lo = portfolio_stats(min_volatility_portfolio(mean_returns, cov)["weights"],
                         mean_returns, cov)["return"]
    hi = float(mean_returns.max()) * TRADING_DAYS
    rows = []
    for target in np.linspace(lo, hi, n_points):
        try:
            res = _optimize(mean_returns, cov, "vol", target_return=target)
            rows.append(res)
        except RuntimeError:
            continue  # infeasible target near the extremes
    return pd.DataFrame(rows)
