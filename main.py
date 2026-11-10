"""End-to-end run: load data, optimize, plot and print results."""

import matplotlib.pyplot as plt

from data_loader import TICKERS, compute_returns, fetch_prices
from optimizer import (efficient_frontier, max_sharpe_portfolio, min_volatility_portfolio,
                       random_portfolios, portfolio_stats)
from plots import plot_frontier, plot_weights


def main() -> None:
    prices = fetch_prices(TICKERS, "2019-01-01", "2024-12-31")
    returns = compute_returns(prices)
    mean_returns = returns.mean()
    cov = returns.cov()

    random_pfs = random_portfolios(mean_returns, cov, n=20000)
    frontier = efficient_frontier(mean_returns, cov, n_points=50)
    max_sharpe = max_sharpe_portfolio(mean_returns, cov)
    min_vol = min_volatility_portfolio(mean_returns, cov)

    print("Max Sharpe portfolio:")
    for ticker, w in sorted(zip(prices.columns, max_sharpe["weights"]), key=lambda x: -x[1]):
        if w > 0.005:
            print(f"  {ticker}: {w:.2%}")
    print(f"  return={max_sharpe['return']:.2%}, vol={max_sharpe['volatility']:.2%}, "
          f"sharpe={max_sharpe['sharpe']:.2f}")

    print("\nMin volatility portfolio:")
    for ticker, w in sorted(zip(prices.columns, min_vol["weights"]), key=lambda x: -x[1]):
        if w > 0.005:
            print(f"  {ticker}: {w:.2%}")
    print(f"  return={min_vol['return']:.2%}, vol={min_vol['volatility']:.2%}, "
          f"sharpe={min_vol['sharpe']:.2f}")

    plot_frontier(random_pfs, frontier, max_sharpe, min_vol).savefig("efficient_frontier.png", dpi=150)
    plot_weights(pd.Series(max_sharpe["weights"], index=prices.columns),
                 "Max Sharpe allocation").savefig("max_sharpe_weights.png", dpi=150)
    plt.close("all")
    print("\nSaved: efficient_frontier.png, max_sharpe_weights.png")


if __name__ == "__main__":
    main()
