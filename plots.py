"""Plotting helpers: random-portfolio cloud, efficient frontier and optimized allocations."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_frontier(random_pfs: pd.DataFrame, frontier: pd.DataFrame,
                  max_sharpe: dict, min_vol: dict) -> plt.Figure:
    """Scatter of random portfolios with the efficient frontier and key portfolios highlighted."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(random_pfs["volatility"], random_pfs["return"], c=random_pfs["sharpe"],
               cmap="viridis", s=10, alpha=0.4, label="random portfolios")
    ax.plot(frontier["volatility"], frontier["return"], "r-", lw=2, label="efficient frontier")
    ax.scatter(max_sharpe["volatility"], max_sharpe["return"], marker="*", s=400, c="gold",
               edgecolors="black", zorder=5, label="max Sharpe")
    ax.scatter(min_vol["volatility"], min_vol["return"], marker="*", s=400, c="orange",
               edgecolors="black", zorder=5, label="min volatility")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.set_title("Markowitz efficient frontier")
    ax.legend()
    return fig


def plot_weights(weights: pd.Series, title: str) -> plt.Figure:
    """Horizontal bar chart of an optimized allocation."""
    fig, ax = plt.subplots(figsize=(8, 4))
    weights.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("weight")
    return fig
