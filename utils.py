"""
Shared helpers for the Credit Risk PD Scorecard project.

Place this file in the PROJECT ROOT (next to README.md) and run notebooks
from the project root, so every notebook can simply `import utils` /
`from utils import ...` instead of redefining the same constants and
plotting boilerplate.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- constants
DATA_PATH = "data/home_credit_data/"
FIG_PATH = "outputs/figures/"

# One consistent palette for every chart in the project
C_RED, C_BLUE, C_GREEN, C_GREY = '#e34948', '#378ADD', '#1D9E75', '#888780'

RANDOM_STATE = 42  # single source of truth for every split / model / sampler


def setup(max_columns: int = 120, max_rows: int = 150) -> None:
    """Standard pandas display options + ensure the figures folder exists."""
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.float_format', '{:.2f}'.format)
    os.makedirs(FIG_PATH, exist_ok=True)


# ---------------------------------------------------------------- plotting
def save_show(fig_name: str) -> None:
    """tight_layout -> save under outputs/figures -> show. One call per chart."""
    plt.tight_layout()
    plt.savefig(FIG_PATH + fig_name, dpi=150, bbox_inches='tight')
    plt.show()


def bar_labeled(series: pd.Series, title: str, ylabel: str = 'Default rate (%)',
                color: str = C_RED, fig_name: str | None = None,
                figsize: tuple = (8, 4), fmt: str = '{:.1f}') -> None:
    """Vertical bar chart with value labels on top of each bar."""
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(series)), series.values, color=color)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([str(i) for i in series.index], rotation=15, fontsize=9)
    for b, v in zip(bars, series.values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if fig_name:
        save_show(fig_name)
    else:
        plt.tight_layout()
        plt.show()


def barh_signed(series: pd.Series, title: str, xlabel: str,
                fig_name: str | None = None, figsize: tuple = (8, 5)) -> None:
    """Horizontal bar chart, blue for negative values and red for positive —
    used for correlation-with-TARGET style charts."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = [C_BLUE if v < 0 else C_RED for v in series.values]
    ax.barh(series.index, series.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if fig_name:
        save_show(fig_name)
    else:
        plt.tight_layout()
        plt.show()


def barh_topn(series: pd.Series, title: str, xlabel: str, color: str = C_GREEN,
              vline: float | None = None, vline_label: str | None = None,
              fig_name: str | None = None, figsize: tuple = (8, 5)) -> None:
    """Horizontal bar chart of a (already sorted ascending) series, with an
    optional threshold line — used for IV / MI / VIF rankings."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(series.index, series.values, color=color)
    if vline is not None:
        ax.axvline(x=vline, color=C_RED, linestyle='--', linewidth=1,
                   label=vline_label or f'threshold = {vline}')
        ax.legend(fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if fig_name:
        save_show(fig_name)
    else:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------- analysis
def rate_by(df: pd.DataFrame, col: str, target: str = 'TARGET') -> pd.Series:
    """Default rate (%) by the values of one column."""
    return df.groupby(col, observed=True)[target].mean() * 100


def is_text_column(s: pd.Series) -> bool:
    """Robust categorical check — catches both 'object' and the newer 'str'
    dtype (the exact distinction behind the notebook-04 IV bug)."""
    return s.dtype in ['object', 'str'] or pd.api.types.is_string_dtype(s)
