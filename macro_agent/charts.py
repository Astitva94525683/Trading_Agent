"""Visual matrix charts for the macro news study (docs/design.md section 10).

Each public function draws one chart, saves it as a 150 dpi PNG on the Surface
colour and returns the path. Empty input still writes a figure that says
"No data", so a report never points at a missing file.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import FuncFormatter

from macro_agent.config import ASSETS, CATEGORIES, ET, RULES

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
DIVERGING = LinearSegmentedColormap.from_list(
    "macro_diverging", ["#e34948", "#f0efec", "#2a78d6"]
).with_extremes(bad=SURFACE)

DPI = 150
NO_DATA = "No data"
LEAD_THEMES = ("fed", "inflation", "labor", "growth")
MAX_LABELS = 12
MAX_ANNOTATED_CELLS = 120
_LINE = {"linewidth": 2, "solid_capstyle": "round", "solid_joinstyle": "round"}
_DOT = {
    "marker": "o",
    "markersize": 8,
    "markerfacecolor": INK_2,
    "markeredgecolor": SURFACE,
    "markeredgewidth": 2,
}


def _style_axes(ax: Axes, gridlines: bool = False) -> None:
    """Apply the shared look: Surface background, GRID spines and INK_2 text."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(color=MUTED, labelcolor=INK_2)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.set_axisbelow(True)
    ax.grid(False)
    if gridlines:
        ax.yaxis.grid(True, color=GRID, linewidth=1, linestyle="-")


def _title(fig: Figure, text: str) -> None:
    """Left-aligned figure title in INK."""
    fig.suptitle(text, x=0.02, y=0.98, ha="left", va="top", color=INK, fontsize=13, fontweight="bold")


def _note(ax: Axes, text: str, size: float = 10) -> None:
    """Centre a short INK_2 note in an axes."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=INK_2, fontsize=size)


def _save(fig: Figure, path: Path | str) -> Path:
    """Write the figure as a PNG, close it and return its path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def _no_data(path: Path | str, title: str) -> Path:
    """Write a placeholder figure with 'No data' centred in INK_2."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    _style_axes(ax)
    ax.set_axis_off()
    _note(ax, NO_DATA, size=12)
    _title(fig, title)
    return _save(fig, path)


def _thin(labels: list[str]) -> tuple[list[int], list[str]]:
    """Pick at most MAX_LABELS evenly spaced tick positions and their labels."""
    step = max(1, math.ceil(len(labels) / MAX_LABELS))
    positions = list(range(0, len(labels), step))
    return positions, [labels[i] for i in positions]


def _symmetric_limit(values: np.ndarray) -> float:
    """Return m = max |finite value|, or 1.0 when no value is finite or m is zero."""
    finite = np.abs(values[np.isfinite(values)])
    limit = float(finite.max()) if finite.size else 0.0
    return limit if limit > 0 else 1.0


def _draw_heatmap(ax: Axes, grid: np.ndarray, limit: float) -> AxesImage:
    """Draw a diverging heatmap on the scale -limit..limit, with NaN cells in Surface."""
    return ax.imshow(
        np.ma.masked_invalid(grid),
        cmap=DIVERGING,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
    )


def _annotate(ax: Axes, grid: np.ndarray, fmt: Callable[[float], str], nan_text: str) -> None:
    """Write each finite cell's value in INK; NaN cells get nan_text, blank when empty."""
    for (row, col), value in np.ndenumerate(grid):
        if np.isfinite(value):
            ax.text(col, row, fmt(float(value)), ha="center", va="center", color=INK, fontsize=9)
        elif nan_text:
            ax.text(col, row, nan_text, ha="center", va="center", color=INK_2, fontsize=9)


def _colorbar(fig: Figure, mappable: AxesImage, axes: list[Axes], label: str) -> None:
    """Attach a labelled colorbar in the house style."""
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label(label, color=INK_2)
    cbar.ax.tick_params(color=MUTED, labelcolor=INK_2)
    cbar.outline.set_edgecolor(GRID)


def sentiment_heatmap(weekly: pd.DataFrame, path: Path | str, value: str = "macro_tone") -> Path:
    """Heatmap of a weekly sentiment value by theme (rows) and week (columns)."""
    title = "News tone by theme and week"
    body = weekly.loc[weekly["theme"] != "all"]
    if body.empty:
        return _no_data(path, title)
    present = set(body["theme"])
    rows = [t for t in LEAD_THEMES if t in present] + sorted(present - set(LEAD_THEMES))
    grid = body.groupby(["theme", "period_end"])[value].mean().unstack("period_end").reindex(index=rows)
    arr = grid.to_numpy(dtype=float)
    dates = pd.to_datetime(pd.Index(list(grid.columns)))
    positions, labels = _thin(list(dates.strftime("%b %d")))
    limit = _symmetric_limit(arr)

    width = min(16.0, max(6.0, 0.32 * arr.shape[1] + 3.0))
    height = max(3.0, 0.55 * arr.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    _style_axes(ax)
    image = _draw_heatmap(ax, arr, limit)
    ax.set_xticks(positions, labels=labels)
    ax.set_yticks(range(len(rows)), labels=rows)
    ax.tick_params(length=0)
    _title(fig, title)
    _colorbar(fig, image, [ax], "mean macro tone" if value == "macro_tone" else value)
    return _save(fig, path)


def surprise_heatmap(events: pd.DataFrame, path: Path | str) -> Path:
    """Mean surprise z by category (rows) and New York release month (columns)."""
    title = "Release surprises by category and month (z, + = stronger or hotter)"
    if events.empty:
        return _no_data(path, title)
    months = pd.to_datetime(events["release_utc"], utc=True).dt.tz_convert(ET).dt.strftime("%Y-%m")
    frame = pd.DataFrame(
        {
            "category": events["category"].to_numpy(),
            "month": months.to_numpy(),
            "z": pd.to_numeric(events["surprise_z"], errors="coerce").to_numpy(),
        }
    )
    cells = frame.groupby(["category", "month"])["z"].mean().unstack("month")
    rows = [c for c in CATEGORIES if c in cells.index]
    if not rows:
        return _no_data(path, title)
    grid = cells.reindex(index=rows).sort_index(axis=1)
    arr = grid.to_numpy(dtype=float)
    positions, labels = _thin([str(m) for m in grid.columns])
    limit = _symmetric_limit(arr)

    width = min(16.0, max(6.0, 0.6 * arr.shape[1] + 3.0))
    height = max(3.0, 0.6 * arr.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    _style_axes(ax)
    image = _draw_heatmap(ax, arr, limit)
    if arr.size <= MAX_ANNOTATED_CELLS:
        _annotate(ax, arr, lambda v: f"{v:.1f}", nan_text="")
    ax.set_xticks(positions, labels=labels)
    ax.set_yticks(range(len(rows)), labels=rows)
    ax.tick_params(length=0)
    _title(fig, title)
    _colorbar(fig, image, [ax], "mean surprise (z)")
    return _save(fig, path)


def sentiment_timeline(weekly: pd.DataFrame, events: pd.DataFrame, path: Path | str) -> Path:
    """Weekly macro tone and sentiment above a strip of high-impact release times."""
    title = "Macro news sentiment and high-impact releases"
    weeks = weekly.loc[weekly["theme"] == "all"].sort_values("period_end")
    if "weight" in events.columns:
        events = events.loc[events["weight"] == 1.0]
    present = set(events["category"]) if "category" in events.columns else set()
    rows = [c for c in CATEGORIES if c in present]
    if weeks.empty and not rows:
        return _no_data(path, title)

    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE
    )
    fig.subplots_adjust(hspace=0.12)
    _style_axes(top, gridlines=True)
    _style_axes(bottom)

    if weeks.empty:
        _note(top, NO_DATA)
    else:
        dates = pd.to_datetime(weeks["period_end"]).to_numpy()
        top.axhline(0.0, color=INK_2, linewidth=1, zorder=1)
        top.plot(dates, weeks["macro_tone"].to_numpy(dtype=float), color=SERIES[0], label="macro tone", **_LINE)
        if weeks["sentiment"].notna().any():
            top.plot(dates, weeks["sentiment"].to_numpy(dtype=float), color=SERIES[1], label="sentiment", **_LINE)
            top.legend(frameon=False, labelcolor=INK_2)

    if rows:
        times = (
            pd.to_datetime(events["release_utc"], utc=True)
            .dt.tz_convert(ET)
            .dt.tz_localize(None)
            .to_numpy()
        )
        for row, category in enumerate(rows):
            mask = (events["category"] == category).to_numpy()
            bottom.plot(times[mask], np.full(int(mask.sum()), row), linestyle="none", **_DOT)
        bottom.set_yticks(range(len(rows)), labels=rows)
        bottom.set_ylim(len(rows) - 0.5, -0.5)
    else:
        _note(bottom, "No releases")

    locator = mdates.AutoDateLocator()
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    _title(fig, title)
    return _save(fig, path)


def strategy_matrix(
    values: pd.DataFrame,
    path: Path | str,
    title: str = "Out-of-sample mean net return per trade (bps)",
) -> Path:
    """Asset-by-rule grids of mean net return per trade in bps, one panel per hold horizon."""
    frame = values.assign(bps=pd.to_numeric(values["value"], errors="coerce") * 1e4)
    rows = [a for a in ASSETS if a in set(frame["asset"])]
    cols = [r for r in RULES if r in set(frame["rule"])]
    horizons = sorted(frame["horizon"].unique())
    if not (rows and cols and horizons):
        return _no_data(path, title)

    cells = frame.groupby(["horizon", "asset", "rule"])["bps"].mean()
    tables = {
        h: cells.xs(h, level="horizon").unstack("rule").reindex(index=rows, columns=cols).to_numpy(dtype=float)
        for h in horizons
    }
    limit = _symmetric_limit(np.stack(list(tables.values())))

    width = 1.6 + len(horizons) * max(2.4, 1.15 * len(cols))
    height = 2.2 + 0.65 * len(rows)
    fig, axes = plt.subplots(
        1, len(horizons), sharey=True, figsize=(width, height), facecolor=SURFACE, squeeze=False
    )
    fig.subplots_adjust(top=0.82)
    panels = list(axes.ravel())
    image = None
    for ax, horizon in zip(panels, horizons):
        table = tables[horizon]
        _style_axes(ax)
        image = _draw_heatmap(ax, table, limit)
        _annotate(ax, table, lambda v: f"{round(v)}", nan_text="n/a")
        ax.set_xticks(range(len(cols)), labels=[c.replace("_on_", " on\n") for c in cols])
        ax.set_yticks(range(len(rows)), labels=rows)
        ax.tick_params(length=0)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_title(f"{int(horizon)}-session hold", loc="left", color=INK, fontsize=11, pad=10)
    _title(fig, title)
    _colorbar(fig, image, panels, "net return (bps)")
    return _save(fig, path)


def equity_curves(curves: Mapping[str, pd.Series], path: Path | str) -> Path:
    """Equity and drawdown (%) for up to four daily-return series, out of sample."""
    if len(curves) > len(SERIES):
        raise ValueError(f"equity_curves takes at most {len(SERIES)} series, got {len(curves)}")
    title = "Equity and drawdown, out of sample"
    returns: dict[str, pd.Series] = {}
    for name, series in curves.items():
        clean = pd.Series(series, dtype=float).dropna().sort_index()
        if not clean.empty:
            clean.index = pd.to_datetime(clean.index)
            returns[name] = clean
    if not returns:
        return _no_data(path, title)

    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6.5), gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE
    )
    fig.subplots_adjust(hspace=0.12)
    _style_axes(top, gridlines=True)
    _style_axes(bottom, gridlines=True)
    for colour, (name, daily) in zip(SERIES, returns.items()):
        equity = (1.0 + daily).cumprod()
        peak = equity.cummax().clip(lower=1.0)  # the starting 1.0 counts as a peak
        drawdown = (equity / peak - 1.0) * 100.0
        top.plot(equity.index, equity.to_numpy(), color=colour, label=name, **_LINE)
        bottom.plot(drawdown.index, drawdown.to_numpy(), color=colour, **_LINE)
    if len(returns) >= 2:
        top.legend(frameon=False, labelcolor=INK_2)
    top.set_ylabel("equity (start = 1.0)", color=INK_2)
    bottom.set_ylabel("drawdown (%)", color=INK_2)
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.0f}%"))
    locator = mdates.AutoDateLocator()
    bottom.xaxis.set_major_locator(locator)
    bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    _title(fig, title)
    return _save(fig, path)
