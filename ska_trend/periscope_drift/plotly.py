"""
Functions to create plotly plots.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import scipy
import webcolors
from astropy.table import Table
from cxotime import CxoTime
from plotly.subplots import make_subplots

from ska_trend.periscope_drift import observation

# x-rays within dr around the source are included in the fit
DR = 4


def add_ska_template():
    layout = pio.templates["plotly_white"].layout.to_plotly_json()

    pio.templates["ska"] = go.layout.Template(layout=layout)

    grid_color = "#A4A8AD"
    grid_color = "#BCC0C6"
    pio.templates["ska"].layout["xaxis"].update(
        {
            "gridcolor": grid_color,
            "linecolor": grid_color,
            "zerolinecolor": grid_color,
            "mirror": True,
            "showline": True,
        }
    )
    pio.templates["ska"].layout["yaxis"].update(
        {
            "gridcolor": grid_color,
            "linecolor": grid_color,
            "zerolinecolor": grid_color,
            "mirror": True,
            "showline": True,
        }
    )


add_ska_template()

pio.templates.default = "ska"


def round_to_uncertainty(x, err):
    """
    Round a value to the largest significant digit of the uncertainty.
    """
    x = np.asarray(x)
    scale = 10 ** np.floor(np.log10(err) - 1)
    return np.round(x / scale) * scale


def plot_sources(periscope_drift_data, ids=None):
    """
    Plot the (yag, zag) event distribution for a list of sources in the given periscope drift data.

    If no sources are given, all sources in the data are plotted.

    Parameters
    ----------
    periscope_drift_data : dict
        The periscope drift data dictionary.
    ids : list, optional
        The list of source IDs to plot. If None, all sources are plotted.

    Returns
    -------
    figure
    """
    ids = periscope_drift_data.sources["id"] if ids is None else ids
    src = periscope_drift_data.sources
    src = src[np.in1d(src["id"], ids)]

    n_x = 6
    n_y = len(src) // n_x + int(len(src) % n_x > 0)
    if n_y == 1:
        n_x = len(src)

    fig, axes = plt.subplots(n_y, n_x, figsize=(4 * n_x, 4 * n_y))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    margin = 5
    bins = np.linspace(-margin - 0.5, margin + 0.5, 10 * int(2 * margin + 1) + 1)
    axes[0].set_ylabel("yag")
    for idx in range(n_x * n_y):
        plt.sca(axes[idx])
        if idx >= len(src):
            plt.axis("off")
            continue
        row = src[idx]
        vals, bins_x, bins_y, _ = plt.hist2d(
            periscope_drift_data.data[row["id"]].events["yag"],
            periscope_drift_data.data[row["id"]].events["zag"],
            bins=(row["yag"] + bins, row["zag"] + bins),
            cmap="Reds",
        )
        plt.xlim((row["yag"] - margin, row["yag"] + margin))
        plt.ylim((row["zag"] - margin, row["zag"] + margin))
        plt.xlabel("yag")
        plt.scatter([row["yag"]], [row["zag"]], marker=".")
        plt.title(
            f"i={idx}, N={row['net_counts']:.0f} SNR={row['snr']:.1f}",
            verticalalignment="top",
        )


def plot_pc1(periscope_drift_data):
    viridis = mpl.colormaps["viridis"]

    bd = periscope_drift_data.binned_data_1d
    fits = periscope_drift_data.fits_1d
    x = np.linspace(np.min(bd["OOBAGRD_pc1_mean"]), np.max(bd["OOBAGRD_pc1_mean"]), 100)

    fig, axes = plt.subplot_mosaic(
        [
            [
                "yag_pc1",
                "zag_pc1",
                "grad6_grad3",
            ]
        ],
        figsize=(12, 4),
    )

    plt.sca(axes["grad6_grad3"])

    tmin = np.min(bd["rel_time_mean"])
    tmax = np.max(bd["rel_time_mean"])
    colors = [viridis((t - tmin) / (tmax - tmin)) for t in bd["rel_time_mean"]]
    plt.scatter(
        bd["OOBAGRD3_mean"][np.newaxis],
        bd["OOBAGRD6_mean"][np.newaxis],
        marker="o",
        c=colors,
    )

    x0, y0 = np.mean(bd["OOBAGRD3_mean"]), np.mean(bd["OOBAGRD6_mean"])
    scale = (
        np.sqrt(
            np.max((bd["OOBAGRD3_mean"] - x0) ** 2 + (bd["OOBAGRD6_mean"] - y0) ** 2)
        )
        / 2
    )
    u = scale * np.array(
        [
            np.cos(periscope_drift_data.summary["OOBAGRD_corr_angle"]),
            np.sin(periscope_drift_data.summary["OOBAGRD_corr_angle"]),
        ]
    )
    plt.quiver(
        x0,
        y0,
        u[0],
        u[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="tab:orange",
        zorder=10,
    )
    plt.ylabel("OOBAGRD6")
    plt.xlabel("OOBAGRD3")
    plt.title("Trajectory in gradient space")

    plt.sca(axes["yag_pc1"])
    plt.errorbar(
        bd["OOBAGRD_pc1_mean"], bd["yag"], yerr=bd["d_yag"], fmt="o", label="yag"
    )
    fit = fits[
        (fits["bin_col"] == "rel_time")
        & (fits["x_col"] == "OOBAGRD_pc1")
        & (fits["target_col"] == "yag")
    ][0]
    fit["parameters"]
    plt.plot(x, observation.line(x, *fit["parameters"]), color="tab:orange")
    # plt.legend(loc="best")
    plt.xlabel("OOBAGRD PC1")
    plt.title("YAG")

    plt.sca(axes["zag_pc1"])
    plt.errorbar(
        bd["OOBAGRD_pc1_mean"], bd["zag"], yerr=bd["d_zag"], fmt="o", label="zag"
    )
    fit = fits[
        (fits["bin_col"] == "rel_time")
        & (fits["x_col"] == "OOBAGRD_pc1")
        & (fits["target_col"] == "zag")
    ][0]
    fit["parameters"]
    plt.plot(x, observation.line(x, *fit["parameters"]), color="tab:orange")

    axes["zag_pc1"].sharex(axes["yag_pc1"])
    axes["zag_pc1"].sharey(axes["yag_pc1"])

    ymax = np.max([[bd["yag"] + bd["d_yag"]], [bd["zag"] + bd["d_zag"]]])
    ymin = np.min([[bd["yag"] - bd["d_yag"]], [bd["zag"] - bd["d_zag"]]])
    y_margin = 0.1 * (ymax - ymin)
    plt.ylim((ymin - y_margin, ymax + y_margin))

    plt.title("ZAG")
    plt.xlabel("OOBAGRD PC1")
    plt.ylabel("Residuals (arcsec)")

    plt.suptitle("Fit over First Principal Component in Gradient Space")

    plt.tight_layout()


def plot_gaussian_extreme(dt, col, y_col, bin_col=None):
    bin_col = "rel_time" if bin_col is None else bin_col
    binned_data = dt.binned_data_1d
    sel = (binned_data["bin_col"] == bin_col) & np.isfinite((binned_data[y_col]))
    binned_data = binned_data[sel]

    if len(binned_data[y_col]) == 0:
        # no data to plot
        return

    matches = dt.events

    i1, i2 = np.argmin(binned_data[y_col]), np.argmax(binned_data[y_col])

    selections = {
        i1: (matches[bin_col] >= binned_data[f"{bin_col}_min"][i1])
        & (matches[bin_col] < binned_data[f"{bin_col}_max"][i1]),
        i2: (matches[bin_col] >= binned_data[f"{bin_col}_min"][i2])
        & (matches[bin_col] < binned_data[f"{bin_col}_max"][i2]),
    }
    # choose the gradient binning
    # using the Freedman–Diaconis rule for the bin size

    sel = selections[i1] | selections[i2]

    q1, q3 = np.percentile(matches[f"residual_{y_col}"][sel], [25, 75])
    dx = 2 * (q3 - q1) * (np.sum(sel)) ** (-1 / 3)
    n = int(2 * DR // dx)
    y_bins = np.linspace(-DR, DR, n)
    # y_bins = np.linspace(-dr, dr, 24)

    x = np.linspace(y_bins[0], y_bins[-1], 101)
    colors = ["tab:blue", "tab:orange"]
    for i, idx in enumerate([i1, i2]):
        col_mean = f"{col}_mean"
        label = f"{col}={binned_data[col_mean][idx]:5.2f}, {y_col}={binned_data[y_col][idx]:.2f}"
        label = label.replace("OOBA", "")

        sel = selections[idx]
        plt.hist(
            matches[f"residual_{y_col}"][sel],
            histtype="step",
            bins=y_bins,
            density=True,
            color=colors[i],
            label=label,
        )
        # gaussian_fit = fits[(col, y_col)]["gaussian_fits"][idx]
        # result = gaussian_fit["result"]
        y = observation.prob(x, *binned_data[f"params_{y_col}"][idx])
        plt.plot(x, y, color=colors[i])
    plt.xlim(
        (
            -3 * binned_data[f"params_{y_col}"][idx][1],
            3 * binned_data[f"params_{y_col}"][idx][1],
        )
    )


def plot_gaussian_extremes(dt):
    fig, axes = plt.subplot_mosaic(
        [
            ["residual_yag_v_OOBAGRD3", "residual_yag_v_OOBAGRD6"],
            ["residual_zag_v_OOBAGRD3", "residual_zag_v_OOBAGRD6"],
        ],
        sharex=False,
        sharey=True,
        figsize=(12, 8),
    )
    plt.sca(axes["residual_yag_v_OOBAGRD3"])
    plot_gaussian_extreme(dt, "OOBAGRD3", "yag")
    plt.title("yag/OOBAGRD3")
    if np.any(
        (dt.binned_data_1d["bin_col"] == "OOBAGRD3")
        & (np.isfinite(dt.binned_data_1d["yag"]))
    ):
        plt.legend(loc="upper left")
    plt.sca(axes["residual_zag_v_OOBAGRD3"])
    plot_gaussian_extreme(dt, "OOBAGRD3", "zag")
    plt.xlabel("OOBAGRD3")
    plt.title("zag/OOBAGRD3")
    if np.any(
        (dt.binned_data_1d["bin_col"] == "OOBAGRD3")
        & (np.isfinite(dt.binned_data_1d["zag"]))
    ):
        plt.legend(loc="upper left")
    plt.sca(axes["residual_yag_v_OOBAGRD6"])
    plot_gaussian_extreme(dt, "OOBAGRD6", "yag")
    plt.ylabel("y residual (arcsec)")
    plt.title("yag/OOBAGRD6")
    if np.any(
        (dt.binned_data_1d["bin_col"] == "OOBAGRD6")
        & (np.isfinite(dt.binned_data_1d["yag"]))
    ):
        plt.legend(loc="upper right")
    plt.sca(axes["residual_zag_v_OOBAGRD6"])
    plot_gaussian_extreme(dt, "OOBAGRD6", "zag")
    plt.xlabel("OOBAGRD6")
    plt.ylabel("z residual (arcsec)")
    plt.title("zag/OOBAGRD6")
    if np.any(
        (dt.binned_data_1d["bin_col"] == "OOBAGRD6")
        & (np.isfinite(dt.binned_data_1d["zag"]))
    ):
        plt.legend(loc="upper right")

    plt.suptitle("Bins with largest difference")
    plt.tight_layout()


def plot_linear_fit(
    periscope_drift_data, col, y_col, bin_col=None, color=None, set_lims=False
):
    bin_col = "rel_time" if bin_col is None else bin_col
    sel = (periscope_drift_data.binned_data_1d["bin_col"] == bin_col) & (
        np.isfinite(periscope_drift_data.binned_data_1d[y_col])
    )
    binned_data = periscope_drift_data.binned_data_1d[sel]

    sel = (periscope_drift_data.fits_1d["x_col"] == col) & (
        periscope_drift_data.fits_1d["target_col"] == y_col
    )
    if np.any(sel):
        line_fit = dict(periscope_drift_data.fits_1d[sel][0])
    else:
        line_fit = None

    y_vals = binned_data[y_col]

    x_vals = binned_data[f"{col}_mean"]
    sigma_vals = binned_data[f"d_{y_col}"]

    plt.errorbar(x_vals, y_vals, yerr=sigma_vals, fmt="o", color=color)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    label_x = xmin + 0.95 * (xmax - xmin)
    label_y = ymin + 0.05 * (ymax - ymin)

    ndf_0 = len(binned_data[y_col])
    ndf = len(binned_data[y_col]) - 2

    chi2_0 = np.sum(y_vals**2 / sigma_vals**2) / ndf_0
    p_value_0 = scipy.stats.chi2.sf(chi2_0 * ndf_0, ndf_0)

    # intercept = round_to_uncertainty(params[1], np.sqrt(cov[1,1]))

    if line_fit is not None and len(binned_data) > 0:
        params = line_fit["parameters"]
        cov = line_fit["covariance"]
        x = np.linspace(
            binned_data[f"{col}_mean"].min(), binned_data[f"{col}_mean"].max(), 100
        )
        plt.plot(x, observation.line(x, *params), color=color)

        slope = round_to_uncertainty(params[0], np.sqrt(cov[0, 0]))
        slope_err = round_to_uncertainty(np.sqrt(cov[0, 0]), np.sqrt(cov[0, 0]))
        chi2 = (
            np.sum((y_vals - observation.line(x_vals, *params)) ** 2 / sigma_vals**2)
            / ndf
        )
        p_value = scipy.stats.chi2.sf(chi2 * ndf, ndf)
    else:
        slope = np.nan
        slope_err = np.nan
        chi2 = np.nan
        p_value = np.nan

    ymin = np.min([[y_vals - sigma_vals]])
    ymax = np.max([[y_vals + sigma_vals]])

    if set_lims:
        dy = np.max([np.abs(ymin), np.abs(ymax), 0.3])
        plt.ylim((-dy, dy))

    plt.text(
        label_x,
        label_y,
        f"slope: {slope:.2} $\\pm$ {slope_err:.1}\n"
        f"null chi2 = {chi2_0:.2f} (p={p_value_0:.2})\n"
        f"fit chi2 = {chi2:.2f} (p={p_value:.2})",
        horizontalalignment="right",
    )


def plot_telem(obs):
    telem = observation.fetch_telemetry(
        obs.get_obspar()["tstart"], obs.get_obspar()["tstop"]
    )
    events = obs.periscope_drift.get_events()

    start = np.min(events["time"])

    fig, axes = plt.subplot_mosaic(
        [
            [
                "OOBAGRD3_v_time",
            ],
            [
                "OOBAGRD6_v_time",
            ],
        ],
        sharex=True,
        sharey=False,
        figsize=(12, 4),
    )

    time = telem["time"] - start
    plt.sca(axes["OOBAGRD3_v_time"])
    plt.plot(time, telem["OOBAGRD3"], ".")
    plt.ylabel("OOBAGRD3")

    plt.sca(axes["OOBAGRD6_v_time"])
    plt.plot(time, telem["OOBAGRD6"], ".")
    plt.ylabel("OOBAGRD6")
    plt.xlabel("Time from start (sec)")

    plt.suptitle("Telemetry")
    plt.tight_layout()


def get_plot_3d_figures(periscope_drift_data):
    """
    Make a 3d plots of yag and zag vs OOBAGRD3/OOBAGRD6 for one x-ray source.

    Parameters
    ----------
    periscope_drift_data : dict
        The periscope drift data dictionary.

    Returns
    -------
    figure : plotly.graph_objects.Figure
    """

    binned_data = periscope_drift_data.binned_data_1d
    sel = (
        (binned_data["bin_col"] == "rel_time")
        & np.isfinite(binned_data["yag"])
        & np.isfinite(binned_data["zag"])
    )
    binned_data = binned_data[sel]

    x0 = binned_data["OOBAGRD3_mean"] - np.mean(binned_data["OOBAGRD3_mean"])
    x1 = binned_data["OOBAGRD6_mean"] - np.mean(binned_data["OOBAGRD6_mean"])
    t = binned_data["rel_time_mean"]

    scatter = {
        y_col: go.Scatter3d(
            x=x0,
            y=x1,
            z=binned_data[y_col],
            mode="markers",
            marker={
                "size": 4,
                "color": t,
                "colorscale": "Viridis",
                "opacity": 0.8,
            },
        )
        for y_col in ["yag", "zag"]
    }

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["YAG residual", "ZAG residual"],
    )

    figure.add_traces([scatter["yag"]], rows=1, cols=1)
    figure.add_traces([scatter["zag"]], rows=1, cols=2)

    # figure.update_xaxes(title_text="OOBAGRD3", row=1, col=1)
    # figure.update_yaxes(title_text="OOBAGRD6", row=1, col=1)
    # figure.update_xaxes(title_text="OOBAGRD3", row=1, col=2)
    # figure.update_yaxes(title_text="OOBAGRD6", row=1, col=2)
    # figure.update_zaxes(title_text="YAG", row=1, col=1)
    # figure.update_zaxes(title_text="ZAG", row=1, col=2)

    figure.update_layout(
        scene1={
            "xaxis": {
                "title": "OOBAGRD3",
            },
            "yaxis": {
                "title": "OOBAGRD6",
            },
            "zaxis": {"title": "YAG"},
            "camera": {"eye": {"x": 1.7, "y": 1.7, "z": 1.7}},
        },
        scene2={
            "xaxis": {
                "title": "OOBAGRD3",
            },
            "yaxis": {
                "title": "OOBAGRD6",
            },
            "zaxis": {"title": "ZAG"},
            "camera": {"eye": {"x": 1.7, "y": 1.7, "z": 1.7}},
        },
        showlegend=False,
        # margin={"l": 80, "r": 80, "b": 100, "t": 80},  # default layout
        margin={"l": 0, "r": 0, "b": 0, "t": 0},  # tight layout
    )

    for idx in range(2):
        figure.layout.annotations[idx].update(y=0.85, yanchor="top")

    return figure


def get_plot_3d_figure(periscope_drift_data, y_col):
    """
    Make a 3d plot of yag/zag vs OOBAGRD3/OOBAGRD6 for one x-ray source.

    Parameters
    ----------
    periscope_drift_data : dict
        The periscope drift data dictionary.
    y_col : str
        The y column to plot. This should be one "yag" or "zag".

    Returns
    -------
    figure : plotly.graph_objects.Figure
    """
    binned_data = periscope_drift_data.binned_data_1d
    sel = (binned_data["bin_col"] == "rel_time") & np.isfinite(binned_data[y_col])
    binned_data = binned_data[sel]

    y = binned_data[y_col]
    x0 = binned_data["OOBAGRD3_mean"] - np.mean(binned_data["OOBAGRD3_mean"])
    x1 = binned_data["OOBAGRD6_mean"] - np.mean(binned_data["OOBAGRD6_mean"])
    t = binned_data["rel_time_mean"]

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=x0,
                y=x1,
                z=y,
                mode="markers",
                marker={
                    "size": 4,
                    "color": t,
                    "colorscale": "Viridis",
                    "opacity": 0.8,
                },
            )
        ]
    )

    figure.update_layout(
        scene={
            "xaxis": {
                "title": "OOBAGRD3",
            },
            "yaxis": {
                "title": "OOBAGRD6",
            },
            "zaxis": {"title": y_col},
        },
        title=f"{y_col}",
    )
    # tight layout
    figure.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})
    return figure


def plot_linear_fits(fits, corr=None):
    fig, axes = plt.subplot_mosaic(
        [["residual_yag_v_time"], ["residual_zag_v_time"]],
        sharex=True,
        sharey=True,
        figsize=(12, 8),
    )

    plt.sca(axes["residual_yag_v_time"])
    plot_linear_fit(fits, "rel_time", "yag", set_lims=True)
    if corr is not None:
        plt.plot(
            corr["times"] - np.min(corr["times"]),
            corr["ang_y_corr"],
        )
    plt.ylabel("y residual (arcsec)")

    plt.sca(axes["residual_zag_v_time"])
    plot_linear_fit(fits, "rel_time", "zag", set_lims=True)
    if corr is not None:
        plt.plot(
            corr["times"] - np.min(corr["times"]),
            corr["ang_z_corr"],
        )
    plt.ylabel("z residual (arcsec)")
    plt.xlabel("Time from start (sec)")

    plt.suptitle("Residuals Vs Time (fits)")
    plt.tight_layout()

    fig, axes = plt.subplot_mosaic(
        [
            ["residual_yag_v_OOBAGRD3", "residual_yag_v_OOBAGRD6"],
            ["residual_zag_v_OOBAGRD3", "residual_zag_v_OOBAGRD6"],
        ],
        sharex=False,
        sharey=True,
        figsize=(12, 8),
    )

    plt.sca(axes["residual_yag_v_OOBAGRD3"])
    plot_linear_fit(fits, "OOBAGRD3", "yag", set_lims=True)
    plt.sca(axes["residual_zag_v_OOBAGRD3"])
    plot_linear_fit(fits, "OOBAGRD3", "zag", set_lims=True)
    plt.xlabel("OOBAGRD3")
    plt.sca(axes["residual_yag_v_OOBAGRD6"])
    plot_linear_fit(fits, "OOBAGRD6", "yag", set_lims=True)
    plt.ylabel("y residual (arcsec)")
    plt.sca(axes["residual_zag_v_OOBAGRD6"])
    plot_linear_fit(fits, "OOBAGRD6", "zag", set_lims=True)
    plt.ylabel("z residual (arcsec)")
    plt.xlabel("OOBAGRD6")

    plt.suptitle("Residuals Vs Gradients (fits)")
    plt.tight_layout()


def scatter_plots(matches):
    fig, axes_1 = plt.subplot_mosaic(
        [["residual_yag_v_time"], ["residual_zag_v_time"]],
        sharex=True,
        sharey=True,
        figsize=(12, 8),
    )
    plt.sca(axes_1["residual_yag_v_time"])
    plt.plot(matches["rel_time"], matches["residual_yag"], ".")
    plt.ylabel("y residual (arcsec)")

    plt.sca(axes_1["residual_zag_v_time"])
    plt.plot(matches["rel_time"], matches["residual_zag"], ".")

    plt.xlabel("time from start (sec)")
    plt.ylabel("z residual (arcsec)")

    plt.suptitle("Residuals Vs Time")

    plt.tight_layout()

    fig, axes_2 = plt.subplot_mosaic(
        [
            ["residual_yag_v_OOBAGRD3", "residual_yag_v_OOBAGRD6"],
            ["residual_zag_v_OOBAGRD3", "residual_zag_v_OOBAGRD6"],
        ],
        sharex=False,
        sharey=True,
        figsize=(12, 8),
    )

    plt.sca(axes_2["residual_yag_v_OOBAGRD3"])
    plt.plot(matches["OOBAGRD3"], matches["residual_yag"], ".")
    plt.ylabel("y residual (arcsec)")

    plt.sca(axes_2["residual_yag_v_OOBAGRD6"])
    plt.plot(matches["OOBAGRD6"], matches["residual_yag"], ".")
    plt.ylabel("y residual (arcsec)")

    plt.sca(axes_2["residual_zag_v_OOBAGRD3"])
    plt.plot(matches["OOBAGRD3"], matches["residual_zag"], ".")
    plt.xlabel("OOBAGRD3")
    plt.ylabel("z residual (arcsec)")

    plt.sca(axes_2["residual_zag_v_OOBAGRD6"])
    plt.plot(matches["OOBAGRD6"], matches["residual_zag"], ".")
    plt.xlabel("OOBAGRD6")
    plt.ylabel("z residual (arcsec)")

    plt.suptitle("Residuals Vs Gradients")

    plt.tight_layout()

    axes = {}
    axes.update(axes_1)
    axes.update(axes_2)

    return axes


def plot(obs, periscope_drift_data, src_id):
    from ska_trend.periscope_drift.reports import expected_corr  # noqa: PLC0415

    cols = [
        "obsid",
        "id",
        "net_counts",
        "snr",
        "max_yang_corr",
        "max_zang_corr",
        "near_neighbor_dist",
        "psfratio",
        "pileup",
        "pileup_size",
        "max_pileup",
        "y_angle",
        "z_angle",
    ]
    src_id = periscope_drift_data.data[src_id].source["id"]
    src = Table([periscope_drift_data.data[src_id].source])[cols]
    src["max_yang_corr"].format = ".2f"
    src["max_zang_corr"].format = ".2f"
    src["max_pileup"].format = ".2f"
    src["pileup"].format = ".2f"
    src["psfratio"].format = ".2f"
    src["near_neighbor_dist"].format = ".2f"
    src["snr"].format = ".1f"
    src["y_angle"].format = ".2f"
    src["z_angle"].format = ".2f"
    src["net_counts"].format = ".0f"
    src.pprint(max_width=-1)

    plot_sources(periscope_drift_data, [src_id])

    matches = periscope_drift_data.data[src_id].events

    axes = scatter_plots(matches)

    corr = expected_corr(
        {
            "kalman_datestart": np.min(matches["time"]),
            "kalman_datestop": np.max(matches["time"]),
        }
    )

    dt = periscope_drift_data.data[src_id]

    plt.sca(axes["residual_yag_v_time"])
    plot_linear_fit(dt, "rel_time", "yag", color="red")
    plt.plot(
        corr["times"] - np.min(corr["times"]),
        corr["ang_y_corr"],
    )
    plt.sca(axes["residual_zag_v_time"])
    plot_linear_fit(dt, "rel_time", "zag", color="red")
    plt.plot(
        corr["times"] - np.min(corr["times"]),
        corr["ang_z_corr"],
    )

    plt.sca(axes["residual_yag_v_OOBAGRD3"])
    plot_linear_fit(dt, "OOBAGRD3", "yag", color="red")
    plt.sca(axes["residual_yag_v_OOBAGRD6"])
    plot_linear_fit(dt, "OOBAGRD6", "yag", color="red")
    plt.sca(axes["residual_zag_v_OOBAGRD3"])
    plot_linear_fit(dt, "OOBAGRD3", "zag", color="red")
    plt.sca(axes["residual_zag_v_OOBAGRD6"])
    plot_linear_fit(dt, "OOBAGRD6", "zag", color="red")

    plot_telem(obs)

    plot_pc1(dt)

    plot_linear_fits(dt, corr)

    plot_gaussian_extremes(dt)


def get_fit_1d_line_annotation(periscope_drift_data, col, y_col, bin_col="rel_time"):
    binned_data = periscope_drift_data.binned_data_1d[
        (periscope_drift_data.binned_data_1d["bin_col"] == bin_col)
        & (np.isfinite(periscope_drift_data.binned_data_1d[y_col]))
    ]

    fits_1d = periscope_drift_data.fits_1d[
        (periscope_drift_data.fits_1d["x_col"] == col)
        & (periscope_drift_data.fits_1d["target_col"] == y_col)
    ]

    line_fit = dict(fits_1d[0]) if len(fits_1d) > 0 else None

    y_vals = binned_data[y_col]

    x_vals = binned_data[f"{col}_mean"]
    sigma_vals = binned_data[f"d_{y_col}"]

    ndf_0 = len(binned_data[y_col])
    ndf = len(binned_data[y_col]) - 2

    chi2_0 = np.sum(y_vals**2 / sigma_vals**2) / ndf_0
    p_value_0 = scipy.stats.chi2.sf(chi2_0 * ndf_0, ndf_0)

    if line_fit is not None and len(binned_data) > 0:
        params = line_fit["parameters"]
        cov = line_fit["covariance"]

        slope = round_to_uncertainty(params[0], np.sqrt(cov[0, 0]))
        slope_err = round_to_uncertainty(np.sqrt(cov[0, 0]), np.sqrt(cov[0, 0]))
        chi2 = (
            np.sum((y_vals - observation.line(x_vals, *params)) ** 2 / sigma_vals**2)
            / ndf
        )
        p_value = scipy.stats.chi2.sf(chi2 * ndf, ndf)

        annotation = {
            "x": 1,
            "y": 0,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "right",
            "yanchor": "bottom",
            "showarrow": False,
            "text": (
                rf"slope: {slope:.2} \pm {slope_err:.1}<br>"
                f"null chi2 = {chi2_0:.2f} (p={p_value_0:.2})<br>"
                f"fit chi2 = {chi2:.2f} (p={p_value:.2})"
            ),
        }

    else:
        annotation = {}

    return annotation


def get_fit_1d_line(periscope_drift_data, col, y_col, bin_col="rel_time", color=None):
    binned_data = periscope_drift_data.binned_data_1d[
        (periscope_drift_data.binned_data_1d["bin_col"] == bin_col)
        & (np.isfinite(periscope_drift_data.binned_data_1d[y_col]))
    ]

    fits_1d = periscope_drift_data.fits_1d[
        (periscope_drift_data.fits_1d["x_col"] == col)
        & (periscope_drift_data.fits_1d["target_col"] == y_col)
    ]

    line_fit = dict(fits_1d[0]) if len(fits_1d) > 0 else None

    if line_fit is None or len(binned_data) == 0:
        trace = {}
    else:
        params = line_fit["parameters"]
        x = np.linspace(
            binned_data[f"{col}_mean"].min(), binned_data[f"{col}_mean"].max(), 100
        )

        trace = {
            "x": x,
            "y": observation.line(x, *params),
            "mode": "lines",
            "name": "1d-fit",
            "line": {"color": color},
            "hoverinfo": "skip",
        }
    return go.Scatter(trace)


def get_smooth_residual_v_time_line(periscope_drift_data, y_col, color=None):
    function = {
        "yag": periscope_drift_data.yag_vs_time,
        "zag": periscope_drift_data.zag_vs_time,
    }

    x = np.linspace(
        periscope_drift_data.events["rel_time"].min(),
        periscope_drift_data.events["rel_time"].max(),
        100,
    )

    trace = {
        "x": x,
        "y": function[y_col](x),
        "mode": "lines",
        "name": "smooth residual vs time",
        "line": {"color": color},
        "hoverinfo": "skip",
    }
    return go.Scatter(trace)


def get_binned_data_1d_scatter(
    periscope_drift_data, col, y_col, bin_col="rel_time", color=None
):
    binned_data = periscope_drift_data.binned_data_1d[
        (periscope_drift_data.binned_data_1d["bin_col"] == bin_col)
        & (np.isfinite(periscope_drift_data.binned_data_1d[y_col]))
    ]

    y_vals = binned_data[y_col]

    x_vals = binned_data[f"{col}_mean"]
    sigma_vals = binned_data[f"d_{y_col}"]

    # format and color?
    trace = {
        "name": "binned data",
        "x": x_vals,
        "y": y_vals,
        "mode": "markers",
        "error_y": {
            "type": "data",  # value of error bar given in data coordinates
            "array": sigma_vals,
            "visible": True,
        },
        "marker": {"color": color},
    }
    return go.Scatter(trace)


def get_events_scatter(events, col, y_col, color=None):
    if len(events) == 0:
        trace = {}
    else:
        trace = {
            "name": "events",
            "x": events[col],
            "y": events[f"residual_{y_col}"],
            "mode": "markers",
            "marker": {"color": color},
            "hoverinfo": "skip",
        }
    return go.Scatter(trace)


def get_telem_objects(obs, color=None):
    telem = observation.fetch_telemetry(
        obs.get_obspar()["tstart"], obs.get_obspar()["tstop"]
    )
    events = obs.periscope_drift.get_events()

    start = np.min(events["time"])
    time = telem["time"] - start

    traces = [
        go.Scatter(
            {
                "name": "OOBAGRD3_v_time",
                "x": time,
                "y": telem["OOBAGRD3"],
                "mode": "markers",
                "marker": {"color": color},
            }
        ),
        go.Scatter(
            {
                "name": "OOBAGRD6_v_time",
                "x": time,
                "y": telem["OOBAGRD6"],
                "mode": "markers",
                "marker": {"color": color},
            }
        ),
    ]
    return traces


def get_telemetry_figure(obs):
    fig = go.Figure()
    fig.set_subplots(
        rows=2,
        cols=1,
        shared_xaxes="all",
        vertical_spacing=0.03,
    )
    telem = get_telem_objects(obs, color="blue")
    fig.add_traces(telem[0], rows=1, cols=1)
    fig.add_traces(telem[1], rows=2, cols=1)
    fig.update_yaxes(title_text="OOBAGRD3", row=1, col=1)
    fig.update_yaxes(title_text="OOBAGRD6", row=2, col=1)
    fig.update_xaxes(title_text="Time from start (sec)", row=2, col=1)
    fig.update_layout(
        # title_text="Telemetry",
        showlegend=False,
    )
    return fig


def get_scatter_plot_figure(src_pdd):
    fig = go.Figure()

    # this is a figure with two stacked plots
    fig.set_subplots(
        rows=2,
        cols=1,
        shared_yaxes="all",
        shared_xaxes="all",
    )

    col = "rel_time"  # the column that contains the independent variable
    bin_col = "rel_time"  # the column that was used to bin the data

    # get all graphic objects
    evt_scatter_yag = get_events_scatter(src_pdd.events, col, "yag", color="gray")
    evt_scatter_zag = get_events_scatter(src_pdd.events, col, "zag", color="gray")
    scatter_yag = get_binned_data_1d_scatter(src_pdd, col, "yag", bin_col, color="blue")
    scatter_zag = get_binned_data_1d_scatter(src_pdd, col, "zag", bin_col, color="blue")
    line_yag = get_fit_1d_line(src_pdd, col, "yag", bin_col, color="blue")
    line_zag = get_fit_1d_line(src_pdd, col, "zag", bin_col, color="blue")

    smooth_line_yag = get_smooth_residual_v_time_line(src_pdd, "yag", color="black")
    smooth_line_zag = get_smooth_residual_v_time_line(src_pdd, "zag", color="black")

    line_yag.update(
        visible="legendonly",
    )
    line_zag.update(
        visible="legendonly",
    )

    # modify some things for a combined figure
    # the corresponding items from both plots have a "grouped" legend so they toggle together
    # and the legend is shown for only one of them.
    line_zag.update(
        legendgroup="line",
        showlegend=False,
    )
    line_yag.update(
        legendgroup="line",
    )

    scatter_zag.update(
        legendgroup="binned data",
        showlegend=False,
    )
    scatter_yag.update(
        legendgroup="binned data",
    )

    evt_scatter_zag.update(
        legendgroup="events",
        showlegend=False,
        marker={
            "opacity": 0.3,
        },
    )
    evt_scatter_yag.update(
        legendgroup="events",
        marker={
            "opacity": 0.3,
        },
    )

    smooth_line_yag.update(
        legendgroup="smooth residual vs time",
    )
    smooth_line_zag.update(
        legendgroup="smooth residual vs time",
        showlegend=False,
    )

    ymax = np.max(scatter_yag.y + scatter_yag.error_y["array"])
    ymin = np.min(scatter_yag.y - scatter_yag.error_y["array"])
    margin = (ymax - ymin) * 0.05
    dy = np.max([np.abs(ymax + margin), np.abs(ymin - margin), 0.6])
    fig.update_yaxes(range=[-dy / 2, dy / 2], row=2, col=1)

    xmin = np.min(evt_scatter_yag.x)
    xmax = np.max(evt_scatter_yag.x)
    margin = (xmax - xmin) * 0.05
    fig.update_xaxes(range=[xmin - margin, xmax + margin], row=2, col=1)

    fig.update_yaxes(title_text="yag", row=1, col=1)
    fig.update_yaxes(title_text="zag", row=2, col=1)
    fig.update_xaxes(title_text="time since start (sec)", row=2, col=1)

    # annotation_yag = get_fit_1d_line_annotation(src_pdd, col, "yag", bin_col)
    # fig.add_annotation(annotation_yag)
    # annotation_zag = get_fit_1d_line_annotation(src_pdd, col, "zag", bin_col)
    # fig.add_annotation(annotation_zag, row=2, col=1)

    fig.add_traces([evt_scatter_yag, scatter_yag, line_yag], rows=1, cols=1)
    fig.add_traces([evt_scatter_zag, scatter_zag, line_zag], rows=2, cols=1)

    fig.add_traces([smooth_line_yag], rows=1, cols=1)
    fig.add_traces([smooth_line_zag], rows=2, cols=1)

    return fig


def get_scatter_versus_gradients_figure(src_pdd):
    fig = go.Figure()

    # this is a figure with two stacked plots
    fig.set_subplots(
        rows=2,
        cols=2,
        shared_yaxes=True,
        shared_xaxes=True,
    )

    bin_col = "rel_time"  # the column that was used to bin the data

    # get all graphic objects
    evt_scatter = {
        (x_col, y_col): get_events_scatter(src_pdd.events, x_col, y_col, color="gray")
        for x_col in ["OOBAGRD3", "OOBAGRD6"]
        for y_col in ["yag", "zag"]
    }
    scatter = {
        (x_col, y_col): get_binned_data_1d_scatter(
            src_pdd, x_col, y_col, bin_col, color="blue"
        )
        for x_col in ["OOBAGRD3", "OOBAGRD6"]
        for y_col in ["yag", "zag"]
    }
    line = {
        (x_col, y_col): get_fit_1d_line(src_pdd, x_col, y_col, bin_col, color="blue")
        for x_col in ["OOBAGRD3", "OOBAGRD6"]
        for y_col in ["yag", "zag"]
    }

    # modify some things for a combined figure
    # the corresponding items from both plots have a "grouped" legend so they toggle together
    # and the legend is shown for only one of them.
    keys = list(line.keys())
    for x_col, y_col in keys:
        show = (x_col == "OOBAGRD3") and (y_col == "yag")
        line[(x_col, y_col)].update(
            legendgroup="line",
            showlegend=show,
        )
        scatter[(x_col, y_col)].update(
            legendgroup="binned data",
            showlegend=show,
        )
        evt_scatter[(x_col, y_col)].update(
            legendgroup="events",
            showlegend=show,
            marker={
                "opacity": 0.3,
            },
            visible="legendonly",
        )

    for row, y_col in enumerate(["yag", "zag"], start=1):
        key = ("OOBAGRD3", y_col)
        ymax = np.max(scatter[key].y + scatter[key].error_y["array"])
        ymin = np.min(scatter[key].y - scatter[key].error_y["array"])
        margin = (ymax - ymin) * 0.05
        dy = np.max([np.abs(ymax + margin), np.abs(ymin - margin), 0.6])
        fig.update_yaxes(range=[-dy / 2, dy / 2], row=row, col=1)

    fig.update_yaxes(title_text="yag", row=1, col=1)
    fig.update_yaxes(title_text="zag", row=2, col=1)
    fig.update_xaxes(title_text="OOBAGRD3", row=2, col=1)
    fig.update_xaxes(title_text="OOBAGRD6", row=2, col=2)

    # annotation_yag = get_fit_1d_line_annotation(src_pdd, col, "yag", bin_col)
    # fig.add_annotation(annotation_yag)
    # annotation_zag = get_fit_1d_line_annotation(src_pdd, col, "zag", bin_col)
    # fig.add_annotation(annotation_zag, row=2, col=1)

    for x_col in ["OOBAGRD3", "OOBAGRD6"]:
        for y_col in ["yag", "zag"]:
            row = 1 if y_col == "yag" else 2
            col = 1 if x_col == "OOBAGRD3" else 2
            fig.add_traces(
                [
                    evt_scatter[(x_col, y_col)],
                    scatter[(x_col, y_col)],
                    line[(x_col, y_col)],
                ],
                rows=row,
                cols=col,
            )

    return fig


def get_extreme_bin_histograms(dt, y_col, bin_col="rel_time"):
    binned_data = dt.binned_data_1d[
        (dt.binned_data_1d["bin_col"] == bin_col)
        & np.isfinite((dt.binned_data_1d[y_col]))
    ]

    if len(binned_data[y_col]) == 0:
        # nothing to plot
        return []

    matches = dt.events

    i1, i2 = np.argmin(binned_data[y_col]), np.argmax(binned_data[y_col])

    selections = {
        i1: (matches[bin_col] >= binned_data[f"{bin_col}_min"][i1])
        & (matches[bin_col] < binned_data[f"{bin_col}_max"][i1]),
        i2: (matches[bin_col] >= binned_data[f"{bin_col}_min"][i2])
        & (matches[bin_col] < binned_data[f"{bin_col}_max"][i2]),
    }
    sel = selections[i1] | selections[i2]

    # choose the gradient binning
    # using the Freedman–Diaconis rule for the bin size
    q1, q3 = np.percentile(matches[f"residual_{y_col}"][sel], [25, 75])
    dx = 2 * (q3 - q1) * (np.sum(sel)) ** (-1 / 3)
    n = int(2 * DR // dx)
    y_bins = np.linspace(-DR, DR, n)

    x = np.linspace(y_bins[0], y_bins[-1], 101)
    colors = [webcolors.name_to_rgb("blue"), webcolors.name_to_rgb("red")]
    traces = []
    for i, idx in enumerate([i1, i2]):
        label = "hist low" if i == 0 else "hist high"

        sel = selections[idx]
        y = observation.prob(x, *binned_data[f"params_{y_col}"][idx])
        r, g, b = colors[i]

        traces.extend(
            [
                go.Histogram(
                    {
                        "name": f"{label}",
                        "histnorm": "probability density",
                        "x": matches[f"residual_{y_col}"][sel],
                        "xbins": {
                            "start": -DR,
                            "end": DR,
                            "size": 2 * DR / (n + 1),
                        },
                        "marker": {
                            "color": f"rgba({r}, {g}, {b}, 0.1)",
                            "line": {
                                "color": f"rgba({r}, {g}, {b}, 1)",
                                "width": 1,
                            },
                        },
                    }
                ),
                go.Scatter(
                    {
                        "x": x,
                        "y": y,
                        "mode": "lines",
                        "name": "1d-fit",
                        "line": {
                            "color": f"rgba({r}, {g}, {b}, 1)",
                        },
                    }
                ),
            ]
        )
    return traces


def get_extreme_bin_histograms_figure(src_pdd):
    fig = go.Figure()
    fig.set_subplots(rows=1, cols=2, subplot_titles=["yag", "zag"])

    hist = get_extreme_bin_histograms(src_pdd, "yag", "rel_time")
    fig.add_traces(hist, rows=1, cols=1)

    hist = get_extreme_bin_histograms(src_pdd, "zag", "rel_time")
    fig.add_traces(hist, rows=1, cols=2)

    fig.update_layout(
        {
            "barmode": "overlay",
            "showlegend": False,
        }
    )

    fig.update_xaxes(range=[-1.5, 1.5])

    return fig


def get_gradient_space_path(periscope_drift_data):
    bd = periscope_drift_data.binned_data_1d

    path = {
        "name": "GRD6 Vs GRD3",
        "x": bd["OOBAGRD3_mean"],
        "y": bd["OOBAGRD6_mean"],
        "mode": "markers",
        "marker": {
            "color": bd["rel_time_mean"],
        },
    }

    x0, y0 = np.mean(bd["OOBAGRD3_mean"]), np.mean(bd["OOBAGRD6_mean"])
    scale = (
        np.sqrt(
            np.max((bd["OOBAGRD3_mean"] - x0) ** 2 + (bd["OOBAGRD6_mean"] - y0) ** 2)
        )
        / 2
    )
    u = scale * np.array(
        [
            np.cos(periscope_drift_data.summary["OOBAGRD_corr_angle"]),
            np.sin(periscope_drift_data.summary["OOBAGRD_corr_angle"]),
        ]
    )
    eigenvector = {
        "x": [x0, x0 + u[0]],
        "y": [y0, y0 + u[1]],
        "marker": {"symbol": "arrow-bar-up", "angleref": "previous", "color": "black"},
        "showlegend": False,
    }
    return [path, eigenvector]


def get_pc1_figure(src_pdd):
    gradient_path, gradient_pc1 = get_gradient_space_path(src_pdd)
    yag_line = get_fit_1d_line(src_pdd, "OOBAGRD_pc1", "yag", color="red")
    yag_scatter = get_binned_data_1d_scatter(
        src_pdd, "OOBAGRD_pc1", "yag", color="blue"
    )
    zag_line = get_fit_1d_line(src_pdd, "OOBAGRD_pc1", "zag", color="red")
    zag_scatter = get_binned_data_1d_scatter(
        src_pdd, "OOBAGRD_pc1", "zag", color="blue"
    )

    fig = go.Figure()
    fig.set_subplots(
        rows=1,
        cols=3,
        subplot_titles=["yag", "zag", "Trajectory in gradient Space"],
    )

    fig.update_layout(
        # title="Fit over First Principal Component in Gradient Space",
        showlegend=False,
    )

    fig.add_trace(yag_scatter, row=1, col=1)
    fig.add_trace(yag_line, row=1, col=1)

    fig.add_trace(zag_scatter, row=1, col=2)
    fig.add_trace(zag_line, row=1, col=2)

    fig.add_trace(gradient_path, row=1, col=3)
    fig.add_trace(gradient_pc1, row=1, col=3)

    fig.update_yaxes(title_text="Residuals (arcsec)", row=1, col=1)
    fig.update_xaxes(title_text="OOBAGRD PC1")
    fig.update_xaxes(title_text="OOBAGRD3", row=1, col=3)
    fig.update_yaxes(title_text="OOBAGRD6", row=1, col=3)

    bd = src_pdd.binned_data_1d
    ymax = np.max([[bd["yag"] + bd["d_yag"]], [bd["zag"] + bd["d_zag"]]])
    ymin = np.min([[bd["yag"] - bd["d_yag"]], [bd["zag"] - bd["d_zag"]]])
    y_margin = 0.1 * (ymax - ymin)
    y_range = (ymin - y_margin, ymax + y_margin)
    fig.update_yaxes(range=y_range, row=1, col=1)
    fig.update_yaxes(range=y_range, row=1, col=2)

    return fig


def get_source_figure(src_pdd):
    margin = 5
    bins = np.linspace(-margin - 0.5, margin + 0.5, 10 * int(2 * margin + 1) + 1)
    bin_centers = bins[:-1] + np.diff(bins) / 2

    vals, _, _ = np.histogram2d(
        src_pdd.events["zag"],
        src_pdd.events["yag"],
        bins=(src_pdd.source["zag"] + bins, src_pdd.source["yag"] + bins),
    )

    heatmap = go.Heatmap(
        {
            "z": vals,
            "x": src_pdd.source["yag"] + bin_centers,
            "y": src_pdd.source["zag"] + bin_centers,
            "type": "heatmap",
            "colorscale": "Greys",
            "reversescale": False,
            "hoverinfo": "skip",
            "showscale": False,
        }
    )

    fig = go.Figure()

    fig.update_layout(
        {
            "autosize": True,
            "yaxis": {
                "scaleanchor": "x",
                "showgrid": False,
                "zeroline": False,
                "showline": True,
            },
            "xaxis": {
                "showgrid": False,
                "zeroline": False,
                "showline": True,
            },
        }
    )
    fig.add_trace(heatmap)

    fig.add_trace(
        go.Scatter(
            {
                "x": [src_pdd.source["yag"]],
                "y": [src_pdd.source["zag"]],
                "mode": "markers",
                "marker": {
                    "size": 5,
                    "color": "red",
                    "opacity": 0.5,
                },
                "name": "Source",
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )
    )

    fig.update_xaxes(
        range=[src_pdd.source["yag"] - margin, src_pdd.source["yag"] + margin],
        constrain="domain",
        title="yag (arcsec)",
    )
    fig.update_yaxes(
        range=[src_pdd.source["zag"] - margin, src_pdd.source["zag"] + margin],
        constrain="domain",
        title="zag (arcsec)",
    )
    fig.update_layout(
        template="simple_white",
    )
    return fig


def get_drift_history_scatter_object(sources):
    custom_data = sources[["obsid", "id"]]
    custom_data["date"] = [d[:10] for d in CxoTime(sources["tstart"]).iso]
    scatter = go.Scatter(
        {
            "x": CxoTime(sources["tstart"]).datetime,
            "y": sources["drift_actual"],
            "mode": "markers",
            "name": "drift_history",
            "marker": {"color": "blue"},
            "hoverinfo": "skip",
            "customdata": custom_data,
            "hovertemplate": (
                "OBSID: %{customdata[0]}<br>"
                "ID: %{customdata[1]}<extra></extra><br>"
                "Date: %{customdata[2]}"
            ),
        }
    )
    return scatter


def get_drift_history_figure(sources):
    fig = go.Figure()

    drift_history_scatter = get_drift_history_scatter_object(sources)

    fig.add_trace(drift_history_scatter)

    fig.update_layout(
        {
            "showlegend": False,
        }
    )

    fig.update_yaxes(
        {
            "title": "Drift Residual (arcsec)",
        },
        # row=1,
        # col=1,
    )

    return fig


def get_drift_scatter_objects(sources):
    drift_scatter = go.Scatter(
        {
            "x": sources["drift_expected"],
            "y": sources["drift_actual"],
            "mode": "markers",
            "name": "Drift",
            "line": {"color": "blue"},
            "hoverinfo": "skip",
            "customdata": sources[["obsid", "id"]],
            "hovertemplate": "OBSID: %{customdata[0]}<br>ID: %{customdata[1]}<extra></extra>",
        }
    )

    return drift_scatter


def get_drift_histograms(sources):
    r, g, b = webcolors.name_to_rgb("blue")
    histogram_actual = go.Histogram(
        {
            "name": "Drift Residual",
            # "histnorm": "probability density",
            "x": sources["drift_actual"],
            "xbins": {
                "start": 0,
                "end": 4,
                "size": 0.1,
            },
            "marker": {
                "color": f"rgba({r}, {g}, {b}, 0.1)",
                "line": {
                    "color": f"rgba({r}, {g}, {b}, 1)",
                    "width": 1,
                },
            },
        }
    )

    r, g, b = webcolors.name_to_rgb("red")
    histogram_expected = go.Histogram(
        {
            "name": "Expected Drift",
            # "histnorm": "probability density",
            "x": sources["drift_expected"],
            "xbins": {
                "start": 0,
                "end": 4,
                "size": 0.1,
            },
            "marker": {
                "color": f"rgba({r}, {g}, {b}, 0.1)",
                "line": {
                    "color": f"rgba({r}, {g}, {b}, 1)",
                    "width": 1,
                },
            },
        }
    )

    return histogram_actual, histogram_expected


def get_drift_figure(sources):
    fig = go.Figure()
    fig.set_subplots(
        rows=1,
        cols=3,
        shared_yaxes=False,
        shared_xaxes=False,
    )

    histogram_actual, histogram_expected = get_drift_histograms(sources)
    drift_scatter = get_drift_scatter_objects(sources)

    drift_scatter.update(
        {
            "showlegend": False,
        }
    )

    fig.add_traces([drift_scatter], rows=1, cols=1)
    fig.add_traces([histogram_expected], rows=1, cols=2)
    fig.add_traces([histogram_actual], rows=1, cols=3)

    fig.update_layout(
        {
            "barmode": "overlay",
            # "title": "Periscope Drift",
            "showlegend": False,
        }
    )

    fig.update_yaxes(
        {
            # "type": 'log',
            "autorange": True
        },
        row=1,
        col=2,
    )

    fig.update_xaxes(
        {
            # "range": [0, 3],
            "title": "Expected Drift (arcsec)"
        },
        row=1,
        col=2,
    )

    if len(sources) > 1:
        ax_max = max(
            1.1 * max(sources["drift_expected"].max(), sources["drift_actual"].max()),
            1.5,
        )
    else:
        ax_max = 3.0
    axis_range = [0, ax_max]
    fig.update_xaxes(
        {
            "title": "Expected Drift (arcsec)",
            "range": axis_range,
            "constrain": "domain",
        },
        row=1,
        col=1,
    )
    fig.update_yaxes(
        {
            # "scaleanchor": "x",
            "title": "Drift Residual (arcsec)",
            "range": axis_range,
            "constrain": "domain",
        },
        row=1,
        col=1,
    )

    fig.update_xaxes(
        {
            "title": "Drift Residual (arcsec)",
            "matches": "x2",  # x3 matches x2
        },
        row=1,
        col=3,
    )
    fig.update_yaxes(matches="y2", row=1, col=3)  # y3 matches y2

    return fig


def get_pc1_goodnes_of_fit_histograms(sources):
    r, g, b = webcolors.name_to_rgb("blue")
    chi2_yag = go.Histogram(
        {
            "name": "yag",
            "x": sources["OOBAGRD_pc1_yag_null_chi2_corr"],
            "xbins": {
                "start": 0,
                "end": 10,
                "size": 0.4,
            },
            "marker": {
                "color": f"rgba({r}, {g}, {b}, 0.1)",
                "line": {
                    "color": f"rgba({r}, {g}, {b}, 1)",
                    "width": 1,
                },
            },
        }
    )

    p_value_yag = go.Histogram(
        {
            "name": "yag",
            "x": sources["OOBAGRD_pc1_yag_null_p_value_corr"],
            "xbins": {
                "start": 0,
                "end": 1,
                "size": 0.01,
            },
            "marker": {
                "color": f"rgba({r}, {g}, {b}, 0.1)",
                "line": {
                    "color": f"rgba({r}, {g}, {b}, 1)",
                    "width": 1,
                },
            },
        }
    )
    r, g, b = webcolors.name_to_rgb("red")
    chi2_zag = go.Histogram(
        {
            "name": "zag",
            "x": sources["OOBAGRD_pc1_zag_null_chi2_corr"],
            "xbins": {
                "start": 0,
                "end": 10,
                "size": 0.4,
            },
            "marker": {
                "color": f"rgba({r}, {g}, {b}, 0.1)",
                "line": {
                    "color": f"rgba({r}, {g}, {b}, 1)",
                    "width": 1,
                },
            },
        }
    )
    p_value_zag = go.Histogram(
        {
            "name": "zag",
            "x": sources["OOBAGRD_pc1_zag_null_p_value_corr"],
            "xbins": {
                "start": 0,
                "end": 1,
                "size": 0.01,
            },
            "marker": {
                "color": f"rgba({r}, {g}, {b}, 0.1)",
                "line": {
                    "color": f"rgba({r}, {g}, {b}, 1)",
                    "width": 1,
                },
            },
        }
    )
    return {
        "chi2": {
            "yag": chi2_yag,
            "zag": chi2_zag,
        },
        "p_value": {"yag": p_value_yag, "zag": p_value_zag},
    }


def get_chi2_figure(sources):
    histograms = get_pc1_goodnes_of_fit_histograms(sources)
    fig = go.Figure()

    fig.set_subplots(
        rows=1,
        cols=2,
        subplot_titles=["yag", "zag"],
        shared_xaxes="all",
        shared_yaxes="all",
    )

    fig.add_trace(histograms["chi2"]["yag"], row=1, col=1)
    fig.add_trace(histograms["chi2"]["zag"], row=1, col=2)

    fig.update_layout(
        {
            "barmode": "overlay",
            "showlegend": False,
        }
    )

    fig.update_xaxes(
        {
            "title": r"$\chi^2$",
        },
        row=1,
        col=1,
    )
    fig.update_xaxes(
        {
            "title": r"$p_{value}$",
        },
        row=1,
        col=2,
    )
    return fig


def get_p_value_figure(sources):
    histograms = get_pc1_goodnes_of_fit_histograms(sources)
    fig = go.Figure()

    fig.set_subplots(
        rows=1,
        cols=2,
        subplot_titles=["yag", "zag"],
        shared_xaxes="all",
        shared_yaxes="all",
    )

    fig.add_trace(histograms["p_value"]["yag"], row=1, col=1)
    fig.add_trace(histograms["p_value"]["zag"], row=1, col=2)

    fig.update_layout(
        {
            "barmode": "overlay",
            "showlegend": False,
        }
    )

    fig.update_xaxes(
        {
            "title": r"$\chi^2$",
        },
        row=1,
        col=1,
    )
    fig.update_xaxes(
        {
            "title": r"$p_{value}$",
        },
        row=1,
        col=2,
    )
    return fig
