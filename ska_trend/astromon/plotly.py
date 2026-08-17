"""
Plotly figures for the astromon trending reports.

Two kinds of figures:

- :func:`get_offsets_history_figure` -- the dY/dZ vs time timelines shown on the summary page.
  Points carry ``customdata = [obsid, x_id]`` so the page javascript can open the matching
  per-source report on click.
- :func:`get_source_figure` -- the cropped sky view (flux image heatmap + catalog/x-ray markers)
  that is the main element of each per-source page. This mirrors the react ``ObsidImage.js``
  component, but cropped around the source.
"""

import plotly.graph_objects as go
from cxotime import CxoTime

from ska_trend.astromon import data

__all__ = [
    "get_offsets_history_figure",
    "get_source_figure",
]

# axis labels and colors for each offset coordinate (matching celmon: dy blue, dz orange)
COORD_INFO = {
    "dy": {"title": "dY (arcsec)", "color": "#1f77b4"},
    "dz": {"title": "dZ (arcsec)", "color": "#ff7f0e"},
}


def _year_to_iso(years):
    """Convert an array of fractional years to ISO date strings for a plotly date axis."""
    if len(years) == 0:
        return []
    return list(CxoTime(years, format="frac_year").isot)


def _median_band_traces(matches, coord, color):
    """
    Build the binned-median line and ±1-sigma band traces (celmon q-history style).

    The median is drawn as one horizontal segment per time bin (disconnected between bins);
    the band is a single filled step polygon between the 15.8 and 84.2 percentiles.
    """
    binned = data.binned_offsets(matches, coord)
    start_iso = _year_to_iso(binned["start"])
    stop_iso = _year_to_iso(binned["stop"])
    if not start_iso:
        return []

    # ±1-sigma band: upper boundary left->right, then lower boundary right->left, closed.
    x_band, y_band = [], []
    for s, e, hi in zip(start_iso, stop_iso, binned["sigma_plus"], strict=True):
        x_band += [s, e]
        y_band += [hi, hi]
    for s, e, lo in zip(
        reversed(start_iso),
        reversed(stop_iso),
        reversed(binned["sigma_minus"]),
        strict=True,
    ):
        x_band += [e, s]
        y_band += [lo, lo]
    band = go.Scatter(
        x=x_band,
        y=y_band,
        mode="lines",
        line={"width": 0},
        fill="toself",
        fillcolor="rgba(255,127,14,0.25)" if coord == "dz" else "rgba(31,119,180,0.25)",
        hoverinfo="skip",
        name="±1σ",
    )

    # median: one horizontal segment per bin, broken by None so bins are not connected.
    x_med, y_med = [], []
    for s, e, med in zip(start_iso, stop_iso, binned["median"], strict=True):
        x_med += [s, e, None]
        y_med += [med, med, None]
    median = go.Scatter(
        x=x_med,
        y=y_med,
        mode="lines",
        line={"color": color, "width": 3},
        hoverinfo="skip",
        name="median",
    )

    return [band, median]


def get_offsets_history_figure(matches, coord):
    """
    Time series of an offset coordinate (``dy`` or ``dz``) versus time.

    A binned-median line and ±1-sigma band are overlaid (as on the celmon q-history plot).

    Parameters
    ----------
    matches : astropy.table.Table
        Cross-match table (must have ``date_iso``, ``time``, ``obsid``, ``x_id`` and ``coord``).
    coord : str
        Either ``"dy"`` or ``"dz"``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    info = COORD_INFO[coord]
    # Use SVG Scatter (not Scattergl): WebGL traces in an initially-hidden tab pane never build
    # their click layer, so plotly_click would not fire in non-default tabs. Point counts here
    # (hundreds to a couple thousand) are well within SVG's comfort zone.
    scatter = go.Scatter(
        {
            "x": list(matches["date_iso"]),
            "y": list(matches[coord]),
            "mode": "markers",
            "name": coord,
            "marker": {"size": 5, "color": "rgba(0,0,0,0.4)"},
            "customdata": matches[["obsid", "x_id"]],
            "hovertemplate": (
                "OBSID: %{customdata[0]}<br>"
                "x_id: %{customdata[1]}<br>"
                f"{coord}: %{{y:.2f}}<extra></extra>"
            ),
        }
    )
    fig = go.Figure(data=[scatter, *_median_band_traces(matches, coord, info["color"])])
    fig.update_layout({"showlegend": False, "template": "simple_white"})
    # match celmon's initial vertical range (points outside are still reachable by zooming out)
    fig.update_yaxes({"title": info["title"], "range": [-1.1, 1.1]})
    fig.update_xaxes({"title": "Date"})
    return fig


def get_source_figure(fig_data):
    """
    Cropped sky view for a single source.

    Parameters
    ----------
    fig_data : dict
        Output of :func:`ska_trend.astromon.data.get_source_figure_data`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # the cropped flux image
    fig.add_trace(
        go.Heatmap(
            z=fig_data["image"],
            colorscale="Greys",
            showscale=False,
            hoverinfo="skip",
        )
    )

    # vicinity catalog sources (blue, one symbol per catalog)
    by_catalog = {}
    for m in fig_data["vicinity"]:
        by_catalog.setdefault(m["catalog"], []).append(m)
    for catalog, members in by_catalog.items():
        fig.add_trace(
            go.Scatter(
                x=[m["x"] for m in members],
                y=[m["y"] for m in members],
                mode="markers",
                name=catalog,
                text=[f"{m['catalog']}<br>{m['name']}" for m in members],
                customdata=[m["simbad_url"] for m in members],
                hovertemplate="%{text}<br>(click to open SIMBAD)<extra></extra>",
                marker={
                    "symbol": members[0]["symbol"],
                    "size": 10,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "dodgerblue", "width": 2},
                },
            )
        )

    # the matched catalog counterpart for this source (red)
    if fig_data.get("match") is not None:
        fig.add_trace(
            go.Scatter(
                x=[fig_data["match"]["x"]],
                y=[fig_data["match"]["y"]],
                mode="markers",
                name="X-match",
                customdata=[fig_data["match"]["simbad_url"]],
                hovertemplate="matched counterpart<br>(click to open SIMBAD)<extra></extra>",
                marker={
                    "symbol": "circle-dot",
                    "size": 20,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "rgb(204,51,0)", "width": 2},
                },
            )
        )

    # the x-ray source detections (orange)
    xray = fig_data["xray"]
    fig.add_trace(
        go.Scatter(
            x=xray["x"],
            y=xray["y"],
            mode="markers",
            name="X-Ray Source",
            hoverinfo="skip",
            marker={
                "symbol": "circle",
                "size": 8,
                "color": "rgba(0,0,0,0)",
                "line": {"color": "rgb(204,102,0)", "width": 2},
            },
        )
    )

    fig.update_layout(
        {
            "template": "simple_white",
            "legend": {"x": 1, "xanchor": "right", "y": 1},
            "showlegend": True,
        }
    )
    fig.update_yaxes({"scaleanchor": "x", "showgrid": False, "showticklabels": False})
    fig.update_xaxes({"showgrid": False, "showticklabels": False})
    return fig
