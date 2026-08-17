"""
Top-level functions to generate HTML reports for astromon (astrometry) trending.

The structure mirrors ``ska_trend.periscope_drift.reports``: a summary page (``index.html``)
with 90-day / 1-year / 5-year tabs, and one per-source page per (OBSID, x-ray source).
"""

import json
import logging
from pathlib import Path

import jinja2
from astropy import units as u
from tqdm import tqdm

from ska_trend.astromon import data
from ska_trend.astromon import plotly as plots

logger = logging.getLogger("astromon")

__all__ = [
    "write_report",
    "write_html_report",
    "write_source_html_report",
]


JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates" / "astromon")
)

# plotly .to_html kwargs shared by all figures
PLOTLY_KWARGS = {
    "config": {"responsive": True},
    "full_html": False,
    "include_plotlyjs": "cdn",
}


def get_data_for_interval(start, stop, matches, idx=0):
    """
    Build the per-tab data dict (the two offset timelines) for a time interval.
    """
    sel = (matches["time"] >= start) & (matches["time"] < stop)
    interval = matches[sel]

    return {
        "start": start.date[:8],
        "stop": stop.date[:8],
        "start_iso": start.iso[:10],
        "stop_iso": stop.iso[:10],
        "n": len(interval),
        "dz_history": plots.get_offsets_history_figure(interval, "dz").to_html(
            div_id=f"dz_history_{idx}", **PLOTLY_KWARGS
        ),
        "dy_history": plots.get_offsets_history_figure(interval, "dy").to_html(
            div_id=f"dy_history_{idx}", **PLOTLY_KWARGS
        ),
    }


def _source_file(outdir, obsid, x_id):
    return (
        Path(outdir)
        / "sources"
        / f"{float(obsid) // 1e3:02.0f}"
        / f"{obsid}"
        / str(x_id)
        / "index.html"
    )


def write_html_report(
    time_ranges,
    outdir,
    matches,
    archive_dir,
    dbfile=None,
    overwrite=False,
    show_progress=False,
):
    """
    Render and write the html pages (one summary page and one per source).
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # JS format string giving the relative path from the summary page to a source report.
    # Interpolated in the plotly onClick handler from customdata = [obsid, x_id].
    # Must match the layout produced by _source_file / write_source_html_report.
    source_report_path = (
        "`sources/${String(Math.floor(Number(obsid) / 1e3)).padStart(2, '0')}"
        "/${obsid}/${x_id}/index.html`"
    )
    context = {"source_report_path": source_report_path}

    # per-source pages
    if show_progress:
        print("Writing source reports...")
    rows = tqdm(matches) if show_progress else matches
    source_files = []
    for row in rows:
        src_file = _source_file(outdir, row["obsid"], row["x_id"])
        write_source_html_report(
            row, src_file.as_posix(), archive_dir, dbfile=dbfile, overwrite=overwrite
        )
        source_files.append(src_file.relative_to(outdir).as_posix())

    matches["filename"] = source_files

    (outdir / "sources").mkdir(exist_ok=True, parents=True)
    with open(outdir / "sources" / "all.json", "w") as fh:
        json.dump(
            matches[["obsid", "x_id", "filename"]].as_array().tolist(), fh, indent=2
        )

    # summary page
    range_data = [
        get_data_for_interval(tr["start"], tr["stop"], matches=matches, idx=idx)
        for idx, tr in enumerate(time_ranges)
    ]
    for rd, tr in zip(range_data, time_ranges, strict=True):
        rd["title"] = tr["title"]

    template = JINJA_ENV.get_template("index.html")
    page = template.render(time_ranges=range_data, context=context)
    with open(outdir / "index.html", "w") as fh:
        fh.write(page)


def write_source_html_report(row, filename, archive_dir, dbfile=None, overwrite=False):
    """
    Render and write a report for a single source (one x-ray source in one OBSID).
    """
    filename = Path(filename)
    if filename.exists() and not overwrite:
        return

    obsid = int(row["obsid"])
    x_id = int(row["x_id"])

    fig_data = data.get_source_figure_data(
        obsid,
        row["x_loc"],
        archive_dir,
        c_loc=row["c_loc"],
        dbfile=dbfile,
    )
    source_figure = None
    if fig_data is not None:
        fig = plots.get_source_figure(fig_data)
        fig.update_layout({"margin": {"l": 0, "r": 0, "b": 0, "t": 0}})
        source_figure = fig.to_html(div_id="source_figure", **PLOTLY_KWARGS)

    source = {
        "obsid": obsid,
        "x_id": x_id,
        "c_id": row["c_id"],
        "target": row["target"],
        "catalog": row["catalog"],
        "name": row["name"],
        "category": row["category"],
        "detector": row["detector"],
        "grating": row["grating"],
        "dy": float(row["dy"]),
        "dz": float(row["dz"]),
        "dr": float(row["dr"]),
        "snr": float(row["snr"]),
        "r_angle": float(row["r_angle"]),
        "date_obs": str(row["date_obs"]),
        # RA/Dec of the matched catalog source, for the exclude command
        "ra": float(row["c_ra"]),
        "dec": float(row["c_dec"]),
    }

    template = JINJA_ENV.get_template("source_report.html")
    page = template.render(source=source, source_figure=source_figure)

    if not filename.parent.exists():
        filename.parent.mkdir(exist_ok=True, parents=True)
    with open(filename, "w") as fh:
        fh.write(page)


def write_report(
    start,
    stop,
    output_dir,
    archive_dir,
    dbfile=None,
    matches=None,
    selection="mta",
    overwrite=False,
    show_progress=False,
):
    """
    Write reports for a given time interval. Calls all the write_* functions.

    Parameters
    ----------
    start : CxoTime
        Start of the report interval (only sources with ``time >= start`` are included).
    stop : CxoTime
        Stop of the report interval.
    output_dir : str or Path
    archive_dir : str or Path
        Astromon archive directory (for the flux images), e.g.
        $SKA/data/astromon/xray_observations.
    dbfile : str or Path, optional
        Astromon HDF5 db. If None, the astromon default is used.
    matches : astropy.table.Table, optional
        Pre-loaded cross-match table. If None, cross-matches are loaded using ``selection``.
    selection : str
        Cross-match selection ("all", "cal", "mta") used when ``matches`` is None. Default: "mta".
    """
    # 5-year tab, plus an "all" tab covering the full report interval when it goes back further
    five_years = stop - 5 * 365 * u.day
    time_ranges = [
        {"start": five_years, "stop": stop, "title": "5 years"},
    ]
    if start < five_years:
        time_ranges.append({"start": start, "stop": stop, "title": "all"})

    if matches is None:
        matches = data.get_matches(selection=selection, dbfile=dbfile)

    # only sources within the full report interval
    report_matches = matches[(matches["time"] >= start) & (matches["time"] < stop)]
    # sort by time so per-source next/prev navigation is chronological
    report_matches.sort("time")

    write_html_report(
        time_ranges,
        output_dir,
        report_matches,
        archive_dir,
        dbfile=dbfile,
        overwrite=overwrite,
        show_progress=show_progress,
    )

    with open(Path(output_dir) / "args.json", "w") as fh:
        args = {
            "start": start,
            "stop": stop,
            "output_dir": str(output_dir),
            "archive_dir": str(archive_dir),
            "dbfile": str(dbfile),
            "selection": selection,
            "n_sources": len(report_matches),
        }
        json.dump(args, fh, indent=2, default=str)
