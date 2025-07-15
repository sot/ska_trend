"""
Top-level functions to generate HTML reports for periscope drift trending.
"""

import json
from pathlib import Path

import jinja2
import numpy as np
from mica.archive.cda import get_ocat_web, get_proposal_abstract

from ska_trend.periscope_drift import plotly as plots

__all__ = [
    "write_html_report",
    "write_source_html_report",
]


JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(
        Path(__file__).parent / "templates" / "periscope_drift"
    )
)


def get_data_for_interval(start, stop, observations, sources, idx=0):
    """
    Render and write the main page.

    Parameters
    ----------
    observations : list
        List of observations.
    sources : Table
        Table of sources.
    """

    tstart = [observations[f"{obs}"].get_info()["tstart"] for obs in sources["obsid"]]
    observations = {
        key: obs for key, obs in observations.items()
        if obs.get_info()["date_obs"] < stop
        and obs.get_info()["date_obs"] >= start
    }
    sources = sources[(tstart < stop) & (tstart >= start)]

    large_exp_drift_sources = sources.copy()
    large_exp_drift_sources = large_exp_drift_sources[
        large_exp_drift_sources["drift_expected"] > 0.4
    ]
    large_exp_drift_sources.sort("drift_expected", reverse=True)
    large_drift_sources = sources.copy()
    large_drift_sources = large_drift_sources[
        large_drift_sources["drift_expected"] > 0.4
    ]
    large_drift_sources.sort("drift_actual", reverse=True)

    poor_fit_sources = sources.copy()
    poor_fit_sources["OOBAGRD_pc1_null_p_value_corr"] = np.where(
        poor_fit_sources["OOBAGRD_pc1_zag_null_p_value_corr"]
        < poor_fit_sources["OOBAGRD_pc1_zag_null_p_value_corr"],
        poor_fit_sources["OOBAGRD_pc1_zag_null_p_value_corr"],
        poor_fit_sources["OOBAGRD_pc1_zag_null_p_value_corr"],
    )
    poor_fit_sources = poor_fit_sources[
        poor_fit_sources["OOBAGRD_pc1_null_p_value_corr"] < 0.01
    ]
    poor_fit_sources.sort(
        [
            "OOBAGRD_pc1_null_p_value_corr",
            "OOBAGRD_pc1_yag_null_p_value_corr",
            "OOBAGRD_pc1_zag_null_p_value_corr",
        ]
    )

    large_exp_drift_sources.rename_columns(
        ["drift_yag_actual", "drift_zag_actual"],
        ["drift_yag_residual", "drift_zag_residual"],
    )
    large_drift_sources.rename_columns(
        ["drift_yag_actual", "drift_zag_actual"],
        ["drift_yag_residual", "drift_zag_residual"],
    )
    poor_fit_sources.rename_columns(
        ["drift_yag_actual", "drift_zag_actual"],
        ["drift_yag_residual", "drift_zag_residual"],
    )

    source_tables = []
    source_tables.append(
        {
            "id": "expected_drift",
            "title": "Sources with large expected drift (> 0.4)",
            "sources": large_exp_drift_sources[:20],
        }
    )
    source_tables.append(
        {
            "id": "actual_drift",
            "title": "Sources with large drift (> 0.4)",
            "sources": large_drift_sources[:20],
        }
    )
    source_tables.append(
        {
            "id": "p_value",
            "title": "Sources inconsistent with null hypothesis (p-value < 0.01)",
            "sources": poor_fit_sources[:20],
        }
    )

    kwargs = {
        "full_html": False,
        "include_plotlyjs": "cdn",
    }

    result = {
        "start": start.date[:8],
        "stop": stop.date[:8],
        "start_iso": start.iso[:10],
        "stop_iso": stop.iso[:10],
        "chi2_figure": plots.get_chi2_figure(sources).to_html(
            div_id=f"chi2_figure_{idx}", **kwargs
        ),
        "p_value_figure": plots.get_p_value_figure(sources).to_html(
            div_id=f"p_value_figure_{idx}", **kwargs
        ),
        "drift_figure": plots.get_drift_figure(sources).to_html(
            div_id=f"drift_figure_{idx}", **kwargs
        ),
        "tables": source_tables,
    }
    return result


def write_html_report(time_ranges, outdir, observations, sources, overwrite=False):
    """
    Render and write the main page.

    Parameters
    ----------
    outdir : str or Path
        Output directory.
    observations : list
        List of observations.
    sources : Table
        Table of sources.
    """

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # this string is used in the template to link to the source report
    # it should match what is done below for src_file
    source_report_path = (
        "`sources/${String(Math.floor(Number(obsid) / 1e3)).padStart(2, '0')}"
        "/${obsid}/${src_id}/index.html`"
    )

    context = {"source_report_path": source_report_path}

    # the sources
    source_files = []
    for obsid, src_id in sources["obsid", "id"]:
        obs = observations[str(obsid)]
        src_file = (
            Path("sources")
            / f"{float(obsid) // 1e3:02.0f}"
            / f"{obsid}"
            / str(src_id)
            / "index.html"
        )
        src_file.parent.mkdir(exist_ok=True, parents=True)
        write_source_html_report(obs, src_id, outdir / src_file, overwrite=overwrite)
        source_files.append(str(src_file))

    sources["filename"] = source_files

    with open(outdir / "sources" / "all.json", "w") as fh:
        json.dump(
            sources[["obsid", "src_id", "filename"]].as_array().tolist(), fh, indent=2
        )

    # the main page
    range_data = [
        get_data_for_interval(
            tr["start"], tr["stop"], observations=observations, sources=sources, idx=idx
        )
        for idx, tr in enumerate(time_ranges)
    ]
    for rd, tr in zip(range_data, time_ranges, strict=True):
        rd["title"] = tr["title"]

    filename = outdir / "index.html"
    template = JINJA_ENV.get_template("index.html")

    page = template.render(
        time_ranges=range_data,
        context=context,
    )
    with open(filename, "w") as fh:
        fh.write(page)


def write_source_html_report(obs, src_id, filename, overwrite=False):
    """
    Render and write a report for a single source in a single observation.
    """
    filename = Path(filename)

    if filename.exists() and not overwrite:
        return

    src_pdd = obs.periscope_drift.get_periscope_drift_data().data[src_id]
    kwargs = {
        "full_html": False,
        "include_plotlyjs": "cdn",
    }
    template = JINJA_ENV.get_template("source_report.html")

    source = plots.get_source_figure(src_pdd)
    source.update_layout({"margin": {"l": 0, "r": 0, "b": 0, "t": 0}})

    summary = src_pdd.summary
    metrics = {}
    if summary:
        metrics = {
            "yag_slope": summary["OOBAGRD_pc1_yag_slope"],
            "yag_slope_err": summary["OOBAGRD_pc1_yag_slope_err"],
            "zag_slope": summary["OOBAGRD_pc1_zag_slope"],
            "zag_slope_err": summary["OOBAGRD_pc1_yag_slope_err"],
            "yag_null_chi2": summary["OOBAGRD_pc1_yag_null_chi2_corr"],
            "yag_ndf": summary["OOBAGRD_pc1_yag_ndf"],
            "zag_null_chi2": summary["OOBAGRD_pc1_zag_null_chi2_corr"],
            "zag_ndf": summary["OOBAGRD_pc1_zag_ndf"],
            "drift_yag_expected": summary["drift_yag_expected"],
            "drift_zag_expected": summary["drift_zag_expected"],
            "drift_yag_residual": summary["drift_yag_actual"],
            "drift_zag_residual": summary["drift_zag_actual"],
            "yag_null_p_value_corr": summary["OOBAGRD_pc1_yag_null_p_value_corr"],
            "zag_null_p_value_corr": summary["OOBAGRD_pc1_zag_null_p_value_corr"],
        }

    try:
        abstract = get_proposal_abstract(obs.obsid)
    except Exception as exc:
        if str(exc).startswith("got error 404"):
            abstract = ""
        else:
            raise

    ocat = {
        "abstract": abstract,
        "ocat": get_ocat_web(obs.obsid),
    }

    page = template.render(
        source=src_pdd.source,
        metrics=metrics,
        source_figure=source.to_html(config={"displayModeBar": False}, **kwargs),
        scatter_plot=plots.get_scatter_plot_figure(src_pdd).to_html(**kwargs),
        scatter_vs_gradients=plots.get_scatter_versus_gradients_figure(src_pdd).to_html(
            **kwargs
        ),
        pc1_figure=plots.get_pc1_figure(src_pdd).to_html(**kwargs),
        telemetry_figure=plots.get_telemetry_figure(obs).to_html(**kwargs),
        extreme_bin_histograms_figure=plots.get_extreme_bin_histograms_figure(
            src_pdd
        ).to_html(**kwargs),
        plot_3d_figures=plots.get_plot_3d_figures(src_pdd).to_html(**kwargs),
        ocat=ocat,
    )
    filename.parent.mkdir(exist_ok=True, parents=True)
    with open(filename, "w") as fh:
        fh.write(page)
