"""
Top-level functions to generate HTML reports for periscope drift trending.
"""

from pathlib import Path

import jinja2

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


def write_html_report(outdir, observations, sources, overwrite=False):
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

    import json

    with open(outdir / "sources" / "all.json", "w") as fh:
        json.dump(
            sources[["obsid", "src_id", "filename"]].as_array().tolist(), fh, indent=2
        )

    # the main page
    filename = outdir / "index.html"

    template = JINJA_ENV.get_template("index.html")

    kwargs = {
        "full_html": False,
        "include_plotlyjs": "cdn",
    }
    page = template.render(
        chi2_figure=plots.get_chi2_figure(sources).to_html(**kwargs),
        drift_figure=plots.get_drift_figure(sources).to_html(**kwargs),
        sources=sources,
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

    src_pdd = obs.get_periscope_drift_data().data[src_id]
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
            "drift_yag_actual": summary["drift_yag_actual"],
            "drift_zag_actual": summary["drift_zag_actual"],
            "yag_null_p_value_corr": summary["OOBAGRD_pc1_yag_null_p_value_corr"],
            "zag_null_p_value_corr": summary["OOBAGRD_pc1_zag_null_p_value_corr"],
        }

    from mica.archive.cda import get_ocat_web, get_proposal_abstract

    ocat = {
        "abstract": get_proposal_abstract(obs.obsid),
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
        plot_3d_figure_yag=plots.get_plot_3d_figure(src_pdd, "yag").to_html(**kwargs),
        plot_3d_figure_zag=plots.get_plot_3d_figure(src_pdd, "zag").to_html(**kwargs),
        ocat=ocat,
    )
    filename.parent.mkdir(exist_ok=True, parents=True)
    with open(filename, "w") as fh:
        fh.write(page)
