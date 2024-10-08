# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Update wrong-box acquisition plot.

This module provides functions to update and generate a wrong-box acquisition plot and table of
recent anomalies.

"""

import argparse
import os
from pathlib import Path

import jinja2

os.environ["MPLBACKEND"] = "Agg"

import agasc
import matplotlib.pyplot as plt
import mica.stats.acq_stats
import numpy as np
from astropy import units as u
from astropy.table import Table, hstack
from cxotime import CxoTime
from mica.starcheck import get_starcheck_catalog
from Quaternion import Quat
from ska_helpers.logging import basic_logger
from ska_quatutil import yagzag2radec

from ska_trend import __version__

# Constants and file path definitions
FILE_DIR = Path(__file__).parent


def INDEX_TEMPLATE_PATH():
    return FILE_DIR / "index_template.html"


logger = basic_logger(__name__, level="INFO")

BOX_PAD = 20


def get_opt():
    parser = argparse.ArgumentParser(
        description="Wrong box acq trend {}".format(__version__)
    )
    parser.add_argument("--out-dir", type=str, default=".", help="Out directory")
    parser.add_argument(
        "--highlight-recent-days",
        type=float,
        default=30.0,
        help="Number of days to highlight in plots and table (days, default=30)",
    )
    return parser


def make_web_page(opt, anom_table):
    """
    Generate a web page report.

    Parameters
    ----------
    opt : argparse.Namespace
        The command-line arguments.
    anom_table : astropy.table.Table
        The table of anomaly data.

    Returns
    -------
    None
    """
    # Setup for the web report template rendering
    tr_classes = []
    for anom in anom_table:
        recent = (
            CxoTime.now() - CxoTime(anom["guide_tstart"])
            < opt.highlight_recent_days * u.day
        )
        tr_class = 'class="pink-bkg"' if recent else ""
        tr_classes.append(tr_class)
    anom_table["tr_class"] = tr_classes

    index_template_html = INDEX_TEMPLATE_PATH().read_text()
    template = jinja2.Template(index_template_html)
    table = anom_table.copy()
    table.sort("guide_tstart", reverse=True)

    # Filter out non-classic anomalies
    ok = table["classic"]
    table = table[ok]

    out_html = template.render(
        anom_table=table[0:20],  # last 20 rows
    )
    html_path = Path(opt.out_dir) / "index.html"
    logger.info(f"Writing HTML to {html_path}")
    html_path.write_text(out_html)


def make_plot(opt, anoms):
    """
    Generate a plot of wrong-box acquisition anomalies.

    Parameters
    ----------
    opt : argparse.Namespace
        The command-line arguments.
    anoms : astropy.table.Table
        The table of anomaly data.
    """
    frac_year = CxoTime(anoms["guide_tstart"]).frac_year

    bin_width = 0.5
    bins = np.arange(2010, CxoTime.now().frac_year + bin_width, bin_width)
    # just exclude the weird ones for these plots
    ok = anoms["classic"]
    trak = anoms["agasc_star"] & anoms["classic"]
    another_acq = anoms["another_acq_star"] & anoms["classic"]
    plt.figure(figsize=(6, 4))
    plt.hist(
        frac_year[ok], bins=bins, edgecolor="black", linewidth=1.4, label="anomaly"
    )
    plt.hist(
        frac_year[trak],
        bins=bins,
        edgecolor="black",
        linewidth=1.4,
        label="anom with agasc star",
    )
    plt.hist(
        frac_year[another_acq],
        bins=bins,
        edgecolor="black",
        linewidth=1.4,
        label="anom with another acq star",
    )
    plt.grid(color="black")
    plt.margins(0.05)

    plt.xlabel("Year")
    plt.ylabel("Acquisition Anomalies (N)")
    plt.title("Wrong-box Acquisition Anomalies\n (6-month bins)")
    plt.legend(loc="upper left")

    Path(opt.out_dir).mkdir(exist_ok=True, parents=True)
    plt.savefig(Path(opt.out_dir) / "wrong_box.png", dpi=200)


def wrong_box(acqs):
    """
    Check if an acquisition is a wrong-box acquisition.

    Parameters
    ----------
    acqs : astropy.table.Table
        The table of acquisition data.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating if each acquisition is a wrong-box acquisition.
    """
    anom_match = (acqs["img_func"] != "NONE") & (
        (np.abs(acqs["dy"]) >= (acqs["halfw"] + BOX_PAD))
        | (np.abs(acqs["dz"]) >= (acqs["halfw"] + BOX_PAD))
    )
    return anom_match


def get_anom_acq_obs(start="2010:001"):
    """
    Get acquisition observations with wrong-box anomalies.

    Parameters
    ----------
    start : str, optional
        The start time for selecting observations. Defaults to "2010:001".

    Returns
    -------
    astropy.table.Table
        The table of acquisition observations with wrong-box anomalies.
    """
    acqs = mica.stats.acq_stats.get_stats()
    acqs = acqs[acqs["guide_tstart"] > CxoTime(start).secs]
    anom_match = wrong_box(acqs)
    anom_obsids = np.unique(acqs[anom_match]["obsid"])
    # Reduce the data set to just obsids with anomalous acqs
    return Table(acqs[np.in1d(acqs["obsid"], anom_obsids)])


def get_anom_info(anom_row, acqs):
    """
    Get information about a wrong-box anomaly.

    Parameters
    ----------
    anom_row : astropy.table.Row
        The row of anomaly data.
    acqs : astropy.table.Table
        The table of acquisition data.

    Returns
    -------
    dict
        Dictionary containing information about the wrong-box anomaly.

    """
    dat = {
        "classic": False,
        "actual_slot": -1,
        "actual_slot_idd": False,
        "actual_slot_mag_obs": -1,
        "another_acq_star": False,
        "agasc_star": False,
    }

    obs_slots = acqs[(acqs["obsid"] == anom_row["obsid"])]

    # Figure out which search box the star was actually in,
    # and record if that star was id'd.
    other_slots = obs_slots[(obs_slots["slot"] != anom_row["slot"])]
    square_dist = np.sqrt(
        (other_slots["yang"] - anom_row["yang_obs"]) ** 2
        + (other_slots["zang"] - anom_row["zang_obs"]) ** 2
    )
    closest_slot = np.argmin(square_dist)
    if (
        np.abs(other_slots["yang"][closest_slot] - anom_row["yang_obs"])
        < (other_slots["halfw"][closest_slot] + BOX_PAD)
    ) & (
        np.abs(other_slots["zang"][closest_slot] - anom_row["zang_obs"])
        < (other_slots["halfw"][closest_slot] + BOX_PAD)
    ):
        dat.update(
            {
                "classic": True,
                "actual_slot": other_slots["slot"][closest_slot],
                "actual_slot_idd": other_slots["acqid"][closest_slot],
                "actual_slot_mag_obs": other_slots["mag_obs"][closest_slot],
            }
        )

    # Get star catalog for the obsid
    sc = get_starcheck_catalog(anom_row["obsid"])

    # And get the RA and Dec of what was tracked (cdy cdz puts things approx at the
    # target location)
    y = anom_row["yang"] + anom_row["cdy"]
    z = anom_row["zang"] + anom_row["cdz"]

    # Check if the acquired star was actually an acquisition star
    intended = acqs[
        (acqs["obsid"] == anom_row["obsid"]) & (acqs["slot"] != anom_row["slot"])
    ]
    is_another_acq_star = np.any(
        (np.abs(intended["yang"] - y) < 7) & (np.abs(intended["zang"] - z) < 7)
    )
    dat.update({"another_acq_star": is_another_acq_star})

    # Check if an agasc star was tracked
    ra, dec = yagzag2radec(
        y / 3600.0,
        z / 3600.0,
        Quat((sc["obs"]["point_ra"], sc["obs"]["point_dec"], sc["obs"]["point_roll"])),
    )
    stars = agasc.get_agasc_cone(
        ra, dec, radius=7.0 / 3600.0, date=anom_row["guide_tstart"]
    )
    dat.update(
        {
            "agasc_star": (len(stars) > 0)
            & (np.any(stars["MAG_ACA"] <= (anom_row["mag_obs"] + 1.5)))
        }
    )
    return dat


def acq_anom_checks(acqs):
    """
    Perform anomaly checks on acquisition observations.

    Parameters
    ----------
    acqs : astropy.table.Table
        The table of acquisition observations.

    Returns
    -------
    astropy.table.Table
        The table of acquisition observations with additional anomaly information.
    """
    anom = wrong_box(acqs)
    anoms = acqs[anom]
    new_info = [get_anom_info(anom_row, acqs) for anom_row in anoms]
    new_info = Table(new_info)
    anoms = hstack([anoms, new_info])
    return anoms


def add_manual_entries(anom_table):
    """
    Add manual entries to the anomaly table.

    The data is just hardcoded within this function as for this trending application one can edit
    the file and update quickly via non-fsds process.

    Parameters
    ----------
    anom_table : astropy.table.Table
        The table of anomaly data.

    Returns
    -------
    astropy.table.Table
        The table of anomaly data with manual entries added.
    """

    table = anom_table.copy()
    table = table[
        "obsid",
        "slot",
        "classic",
        "acq_start",
        "guide_tstart",
        "mag_obs",
        "actual_slot",
        "actual_slot_idd",
        "actual_slot_mag_obs",
        "another_acq_star",
        "agasc_star",
    ]

    # TODO - move manual data out of the Python code and into an ecsv file
    manual_rows = [
        {
            "obsid": 29221,
            "acq_start": "2024:024:09:42:51.000",
            "guide_tstart": CxoTime("2024:024:09:48:51.000").secs,
            "slot": 7,
            "actual_slot": 0,
            "actual_slot_idd": False,
            "another_acq_star": True,
            "agasc_star": True,
            "classic": True,
            "mag_obs": 8.62,
            "actual_slot_mag_obs": 9.19,
        }
    ]
    for row in manual_rows:
        table.add_row(row)
    return table


def main(sys_args=None):
    opt = get_opt().parse_args(sys_args)

    acqs = get_anom_acq_obs()
    anom_table = acq_anom_checks(acqs)
    anom_table = add_manual_entries(anom_table)
    make_plot(opt, anom_table)
    make_web_page(opt, anom_table)


if __name__ == "__main__":
    main()
