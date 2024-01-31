# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Update wrong-box acquisition plot.
"""

import argparse
import functools
import os

from pathlib import Path
from typing import TypeAlias
import jinja2


os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import mica.stats.acq_stats
from astropy import units as u
from astropy.table import Table, hstack
from ska_matplotlib import plot_cxctime
from cxotime import CxoTime
from ska_quatutil import yagzag2radec
from mica.starcheck import get_starcheck_catalog
from Quaternion import Quat
import agasc
from ska_helpers.logging import basic_logger

from .. import __version__

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
    table.sort('guide_tstart', reverse=True)
    table = table[table['classic'] == True]
    out_html = template.render(
        anom_table=table[0:20],  # last 20 rows
    )
    html_path = Path(opt.out_dir) / "index.html"
    logger.info(f"Writing HTML to {html_path}")
    html_path.write_text(out_html)


def make_plot(opt, anoms):
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
    anom_match = (acqs["img_func"] != "NONE") & (
        (np.abs(acqs["dy"]) >= (acqs["halfw"] + BOX_PAD))
        | (np.abs(acqs["dz"]) >= (acqs["halfw"] + BOX_PAD))
    )
    return anom_match


def get_anom_acq_obs(start="2010:001"):
    acqs = mica.stats.acq_stats.get_stats()
    acqs = acqs[acqs["guide_tstart"] > CxoTime(start).secs]
    anom_match = wrong_box(acqs)
    anom_obsids = np.unique(acqs[anom_match]["obsid"])
    # Reduce the data set to just obsids with anomalous acqs
    return Table(acqs[np.in1d(acqs["obsid"], anom_obsids)])


def get_anom_info(anom_row, acqs):
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
        {"agasc_star": (len(stars) > 0) & (np.any(stars["MAG_ACA"] <= (anom_row["mag_obs"] + 1.5)))})
    return dat


def acq_anom_checks(acqs):
    anom = wrong_box(acqs)
    anoms = acqs[anom]
    new_info = []
    for anom_row in anoms:
        new_info.append(get_anom_info(anom_row, acqs))
    new_info = Table(new_info)
    anoms = hstack([anoms, new_info])
    return anoms


def main(sys_args=None):
    opt = get_opt().parse_args(sys_args)

    acqs = get_anom_acq_obs()
    anom_table = acq_anom_checks(acqs)

    make_plot(opt, anom_table)
    make_web_page(opt, anom_table)


if __name__ == "__main__":
    main()
