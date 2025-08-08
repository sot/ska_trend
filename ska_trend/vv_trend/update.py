#!/usr/bin/env python

import argparse
import shutil
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from astropy.table import Table
import astropy.units as u

from cxotime import CxoTime

from chandra_aca.dark_model import get_warm_fracs
from mica.vv import get_rms_data


def get_options():
    parser = argparse.ArgumentParser(
        description="Update VV/Aspect Solution resid plots"
    )
    parser.add_argument("--outdir", default=".", help="directory for plots")
    return parser


def mission_plots(rms_data):
    norm = mpl.colors.LogNorm()
    my_cm = cm.jet
    figsize = (6, 4)
    data = rms_data
    reasonable = (data["dz_rms"] > 0) & (data["dz_rms"] < 1)
    last_year = data["tstart"] > (CxoTime.now() - 365 * u.day).secs
    since_2015 = data["tstart"] > CxoTime("2015:001").secs

    mag_resid_fig = plt.figure(figsize=figsize)
    year_2007 = (data["tstart"] < CxoTime("2008:001").secs) & (
        data["tstart"] > CxoTime("2007:001").secs
    )
    cmap_blue = matplotlib.colors.ListedColormap(["white", "blue"])
    cmap_red = matplotlib.colors.ListedColormap(["white", "red"])
    cmap_blue._init()
    cmap_red._init()
    # set the white to have alpha 0 and everything else to a half
    cmap_blue._lut[:, -1] = [0, 1, 0, 1, 1]
    cmap_red._lut[:, -1] = [0, 0.8, 0, 0.8, 0.8]
    bounds = [0, 1, 10]
    norm_blue = matplotlib.colors.BoundaryNorm(bounds, cmap_blue.N, 256)
    norm_red = matplotlib.colors.BoundaryNorm(bounds, cmap_red.N, 256)
    ax2 = mag_resid_fig.add_axes([0.14, 0.12, 0.70, 0.78])
    h2, x2, y2 = np.histogram2d(
        data[reasonable & year_2007]["mag_med"],
        data[reasonable & year_2007]["dz_rms"],
        bins=100,
        range=[[5, 11], [0, 0.3]],
    )
    h1, x1, y1 = np.histogram2d(
        data[reasonable & last_year]["mag_med"],
        data[reasonable & last_year]["dz_rms"],
        bins=100,
        range=[[5, 11], [0, 0.3]],
    )
    ax2.pcolorfast(x2, y2, h2.T, cmap=cmap_blue, norm=norm_blue)
    ax2.pcolorfast(x1, y1, h1.T, cmap=cmap_red, norm=norm_red)
    plt.grid()
    plt.title("RMS vs mag: year 2007 (blue), past year (red)")
    plt.xlabel("Median Mag.")
    plt.ylabel("Star Resid RMS in Z (arcsec)")

    hist2d_fig = plt.figure(figsize=figsize)
    H, xedges, yedges = np.histogram2d(
        CxoTime(data[reasonable & since_2015]["tstart"]).frac_year,
        data[reasonable & since_2015]["dz_rms"],
        bins=150,
        range=[[2015, CxoTime.now().frac_year + 0.25], [0, 0.35]],
    )
    # ax1 = hist2d_fig.add_axes([0.125, 0.12, 0.70, 0.78])
    ax1 = hist2d_fig.add_axes([0.14, 0.14, 0.70, 0.78])
    # ax1 = subplot(111)
    ax1.pcolorfast(xedges, yedges, H.T, cmap=my_cm, norm=norm)
    plt.grid()
    plt.ylim(-0.045, 0.35)
    plt.vlines(CxoTime("2018:292").frac_year, -0.045, 0.35)
    plt.annotate(
        "Mixed IRU",
        (CxoTime("2018:292").frac_year - 0.175, -0.04),
        rotation=90,
        fontsize=8,
    )
    plt.vlines(CxoTime("2020:213").frac_year, -0.045, 0.35)
    plt.annotate(
        "Single IRU",
        (CxoTime("2020:213").frac_year - 0.175, -0.04),
        rotation=90,
        fontsize=8,
    )
    smode_date = CxoTime("2022:294").frac_year
    plt.vlines(smode_date, -0.045, 0.35)
    plt.annotate("Safe Mode", (smode_date - 0.175, -0.04), rotation=90, fontsize=8)
    smode_date = CxoTime("2023:044").frac_year
    plt.vlines(smode_date, -0.045, 0.35)
    plt.annotate("Safe Mode", (smode_date - 0.175, -0.04), rotation=90, fontsize=8)

    plt.ylabel("Star Resid RMS in Z (arcsec)")
    plt.suptitle("RMS vs Time")
    plt.xticks(rotation=60)
    ax1.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))

    ax2 = hist2d_fig.add_axes([0.85, 0.14, 0.015, 0.78])
    tick_locs = [norm.vmin, 2, 5, 10, norm.vmax]
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=my_cm, norm=norm, orientation="vertical")
    cb.locator = mpl.ticker.FixedLocator(tick_locs)
    cb.formatter = mpl.ticker.FormatStrFormatter("%d")
    plt.ylabel("N stars")
    cb.update_ticks()

    # plot rms vs warm fraction
    hist2d_fig_n100 = plt.figure(figsize=figsize)
    H, xedges, yedges = np.histogram2d(
        data[reasonable & since_2015]["n100_frac"],
        data[reasonable & since_2015]["dz_rms"],
        bins=100,
        range=[[0.01, np.max(data[reasonable]["n100_frac"])], [0.0, 0.35]],
    )
    ax1n = hist2d_fig_n100.add_axes([0.14, 0.14, 0.70, 0.78])
    ax1n.pcolorfast(xedges, yedges, H.T, cmap=my_cm, norm=norm)
    plt.ylim(-0.04, 0.35)
    plt.grid()
    plt.ylabel("Star Resid RMS in Z (arcsec)")
    plt.xlabel("N100 frac")
    plt.suptitle("RMS vs N100 frac")
    ax2n = hist2d_fig_n100.add_axes([0.85, 0.14, 0.015, 0.78])
    cbn = mpl.colorbar.ColorbarBase(ax2n, cmap=my_cm, norm=norm, orientation="vertical")
    cbn.locator = mpl.ticker.FixedLocator(tick_locs)
    cbn.formatter = mpl.ticker.FormatStrFormatter("%d")
    plt.ylabel("N stars")
    cbn.update_ticks()

    return dict(
        hist2d_fig=hist2d_fig,
        hist2d_fig_n100=hist2d_fig_n100,
        mag_vs_resid=mag_resid_fig,
    )


def main(args=None):
    matplotlib.use("Agg")
    opt = get_options().parse_args(args)

    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Copy index to output directory
    shutil.copyfile(
        Path(__file__).parent / "data" / "index_template.html", outdir / "index.html"
    )

    rms_data = Table(get_rms_data())
    # Filter in place to only use default data in plots
    rms_data = rms_data[rms_data["isdefault"] == 1]
    # And filter to only plot "used" star slots
    rms_data = rms_data[(rms_data["used"] == 1) & (rms_data["type"] != "FID")]

    # Lookup the n100 warm fraction for each slot
    # This doesn't take long enough to be worth optimizing (to do the lookups via obsid
    # or save out the values to another table or the like)
    warm_frac = []
    for slot in rms_data:
        frac = get_warm_fracs(100, slot["tstart"], slot["mean_aacccdpt"])
        warm_frac.append(frac)
    rms_data = Table(rms_data)
    rms_data["n100_frac"] = np.array(warm_frac)

    figures = mission_plots(rms_data)
    for fig in figures:
        plt.figure(figures[fig].number)
        plt.savefig(outdir / f"{fig}.png")


if __name__ == "__main__":
    main()
