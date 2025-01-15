#!/usr/bin/env python

import os
import tempfile
from glob import glob
from pathlib import Path

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import Quaternion
import ska_numpy
from astropy.table import Table
from chandra_aca.transform import radec_to_yagzag
from cheta import fetch
from cxotime import CxoTime
from ska_dbi.sqsh import Sqsh
from ska_helpers import logging
from ska_shell import bash, getenv, tcsh_shell

CIAO_ENV = getenv("source /soft/ciao/bin/ciao.csh", shell="tcsh")

TASK = "periscope_drift_reports"
TASK_SHARE = Path(os.environ["SKA"]) / "data" / TASK


JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(
        Path(__file__).parent / "templates" / "periscope_drift"
    )
)


logger = logging.basic_logger(TASK, level="INFO", format="%(message)s")


SYBASE_ACA = Sqsh(
    dbi="sybase", server="sybase", user="aca_read", database="aca", numpy=True
)


AXAF_PSTAT = Sqsh(dbi="sybase", server="sqlocc", user="aca_ops", database="axafapstat")


GRADIENTS = {
    "OOBAGRD3": {
        "yag": 6.98145650e-04,
        "zag": 9.51578351e-05,
    },
    "OOBAGRD6": {
        "yag": -1.67009240e-03,
        "zag": -2.79084775e-03,
    },
}


class ObiError(Exception):
    """
    Error for cases when no acquisition stars are found.
    """


class NoDataError(Exception):
    """
    Error for cases when no data is found.
    """


class WeirdDataError(Exception):
    """
    Error for cases when the data found makes no sense.
    """


def expected_corr(obs):
    gradients = fetch.MSIDset(
        ["OOBAGRD3", "OOBAGRD6"],
        obs["kalman_datestart"],
        obs["kalman_datestop"],
    )
    corr = {
        "ang_y_corr": np.zeros_like(gradients["OOBAGRD6"].vals),
        "ang_z_corr": np.zeros_like(gradients["OOBAGRD6"].vals),
    }

    for msid in gradients:
        filter_bad_telem(gradients[msid])
        # find a mean gradient, because this calibration is relative to mean
        mean_gradient = np.mean(gradients[msid].vals)
        # and smooth the telemetry to deal with slow changes and large step sizes...
        smooth_gradient = smooth(gradients[msid].vals, window_len=152)
        # Make the actual centroid corrections for the y and z axes
        corr["ang_y_corr"] -= (smooth_gradient - mean_gradient) * GRADIENTS[msid]["yag"]
        corr["ang_z_corr"] -= (smooth_gradient - mean_gradient) * GRADIENTS[msid]["zag"]

    corr["times"] = gradients["OOBAGRD6"].times
    corr["ang_y_corr"] *= 3600
    corr["ang_z_corr"] *= 3600

    return corr


def make_plots(obsdata, outdir="plots", subdir=""):
    outdir = Path(outdir)
    subdir = Path(subdir)
    (outdir / subdir).mkdir(exist_ok=True, parents=True)

    figsize = (5, 3.5)
    h = plt.figure(figsize=figsize)
    default_max = 1.0
    plot_max = np.max([np.max(obsdata["drift"]), default_max])
    plt.hist(obsdata["drift"], bins=np.arange(0, plot_max + 0.05, 0.05), log=True)

    plt.xlabel("Expected Drift (arcsec)")
    plt.ylabel("N observations")
    plt.title("Expected Periscope Drift/Correction")
    plt.subplots_adjust(bottom=0.17)
    plt.ylim(ymin=0.5)
    plt.savefig(outdir / subdir/ "drift_histogram.png")
    plt.close(h)

    largest = obsdata[obsdata["drift"] == np.max(obsdata["drift"])][0]
    corr = expected_corr(largest)
    for corr_dir in ("ang_y_corr", "ang_z_corr"):
        h = plt.figure(figsize=figsize)
        plt.plot((corr["times"] - corr["times"][0]) / 1000.0, corr[corr_dir])
        plt.title("Obsid %d %s" % (largest["obsid"], corr_dir))
        plt.xlabel("Obs Time (ksec)")
        plt.ylabel("Expected Corr (arcsec)")
        plt.grid()
        plt.subplots_adjust(left=0.17, bottom=0.16)
        plt.savefig(outdir / subdir / f"large_drift_{corr_dir}.png")
        plt.close(h)

    plots = {
        "drift_histogram": {
            "filename": str(subdir / "drift_histogram.png"),
        },
        "large_drift_ang_y_corr": {
            "filename": str(subdir / "large_drift_ang_y_corr.png"),
        },
        "large_drift_ang_z_corr": {
            "filename": str(subdir / "large_drift_ang_z_corr.png"),
        },
    }

    return plots


def filter_bad_telem(msid, method="nearest"):
    # use the bad quality field to select
    # and replace bad data in place using the given method
    ok = ~msid.bads
    bad = msid.bads
    fix_vals = ska_numpy.interpolate(
        msid.vals[ok], msid.times[ok], msid.times[bad], method=method
    )
    msid.vals[bad] = fix_vals
    return msid


def smooth(x, window_len=10, window="hanning"):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Example::

      t = linspace(-2, 2, 50)
      y = sin(t) + randn(len(t)) * 0.1
      ys = ska_numpy.smooth(y)
      plot(t, y, t, ys)

    See also::

      numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
      scipy.signal.lfilter

    :param x: input signal
    :param window_len: dimension of the smoothing window
    :param window: type of window ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')

    :rtype: smoothed signal
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    # if x.size < window_len:
    #    raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    ones = np.ones(window_len)
    s = np.r_[
        ones[: window_len - len(x)] * x[-1],
        x[window_len - 1 : 0 : -1],
        x,
        x[-1:-window_len:-1],
        ones[: window_len - (len(x) + 1)] * x[0],
    ]
    # s=numpy.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]

    # Moving average
    if window == "flat":
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1 : -window_len + 1]


def make_html(data, outdir):
    """
    Render and write the basic page.

    Render and write the basic page, where nav_dict is a dictionary of the
    navigation elements (locations of UP_TO_MAIN, NEXT, PREV), rep_dict is
    a dictionary of the main data elements (n failures etc), fail_dict
    contains the elements required for the extra table of failures at the
    bottom of the page, and outdir is the destination directory.
    """

    template = JINJA_ENV.get_template("index.html")
    page = template.render(time_ranges=data)
    f = open(outdir / "index.html", "w")
    f.write(page)
    f.close()


TILT_QUERY = """select
p.obsid,p.obi, p.max_oobagrd3, p.min_oobagrd3, p.min_oobagrd6, p.max_oobagrd6,
readmode, grating, instrume, detector, kalman_datestart,kalman_datestop, ra_nom, dec_nom, roll_nom,
power(power((max_oobagrd6 - min_oobagrd6)*%(ycoeff_oobagrd6)f
    + (max_oobagrd3 - min_oobagrd3)*%(ycoeff_oobagrd3)f ,2)
+ power((max_oobagrd6 - min_oobagrd6)*%(zcoeff_oobagrd6)f
    + (max_oobagrd3 - min_oobagrd3)*%(zcoeff_oobagrd3)f ,2),.5) * 3600
    as drift
from obs_periscope_tilt p left join observations o on p.obsid = o.obsid and p.obi = o.obi
where kalman_datestart >= '%(datestart)s' and kalman_datestart < '%(datestop)s'
"""


def obs_info(tname, datestart, datestop):
    datestart = CxoTime(datestart)
    datestop = CxoTime(datestop)

    # coeffs in oobagrd -> degrees
    coeffs = {
        "ycoeff_oobagrd3": GRADIENTS["OOBAGRD3"]["yag"],
        "ycoeff_oobagrd6": GRADIENTS["OOBAGRD6"]["yag"],
        "zcoeff_oobagrd3": GRADIENTS["OOBAGRD3"]["zag"],
        "zcoeff_oobagrd6": GRADIENTS["OOBAGRD6"]["zag"],
    }

    query_dict = coeffs.copy()
    rep = {
        "name": tname,
        "datestart": datestart.date,
        "datestop": datestop.date,
        "human_date_start": datestart.datetime.strftime("%Y-%b-%d"),
        "human_date_stop": datestop.datetime.strftime("%Y-%b-%d"),
    }

    query_dict["datestart"] = rep["datestart"]
    query_dict["datestop"] = rep["datestop"]

    tilt = SYBASE_ACA.fetchall(TILT_QUERY % query_dict)
    if len(tilt) == 0:
        rep["mean_drift"] = 0
        rep["max_drift"] = 0
        rep["n_obs"] = 0
        rep["max_drift_obs"] = 0
        rep["n_obs_drift"] = 0
        return rep, tilt

    rep["mean_drift"] = float(np.mean(tilt["drift"]))
    rep["max_drift"] = float(np.max(tilt["drift"]))
    rep["n_obs"] = len(tilt)
    rep["max_drift_obs"] = int(tilt["obsid"][tilt["drift"] == rep["max_drift"]][0])
    threshold = 0.1
    rep["n_obs_drift"] = len(np.flatnonzero(tilt["drift"] > threshold))

    return rep, tilt


def get_sources(obs, srcfiles):
    if len(srcfiles) == 0:
        return {}
    maxsrc = None
    for srcfile in srcfiles:
        try:
            srctable = Table.read(srcfile)
            if len(srctable):
                maxcsrc = srctable[np.argmax(srctable["NET_COUNTS"])]
                if maxsrc is None or maxcsrc["NET_COUNTS"] >= maxsrc["NET_COUNTS"]:
                    maxsrc = maxcsrc
        except Exception as exc:
            logger.debug(f"Cannot get sources in {srcfile}: {exc}")
            continue
    if maxsrc is None:
        return {}

    srcdict = {
        "obsid": obs["obsid"],
        "obi": obs["obi"],
        "ra": maxsrc["RA"],
        "dec": maxsrc["DEC"],
        "x": maxsrc["X"],
        "y": maxsrc["Y"],
        "net_counts": maxsrc["NET_COUNTS"],
        "snr": maxsrc["SNR"],
        "detsize": maxsrc["DETSIZE"],
    }

    return srcdict


def find_obsid_src(obsid):
    obs = SYBASE_ACA.fetchone("select * from observations where obsid = %d" % obsid)

    xray_data = TASK_SHARE / "xray_data"
    xray_data.mkdir(exist_ok=True, parents=True)
    obsdir = xray_data / f"obs{obsid}"
    obsdir.mkdir(exist_ok=True, parents=True)

    srcfiles = list(obsdir.glob("*src2*"))
    if len(srcfiles):
        src = get_sources(obs, srcfiles)
        return obs, src

    tempdir = tempfile.mkdtemp()
    # print(tempdir)
    det = (
        "hrc"
        if (obs["detector"] == "HRC-I") or (obs["detector"] == "HRC-S")
        else "acis"
    )

    if obs["detector"] == "HRC-I":
        raise NoDataError("Skip HRC-I observations")
    bash(
        'echo "cd %s\n obsid=%d\n get %s2{src}\n" | arc5gl -stdin'
        % (obsdir, obs["obsid"], det)
    )
    srcfiles = list(obsdir.glob("*src2*"))
    if len(srcfiles):
        src = get_sources(obs, srcfiles)
        return obs, src
    else:
        # no src2, make one
        bash(
            'echo "cd %s\n obsid=%d\n get %s2{evt2}\n" | arc5gl -stdin'
            % (tempdir, obs["obsid"], det)
        )
        event_files = glob("%s/*evt2*gz" % tempdir)
        if not len(event_files):
            raise NoDataError("No evt2 files")
        if len(event_files) > 1:
            raise WeirdDataError("More than 1 evt2 file")
        bash("gunzip %s" % event_files[0])
        event_files = glob("%s/*evt2*" % tempdir)
        pfiles = ";".join([tempdir, CIAO_ENV["PFILES"].split(";")[-1]])
        outlines1, stat = tcsh_shell(
            f"env PFILES={pfiles} punlearn celldetect", env=CIAO_ENV
        )
        envstr = f'env LD_LIBRARY_PATH="" PFILES="{pfiles}"'
        if obs["detector"] == "HRC-S":
            binstr = "[bin x=31744.5:33792.5:1,y=31744.5:33792.5:1]"
        else:
            binstr = "[bin x=3072.5:5120.5:1,y=3072.5:5120.5:1]"
        mkimage = "{} dmcopy infile='{}{}' outfile='{}/{}_center_img.fits'".format(
            envstr, event_files[0], binstr, obsdir, obs["obsid"]
        )
        print(mkimage)
        outlines2, stat = tcsh_shell(mkimage, env=CIAO_ENV)
        cmd = '{} {} infile="{}/{}_center_img.fits" outfile="{}/{}_src2.fits"'.format(
            envstr,
            "celldetect fixedcell=9 maxlogicalwindow=2048 ",
            obsdir,
            obs["obsid"],
            obsdir,
            obs["obsid"],
        )
        print(cmd)
        outlines3, stat = tcsh_shell(cmd, env=CIAO_ENV)
        srcfiles = list(obsdir.glob("*src2*"))
        print(srcfiles)
        if len(srcfiles):
            src = get_sources(obs, srcfiles)
            print(f"Deleting {event_files[0]}")
            os.unlink(event_files[0])
            return obs, src
        else:
            raise ValueError("No src2 file was made")


APSTAT_QUERY = """select
t.obsid, j.obi, j.ap_date, j.ascdsver, j.ap_date_diff, t.ocat_status from target_info t
left join
(select s.obsid, o.obi, s.revision as s2rev, aspect_1.ascdsver, aspect_1.ap_date,
datediff(day, '%s', aspect_1.ap_date) ap_date_diff,
s.science_2_id, science_1.science_1_id, aspect_1.aspect_1_id
from (select obsid, max(revision) as maxrev from science_2 s
where ingested = 'Y' and ap_status = 'DONE' and quality != 'P' and quality != 'R'
group by obsid) as x
inner join science_2 as s on s.obsid = x.obsid and s.revision = x.maxrev
join science_2_obi o on s.science_2_id = o.science_2_id
join science_1 on o.science_1_id = science_1.science_1_id
join aspect_1 on science_1.aspect_1_id = aspect_1.aspect_1_id) as j on t.obsid = j.obsid
where t.obsid = %d order by obsid
"""


def is_corrected(obsid):
    pipeline_corr_date = "Jun 29 2011  00:00:00:000AM"
    apstat_query = APSTAT_QUERY % (pipeline_corr_date, obsid)

    apstat = AXAF_PSTAT.fetchone(apstat_query)
    if apstat["ap_date_diff"] > 0:
        return True, apstat["ascdsver"]
    else:
        return False, apstat["ascdsver"]


def xray_plots(tilt, outdir, subdir=""):  # noqa: PLR0912, PLR0915
    subdir=Path(subdir)
    figsize = (5, 3.5)
    min_detsize = {"HRC": 40, "ACIS": 10}

    # sort in desc by drift
    tilt = np.sort(tilt, order=["drift"])[::-1]
    for tobs in tilt:
        print("Checking for sources for %d" % tobs["obsid"])
        if tobs["readmode"] != "CONTINUOUS":
            try:
                obs, src = find_obsid_src(tobs["obsid"])
            except (WeirdDataError, NoDataError):
                continue
            print(src)
            if "net_counts" in src:
                if (src["net_counts"] > 2500) & (
                    src["detsize"] <= min_detsize[tobs["instrume"]]
                ):
                    break

    print("Using obsid %d for X-ray plots" % tobs["obsid"])
    xray_data = TASK_SHARE / "xray_data"
    xray_data.mkdir(exist_ok=True, parents=True)
    obsdir = xray_data / f"obs{tobs['obsid']:d}"
    obsdir.mkdir(exist_ok=True, parents=True)

    (outdir / subdir).mkdir(exist_ok=True, parents=True)

    point = obsdir / "point_source.fits"
    if not point.exists():
        tempdir = tempfile.mkdtemp()
        det = "acis"
        if (tobs["detector"] == "HRC-I") or (tobs["detector"] == "HRC-S"):
            det = "hrc"
        bash(
            'echo "cd %s\n obsid=%d\n get %s2{evt2}\n" | arc5gl -stdin'
            % (tempdir, tobs["obsid"], det)
        )

        c = open(obsdir / "center.reg", "w")
        regstring = "circle(%f,%f,%f)" % (src["x"], src["y"], src["detsize"])
        c.write("%s\n" % regstring)
        c.close()

        try:
            evt2 = glob("%s/*evt2.fits*" % tempdir)[0]
            # this unused variable causes ruff to complain, it was in the original, is it a bug?
            # reg = obsdir / "center.reg"
            dmstring = "[cols time,ra,dec,x,y]"
            if det == "acis":
                dmstring = dmstring + "[energy=300:7000]"
            print(
                "env LD_LIBRARY_PATH='' dmcopy %s'[(x,y)=%s]%s' %s"
                % (evt2, regstring, dmstring, point)
            )
            outlines, status = tcsh_shell(
                "env LD_LIBRARY_PATH='' dmcopy %s'[(x,y)=%s]%s' %s"
                % (evt2, regstring, dmstring, point),
                env=CIAO_ENV,
            )
            if not status:
                os.unlink(evt2)
        except IndexError:
            pass

    # position data
    evts = Table.read(point)

    q = Quaternion.Quat([tobs["ra_nom"], tobs["dec_nom"], tobs["roll_nom"]])
    yag = []
    zag = []

    corr = expected_corr(tobs)

    poscorr = {
        "yag": ska_numpy.interpolate(corr["ang_y_corr"], corr["times"], evts["time"]),
        "zag": ska_numpy.interpolate(corr["ang_z_corr"], corr["times"], evts["time"]),
    }

    yag, zag = radec_to_yagzag(evts["RA"], evts["DEC"], q)

    pos = {"time": evts["time"], "yag": np.array(yag), "zag": np.array(zag)}

    corrected, ascdsver = is_corrected(tobs["obsid"])

    src_plot = {"obsid": int(tobs["obsid"])}

    if corrected:
        src_plot["label"] = (
            """Obsid %d has been corrected for periscope drift in processing (DS %s)."""
            % (tobs["obsid"], ascdsver)
        )
    else:
        src_plot["label"] = (
            """Obsid %d has NOT been corrected for periscope drift in processing (DS %s)."""
            % (tobs["obsid"], ascdsver)
        )

    evtmarksize = 1
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(1, 1, 1, aspect="equal")
    ax.plot(pos["zag"], pos["yag"], "b.", markersize=evtmarksize)
    plt.title("Obsid %d Evts" % (tobs["obsid"]))
    plt.xlabel("%s (arcsec)" % "zag", fontsize=12)
    plt.ylabel("%s (arcsec)" % "yag", fontsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_size("small") for label in labels]
    plt.subplots_adjust(bottom=0.15, left=0.15)
    pict_name = "%d_real.png" % (tobs["obsid"])
    plt.savefig(outdir / subdir / pict_name)
    plt.close(f)
    src_plot["pict"] = str(subdir / pict_name)

    for pax in ["yag", "zag"]:
        x = (pos["time"] - pos["time"][0]) / 1000
        y = pos[pax]
        (m, b) = np.polyfit(x, y, 1)
        f = plt.figure(figsize=figsize)
        plt.plot(x, y, "b.", markersize=evtmarksize)
        plt.plot(x, m * x + b, "k-", linewidth=2.5)
        plt.grid()
        plt.title("Obsid %d %s Evts" % (tobs["obsid"], pax))
        plt.xlabel("Time (ksec)")
        plt.ylabel("%s (arcsec)" % pax)
        plt.subplots_adjust(bottom=0.15, left=0.2)
        plot_name = "%d_%s_real.png" % (tobs["obsid"], pax)
        plt.savefig(outdir / subdir / plot_name)
        plt.close(f)
        src_plot["%s_plot" % pax] = str(subdir / plot_name)

        f = plt.figure(figsize=figsize)
        plt.plot(x, poscorr[pax], "b.")
        plt.grid()
        plt.title("Periscope Correction Obsid %d %s" % (tobs["obsid"], pax))
        plt.xlabel("Time (ksec)")
        plt.ylabel("%s (arcsec)" % pax)
        plt.subplots_adjust(bottom=0.15, left=0.2)
        corr_plot = "%d_%s_corr.png" % (tobs["obsid"], pax)
        plt.savefig(outdir / subdir / corr_plot)
        plt.close(f)
        src_plot["%s_corr_plot" % pax] = str(subdir / corr_plot)

    return src_plot


def process_interval(start, stop, name, output=None):
    logger.debug("Attempting to update %s" % name)
    logger.debug(f"Output directory: {output}")

    datestart = CxoTime(start)
    datestop = CxoTime(stop)

    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)

    rep, obsdata = obs_info(name, datestart, datestop)

    rep["src_plot"] = xray_plots(obsdata, outdir=output, subdir=rep["name"])
    rep["plots"] = make_plots(obsdata, outdir=output, subdir=rep["name"])

    if output is not None:
        make_html([rep], outdir=output)

    return rep
