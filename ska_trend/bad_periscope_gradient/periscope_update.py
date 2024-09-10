# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Check for bad data and discontinuities in periscope gradients"""

import argparse
from pathlib import Path

import numpy as np
from acdc.common import send_mail
from astropy.table import Table, vstack
from cheta import fetch
from cxotime import CxoTime
from kadi import events
from ska_helpers.logging import basic_logger

from ska_trend import __version__

logger = basic_logger(__name__, level="INFO")


def get_opt():
    parser = argparse.ArgumentParser(
        description="Periscope gradient checker {}".format(__version__)
    )
    parser.add_argument(
        "--email",
        type=str,
        default=[],
        action="append",
        dest="emails",
        help="Email address to send alerts",
    )
    parser.add_argument("--out-dir", type=str, default=".", help="Out directory")
    return parser


def check_for_bad_times(start):
    """
    Review perigee gradient data for discontinuities and bad data.

    This uses 0.0032 K as the quantization and checks that the difference between
    consecutive samples are not more than 5 * quantization.  Times with samples
    that exceed this threshold are put into a table along with the telemetered
    COBSRQID, AOACASEQ, and AOPCADMD.  The obsid of the most recent maneuver is
    also captured, as it looks like for science processing the problem can be that
    the gradient data for the science interval extends into the time of an anomaly,
    even though AOACASEQ and AOPCADMD show that the spacecraft is in anomaly state,
    and COBSRQID is often 0.

    Parameters
    ----------
    start : str
        Start time for search

    Returns
    -------
    bad_events : astropy.table.Table
        Table of bad events
    """
    msids = ["OOBAGRD3", "OOBAGRD6"]
    quantization = 0.0032
    sampling = 32.80000185966492
    quant_mult = 5

    bad_events = []
    for msid in msids:
        dat = fetch.Msid(msid, start, filter_bad=True)
        dat.interpolate(dt=sampling)
        diff = np.diff(dat.vals)
        nok = np.abs(diff) > quantization * quant_mult

        for hit in np.where(nok)[0]:
            event = {
                "time": dat.times[hit],
                "date": CxoTime(dat.times[hit]).date,
                "msid": msid,
            }
            local_manvrs = events.manvrs.filter(
                start=dat.times[hit] - (7 * 86400), stop=dat.times[hit] + (1 * 86400)
            )
            extra_msids = ["COBSRQID", "AOPCADMD", "AOACASEQ"]
            extra_defaults = {"COBSRQID": 0, "AOPCADMD": "N/A", "AOACASEQ": "N/A"}
            for extra_msid in extra_msids:
                mdat = fetch.Msid(extra_msid, dat.times[hit] - 200, dat.times[hit] + 50)
                # get last sample before hit
                hit_idx = np.searchsorted(mdat.times, dat.times[hit])
                if hit_idx == 0:
                    event.update({extra_msid: extra_defaults[extra_msid]})
                else:
                    event.update({extra_msid: mdat.vals[hit_idx - 1]})
            last_manvr = None
            for manvr in local_manvrs:
                if manvr.tstart < dat.times[hit]:
                    last_manvr = manvr
            event.update({"manvr_obsid": last_manvr.obsid})
            event.update({"manvr_tstart": last_manvr.tstart})
            event.update({"manvr_tstop": last_manvr.tstop})
            bad_events.append(event.copy())
    return Table(bad_events)


def send_process_email(opt, bad_science_data):
    subject = "perigee gradient data: bad data or discontinuities found"
    lines = [
        "Discontinuities found in perigee gradient data.",
        "Check V&V for these science observations.",
    ]
    lines.extend(bad_science_data.pformat(max_lines=-1, max_width=-1))
    text = "\n".join(lines)
    logger.info(text)
    send_mail(logger, opt, subject, text, __file__)


def main(sys_args=None):
    opt = get_opt().parse_args(sys_args)

    Path(opt.outdir).mkdir(parents=True, exist_ok=True)
    data_file = Path(opt.out_dir) / "data.ecsv"

    start = "2001:001"
    hiccups = None
    if data_file.exists():
        hiccups = Table.read(data_file)
        start = hiccups["time"].max()

    logger.info("Checking for bad data starting at {}".format(start))
    bads = check_for_bad_times(start)

    if len(bads) > 0:
        # If the obsid associated with the last maneuver before an event
        # was a science obsid then send an email alert.
        bad_science = (
            (bads["manvr_obsid"] != -999)
            & (bads["manvr_obsid"] != 0)
            & (bads["manvr_obsid"] < 38000)
        )
        if np.any(bad_science):
            logger.info(f"Sending email alert for {len(bads[bad_science])} obsids")
            send_process_email(opt, bads[bad_science])

        # Append the new bad data to the existing data file
        if hiccups is not None:
            hiccups = vstack([hiccups, bads])
        else:
            hiccups = bads
        hiccups.sort("time")
        logger.info("Writing data to {}".format(data_file))
        hiccups.write(data_file, format="ascii.ecsv", overwrite=True)


if __name__ == "__main__":
    main()
