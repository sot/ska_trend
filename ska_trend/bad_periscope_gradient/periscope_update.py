# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Check for bad data and discontinuities in periscope gradients"""

import argparse
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
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
    Review periscope gradient data for discontinuities and bad data.

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
            hit_time = dat.times[hit]
            event = {
                "time": hit_time,
                "date": CxoTime(hit_time).date,
                "msid": msid,
            }

            # There is value in getting the telemetered obsid and the state of
            # AOPCADMD and AOACASEQ at the time of the discontiuity/hit.  However,
            # for the times when there are discontiuities in OOBAGRD3 and OOBAGRD6
            # there is often missing data / gaps in telemetry at safe modes.
            # So fetch these from cheta but use trivial defaults in the output table
            # if the data is missing.
            extra_msids = ["COBSRQID", "AOPCADMD", "AOACASEQ"]
            extra_defaults = {"COBSRQID": 0, "AOPCADMD": "N/A", "AOACASEQ": "N/A"}
            for extra_msid in extra_msids:
                mdat = fetch.Msid(extra_msid, hit_time - 200, hit_time + 50)

                # Get last msid sample before discontinuity/hit
                hit_idx = np.searchsorted(mdat.times, hit_time)

                # If the telemetry doesn't cover the hit_time or is empty,
                # searchsorted returns 0.  Use the defaults in those cases.
                if hit_idx == 0:
                    event.update({extra_msid: extra_defaults[extra_msid]})
                else:
                    event.update({extra_msid: mdat.vals[hit_idx - 1]})

            # The last nominal obsid for the maneuver before a discontiuity
            # is the obsid that should be checked in V&V if there is a problem,
            # so fetch this from kadi manvrs. There should be an easier way
            # to get the last one before an event, but, again keeping in mind
            # that there are often gaps in telemetry at safe modes, fetches
            # 7 days before the event and 1 day after the event and tries to
            # get the last maneuver that started before the event.
            local_manvrs = events.manvrs.filter(
                start=hit_time - (7 * 86400), stop=hit_time + (1 * 86400)
            )
            last_manvr = None
            for manvr in local_manvrs:
                if manvr.tstart < hit_time:
                    last_manvr = manvr
            event.update({"manvr_obsid": last_manvr.obsid})
            event.update({"manvr_tstart": last_manvr.tstart})
            event.update({"manvr_tstop": last_manvr.tstop})

            bad_events.append(event.copy())
    return Table(bad_events)


def send_process_email(opt, bad_science_data):
    subject = "periscope gradient data: bad data or discontinuities found"
    cols = ["date", "msid", "AOPCADMD", "AOACASEQ", "COBSRQID", "manvr_obsid"]
    data_html = bad_science_data[cols].pformat(max_lines=-1, max_width=-1, html=True)
    text = [
        "Discontinuities found in periscope gradient data."
        "Check V&V for these science observations."
    ]
    text.extend(data_html)
    msg = MIMEText("\n".join(text), "html")
    msg["Subject"] = subject
    me = "aca@cfa.harvard.edu"
    msg["From"] = me
    msg["To"] = ", ".join(opt.emails)
    s = smtplib.SMTP("localhost")
    s.sendmail(me, [opt.emails], msg.as_string())


def main(sys_args=None):
    opt = get_opt().parse_args(sys_args)

    Path(opt.out_dir).mkdir(parents=True, exist_ok=True)
    data_file = Path(opt.out_dir) / "data.ecsv"

    start = "2001:001"
    hiccups = None
    if data_file.exists():
        hiccups = Table.read(data_file)
        start = hiccups["time"].max()

    logger.info("Checking for bad data starting at {}".format(start))
    bads = check_for_bad_times(start)

    if len(bads) > 0:
        bads.sort("time")

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
