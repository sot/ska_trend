# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Check for new science observations in the ocat with dither disabled or 0"""

import argparse
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from cxotime import CxoTime
from mica.archive.cda import get_ocat_local
from ska_helpers.logging import basic_logger

from ska_trend import __version__

logger = basic_logger(__name__, level="INFO")


def get_opt():
    parser = argparse.ArgumentParser(
        description="No dither check {}".format(__version__)
    )
    parser.add_argument(
        "--email",
        type=str,
        default=[],
        action="append",
        dest="emails",
        help="Email address to send alerts",
    )
    return parser


def check_for_no_dither():
    """
    Check the ocat for science observations with dither disabled or set to 0.

    Returns
    -------
    no_dither_obs : astropy.table.Table
        Table of obsids with no dither
    """
    ocat = get_ocat_local()
    cycle_start = 27
    # get the no dither science observations at/after proposal cycle 27
    no_dither_obs = ocat[(ocat["prop_cycle"] >= str(cycle_start))
                    & (ocat["obsid"] < 38000)
                    & ((ocat["dither"] == "N")
                    | ((ocat["y_amp"] == 0) | (ocat["z_amp"] == 0)))]
    return no_dither_obs


def send_process_email(opt, no_dither_obs):
    subject = "No dither science observations found in ocat for prop cycle >= 27"
    text = [
        "No dither science observations found in ocat for prop cycle >= 27: <br></br>",
    ]
    for obsid in no_dither_obs["obsid"]:
        text.append(f" obsid: {obsid} <br></br>")
    msg = MIMEText("\n".join(text), "html")
    msg["Subject"] = subject
    me = "aca@cfa.harvard.edu"
    msg["From"] = me
    msg["To"] = ", ".join(opt.emails)
    s = smtplib.SMTP("localhost")
    s.sendmail(me, [opt.emails], msg.as_string())


def main(sys_args=None):
    opt = get_opt().parse_args(sys_args)

    logger.info("Checking for no dither observations")
    no_dither_obs = check_for_no_dither()

    if len(no_dither_obs) > 0:

        logger.info(f"Sending email alert for {len(no_dither_obs)} obsids")
        send_process_email(opt, no_dither_obs)


if __name__ == "__main__":
    main()
