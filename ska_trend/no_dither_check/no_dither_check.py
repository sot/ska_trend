# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Check for new science observations in the ocat with dither disabled or 0"""

import argparse
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import requests
from astropy.table import Table
from mica.archive.cda import get_ocat_local
from ska_helpers.logging import basic_logger
from ska_helpers.retry import retry_func

from ska_trend import __version__

LOGGER = basic_logger(__name__, level="INFO")

DOC_ID = "1GoYBTIQAv0qq2vh3jYxHBYHfEq2I8LVGMiScDX7OFvw"
GID = "892949670"
url_start = "https://docs.google.com/spreadsheets/d"
GSHEET_URL = f"{url_start}/{DOC_ID}/export?format=csv&id={DOC_ID}&gid={GID}"
GSHEET_USER_URL = f"{url_start}/{DOC_ID}/edit?gid={GID}#gid={GID}"


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
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Root directory for data (trivial for this app)",
    )
    return parser


def get_sheet(data_root) -> Table:
    """
    Get the known dither disabled obsids from the Google sheet.

    Parameters
    ----------
    data_root : str
        The root directory for the data
    Returns
    -------
    dat : astropy.table.Table
        Table of notes
    """
    file = "known_no_dither_obsids.csv"
    LOGGER.info(f"Reading google sheet {GSHEET_URL}")
    req = retry_func(requests.get)(GSHEET_URL, timeout=5)
    if req.status_code != 200:
        LOGGER.error(f"Failed to read {GSHEET_URL} with status code: {req.status_code}")
        if (Path(data_root) / file).exists():
            LOGGER.info(f"Reading local {file} file")
            dat = Table.read(Path(data_root) / file, format="ascii.csv")
    else:
        dat = Table.read(req.text, format="ascii.csv")
        LOGGER.info(f"Writing google sheet to {Path(data_root) / file}")
        # Make sure the data root directory exists
        Path(data_root).mkdir(parents=True, exist_ok=True)
        dat.write(
            Path(data_root) / file,
            format="ascii.csv",
            overwrite=True,
        )
    return dat


def check_for_no_dither(data_root) -> Table:
    """
    Check the ocat for new science observations with dither disabled or set to 0.

    This function gets the science observations at or after proposal cycle 26 from the ocat with
    dither disabled or set to 0, then filters out the known no dither obsids from the google sheet.

    Parameters
    ----------
    data_root : str
        The root directory for the data (used to read/write the known no dither obsids

    Returns
    -------
    no_dither_obs : astropy.table.Table
        Table of obsids with no dither
    """
    ocat = get_ocat_local()
    cycle_start = 26
    # get the no dither science observations at/after proposal cycle 26
    no_dither_obs = ocat[
        (ocat["prop_cycle"] >= str(cycle_start))
        & (ocat["obsid"] < 38000)
        & ((ocat["dither"] == "N") | ((ocat["y_amp"] == 0) | (ocat["z_amp"] == 0)))
    ]
    # get the known no dither obsids from the google sheet
    known_no_dither_obsids = get_sheet(data_root)["obsid"].tolist()

    # filter out the known no dither obsids
    mask = ~np.isin(no_dither_obs["obsid"], known_no_dither_obsids)
    no_dither_obs = no_dither_obs[mask]
    return no_dither_obs


def send_process_email(opt, no_dither_obs):
    subject = "New no-dither science observations found in ocat"
    text = [
        "New no-dither science observations found in ocat: <br></br>",
    ]
    text.extend([f" obsid: {obsid} <br></br>" for obsid in no_dither_obs["obsid"]])
    text.extend([f"<br></br>See {GSHEET_USER_URL} for known no-dither obsids."])
    msg = MIMEText("\n".join(text), "html")
    msg["Subject"] = subject
    me = "aca@cfa.harvard.edu"
    msg["From"] = me
    msg["To"] = ", ".join(opt.emails)
    s = smtplib.SMTP("localhost")
    s.sendmail(me, [opt.emails], msg.as_string())


def main(sys_args=None):
    opt = get_opt().parse_args(sys_args)

    LOGGER.info("Checking for no dither observations")
    no_dither_obs = check_for_no_dither(opt.data_root)

    if len(no_dither_obs) > 0:
        LOGGER.info(f"Sending email alert for {len(no_dither_obs)} obsids")
        send_process_email(opt, no_dither_obs)


if __name__ == "__main__":
    main()
