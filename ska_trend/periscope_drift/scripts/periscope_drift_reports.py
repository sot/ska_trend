#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path

import matplotlib
from cxotime import CxoTime
from cxotime import units as u

from ska_trend.periscope_drift.reports import TASK, logger, make_html, process_interval


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=Path(os.environ["SKA"]) / "www" / "ASPECT" / TASK,
        type=Path,
        help="Output directory",
    )
    parser.add_argument("--stop", default=None)
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
        ],
        help="Verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser


def main():
    matplotlib.use("Agg")

    opt = get_parser().parse_args()
    logger.setLevel(opt.log_level.upper())

    now = CxoTime()
    logger.info(
        "---------- periscope drift reports update at %s ----------" % (now.iso)
    )

    stop = now if opt.stop is None else CxoTime(opt.stop)

    time_ranges = [
        {
            "name": "month",
            "start": stop - 30 * u.day,
            "stop": stop,
        },
        {
            "name": "quarter",
            "start": stop - 90 * u.day,
            "stop": stop,
        },
        {
            "name": "half-year",
            "start": stop - 182 * u.day,
            "stop": stop,
        },
        {
            "name": "year",
            "start": stop - 365 * u.day,
            "stop": stop,
        },
    ]

    data = []
    for tr in time_ranges:
        logger.debug("Attempting to update %s" % tr["name"])

        rep = process_interval(
            start=tr["start"],
            stop=tr["stop"],
            name=tr["name"],
            output=opt.output,
        )
        data.append(rep)

    make_html(data, outdir=opt.output)

    with open(opt.output / "data.json", "w") as rep_file:
        rep_file.write(json.dumps(data, sort_keys=True, indent=2))



if __name__ == "__main__":
    main()
