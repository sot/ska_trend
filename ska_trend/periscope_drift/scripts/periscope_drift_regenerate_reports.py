#!/usr/bin/env python

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import ska_helpers
from cxotime import CxoTime
from cxotime import units as u

from ska_trend.periscope_drift import reports

logger = logging.getLogger("periscope_drift")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=Path(os.environ["SKA"]) / "www" / "ASPECT" / "periscope_drift",
        type=Path,
        help="Output directory",
    )
    parser.add_argument("--start", default="-1825d", help="Start of report interval")
    parser.add_argument("--stop", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output directory",
        default=False,
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Working directory (the default is a temporary directory)",
    )
    parser.add_argument(
        "--archive-dir",
        default=Path(os.environ["SKA"]) / "data" / "astromon" / "archive",
        type=Path,
        help="Astromon archive directory",
    )
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
    parser.add_argument(
        "--log-file",
        default=None,
        type=Path,
        help="Log file. If not specified, log to stdout.",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(Path(args.output) / "args.json", "r") as fh:
        previous_args = json.load(fh)
        parser.set_defaults(
            start=previous_args["start"],
            stop=previous_args["stop"],
            output=previous_args["output_dir"],
            archive_dir=previous_args["archive_dir"],
        )

    log_args = {
        "level": args.log_level.upper(),
        "format": "%(message)s",
    }
    log_args.update(
        {"stream": sys.stdout} if args.log_file is None else {"filename": args.log_file}
    )
    logger = ska_helpers.logging.basic_logger("periscope_drift", **log_args)
    ska_helpers.logging.basic_logger("astromon", **log_args)

    now = CxoTime()
    logger.info(
        "---------- periscope drift reports update at %s ----------" % (now.iso)
    )

    stop = now if args.stop is None else CxoTime(args.stop)

    if re.match(r"[-+]? [0-9]* \.? [0-9]+ d", args.start, re.VERBOSE):
        start_report = stop + float(args.start[:-1]) * u.d
    else:
        start_report = CxoTime(args.start)

    reports.write_report(
        start=start_report,
        stop=stop,
        output_dir=args.output,
        archive_dir=args.archive_dir,
        workdir=args.workdir,
        overwrite=args.overwrite,
        show_progress=True,
    )


if __name__ == "__main__":
    main()
