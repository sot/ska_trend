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

from ska_trend.periscope_drift import processing, reports

logger = logging.getLogger("periscope_drift")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=Path(os.environ["SKA"]) / "www" / "ASPECT" / "periscope_drift",
        type=Path,
        help="Output directory",
    )
    parser.add_argument("--start", default="-365d")
    parser.add_argument("--stop", default=None)
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Working directory",
    )
    parser.add_argument(
        "--archive-dir",
        default=Path("/data/aca/periscope_drift/data"),
        type=Path,
        help="Archive directory",
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
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bar",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write output (reports and JSON files)",
    )
    return parser


def main():
    args = get_parser().parse_args()

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
        start = stop + float(args.start[:-1]) * u.d
    else:
        start = CxoTime(args.start)

    observations, sources, errors = processing.process_interval(
        start,
        stop,
        archive_dir=args.archive_dir,
        workdir=args.workdir,
        log_level=args.log_level,
        show_progress=args.show_progress,
    )

    if args.workdir is not None:
        with open(args.workdir / "errors.json", "w") as fh:
            json.dump(errors, fh)

    if not args.no_output:
        time_ranges = [
            {"start": stop - 30 * u.day, "stop": stop, "title": "30 days"},
            {"start": stop - 90 * u.day, "stop": stop, "title": "90 days"},
            {"start": stop - 180 * u.day, "stop": stop, "title": "180 days"},
            {"start": stop - 365 * u.day, "stop": stop, "title": "1 year"},
            {"start": stop - 5 * 365 * u.day, "stop": stop, "title": "5 year"},
        ]
        # exclude time ranges that do not add any data
        time_ranges = [
            time_ranges[idx] for idx in range(1, len(time_ranges))
            if (time_ranges[idx - 1]["start"] > start)
        ]

        reports.write_html_report(time_ranges, args.output, observations, sources)

        with open(args.output / "sources.json", "w") as fh:
            json.dump(sources.to_pandas().to_json(), fh)


if __name__ == "__main__":
    main()
