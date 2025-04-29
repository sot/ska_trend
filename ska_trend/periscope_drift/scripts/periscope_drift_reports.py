#!/usr/bin/env python

import argparse
import os
import re
import sys
from pathlib import Path

import ska_helpers
from cxotime import CxoTime
from cxotime import units as u

from ska_trend.periscope_drift import processing, reports

logger = ska_helpers.logging.basic_logger(
    "periscope_drift_reports", level="WARNING", format="%(message)s"
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=Path(os.environ["SKA"]) / "www" / "ASPECT" / "periscope_drift",
        type=Path,
        help="Output directory",
    )
    parser.add_argument("--start", default="-30d")
    parser.add_argument("--stop", default=None)
    parser.add_argument(
        "--workdir",
        default=Path("/data/aca/periscope_drift/work"),
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
    return parser


def main():
    args = get_parser().parse_args()

    if args.log_file is not None:
        filename = args.log_file
        stream = None
    else:
        filename = None
        stream = sys.stdout

    logger = ska_helpers.logging.basic_logger(
        "astromon",
        level=args.log_level.upper(),
        format="%(message)s",
        stream=stream,
    )

    logger = ska_helpers.logging.basic_logger(
        "periscope_drift_reports",
        level=args.log_level.upper(),
        format="%(message)s",
        stream=stream,
    )

    now = CxoTime()
    logger.info(
        "---------- periscope drift reports update at %s ----------" % (now.iso)
    )

    stop = now if args.stop is None else CxoTime(args.stop)

    if re.match(r"[-+]? [0-9]* \.? [0-9]+ d", args.start, re.VERBOSE):
        start = stop + float(args.start[:-1]) * u.d
    else:
        start = CxoTime(start)

    observations, sources, errors = processing.process_interval(
        start,
        stop,
        archive_dir=args.archive_dir,
        workdir=args.workdir,
        log_level=args.log_level,
        show_progress=False,
    )

    reports.write_html_report(args.output, observations, sources)

    import json
    with open(args.output / "sources.json", "w") as fh:
        json.dump(sources.to_pandas().to_json(), fh)

    with open(args.workdir / "errors.json", "w") as fh:
        json.dump(errors, fh)


if __name__ == "__main__":
    main()
