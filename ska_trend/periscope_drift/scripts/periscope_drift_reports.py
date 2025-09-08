#!/usr/bin/env python

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ska_helpers
from astropy.table import Table, vstack
from cxotime import CxoTime
from cxotime import units as u

from ska_trend.periscope_drift import observation, processing, reports

logger = logging.getLogger("periscope_drift")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=Path(os.environ["SKA"]) / "www" / "ASPECT" / "periscope_drift",
        type=Path,
        help="Output directory",
    )
    parser.add_argument("--start", default="-60d", help="Start of processing interval")
    parser.add_argument("--stop", default=None)
    parser.add_argument(
        "--start-report", default="-1825d", help="Start of report interval"
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

    if re.match(r"[-+]? [0-9]* \.? [0-9]+ d", args.start_report, re.VERBOSE):
        start_report = stop + float(args.start_report[:-1]) * u.d
    else:
        start_report = CxoTime(args.start_report)

    if re.match(r"[-+]? [0-9]* \.? [0-9]+ d", args.start, re.VERBOSE):
        start_process = stop + float(args.start[:-1]) * u.d
    else:
        start_process = CxoTime(args.start)

    errors = processing.process_interval(
        start_process,
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
            time_ranges[idx]
            for idx in range(len(time_ranges))
            if idx == 0 or (time_ranges[idx - 1]["start"] > start_report)
        ]

        all_sources = processing.get_sources()

        report_sources = all_sources[
            np.in1d(all_sources["obsid"], processing.get_obsids(start_report, stop))
        ]

        report_observations = {
            str(obsid): observation.Observation(
                obsid, workdir=args.workdir, archive_dir=args.archive_dir
            )
            for obsid in np.unique(report_sources["obsid"])
        }

        reports.write_html_report(
            time_ranges, args.output, report_observations, report_sources
        )

        with open(args.output / "errors.json", "w") as fh:
            json.dump(errors, fh)

        with open(args.output / "sources.json", "w") as fh:
            fh.write(all_sources.to_pandas().to_json())

if __name__ == "__main__":
    main()
