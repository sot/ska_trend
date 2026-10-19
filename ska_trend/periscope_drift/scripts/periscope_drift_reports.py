#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import ska_helpers
from cxotime import CxoTime

from ska_trend.periscope_drift import processing, reports
from ska_trend.time_utils import parse_time_arg

logger = logging.getLogger("periscope_drift")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=Path(os.environ["SKA"]) / "www" / "ASPECT" / "periscope_drift",
        type=Path,
        help="Report output directory. Default: $SKA/www/ASPECT/periscope_drift",
    )
    parser.add_argument(
        "--start",
        default="stop-60d",
        help="Start of processing interval (e.g. stop-60d or a date). Default: stop-60d",
    )
    parser.add_argument("--stop", default=None)
    parser.add_argument(
        "--start-report",
        default="stop-1825d",
        help="Start of report interval (e.g. stop-1825d or a date). Default: stop-1825d",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Working directory (the default is a temporary directory)",
    )
    parser.add_argument(
        "--astromon-archive-dir",
        default=Path(os.environ["SKA"]) / "data" / "astromon" / "xray_observations",
        type=Path,
        help="Astromon archive directory. Default: $SKA/data/astromon/xray_observations",
    )
    parser.add_argument(
        "--archive-dir",
        default=Path(os.environ["SKA"])
        / "data"
        / "periscope_drift"
        / "xray_observations",
        type=Path,
        help="Astromon archive directory. Default: $SKA/data/periscope_drift/xray_observations",
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
        help="Verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: DEBUG.",
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

    start_report = parse_time_arg(args.start_report, stop=stop)
    start_process = parse_time_arg(args.start, stop=stop)

    errors = processing.process_interval(
        start_process,
        stop,
        archive_dir=args.archive_dir,
        astromon_archive_dir=args.astromon_archive_dir,
        workdir=args.workdir,
        log_level=args.log_level.upper(),
        show_progress=args.show_progress,
    )

    if args.workdir is not None:
        with open(args.workdir / "errors.json", "w") as fh:
            json.dump(errors, fh)

    if not args.no_output:
        reports.write_report(
            start=start_report,
            stop=stop,
            output_dir=args.output,
            archive_dir=args.archive_dir,
            astromon_archive_dir=args.astromon_archive_dir,
            workdir=args.workdir,
            show_progress=args.show_progress,
        )

        with open(args.output / "errors.json", "w") as fh:
            json.dump(errors, fh)

        with open(args.output / "sources.json", "w") as fh:
            all_sources = processing.get_sources()
            fh.write(all_sources.to_pandas().to_json())


if __name__ == "__main__":
    main()
