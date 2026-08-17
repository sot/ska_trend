#!/usr/bin/env python

import argparse
import logging
import os
import sys
from pathlib import Path

import ska_helpers
from cxotime import CxoTime

from ska_trend.astromon import reports
from ska_trend.time_utils import parse_time_arg

logger = logging.getLogger("astromon")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=Path(os.environ["SKA"]) / "www" / "ASPECT" / "astromon",
        type=Path,
        help="Report output directory. Default: $SKA/www/ASPECT/astromon",
    )
    parser.add_argument(
        "--start-report",
        default="stop-5yr",
        help="Start of report interval (e.g. stop-5yr or a date). Default: stop-5yr",
    )
    parser.add_argument("--stop", default=None, help="Stop time. Default: NOW")
    parser.add_argument(
        "--matches",
        choices=["all", "cal", "mta"],
        default="mta",
        help="Cross-match selection: all (unfiltered), cal, or mta. Default: mta",
    )
    parser.add_argument(
        "--astromon-archive-dir",
        default=Path(os.environ["SKA"]) / "data" / "astromon" / "xray_observations",
        type=Path,
        help="Astromon archive directory (flux images). "
        "Default: $SKA/data/astromon/xray_observations",
    )
    parser.add_argument(
        "--dbfile",
        default=None,
        type=Path,
        help="Astromon HDF5 db file. Default: astromon default "
        "($ASTROMON_FILE or $SKA/data/astromon/astromon.h5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        type=str.upper,
        help="Verbosity. Default: INFO.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-source pages",
    )
    return parser


def main():
    args = get_parser().parse_args()

    log_args = {"level": args.log_level, "format": "%(message)s"}
    log_args.update(
        {"stream": sys.stdout} if args.log_file is None else {"filename": args.log_file}
    )
    logger = ska_helpers.logging.basic_logger("astromon", **log_args)

    now = CxoTime()
    logger.info("---------- astromon reports update at %s ----------" % (now.iso))

    stop = now if args.stop is None else CxoTime(args.stop)
    start_report = parse_time_arg(args.start_report, stop=stop)

    reports.write_report(
        start=start_report,
        stop=stop,
        output_dir=args.output,
        archive_dir=args.astromon_archive_dir,
        dbfile=args.dbfile,
        selection=args.matches,
        overwrite=args.overwrite,
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    main()
