"""
Top-level functions to process periscope drift trending.
"""

import logging
import os
import re
import sys
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from astromon.utils import CiaoProcessFailure
from astropy.table import Table, vstack
from cxotime import CxoTime
from ska_arc5gl import Arc5gl
from ska_helpers.logging import basic_logger
from tqdm import tqdm

from ska_trend.periscope_drift import observation

__all__ = [
    "get_obsids",
    "process_observation",
    "run_multiprocess",
    "process_interval",
    "SOURCES_FILE",
    "get_sources",
    "update_sources",
]


DATA_DIR = Path(os.environ["SKA"]) / "data" / "periscope_drift"

SOURCES_FILE = DATA_DIR / "sources.fits"


def get_obsids(tstart, tstop):
    """
    Get a list of obsids from the archive for a given time interval.

    Parameters
    ----------
    tstart : CxoTime-like
        Start of the time interval.
    tstop : CxoTime-like
        End of the time interval.

    Returns
    -------
    obsids : list
        List of obsids in the time interval.
    """
    tstart = CxoTime(tstart)
    tstop = CxoTime(tstop)
    arc5gl = Arc5gl()
    arc5gl.sendline(f"tstart={tstart.date}")
    arc5gl.sendline(f"tstop={tstop.date}")
    arc5gl.sendline("operation=browse")
    arc5gl.sendline("dataset=flight")
    arc5gl.sendline("level=0.5")
    arc5gl.sendline("detector=obi")
    arc5gl.sendline("subdetector=obspar")
    arc5gl.sendline("filetype=actual")
    arc5gl.arc5gl.sendline("go")

    text = ""
    while True:
        line = arc5gl.arc5gl.read_nonblocking(10000, 10)
        text += line
        if line.find(arc5gl.prompt) >= 0:
            break

    regex = [re.search("axaff([0-9]+)_", name) for name in text.splitlines()]
    return [m.group(1) for m in regex if m]


def process_observation(obsid, work_dir, archive_dir, astromon_archive_dir, log_level):
    """
    Process a single observation.

    Parameters
    ----------
    obsid : int
        Observation ID.
    work_dir : str
        Working directory. If this is None, a temporary directory will be created.
        If it is not None, a subdirectory will be created to store temporary files.
    archive_dir : str
        Archive directory. The final location where to archive data for future use.
    astromon_archive_dir : str
        Archive directory for astromon data. The final location where to archive astromon data for
        future use.
    log_level : str
        Logging level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    logger = basic_logger("periscope_drift", level=log_level)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="A coordinate frame was not found for region"
        )
        warnings.filterwarnings(
            "ignore",
            message='"physical" frame or shape is not supported by the regions package, skipping.',
        )

        ok = True
        msg = ""
        obs = None
        try:
            obs = observation.Observation(
                obsid,
                workdir=work_dir,
                archive_dir=archive_dir,
                astromon_archive_dir=astromon_archive_dir,
            )

            if obs.is_selected:
                # caching both cases of get_sources explicitly
                obs.periscope_drift.get_sources(apply_filter=True)
                obs.periscope_drift.get_sources(apply_filter=False)
                obs.periscope_drift.get_periscope_drift_data()
        except CiaoProcessFailure as exc:
            ok = False
            exc_type, exc_value, exc_traceback = sys.exc_info()
            msg = f"OBSID={obs.obsid} CIAO process fail: {exc_value}"
            logger.error(msg)
            for line in exc.lines:
                level = (
                    logging.ERROR
                    if "ERROR" in line
                    else (logging.WARNING if "WARNING" in line else logging.DEBUG)
                )
                logger.log(level, f"OBSID={obs.obsid} CIAO process fail: {line}")
            trace = traceback.extract_tb(exc_traceback)
            for step in trace:
                logger.debug(
                    f"OBSID={obsid}         in {step.filename}:{step.lineno}/{step.name}:"
                )
                logger.debug(f"OBSID={obsid}           {step.line}")
        except Exception as exc:
            ok = False
            msg = f"OBSID={obsid} FAIL - skipped: {exc}"
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"OBSID={obsid} {exc_type}: {exc_value}")
            trace = traceback.extract_tb(exc_traceback)
            for step in trace:
                logger.error(
                    f"OBSID={obsid}         in {step.filename}:{step.lineno}/{step.name}:"
                )
                logger.error(f"OBSID={obsid}           {step.line}")
        finally:
            if obs is not None:
                obs.archive()
                obs.periscope_drift.archive()
        return {
            "obsid": obsid,
            "ok": ok,
            "msg": msg,
        }


def split(arg_list, n_per_chunk):
    """
    Split a list into chunks of size n_per_chunk.
    """
    args = [arg_list[i : i + n_per_chunk] for i in range(0, len(arg_list), n_per_chunk)]
    return [arg for arg in args if arg]


def run_multiprocess(
    obsids,
    *,
    log_level="DEBUG",
    archive_dir=None,
    astromon_archive_dir=None,
    n_threads=8,
    n_per_iter=10,
    show_progress=False,
    workdir=None,
):
    """
    Run the process_observation function in parallel using multiprocessing.

    Parameters
    ----------
    obsids : list
        List of obsids to process.
    log_level : str
        Logging level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    archive_dir : str
        Archive directory. The final location where to archive data for future use.
    astromon_archive_dir : str
        Archive directory for astromon data. The final location where to archive astromon data.
    n_threads : int
        Number of threads to use for multiprocessing.
    n_per_iter : int
        Number of observations to process in each iteration.
    show_progress : bool
        If True, show a progress bar.
    workdir : str
        Working directory. If this is None, a temporary directory will be created.
        If it is not None, a subdirectory will be created to store temporary files.

    Returns
    -------
    results : list
        List of dictionaries with the results of processing each observation.
    """
    logger = basic_logger("astromon", level=log_level)

    task_args = [
        (int(obsid), workdir, archive_dir, astromon_archive_dir, log_level)
        for obsid in obsids
    ]

    with Pool(processes=n_threads) as pool:
        if show_progress:
            # this is a print statement because that is where the tqdm progress bar goes
            print("Processing observations...")
            results = []
            n_per_iter = n_per_iter if n_per_iter else 20 * n_threads
            for t_args in tqdm(split(task_args, n_per_iter)):
                results += pool.starmap(process_observation, t_args)
        else:
            results = pool.starmap(process_observation, task_args)

    logger.info(f"Processed {len(results)} observations")
    return results


def process_interval(
    start,
    stop,
    log_level=None,
    archive_dir=None,
    astromon_archive_dir=None,
    n_threads=8,
    show_progress=True,
    workdir=None,
    no_output=False,
):
    """
    Process all observations in the given time interval.

    Parameters
    ----------
    start : str
        Start of the time interval. Can be a CxoTime-like string.
    stop : str
        End of the time interval. Can be a CxoTime-like string.
    log_level : str
        Logging level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    archive_dir : str
        Archive directory. The final location where to archive data for future use.
    astromon_archive_dir : str
        Archive directory for astromon data. The final location where to archive astromon data.
    n_threads : int
        Number of threads to use for multiprocessing.
    show_progress : bool
        If True, show a progress bar.
    workdir : str
        Working directory. If this is None, a temporary directory will be created.
        If it is not None, a subdirectory will be created to store temporary files.
    no_output : bool
        If True, do not write output.

    Returns
    -------
    errors : list
        List of tuples with the errors encountered during processing. Each tuple
        contains the obsid and the error message.
    """

    return process_obsids(
        get_obsids(start, stop),
        log_level=log_level,
        archive_dir=archive_dir,
        astromon_archive_dir=astromon_archive_dir,
        n_threads=n_threads,
        show_progress=show_progress,
        workdir=workdir,
        no_output=no_output,
    )


def process_obsids(
    obsids,
    log_level="WARNING",
    archive_dir=None,
    astromon_archive_dir=None,
    n_threads=8,
    show_progress=True,
    workdir=None,
    no_output=False,
):
    """
    Process all observations in the given time interval.

    Parameters
    ----------
    obsids : list
        List of obsids to process.
    log_level : str
        Logging level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    archive_dir : str
        Archive directory. The final location where to archive data for future use.
    astromon_archive_dir : str
        Archive directory for astromon data. The final location where to archive astromon data.
    n_threads : int
        Number of threads to use for multiprocessing.
    show_progress : bool
        If True, show a progress bar.
    workdir : str
        Working directory. If this is None, a temporary directory will be created.
        If it is not None, a subdirectory will be created to store temporary files.
    no_output : bool
        If True, do not write output.

    Returns
    -------
    errors : list
        List of tuples with the errors encountered during processing. Each tuple
        contains the obsid and the error message.
    """
    logger = basic_logger("astromon", level=log_level)
    if log_level is not None:
        logger.setLevel(log_level)

    logger.info(f"Processing {len(obsids)} observations with {n_threads} threads")
    results = run_multiprocess(
        obsids,
        log_level=log_level,
        archive_dir=archive_dir,
        workdir=workdir,
        show_progress=show_progress,
        n_threads=n_threads,
    )

    # obsids = [result["obsid"] for result in results if result["ok"]]
    errors = {str(r["obsid"]): r["msg"] for r in results if not r["ok"]}
    observations = {}
    summary = []
    if show_progress:
        # this is a print statement because that is where the tqdm progress bar goes
        print("Summarizing observations...")
    obsid_iter = tqdm(obsids) if show_progress else obsids
    for obsid in obsid_iter:
        obs = observation.Observation(
            obsid,
            workdir=workdir,
            archive_dir=archive_dir,
            astromon_archive_dir=astromon_archive_dir,
        )

        if obs.obsid in errors:
            continue

        if not obs.periscope_drift.is_selected():
            continue

        try:
            summary.extend(
                [
                    src_tdd.summary
                    for src_tdd in obs.periscope_drift.get_periscope_drift_data().data.values()
                ]
            )
            observations[obs.obsid] = obs
        except Exception as exc:
            errors[obs.obsid] = str(exc)

    summary = Table(summary)
    msg = f"Processed {len(obsids)} observations with {len(summary)} sources"
    if len(errors) > 0:
        msg += f" and {len(errors)} errors"
    logger.debug(msg)
    for obsid, error in errors.items():
        logger.debug(f"    OBSID={obsid}: {error}")

    if not no_output:
        update_sources(Table(summary), obsids)

    return errors


def get_sources(filename=None):
    """
    Get sources on file.
    """
    filename = SOURCES_FILE if filename is None else filename
    if not filename.exists():
        raise FileNotFoundError(f"Sources file {filename} not found")
    # if the file is in fits format, the byte order will always be big-endian,
    # whereas OSX systems are little-endian. Later on, one creates plotly figures
    # and often calls `.to_html()` on them, which fails if the byte order is not native.
    # instead of trying to fix the byte order in various places, we just convert it here
    # by calling `as_array()`
    return Table(Table.read(filename).as_array())


def update_sources(sources, obsids, filename=None):
    """
    Update the given sources, corresponding to the given obsids, in the sources file.

    The obsids parameters is used to know which entries need to be modified. This matters only if
    the observation has no sources, in which case we want to remove any previous entries.
    """

    filename = SOURCES_FILE if filename is None else filename

    if filename.exists():
        # if the file exists, replace the entries with the newly processed ones
        prev_sources = get_sources(filename)
        prev_sources = prev_sources[~np.in1d(prev_sources["obsid"], obsids)]
        if not sources or len(sources) == 0:
            all_sources = prev_sources
        else:
            all_sources = vstack([prev_sources, sources], metadata_conflicts="silent")
    else:
        all_sources = sources

    all_sources.write(filename, overwrite=True)
