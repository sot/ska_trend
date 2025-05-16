"""
Top-level functions to process periscope drift trending.
"""

import re
import sys
import tempfile
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import astromon.observation
from astropy.table import Table
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
]


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
    with tempfile.TemporaryDirectory() as td, astromon.utils.chdir(td):
        path = Path(td)
        arc5gl = Arc5gl()
        arc5gl.sendline(f"tstart={tstart.date}")
        arc5gl.sendline(f"tstop={tstop.date}")
        arc5gl.sendline("get obspar")
        names = [str(p) for p in path.glob("*")]
        return [
            m.group(1)
            for m in [re.search("axaff([0-9]+)_", name) for name in names]
            if m
        ]


def process_observation(obsid, work_dir, archive_dir, log_level):
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
                obsid, workdir=work_dir, archive_dir=archive_dir
            )
            # The first step (commented out) is to make sure all required steps are done,
            # but this should not be needed.
            # obs.process()
            obs.get_sources()
            obs.get_periscope_drift_data()
        except astromon.observation.Skipped as exc:
            msg = f"skipped: {exc}"
            msg = f"OBSID={obsid} - skipped: {exc}"
        except astromon.observation.SkippedWithWarning as exc:
            msg = f"skipped: {exc}"
            msg = f"OBSID={obsid} WARNING - skipped: {exc}"
        except Exception as exc:
            ok = False
            msg = f"OBSID={obs.obsid} FAIL - skipped: {exc}"
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"OBSID={obs.obsid} {exc_type}: {exc_value}")
            trace = traceback.extract_tb(exc_traceback)
            for step in trace:
                logger.error(
                    f"OBSID={obsid}            in {step.filename}:{step.lineno}/{step.name}:"
                )
                logger.error(f"OBSID={obsid}           {step.line}")
        finally:
            if obs is not None:
                obs.archive()
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
        # (int(obsid), workdir, archive_dir, log_level)
        (int(obsid), workdir, archive_dir, log_level)
        for obsid in obsids
    ]

    with Pool(processes=n_threads) as pool:
        if show_progress:
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
    log_level="WARNING",
    archive_dir=None,
    n_threads=8,
    show_progress=True,
    workdir=None,
):
    """
    Process all observations in the given time interval.

    Parameters
    ----------
    start : str
        Start of the time interval. Can be a CxoTime-like string or a relative time string.
    stop : str
        End of the time interval. Can be a CxoTime-like string or a relative time string.
    log_level : str
        Logging level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    archive_dir : str
        Archive directory. The final location where to archive data for future use.
    n_threads : int
        Number of threads to use for multiprocessing.
    show_progress : bool
        If True, show a progress bar.
    workdir : str
        Working directory. If this is None, a temporary directory will be created.
        If it is not None, a subdirectory will be created to store temporary files.

    Returns
    -------
    observations : dict
        Dictionary of observations, where the keys are the obsids and the values are
        the corresponding Observation objects.
    sources : Table
        Table of sources, where each row corresponds to a source and the columns
        contain the source data.
    errors : list
        List of tuples with the errors encountered during processing. Each tuple
        contains the obsid and the error message.
    """
    logger = basic_logger("astromon", level=log_level)

    obsids = get_obsids(start, stop)
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
    obsid_iter = tqdm(obsids) if show_progress else obsids
    for obsid in obsid_iter:
        obs = observation.Observation(obsid, archive_dir=archive_dir, workdir=workdir)

        if obs.obsid in errors:
            continue

        if not obs.is_selected():
            continue

        try:
            # running process here might take extra time, and in principle should not be needed
            # obs.process()
            summary.extend(
                [
                    src_tdd.summary
                    for src_tdd in obs.get_periscope_drift_data().data.values()
                ]
            )
            observations[obs.obsid] = obs
        except astromon.observation.Skipped:
            pass
        except astromon.observation.SkippedWithWarning:
            pass
        except Exception as exc:
            errors[obs.obsid] = str(exc)

    logger.debug(
        f"Processed {len(obsids)} observations with {len(summary)} sources and {len(errors)} errors"
    )
    for error in errors:
        logger.debug(f"    OBSID={error[0]}: {error[1]}")

    return observations, Table(summary), errors
