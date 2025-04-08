import argparse
import copy
import functools
import json
import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import agasc
import astropy.units as u
import chandra_aca.plot
import kadi.commands as kc
import kadi.events as ke
import numpy as np
import numpy.typing as npt
import parse_cm.paths
import razl.observations
from astropy.table import Table
from chandra_aca.centroid_resid import CentroidResiduals
from chandra_aca.transform import yagzag_to_pixels
from cheta import fetch, fetch_eng, fetch_sci
from cxotime import CxoTime, CxoTimeLike
from jinja2 import Environment
from matplotlib import pyplot as plt
from mica.archive import asp_l1
from Quaternion import Quat
from ska_helpers.logging import basic_logger
from ska_matplotlib import plot_cxctime
from starcheck.state_checks import calc_man_angle_for_duration

if TYPE_CHECKING:
    from proseco.catalog import ACATable

# Update guide metrics file with new obsids between NOW and (NOW - NDAYS_DEFAULT) days
NDAYS_DEFAULT = 7
SKA = Path(os.environ["SKA"])

# Count of sporadic exceptions for testing. See `raise_sporadic_exc_for_testing`.
SPORADIC_EXC_COUNT = 0

logger = basic_logger("centroid_dashboard")


def get_opt():
    parser = argparse.ArgumentParser(description="Centroid dashboard")
    parser.add_argument(
        "--start",
        help=f"Processing start date (default=stop - {NDAYS_DEFAULT} days)",
    )
    parser.add_argument(
        "--stop",
        default=CxoTime.NOW,
        help="Processing stop date (default=NOW)",
    )
    parser.add_argument(
        "--data-root",
        default=".",
        help="Root directory for data files (default=.)",
    )
    parser.add_argument(
        "--force",
        help="Force processing even if data exists",
        action="store_true",
    )
    parser.add_argument(
        "--remote-copy",
        help="Copy asp L1 data from remote archive if not available locally",
        action="store_true",
    )
    parser.add_argument(
        "--skip-plots",
        help="Skip generating plots (default=False)",
        action="store_true",
    )
    # Add obsid argument for optional single obsid to process
    parser.add_argument(
        "--obsid",
        type=int,
        help="Observation ID to process (default=all), still need start/stop times.",
    )
    parser.add_argument(
        "--raise-exc",
        help="Raise exceptions during processing for debugging (default=False)",
        action="store_true",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default=INFO)"
    )
    return parser


def raise_sporadic_exc_for_testing():
    """Raise a quasi-sporadic exception for testing.

    A handful of calls to this function are embedded in the code to raise an exception
    sporadically for testing purposes. The exception is raised when the environment
    variable `CENTROID_DASHBOARD_RAISE_EXC` is set to a positive integer.

    For testing the exception handling, something like this::

       env CENTROID_DASHBOARD_RAISE_EXC=11 python -m ska_trend.centroid_dashboard.app \
          --start 2025:015 --stop 2025:017  --force
    """
    global SPORADIC_EXC_COUNT  # noqa: PLW0603
    if not (n_ok := os.environ.get("CENTROID_DASHBOARD_RAISE_EXC")):
        return

    SPORADIC_EXC_COUNT += 1
    if SPORADIC_EXC_COUNT % int(n_ok) == 0:
        raise ValueError("Sporadic exception for testing")


@functools.lru_cache()
def get_index_template():
    path = Path(__file__).parent / "index_template.html"
    return path.read_text()


class ReportDirMixin:
    @functools.cached_property
    def report_dir(self):
        return Path(self.opt["data_root"]) / "reports" / self.report_subdir

    @functools.cached_property
    def report_subdir(self):
        year = parse_cm.paths.parse_load_name(self.source)[-1]
        return Path(str(year), self.source, f"{self.obsid:05d}")


class ObservationFromInfo(ReportDirMixin):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass(repr=False, kw_only=True)
class Observation(razl.observations.Observation, ReportDirMixin):
    obs_next: Optional["Observation"] = None
    obs_prev: Optional["Observation"] = None

    @functools.cached_property
    def manvr_event(self):
        """Provide the kadi maneuver event leading to this observation."""
        raise_sporadic_exc_for_testing()
        # Manvr leading to this observation. Remember the Manvr class includes the
        # maneuver and info about the dwell.
        manvr = self.manvrs[-1]
        manvrs = ke.manvrs.filter(manvr.start, manvr.stop)
        if len(manvrs) == 0:
            # Most commonly because telemetry does not include this maneuver yet.
            out = None
        elif len(manvrs) > 2:
            # Should never have 3 or more manvrs in a row.
            raise ValueError(
                f"Multiple manvrs found between {manvr.start} and {manvr.stop}:\n{manvrs}"
            )
        else:
            # Take the last maneuver before the observation. Segmented maneuvers or
            # the high-IR zone dwell are common cases of 2 manvrs.
            out = manvrs[len(manvrs) - 1]  # Negative indexing not supported
        return out

    @functools.cached_property
    def aber(self) -> dict:
        return get_aberration_correction(self.obsid, self.source)

    @functools.cached_property
    def kalman_start(self) -> CxoTime:
        """Date of start of KALMAN from telemetry."""
        if self.manvr_event is None:
            raise ValueError("manvr_event is None")
        return CxoTime(self.manvr_event.kalman_start)

    @functools.cached_property
    def kalman_stop(self) -> CxoTime:
        """Date of stop of KALMAN from telemetry."""
        if self.manvr_event is None:
            raise ValueError("manvr_event is None")
        return CxoTime(self.manvr_event.npnt_stop)

    @functools.cached_property
    def info(self) -> str:
        attrs = [
            "obsid",
            "source",
            "aber",
            "att_stats",
            "manvr_angle",
            "obs_links",
            "one_shot",
        ]
        out = {attr: getattr(self, attr) for attr in attrs}
        for attr in ["date_starcat", "kalman_start", "kalman_stop"]:
            out[attr] = getattr(self, attr).date

        return out

    @functools.cached_property
    def one_shot(self) -> dict[str] | None:
        if (manvr_event := self.manvr_event) is None:
            return None

        aber_corrected = (
            np.hypot(
                manvr_event.one_shot_pitch - self.aber["y"],
                manvr_event.one_shot_yaw - self.aber["z"],
            )
            if self.aber["status"] == "OK"
            else None
        )

        return {
            "total": manvr_event.one_shot,
            "pitch": manvr_event.one_shot_pitch,
            "yaw": manvr_event.one_shot_yaw,
            "aber_corrected": aber_corrected,
        }

    @functools.cached_property
    def manvr_angles(self):
        angles = []
        for manvr in self.manvrs:
            dq = manvr.att_stop.dq(manvr.att_start)
            angles.append(np.rad2deg(np.arccos(dq.q[3]) * 2))
        return angles

    @functools.cached_property
    def manvr_angles_text(self) -> str | None:
        if len(self.manvr_angles) == 1:
            out = None
        else:
            lines = []
            lines.append(" Segments:")
            for manvr, manvr_angle in zip(self.manvrs, self.manvr_angles, strict=True):
                lines.append(f"  {manvr_angle:5.1f} deg in {manvr.dur:.1f} sec")
            out = "\n".join(lines)
        return out

    @functools.cached_property
    def manvr_angle(self) -> float:
        if (n_manvrs := len(self.manvr_angles)) == 1:
            manvr = self.manvrs[0]
            dq = manvr.att_stop.dq(manvr.att_start)
            angle = np.rad2deg(np.arccos(dq.q[3]) * 2)
        elif n_manvrs > 1:
            duration = self.manvrs[-1].stop.secs - self.manvrs[0].start.secs
            angle = calc_man_angle_for_duration(duration)
        else:
            angle = -999.0
            logger.warning(f"Observation {self.obsid} has no maneuvers")
        return angle

    @functools.cached_property
    def obs_links(self) -> dict[str, dict | None]:
        """Get subset of previous and next observation info.

        This is used to create and maintain the links to the previous and next
        observations in the HTML report along with the previous ending roll error.
        """
        out = {}
        for link, obs in [("prev", self.obs_prev), ("next", self.obs_next)]:
            if obs is None:
                out[link] = None
            else:
                out[link] = {"obsid": obs.obsid, "source": obs.source}
                if link == "prev":
                    out[link]["att_stats"] = obs.att_stats

        return out

    def obs_link_from_info(self, link: str) -> ObservationFromInfo | None:
        """Get a ObservationFromInfo object or None for next or prev observation.

        This uses the current obs info.json file to get the obsid and source for the
        next or previous observation. If the info.json file for the next or previous
        observation exists, it uses that info to create the ObservationFromInfo object.

        This method called for the first and last observations in the processing to get
        obs_next and obs_prev. It is not called for the rest of the observations in the
        processing since they already have the prev/next obs objects.

        The ObservationFromInfo object is a minimal stub that looks like an Observation.

        Parameters
        ----------
        link : str
            "next" or "prev"
        """
        # If the current obs does not have an info.json file then we have no way to
        # get info about the prev/next obs. In this case return None. This normally
        # happens for the last observation when processing new observations.
        if not (info_json := self.report_dir / "info.json").exists():
            logger.info(f"No {info_json} file found, obs=None")
            return None

        # info.json is available, so it has an "obs_links" dict with "next" and "prev"
        # keys. Each of these can be either None (meaning not available) or a dict with
        # obsid, source, att_stats keys. This is not common but happens if you re-run
        # processing over the same date range.
        info = json.loads(info_json.read_text())
        obs_link = info["obs_links"][link]
        if obs_link is None:
            logger.info(f"No {link} obs info found, obs=None")
            return None

        obs = ObservationFromInfo(opt=self.opt, **obs_link)
        if (info_json_link := obs.report_dir / "info.json").exists():
            info_link = json.loads(info_json_link.read_text())
            obs = ObservationFromInfo(opt=self.opt, **info_link)
        logger.info(f"Found {link} obsid {obs.obsid}")

        return obs

    @functools.cached_property
    def att_deltas(self) -> dict[str, npt.NDArray] | None:
        out = get_obc_gnd_att_deltas(
            self.obsid, self.q_att_obc, remote_copy=self.opt["remote_copy"]
        )
        return out

    @functools.cached_property
    def att_stats(self) -> dict[str, float]:
        raise_sporadic_exc_for_testing()
        if self.att_deltas:
            out = {
                "d_roll50": np.percentile(np.abs(self.att_deltas["d_roll"]), 50),
                "d_roll95": np.percentile(np.abs(self.att_deltas["d_roll"]), 95),
                "d_roll_end": self.att_deltas["d_roll"][-1],
            }
        else:
            out = {}
        return out

    @functools.cached_property
    def aacccdpt_msid(self) -> fetch.Msid | None:
        if self.manvr_event is None:
            return None
        out = fetch_sci.Msid("aacccdpt", self.kalman_start, self.kalman_stop)
        return out

    @functools.cached_property
    def t_ccd_mean(self) -> float | None:
        """Mean temperature of the CCDs during the observation."""
        if self.aacccdpt_msid is None:
            return None
        # Get the mean temperature over the observation period
        t_ccd = np.mean(self.aacccdpt_msid.vals)
        return t_ccd

    @functools.cached_property
    def t_ccd_max(self) -> float | None:
        """Max temperature of the CCDs during the observation."""
        if self.aacccdpt_msid is None:
            return None
        # Get the max temperature over the observation period
        t_ccd = np.max(self.aacccdpt_msid.vals)
        return t_ccd

    @functools.cached_property
    def q_att_obc(self) -> fetch.Msid | None:
        # Need manvr_event for kalman_start and kalman_stop
        if self.manvr_event is None:
            return None

        try:
            out = fetch.Msid("quat_aoattqt", self.kalman_start, self.kalman_stop)
        except IndexError:
            # No telemetry unfortunately raises IndexError instead of zero-length Msid
            return None
        # Enough data and sampling to end of observation
        if len(out) == 0 or self.kalman_stop.secs - out.times[-1] > 10:
            return None
        return out

    @functools.cached_property
    def date_starcat(self) -> CxoTime:
        """Date of MP_STARCAT command"""
        return CxoTime(self.starcat.date)

    @functools.cached_property
    def starcat_summary(self):
        return self.starcat.copy()


def processed(info_json_path: Path):
    """Check if the observation has already been fully processed.

    The check is based on the existence of the info.json file and the non-None values
    obsid_next and obsid_prev keys in the file. This file is created/updated at the end
    of the processing.
    """
    if not info_json_path.exists():
        return False
    info = json.loads(info_json_path.read_text())
    # Check that info has a few key fields. Any processing exception will delete the
    # info.json file.
    out = (
        info["kalman_start"]
        and info["obs_links"]["next"]
        and info["obs_links"]["prev"]
        and info["one_shot"]
        and (info["obsid"] >= 38000 or info["att_stats"])
    )
    # Last check requires that every OR has values of att_stats (from OBC vs GND
    # attitude deltas). For planned OR's that do not run due to SCS-107, the att_stats
    # will never be computed so these obsids get reprocessed every time.
    return out


def get_gnd_atts(
    obsid: int, remote_copy: bool = False
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Get ground attitude solution for the observation.

    If `remote_copy` is True and the data are not available locally, copy the data from
    the remote archive on HEAD.

    Parameters
    ----------
    obsid : int
        Observation ID. If obsid >= 38000, return None.
    remote_copy : bool, optional
        If True, copy the data from the remote archive on HEAD.

    Returns
    -------
    tuple
        Tuple of ground attitude quaternions (Nx4) and times (N).
    """
    obsid_str = f"{obsid:05d}"
    obs2_dir = f"data/mica/archive/asp1/{obsid_str[:2]}"
    obs2_dir_local = SKA / obs2_dir
    obsid_dir_remote = f"kady:/proj/sot/ska/{obs2_dir}/{obsid_str}*"
    obsid_dir_local = obs2_dir_local / obsid_str

    if not obsid_dir_local.exists():
        if not remote_copy:
            return [], []

        # Get a limited copy of the aspect solution data from the remote archive. This
        # is custom to this application because it only keeps two columns of the ASOL
        # file for disk space considerations.
        obs2_dir_local.mkdir(parents=True, exist_ok=True)
        cmds = [
            "rsync",
            "-av",
            # Ignore files like pcadf30109_001N001_asol1.fits.gz
            "--include=pcadf[0-9][0-9][0-9][0-9][0-9][0-9]*_asol1.fits.gz",
            "--include=pcadf*_acal1.fits.gz",
            "--include=pcadf*_aqual1.fits.gz",
            f"--include={obsid_str}",
            f"--include={obsid_str}_v*",
            "--exclude=*",
            obsid_dir_remote,
            f"{obs2_dir_local}/",
        ]
        logger.info(f"Copying remote data with command: {' '.join(cmds)}")
        completed_process = subprocess.run(cmds, check=False)
        if completed_process.returncode != 0:
            logger.warning("Could not copy remote aspect solution data")
            return [], []

        for path in obsid_dir_local.glob("*_asol1.fits.gz"):
            logger.info(f"Overwriting existing file {path} with only time, q_att_raw cols")
            dat = Table.read(path)
            dat = dat["time", "q_att_raw"]
            dat.write(path, overwrite=True)

        obsid_dir_resolve = obsid_dir_local.resolve().name
        for path in obs2_dir_local.glob(f"{obsid_str}_v*"):
            if path.name != obsid_dir_resolve:
                # Remove that directory
                logger.info(f"Removing directory {path}")
                shutil.rmtree(path)

    raise_sporadic_exc_for_testing()
    atts, atts_times, _ = asp_l1.get_atts(obsid=obsid)

    return atts, atts_times


def get_obc_gnd_att_deltas(
    obsid: int, q_att_obc: fetch.Msid | None, remote_copy: bool = False
) -> dict[str] | None:
    """
    Get OBC pitch, yaw, roll errors with respect to the ground aspect solution.

    Parameters
    ----------
    obsid : int
        Observation ID. If obsid >= 38000, return None.

    """
    if obsid >= 38000 or q_att_obc is None:
        return None

    # Get ground attitude solution and times
    atts_gnd, atts_gnd_times = get_gnd_atts(obsid, remote_copy=remote_copy)

    # If data are not available `get_atts` returns empty arrays.
    if len(atts_gnd_times) == 0:
        return None

    # Get OBC attitude solution and times
    atts_obc = q_att_obc.vals.q
    atts_obc_times = q_att_obc.times

    tstart = max(atts_gnd_times[0], atts_obc_times[0])
    tstop = min(atts_gnd_times[-1], atts_obc_times[-1])

    # Ensure that endpoints of telemetry and ground solution are within 5 minutes
    if atts_gnd_times[-1] - tstop > 300 or atts_obc_times[-1] - tstop > 300:
        return None

    # Trim OBC telemetry to common time range
    i0, i1 = np.searchsorted(atts_obc_times, [tstart, tstop])
    atts_obc = atts_obc[i0:i1]
    atts_obc_times = atts_obc_times[i0:i1]

    # Sample ground solution at OBC times. Since ground solution is sampled at 0.256 sec
    # this is good enough for this application.
    idxs = np.searchsorted(atts_gnd_times, atts_obc_times)
    atts_gnd = atts_gnd[idxs]

    # Compute the quaternion difference between the OBC and ground solutions
    q_obc = Quat(q=atts_obc)
    q_gnd = Quat(q=atts_gnd)
    dq = q_gnd.dq(q_obc)

    out = {
        "time": atts_obc_times,
        "d_roll": dq.roll0 * 3600,
        "d_pitch": dq.pitch * 3600,
        "d_yaw": dq.yaw * 3600,
    }
    return out


def get_observations(
    start: CxoTimeLike,
    stop: CxoTimeLike,
    opt: argparse.Namespace | None = None,
) -> list[Observation]:
    """
    Get observations between the specified start and stop times.

    This function uses the kadi get_cmds() to retrieve commands and the `razl` module to
    convert those commands into observations. It also logs information about each
    observation found.

    Parameters
    ----------
    start : CxoTimeLike
        The start time for the observation retrieval.
    stop : CxoTimeLike
        The stop time for the observation retrieval.
    opt : argparse.Namespace, optional
        Command line options. If None, no options are used.

    Returns
    -------
    list of Observation
        A list of Observation objects found between the specified start and stop times.
    """
    start = CxoTime(start)
    stop = CxoTime(stop)

    # Get planned commands as if no SCS-107 events occurred. This gives the planned
    # obsids and other observation information from commands.
    lookback_days = (stop - start + 14 * u.day).to_value(u.day)
    with kc.set_time_now(stop), kc.conf.set_temp("default_lookback", lookback_days):
        cmds = kc.get_cmds(start, stop, event_filter=kc.filter_scs107_events)

    obss_razl = razl.observations.get_observations_from_cmds(
        cmds,
        allow_skip_first_obs=True,
    )

    obss = []
    for obs_razl in obss_razl:
        # Create local Observation object from razl Observation object
        kwargs = {
            k: getattr(obs_razl, k)
            for k in razl.observations.Observation.__annotations__
        }
        obs = Observation(**kwargs)

        # Ignore intermediate attitude observations without a star catalog
        if obs.starcat is None:
            continue

        # In this application we only care about guide star slots
        obs.starcat = obs.starcat[np.isin(obs.starcat["type"], ["BOT", "GUI"])]
        if opt is not None:
            obs.opt.update(vars(opt))
        obss.append(obs)
        logger.info(
            f"Found observation {obs.obsid} at {obs.obs_start} with {len(obs.manvrs)} manvrs"
        )

    return obss


def make_html(obs: Observation, traceback=None):
    """Make the HTML file for the observation."""
    raise_sporadic_exc_for_testing()
    logger.debug(f"Making HTML for observation {obs.obsid}")
    # Get the template from index_template.html
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(get_index_template())
    context = {
        "MICA_PORTAL": "https://kadi.cfa.harvard.edu/mica/",
        "obs": obs,
        "traceback": traceback,  # For error pages
    }
    html = template.render(**context)

    path = obs.report_dir / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {path}")
    path.write_text(html)


def get_centroid_resids(
    start: CxoTimeLike,
    stop: CxoTimeLike,
    starcat: "ACATable",
    q_att: fetch.Msid,
) -> dict[int, CentroidResiduals]:
    """
    Get OBC centroid residuals for all guide slots.

    This is relative to the OBC attitude solution.

    Parameters
    ----------
    start : CxoTimeLike
        Start time for the centroid residuals.
    stop : CxoTimeLike
        Stop time for the centroid residuals.
    starcat : ACATable
        Star catalog table.
    q_att : fetch.Msid
        Attitude quaternion telemetry for `quat_aoattqt` MSID.

    Returns
    -------
    dict
        Dictionary of CentroidResiduals objects keyed by slot.
    """
    crs = {}
    cr = CentroidResiduals(start, stop)

    # Grab attitude telemetry once for all slots, copying each time. This is basically
    # equivalent to cr.set_atts("obc"), but using quat_aoattqt is more robust.
    cr.att_source = "obc"
    cr.atts = q_att.vals.q
    cr.att_times = q_att.times
    cr.obsid = starcat.obsid

    for slot, agasc_id in zip(starcat["slot"], starcat["id"], strict=True):
        try:
            cr.set_centroids("obc", slot=slot)
            cr.set_star(agasc_id=agasc_id)
            cr.calc_residuals()
            if len(cr.yag_times) > 10:
                crs[slot] = copy.copy(cr)
        except Exception:
            logger.info(f"Could not compute crs for slot {slot})")

    return crs


def get_q_att_obc(start: CxoTimeLike, stop: CxoTimeLike) -> fetch.Msid | None:
    """
    Get the attitude quaternion telemetry for the observation.

    Parameters
    ----------
    start : CxoTimeLike
        Start time for the observation (kalman_start).
    stop : CxoTimeLike
        Stop time for the observation (kalman_stop).

    Returns
    -------
    fetch.Msid | None
        The attitude quaternion telemetry for the observation or None if the telemetry
        does not cover start/stop within 10 seconds.
    """
    q_att = fetch.Msid("quat_aoattqt", start, stop)

    tstart = CxoTime(start).secs
    tstop = CxoTime(stop).secs
    if abs(tstart - q_att.times[0]) > 10 or abs(tstop - q_att.times[-1]) > 10:
        return None

    return q_att


def get_aberration_correction(obsid, source):
    """
    Get the aberration correction values for a given observation ID (obsid).

    This function retrieves the aberration correction values (aber-Y and aber-Z)
    from the ManErr.txt file located in the mission planning directory for the
    specified obsid. If the directory or file is not found, or if there are
    issues with the data, appropriate flags and default values are returned.

    Parameters
    ----------
    obsid : int
        Observation ID for which to retrieve aberration correction values.

    Returns
    -------
    dict
        - status (str): Status of the aberration correction retrieval:
            - "OK": Aberration correction values successfully retrieved.
            - "No ManErr directory": ManErr directory not found.
            - "No ManErr files": No ManErr files found in the directory.
            - "Multiple ManErr files": Multiple ManErr files found in the directory.
            - "Multiple entries in ManErr": Multiple entries found for the obsid.
        - y (float): Aberration correction for the Y-axis if found.
        - z (float): Aberration correction for the Z-axis if found.
    """
    load_dir = parse_cm.paths.load_dir_from_load_name(source)
    manerr_dir = load_dir / "output"

    if not manerr_dir.exists():
        logger.warning(f"No directory {manerr_dir}, Skipping aber correction.")
        return {"status": "No ManErr directory"}

    # Find unique ManErr file in the directory
    manerr_files = list(manerr_dir.glob("*_ManErr.txt"))
    if (n_files := len(manerr_files)) != 1:
        word = "Multiple" if n_files > 1 else "No"
        logger.warning(
            f"{word} ManErr file(s) in {manerr_dir}, skipping aber correction."
        )
        return {"status": f"{word} ManErr files"}

    manerr_file = manerr_files[0]
    dat = Table.read(
        manerr_file, format="ascii.basic", guess=False, header_start=2, data_start=3
    )
    ok = dat["obsid"] == obsid

    if (n_ok := np.count_nonzero(ok)) == 0:
        logger.warning(
            f"No entry for {obsid} in {manerr_file}. Skipping aber correction"
        )
        return {"status": "No entry in ManErr"}
    elif n_ok > 1:
        logger.info(
            f"More than one entry for {obsid} in {manerr_file}. Skipping aber correction"
        )
        return {"status": "Multiple entries in ManErr"}

    aber_y = dat["aber-Y"][ok][0]
    aber_z = dat["aber-Z"][ok][0]
    return {"status": "OK", "y": aber_y, "z": aber_z}


def plot_n_kalman_delta_roll(
    start,
    stop,
    obc_gnd_att_deltas: dict[str, npt.NDArray] | None,
    save_path: Path | None = None,
):
    """
    Fetch and plot number of Kalman stars for the obsid.
    """
    n_kalman = fetch_eng.Msid("aokalstr", start, stop)

    fig, ax = plt.subplots(figsize=(8, 2.5))

    if obc_gnd_att_deltas:
        ax2 = ax.twinx()
        color2 = "C1"
        plot_cxctime(
            obc_gnd_att_deltas["time"],
            obc_gnd_att_deltas["d_roll"],
            color=color2,
            alpha=0.5,
            lw=3,
            ax=ax2,
        )
        ax2.set_ylabel("OBC - GND d_roll (arcsec)", color=color2)
        ax2.set_ylim(-100, 100)
        ax2.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
        ax2.axhline(0, color=color2, lw=2, linestyle=":")
        ax2.tick_params(axis="y", labelcolor=color2)
    else:
        logger.info("No OBC - GND deltas available, not plotting")

    # The Kalman vals are strings, so use raw_vals.
    color1 = "k"
    plot_cxctime(n_kalman.times, n_kalman.raw_vals, color=color1, ax=ax)
    ax.set_ylabel("# Kalman stars", color=color1)
    ax.set_ylim(-0.2, 8.2)

    # Draw a line to indicate 1 ksec
    t0 = n_kalman.times[0]
    plot_cxctime([t0, t0 + 1000], [0.5, 0.5], lw=2, color="red", ax=ax)
    ax.text(CxoTime(t0).plot_date, 0.7, "1 ksec")

    ax.grid(ls=":")
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.95)
    plt.tight_layout()

    if save_path:
        logger.info(f"Writing plot file {save_path}")
        fig.savefig(save_path)
        plt.close()


def plot_crs_time(crs: CentroidResiduals, save_path: Path | None = None):
    """
    Make png plot of OBC centroid residuals in each slot.

    Residuals computed using ground attitude solution for science observations
    and OBC attitude solution for ER observations.

    Parameters
    ----------
    crs : dict
        Dictionary of CentroidResiduals objects keyed by slot.
    save_path : Path, optional
        Path to save the plot if not None.
    """

    colors = {"yag": "k", "zag": "slategray"}

    n_slots = len(crs)
    fig, axes = plt.subplots(nrows=n_slots, ncols=1, figsize=(8, n_slots * 7 / 8))

    legend = False

    for slot, ax in zip(crs, axes, strict=True):
        for coord in ["yag", "zag"]:
            resids_obc = getattr(crs[slot], f"d{coord}s")
            times_obc = getattr(crs[slot], f"{coord}_times")

            if len(times_obc) == 0:
                continue

            ok = np.abs(resids_obc) <= 5

            ax.plot(
                times_obc - times_obc[0],
                np.ma.array(resids_obc, mask=~ok),
                color=colors[coord],
                alpha=0.9,
                label=f"d{coord}",
            )

            if np.sum(~ok) > 0:
                ax.plot(
                    times_obc - times_obc[0],
                    np.ma.array(resids_obc, mask=ok),
                    color="crimson",
                    alpha=0.9,
                )

        ax.grid(ls=":")
        ax.set_ylim(-8, 8)
        ax.set_yticks([-5, 0, 5], ["-5", "0", "5"])
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel(f"Slot {slot}\n(arcsec)")

        ax.get_yaxis().set_label_coords(-0.065, 0.5)
        _, labels = ax.get_legend_handles_labels()
        if len(labels) > 0 and not legend:
            ax.legend(loc=1)
            legend = True

    plt.subplots_adjust(left=0.1, right=0.95, hspace=0, bottom=0.10, top=0.98)

    if save_path:
        logger.info(f"Writing plot file {save_path}")
        fig.savefig(save_path)
        plt.close(fig)


def plot_crs_scatter(
    starcat: "ACATable",
    crs: dict[int, CentroidResiduals],
    scale: float = 20,
    save_path: Path | None = None,
):
    """
    Make visual plot of OBC centroid residuals.

    Plot visualization of OBC centroid residuals with respect to ground (obc)
    aspect solution for science (ER) observations in the yang/zang plain.

    Parameters
    ----------
    starcat : ACATable
        Star catalog table.
    crs : dict
        Dictionary of CentroidResiduals objects keyed by slot.
    scale : float, optional
        Scale factor for residuals.
    save_path : Path, optional
        Path to save the plot if not None.
    """
    # Relabel idx in a copy of starcat so plot is numbered by slot.
    cat = starcat.copy()
    cat["idx"] = cat["slot"]

    # Use star catalog for field stars
    stars = agasc.get_stars(cat["id"])

    fig, ax = plt.subplots(figsize=(6, 6))
    chandra_aca.plot.plot_stars(cat.att, cat, stars, ax=ax)
    colors = ["orange", "forestgreen", "steelblue", "maroon", "gray"]

    for entry in cat:
        yag = entry["yang"]
        zag = entry["zang"]
        slot = entry["slot"]
        # Skip if no centroid residuals
        if slot not in crs:
            continue
        row, col = yagzag_to_pixels(yag, zag)
        # 1 px -> factor px; 5 arcsec = 5 * factor arcsec
        # Minus sign for y-coord to reflect sign flip in the pixel
        # to yag conversion and yag scale going from positive to negative
        yy = row - crs[slot].dyags * scale
        zz = col + crs[slot].dzags * scale
        ax.plot(yy, zz, alpha=0.3, marker=",", color=colors[slot - 3])
        circle = plt.Circle((row, col), 5 * scale, color="darkorange", fill=False)
        ax.add_artist(circle)

    plt.text(-511, 530, "ring radius = 5 arcsec (scaled)", color="darkorange")

    if save_path:
        logger.info(f"Writing plot file {save_path}")
        fig.savefig(save_path)
        plt.close()


def update_starcat_summary(
    start,
    stop,
    starcat: "ACATable",
    crs: dict[int, CentroidResiduals],
):
    """Update starcat in place with median observed mag, dyag, dzag values."""
    for name in ["dyag", "dzag", "mag"]:
        starcat[f"{name}_median"] = np.nan

    for entry in starcat:
        slot = entry["slot"]
        if slot not in crs:
            continue
        entry["dyag_median"] = np.median(crs[slot].dyags)
        entry["dzag_median"] = np.median(crs[slot].dzags)
        mags = fetch.Msid(f"aoacmag{slot}", start, stop)
        entry["mag_median"] = np.median(mags.vals)


def write_info_json(obs: Observation, info_json_path: Path):
    """Write the info.json file for the observation."""
    out = obs.info.copy()
    for key, val in out.items():
        if isinstance(val, float):
            out[key] = round(val, 3)
    logger.info(f"Writing {info_json_path}")
    info_json_path.write_text(json.dumps(out, indent=2, sort_keys=True))


def write_centroid_resids(crs: dict[int, CentroidResiduals], save_path: Path) -> None:
    """Write the centroid residuals to a file.

    This creates a dictionary with the slot number as the key and a dictionary with the
    start and stop times, the number of times, and the dyags and dzags as the values.
    The structure is::

    {
        slot: {
            "tstart": t0,
            "tstop": t1,
            "n_times": n_times,
            "dyags": dyags,
            "dzags": dzags,
        },
    }

    Notes:
    - The dyags and dzags are interpolated to a common time grid based on the median
      difference between the yag times.
    - The times are in CXC seconds.
    - The dyags and dzags are in arcseconds.
    - The output is a pickle file that can be loaded later.
    - The file is saved in the report directory for the observation.
    - The dyags and dzags are saved as float16 to save space. The max error is around
      0.002 arcsec.

    Parameters
    ----------
    crs : dict
        Dictionary of CentroidResiduals objects keyed by slot.
    save_path : Path
        Path to save the centroid residuals file.
    """
    out = {}
    for slot, cr in crs.items():
        t0 = max(cr.yag_times[0], cr.zag_times[0])
        t1 = min(cr.yag_times[-1], cr.zag_times[-1])
        ok = (cr.yag_times >= t0) & (cr.yag_times <= t1)
        if np.count_nonzero(ok) < 4:
            logger.warning(
                f"Not enough points in slot {slot} to write centroid residuals"
            )
            continue
        dt_median = np.median(np.diff(cr.yag_times[ok]))
        n_times = int(np.round(t1 - t0) / dt_median) + 1
        times = np.linspace(t0, t1, n_times)
        info_slot = {"tstart": t0, "tstop": t1, "n_times": n_times}
        info_slot["dyags"] = np.interp(times, cr.yag_times, cr.dyags).astype(np.float16)
        info_slot["dzags"] = np.interp(times, cr.zag_times, cr.dzags).astype(np.float16)
        out[slot] = info_slot

    logger.info(f"Writing {save_path} with {len(out)} slots")
    save_path.write_bytes(pickle.dumps(out))


class SkipObservation(Exception):
    """Skip observation exception.

    This is used to skip observations that have already been processed or have no
    telemetry.
    """

    def __init__(self, message: str):
        super().__init__(message)
        logger.info(message)


def process_obs(obs: Observation, opt: argparse.Namespace):
    """Process the observation."""
    if opt.obsid and obs.obsid != opt.obsid:
        logger.info(f"ObsID {obs.obsid} does not match requested obsid {opt.obsid}")
        return

    report_dir = obs.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    info_json_path = obs.report_dir / "info.json"
    if opt.force:
        info_json_path.unlink(missing_ok=True)

    # Skip processing if next_obsid is already in info.json. This implies that the
    # observation has already been processed.
    if processed(info_json_path):
        logger.info(f"ObsID {obs.obsid} already processed, skipping")
        return

    if obs.manvr_event is None:
        raise SkipObservation(
            f"ObsID {obs.obsid} has no maneuver event in telemetry, skipping"
        )

    # Check if telemetry is available, using AOATTQT as a proxy for all telemetry.
    if obs.q_att_obc is None:
        raise SkipObservation(f"ObsID {obs.obsid} has insufficient telemetry, skipping")

    start, stop = obs.kalman_start, obs.kalman_stop
    crs = get_centroid_resids(start, stop, obs.starcat, obs.q_att_obc)

    if not opt.skip_plots:  # for testing
        plot_crs_time(crs, report_dir / "centroid_resids_time.png")
        plot_n_kalman_delta_roll(
            start, stop, obs.att_deltas, report_dir / "n_kalman_delta_roll.png"
        )
        plot_crs_scatter(
            obs.starcat, crs, save_path=report_dir / "centroid_resids_scatter.png"
        )

    update_starcat_summary(start, stop, obs.starcat, crs)
    make_html(obs)
    write_centroid_resids(crs, report_dir / "centroid_resids.pkl")
    write_info_json(obs, info_json_path)


def main(args=None):
    # Always non-interactive plots for command-line app
    plt.switch_backend("agg")

    opt = get_opt().parse_args(args)
    logger.setLevel(opt.log_level)

    stop = CxoTime(opt.stop)
    start = CxoTime(opt.start) if opt.start else stop - NDAYS_DEFAULT * u.day
    logger.info(f"Processing from {start} to {stop}")

    obss = get_observations(start, stop, opt)
    logger.info(f"Found {len(obss)} observations")

    for idx, obs in enumerate(obss):
        logger.info("*" * 80)
        logger.info(f"Processing observation {obs.obsid}")
        logger.info("*" * 80)

        obs.obs_prev = obss[idx - 1] if idx > 0 else obs.obs_link_from_info("prev")
        obs.obs_next = (
            obss[idx + 1] if idx < len(obss) - 1 else obs.obs_link_from_info("next")
        )

        try:
            process_obs(obs, opt)

        except SkipObservation as err:
            try:
                make_html(obs, traceback=str(err))
            except Exception as err:
                logger.error(f"Error making traceback HTML for {obs.obsid}: {err}")
            (obs.report_dir / "info.json").unlink(missing_ok=True)

        except Exception as err:
            if opt.raise_exc:
                raise
            logger.info(f"Error processing {obs.obsid}: {err}")
            import traceback

            tb = traceback.format_exc()
            logger.error(f"Error processing {obs.obsid}:\n{tb}")
            try:
                # This SHOULD always work to preserve navigation and the traceback, but
                # if it fails we'll just have to live with it.
                make_html(obs, traceback=tb)
            except Exception as err:
                logger.error(f"Error making traceback HTML for {obs.obsid}: {err}")

            # Just in case, remove info file so observation is reprocessed next time.
            (obs.report_dir / "info.json").unlink(missing_ok=True)

        logger.info("")


if __name__ == "__main__":
    main()
