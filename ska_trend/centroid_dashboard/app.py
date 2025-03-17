import argparse
import copy
import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
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
from chandra_aca.centroid_resid import CentroidResiduals
from chandra_aca.transform import yagzag_to_pixels
from cheta import fetch, fetch_eng
from cxotime import CxoTime, CxoTimeLike
from jinja2 import Environment
from matplotlib import pyplot as plt
from mica.archive import asp_l1
from Quaternion import Quat
from ska_helpers.logging import basic_logger
from ska_matplotlib import plot_cxctime
from starcheck.state_checks import calc_man_angle_for_duration

from . import paths

if TYPE_CHECKING:
    from proseco.catalog import ACATable

# Update guide metrics file with new obsids between NOW and (NOW - NDAYS_DEFAULT) days
NDAYS_DEFAULT = 7
SKA = Path(os.environ["SKA"])

logger = basic_logger("centroid_dashboard")


def get_opt():
    parser = argparse.ArgumentParser(description="Centroid dashboard")
    parser.add_argument("--obsid", help="Processing obsid (default=None)")
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
        help="Copy data from remote archive if not available locally",
        action="store_true",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default=INFO)"
    )
    return parser


@functools.lru_cache()
def get_index_template():
    path = Path(__file__).parent / "index_template.html"
    return path.read_text()


@dataclass(repr=False, kw_only=True)
class Observation(razl.observations.Observation):
    obs_next: Optional["Observation"] = None
    obs_prev: Optional["Observation"] = None

    @functools.cached_property
    def report_subdir(self):
        year = parse_cm.paths.parse_load_name(self.source)[-1]
        return Path(str(year), self.source, f"{self.obsid:05d}")

    @functools.cached_property
    def manvr_event(self):
        """Provide the maneuver event leading to this observation."""
        # Manvr leading to this observation. Remember the Manvr class includes the
        # maneuver and info about the dwell.
        manvr = self.manvrs[-1]
        start = manvr.start - 1 * u.min
        stop = manvr.stop + 5 * u.min
        manvrs = ke.manvrs.filter(start, stop)
        if len(manvrs) == 0:
            # Most commonly because telemetry does not include this maneuver yet.
            out = None
        elif len(manvrs) > 2:
            # Should never have 3 or more manvrs in a row.
            raise ValueError(
                f"Multiple manvrs found between {start} and {stop}:\n{manvrs}"
            )
        else:
            # Take the last maneuver before the observation. Segmented maneuvers or
            # the high-IR zone dwell are common cases of 2 manvrs.
            out = manvrs[len(manvrs) - 1]  # Negative indexing not supported
        return out

    @functools.cached_property
    def aber_y(self) -> float:
        return 0.0

    @functools.cached_property
    def aber_z(self) -> float:
        return 0.0

    @functools.cached_property
    def kalman_start(self) -> str:
        """Date of start of KALMAN from telemetry."""
        return self.manvr_event.kalman_start

    @functools.cached_property
    def kalman_stop(self) -> str:
        """Date of stop of KALMAN from telemetry."""
        return self.manvr_event.npnt_stop

    @functools.cached_property
    def info(self) -> str:
        attrs = [
            "obsid",
            "aber_y",
            "aber_z",
            "date_starcat",
            "kalman_start",
            "kalman_stop",
            "manvr_angle",
            "obsid_next",
            "obsid_prev",
            "one_shot",
            "one_shot_aber_corrected",
            "one_shot_pitch",
            "one_shot_yaw",
            "roll_err_prev",
        ]
        out = {}
        for attr in attrs:
            out[attr] = getattr(self, attr)

        return out

    @functools.cached_property
    def one_shot_aber_corrected(self):
        return 0.0

    @functools.cached_property
    def one_shot(self):
        return self.manvr_event.one_shot

    @functools.cached_property
    def one_shot_pitch(self):
        return self.manvr_event.one_shot_pitch

    @functools.cached_property
    def one_shot_yaw(self):
        return self.manvr_event.one_shot_yaw

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
    def manvr_angle(self):
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
    def obsid_next(self):
        return self.obs_next.obsid if self.obs_next else None

    @functools.cached_property
    def obsid_prev(self):
        return self.obs_prev.obsid if self.obs_prev else None

    @functools.cached_property
    def roll_err_prev(self):
        return 0.0

    @functools.cached_property
    def date_starcat(self):
        """Date of MP_STARCAT command"""
        return self.starcat.date

    @functools.cached_property
    def starcat_summary(self):
        summary = self.starcat.copy()
        summary["median_mag"] = 0.0
        summary["median_dy"] = 0.0
        summary["median_dz"] = 0.0
        return summary


def processed(info_json_path: Path):
    """Check if the observation has already been fully processed.

    The check is based on the existence of the info.json file and the non-None values
    obsid_next and obsid_prev keys in the file. This file is created/updated at the end
    of the processing.
    """
    if not info_json_path.exists():
        return False
    info = json.loads(info_json_path.read_text())
    return info["obsid_next"] is not None and info["obsid_prev"] is not None


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
    if obsid >= 38000:
        return None

    obsid_str = f"{obsid:05d}"
    obs2_dir = f"data/mica/archive/asp1/{obsid_str[:2]}"
    obs2_dir_local = SKA / obs2_dir
    obsid_dir_remote = f"kady:/proj/sot/ska/{obs2_dir}/{obsid_str}*"
    obsid_dir_local = obs2_dir_local / obsid_str

    if not obsid_dir_local.exists():
        if remote_copy:
            obs2_dir_local.mkdir(parents=True, exist_ok=True)
            cmds = ["rsync", "-av", obsid_dir_remote, f"{obs2_dir_local}/"]
            logger.info(f"Copying remote data with command: {' '.join(cmds)}")
            subprocess.check_call(cmds)
        else:
            return [], []

    atts, atts_times, _ = asp_l1.get_atts(obsid=obsid)

    return atts, atts_times


def get_obc_gnd_att_deltas(
    obsid: int, q_att_obc: fetch.Msid, remote_copy: bool = False
) -> dict[str]:
    """
    Get OBC pitch, yaw, roll errors with respect to the ground aspect solution.

    Parameters
    ----------
    obsid : int
        Observation ID. If obsid >= 38000, return None.

    """
    if obsid >= 38000:
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
    out["d_roll50"] = np.percentile(np.abs(out["d_roll"]), 50)
    out["d_roll95"] = np.percentile(np.abs(out["d_roll"]), 95)
    out["d_roll_end"] = out["d_roll"][-1]

    return out


def get_observations(start: CxoTimeLike, stop: CxoTimeLike) -> list[Observation]:
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
        opt={"aca_kwargs": None, "raise_exc": False},
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
        # In this application we only care about guide star slots
        obs.starcat = obs.starcat[np.isin(obs.starcat["type"], ["BOT", "GUI"])]
        obss.append(obs)
        logger.info(
            f"Found observation {obs.obsid} at {obs.obs_start} with {len(obs.manvrs)} manvrs"
        )

    return obss


def path_exists(path: Path | str) -> bool:
    """Check if the path exists."""
    return Path(path).exists()


def make_html(obs: Observation, opt: argparse.Namespace):
    """Make the HTML file for the observation."""
    logger.debug(f"Making HTML for observation {obs.obsid}")
    # Get the template from index_template.html
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(get_index_template())
    context = {
        "MICA_PORTAL": "https://icxc.harvard.edu/mica",
        "obs": obs,
    }
    html = template.render(**context)

    path = paths.index_html(obs, opt.data_root)
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
            crs[slot] = copy.copy(cr)
        except Exception:
            crs[slot] = None
            print(f"Could not compute crs for slot {slot})")

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

    if obc_gnd_att_deltas is not None:
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
        ylim = max(np.abs(ax2.get_ylim()).max(), 50)
        ax2.set_ylim(-ylim, ylim)
        ax2.tick_params(axis="y", labelcolor=color2)

    # The Kalman vals are strings, so use raw_vals.
    color1 = "k"
    plot_cxctime(n_kalman.times, n_kalman.raw_vals, color=color1, ax=ax)
    ax.set_ylabel("# Kalman stars", color=color1)
    ax.set_ylim(-0.2, 8.2)

    # Draw a line to indicate 1 ksec
    t0 = n_kalman.times[0]
    plot_cxctime([t0, t0 + 1000], [0.5, 0.5], lw=3, color="orange", ax=ax)
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
        ylims = ax.get_ylim()
        if max(np.abs(ylims)) < 5:
            ax.set_ylim(-6, 6)
            ax.set_yticks([-5, 0, 5], ["-5", "0", "5"])
        else:
            ax.set_ylim(-12, 12)
            ax.set_yticks([-10, -5, 0, 5, 10], ["-10", "-5", "0", "5", "10"])
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel(f"Slot {slot}\n(arcsec)")

        ax.get_yaxis().set_label_coords(-0.065, 0.5)
        _, labels = ax.get_legend_handles_labels()
        if len(labels) > 0 and not legend:
            ax.legend(loc=1)
            legend = True

    plt.subplots_adjust(left=0.1, right=0.95, hspace=0, bottom=0.08, top=0.98)

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

    fig, ax = plt.subplots(figsize=(5, 5))
    chandra_aca.plot.plot_stars(cat.att, cat, stars, ax=ax)
    colors = ["orange", "forestgreen", "steelblue", "maroon", "gray"]

    for entry in cat:
        yag = entry["yang"]
        zag = entry["zang"]
        slot = entry["slot"]
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


def write_info_json(obs: Observation, info_json_path: Path):
    """Write the info.json file for the observation."""
    out = obs.info.copy()
    for key, val in out.items():
        if isinstance(val, float):
            out[key] = round(val, 3)
    info_json_path.write_text(json.dumps(out, indent=2, sort_keys=True))


def process_obs(obs: Observation, opt: argparse.Namespace):
    """Process the observation."""
    if obs.manvr_event is None:
        logger.info(f"ObsID {obs.obsid} has no maneuver event in telemetry, skipping")
        return

    info_json_path = paths.info_json(obs, opt.data_root)
    if opt.force:
        info_json_path.unlink(missing_ok=True)

    # Skip processing if next_obsid is already in info.json. This implies that the
    # observation has already been processed.
    if processed(info_json_path):
        logger.info(f"ObsID {obs.obsid} already processed, skipping")
        return

    report_dir = paths.report_dir(obs, opt.data_root)
    report_dir.mkdir(parents=True, exist_ok=True)

    start, stop = obs.kalman_start, obs.kalman_stop
    if (q_att_obc := fetch.Msid("quat_aoattqt", start, stop)) is None:
        logger.info(f"ObsID {obs.obsid} has insufficient attitude telemetry, skipping")
        return

    obc_gnd_att_deltas = get_obc_gnd_att_deltas(
        obs.obsid, q_att_obc, remote_copy=opt.remote_copy
    )
    if obc_gnd_att_deltas is not None:
        for key in ["d_roll50", "d_roll95", "d_roll_end"]:
            obs.info[key] = obc_gnd_att_deltas[key]

    crs = get_centroid_resids(start, stop, obs.starcat, q_att_obc)

    plot_crs_time(crs, report_dir / "centroid_resids_time.png")
    plot_n_kalman_delta_roll(
        start, stop, obc_gnd_att_deltas, report_dir / "n_kalman_delta_roll.png"
    )
    plot_crs_scatter(
        obs.starcat, crs, save_path=report_dir / "centroid_resids_scatter.png"
    )

    make_html(obs, opt)
    write_info_json(obs, info_json_path)


def main(args=None):
    # Always non-interactive plots for command-line app
    plt.switch_backend("agg")

    opt = get_opt().parse_args(args)
    logger.setLevel(opt.log_level)

    stop = CxoTime(opt.stop)
    start = CxoTime(opt.start) if opt.start else stop - NDAYS_DEFAULT * u.day
    logger.info(f"Processing from {start} to {stop}")

    obss = get_observations(start, stop)
    logger.info(f"Found {len(obss)} observations")

    for idx, obs in enumerate(obss):
        obs.obs_prev = obss[idx - 1] if idx > 0 else None
        obs.obs_next = obss[idx + 1] if idx < len(obss) - 1 else None

        logger.info(
            f"Processing observation {obs.obsid} prev={obs.obsid_prev} next={obs.obsid_next}"
        )
        try:
            process_obs(obs, opt)
        except Exception as err:
            print(f"Error processing {obs.obsid}: {err}")
            import traceback

            tb = traceback.format_exc()
            logger.error(f"Error processing {obs.obsid}:\n{tb}")


if __name__ == "__main__":
    main()
