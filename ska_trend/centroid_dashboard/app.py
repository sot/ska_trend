import argparse
import copy
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import astropy.units as u
import kadi.commands as kc
import kadi.events as ke
import numpy as np
import parse_cm.paths
import razl.observations
from chandra_aca.centroid_resid import CentroidResiduals
from cxotime import CxoTime, CxoTimeLike  # , CxoTimeDescriptor
from jinja2 import Template
from matplotlib import pyplot as plt
from ska_helpers.logging import basic_logger
from starcheck.state_checks import calc_man_angle_for_duration

from . import paths

# Update guide metrics file with new obsids between NOW and (NOW - NDAYS_DEFAULT) days
NDAYS_DEFAULT = 7

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

    def processed(self, info_json_path: Path):
        """Check if the observation has already been processed.

        The check is based on the existence of the info.json file and the non-None
        values obsid_next and obsid_prev keys in the file.
        """
        if not info_json_path.exists():
            return False
        info = json.loads(info_json_path.read_text())
        return info["obsid_next"] is not None and info["obsid_prev"] is not None

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
    def date(self) -> str:
        """Date of start of KALMAN from telemetry.

        TODO: change name to date_kalman_start?
        """
        return self.manvr_event.kalman_start

    @functools.cached_property
    def info(self) -> str:
        attrs = [
            "obsid",
            "aber_y",
            "aber_z",
            "date_starcat",
            "date",
            "manvr_angle",
            "obsid_next",
            "obsid_prev",
            "one_shot",
            "one_shot_aber_corrected",
            "one_shot_pitch",
            "one_shot_yaw",
            "roll_err_ending",
            "roll_err_prev",
        ]
        out = {}
        for attr in attrs:
            val = getattr(self, attr)
            if isinstance(val, float):
                val = round(val, 3)
            out[attr] = val

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
    def manvr_angles_text(self):
        if len(self.manvr_angles) == 1:
            out = ""
        else:
            angles = " + ".join([f"{angle:.1f}" for angle in self.manvr_angles])
            out = f" Segmented: {angles} deg"
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
    def dr50(self):
        return 0.0

    @functools.cached_property
    def dr95(self):
        return 0.0

    @functools.cached_property
    def roll_err_ending(self):
        return 0.0

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
        obss.append(obs)
        logger.info(
            f"Found observation {obs.obsid} at {obs.obs_start} with {len(obs.manvrs)} manvrs"
        )

    return obss


def make_html(obs: Observation, opt: argparse.Namespace):
    """Make the HTML file for the observation."""
    logger.debug(f"Making HTML for observation {obs.obsid}")
    # Get the template from index_template.html
    template = Template(get_index_template())
    context = {"MICA_PORTAL": "https://icxc.harvard.edu/mica", "obs": obs}
    html = template.render(**context)

    path = paths.index_html(obs, opt.data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {path}")
    path.write_text(html)


def get_crs(obs):
    """
    Get OBC centroid residuals per obsid for all slots.

    This is with respect to both the OBC and ground aspect solution.

    :param obsid: obsid
    """
    crs = {}
    ok = np.isin(obs.starcat["type"], ["BOT", "GUI"])
    cat = obs.starcat[ok]

    cr = CentroidResiduals(
        start=obs.manvr_event.kalman_start,
        stop=obs.manvr_event.npnt_stop,
    )
    cr.set_atts("obc")

    for slot, agasc_id in zip(cat["slot"], cat["id"], strict=True):
        try:
            cr.set_centroids("obc", slot=slot)
            cr.set_star(agasc_id=agasc_id)
            cr.calc_residuals()
            crs[slot] = copy.copy(cr)
        except Exception:
            crs[slot] = None
            print(f"Could not compute crs for {obs.obsid} slot {slot})")

    return crs


def plot_crs_per_obsid(crs, save_path=None):
    """
    Make png plot of OBC centroid residuals in each slot.

    Residuals computed using ground attitude solution for science observations
    and OBC attitude solution for ER observations.

    :param crs: dictionary with keys 'ground' and 'obc', dictionary values
                are also dictionaries keyed by slot number containing
                corresponding CentroidResiduals objects
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
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    # return crs


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
    if obs.processed(info_json_path):
        logger.info(f"ObsID {obs.obsid} already processed, skipping")
        return

    crs = get_crs(obs)
    plot_crs_per_obsid(
        crs,
        save_path=paths.report_dir(obs, opt.data_root) / "centroid_resids.png",
    )

    make_html(obs, opt)
    info_json_path.write_text(json.dumps(obs.info, indent=2, sort_keys=True))


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
            f"Processing observation {obs.obsid} {obs.obsid_prev} {obs.obsid_next}"
        )
        process_obs(obs, opt)


if __name__ == "__main__":
    main()
