import argparse
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import astropy.units as u
import kadi.commands as kc
import kadi.events as ke
import numpy as np
import razl.observations
from cxotime import CxoTime, CxoTimeLike  # , CxoTimeDescriptor
from jinja2 import Template
from ska_helpers.logging import basic_logger
from starcheck.state_checks import calc_man_angle_for_duration

# Update guide metrics file with new obsids between NOW and (NOW - NDAYS_DEFAULT) days
NDAYS_DEFAULT = 7


# @dataclass
# class Observation:
#     obsid: int  # Obsid planned (from schedule)
#     obsid_tlm: int  # Obsid from telemetry
#     starcat_date: CxoTimeDescriptor | None = CxoTimeDescriptor()

# droll_50: float | None = None
# droll_95: float | None = None
# one_shot: float
# one_shot_pitch: float
# one_shot_yaw: float
# manvr_angle: float
# obsid_prev: int
# roll_err_ending: float
# roll_err_prev: float
# aber_y: float
# aber_z: float
# aber_flag: int
# one_shot_aber_corrected: float
# obsid_next: int
# att_errors: dict
# att_flag: int
# dwell: bool

logger = basic_logger("centroid_dashboard")


def index_html_path(obs, opt):
    return Path(opt.data_root) / "reports" / str(obs.obsid) / "index.html"


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

    def processed(self):
        return False

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
        return self.manvr_event.kalman_start

    @functools.cached_property
    def context(self):
        context = {"MICA_PORTAL": "https://icxc.harvard.edu/mica", "obs": self}
        return context

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
    """Make the HTML file for the observation.

    It includes the following Jinja variable references:
    - aber_y
    - aber_z
    - dr50
    - dr95
    - roll_err_ending
    - manvr_angle
    - mean_date
    - MICA_PORTAL
    - obsid_next_url
    - obsid
    - obsid_next
    - obsid_prev
    - one_shot
    - one_shot_aber_corrected
    - one_shot_pitch
    - one_shot_yaw
    - obsid_prev_url
    - roll_err_prev
    - starcat
    """
    logger.info(f"Making HTML for observation {obs.obsid}")
    # Get the template from index_template.html
    template = Template(get_index_template())

    html = template.render(**obs.context)
    path = index_html_path(obs, opt)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {path}")
    path.write_text(html)


def main(args=None):
    opt = get_opt().parse_args(args)
    logger.setLevel(opt.log_level)

    stop = CxoTime(opt.stop)
    start = CxoTime(opt.start) if opt.start else stop - NDAYS_DEFAULT * u.day
    logger.info(f"Processing from {start} to {stop}")

    obss = get_observations(start, stop)
    logger.info(f"Found {len(obss)} observations")

    for idx, obs in enumerate(obss):
        obs_prev = obss[idx - 1] if idx > 0 else None
        obs_next = obss[idx + 1] if idx < len(obss) - 1 else None
        if obs_next is not None:
            obs.obs_next = obs_next
        if obs_prev is not None:
            obs.obs_prev = obs_prev

        if obs.processed():
            logger.info(f"Skipping processed observation {obs.obsid}")
            continue
        # if obs_prev is None:
        #     obs_prev = Observation.from_json()
        logger.info(f"Processing observation {obs.obsid}")
        make_html(obs, opt)


if __name__ == "__main__":
    main()
