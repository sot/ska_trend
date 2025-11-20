"""
Observation class.

This module contains the Observation class for processing periscope drift data.
The Observation class is a subclass of astromon.observation.Observation.
"""

import functools
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astromon
import astromon.observation
import numpy as np
import scipy
import scipy.interpolate
import ska_numpy
from astromon import source_detection
from astromon.db import is_in_excluded_region
from astromon.stored_result import StorableClass, stored_result
from astromon.task import ReturnCode, run_tasks
from astropy import table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from chandra_aca.transform import radec_to_yagzag
from cheta import fetch
from cxotime import CxoTime
from Quaternion import Quat
from scipy.interpolate import BSpline, make_smoothing_spline

from ska_trend.periscope_drift.correction import (
    get_expected_correction as _get_expected_correction,
)

logger = logging.getLogger("periscope_drift")


ARCHIVE_DIR = (
    Path(os.environ["SKA"]) / "data" / "periscope_drift_reports" / "xray_observations"
)


EXCLUDED_SOURCES_FILE = (
    Path(os.environ["SKA"]) / "data" / "periscope_drift_reports" / "excluded_sources.h5"
)


GRADIENTS = [
    "SourceData",
    "ObservationData",
    "Observation",
    "fetch_telemetry",
    "smooth",
    "line",
    "prob",
]


@dataclass
class SourceData:
    source: dict[str, Any]
    summary: dict[str, Any]
    events: table.Table
    binned_data_1d: table.Table
    fits_1d: table.Table
    yag_vs_time: Any
    zag_vs_time: Any
    spline_fit: dict[str, Any]


@dataclass
class ObservationData:
    summary: table.Table
    sources: table.Table
    data: dict[int, SourceData]
    expected_correction: dict[str, Any]


class PeriscopeDriftData(StorableClass):
    def __init__(self, obs):
        """
        Class with methods related to periscope drift data.

        Parameters
        ----------
        obs : Observation
            Astromon Observation object.
        """
        self.obs = obs
        super().__init__(
            archive_dir=ARCHIVE_DIR,
            subdir=Path(f"obs{int(obs.obsid) // 1000:02d}") / f"{obs.obsid}",
        )

    def is_selected(self):
        obsid_info = self.obs.get_info()
        return (
            not self.obs.is_multi_obi
            # and obsid_info['category_id'] not in [110]
            and obsid_info["obs_mode"] == "POINTING"
            and obsid_info["grating"] == "NONE"
            and (
                obsid_info["instrume"] == "HRC"
                or (
                    obsid_info["instrume"] == "ACIS"
                    and obsid_info["readmode"] == "TIMED"
                    and int(obsid_info["dtycycle"]) == 0
                )
            )
        )

    @staticmethod
    def is_selected_source(source, exclude_regions=True):
        """
        Return a boolean array indicating which sources are selected.

        The function can optionally exclude sources in astromon excluded regions. Excluded regions
        can change at any time, so by default we do not use them when caching source data.

        Parameters
        ----------
        source : astropy.table.Table
            Table with source data.
        exclude_regions : bool, optional
            Whether to exclude sources in excluded regions, by default True.
        Returns
        -------
        numpy.ndarray
            Boolean array indicating which sources are selected.
        """
        result = (
            PeriscopeDriftData.is_pre_selected_source(source)
            & (source["snr"] > 90)
            & (source["psfratio"] < 1.2)
            & (source["n_points"] > 2)
        )
        if exclude_regions:
            pos = SkyCoord(source["ra"] * u.deg, source["dec"] * u.deg)
            result &= ~is_in_excluded_region(
                pos, source["obsid"], dbfile=EXCLUDED_SOURCES_FILE
            )
        return result

    @staticmethod
    def is_pre_selected_source(source):
        """
        Return a boolean array indicating which sources are pre-selected.

        This method is intended to to reduce the number of sources before processing. It is not the
        final selection. As such, the criteria are looser than those in is_selected_source. The idea
        is that we often want to see how the pre-selected sources behave even if they do not pass.

        Parameters
        ----------
        source : astropy.table.Table
            Table with source data.

        Returns
        -------
        numpy.ndarray
            Boolean array indicating which sources are pre-selected.
        """
        result = (
            (source["snr"] > 3)
            & (source["net_counts"] > 200)
            # distance to closest source. Extended sources can be split into several sources
            # and we currently model sources in a 8x8 arcsec box.
            & (source["near_neighbor_dist"] > 6)
            # psfratio is the ratio of the source ellipse to the PSF size
            & (source["psfratio"] < 3.0)
        )
        return result

    def get_events(self):
        # THIS IS A HACK: the "dependencies" decorator does not work on PeriscopeDriftData
        # instances, only on Observation instances.
        rv = run_tasks(obs=self.obs, task_names=["filter_events"])

        errors = {
            name: value
            for name, value in rv.items()
            if value.return_code.value >= ReturnCode.ERROR.value
        }
        if errors:
            msg = ", ".join(f"{name} {value.msg}" for name, value in errors.items())
            raise RuntimeError(
                f"PeriscopeDriftData.get_events failed. Dependency tasks failed: {msg}"
            )

        # get events
        obs_info = self.obs.get_info()
        att = Quat([obs_info["ra_nom"], obs_info["dec_nom"], obs_info["roll_nom"]])
        evtfiles = self.obs.file_glob("primary/*_evt2_filtered.fits*")
        if not evtfiles:
            return table.Table()
        events_file = evtfiles[0]

        # add yag/zag columns to events, because residuals will be in yag/zag
        outfile = self.obs.workdir / events_file.name.replace(".fits", "_radec.fits")
        self.obs.ciao(
            "dmcopy",
            f"{events_file}[cols time,ra,dec,x,y]",
            outfile,
            clobber="yes",
            logging_tag=str(self),
        )
        # calling as_array converts to native byteorder
        # (some tools like pandas and seaborn do not support big endian on little endian machines)
        events = table.Table(table.Table.read(outfile, hdu=1).as_array())
        events["yag"], events["zag"] = radec_to_yagzag(events["RA"], events["DEC"], att)

        if len(events) > 0:
            # add a column with time in seconds since start of observation
            events["rel_time"] = events["time"] - np.min(events["time"])

        # fetch telemetry and add telemetry values to events
        telem = fetch_telemetry(
            CxoTime(float(obs_info["tstart"])),
            CxoTime(float(obs_info["tstop"])),
            times=events["time"],
        )
        events["OOBAGRD3"] = telem["OOBAGRD3"]
        events["OOBAGRD6"] = telem["OOBAGRD6"]
        events["OOBAGRD3_raw"] = telem["OOBAGRD3_raw"]
        events["OOBAGRD6_raw"] = telem["OOBAGRD6_raw"]

        if len(events) > 0:
            oobagrd3 = telem["OOBAGRD3"] - np.mean(telem["OOBAGRD3"])
            oobagrd6 = telem["OOBAGRD6"] - np.mean(telem["OOBAGRD6"])
            covariance = np.cov([oobagrd3, oobagrd6])
            if np.all(np.isfinite(covariance)):
                # this happens in rare cases, at least if there is no telemetry (obsid 28065)
                eig_vals, eig_vec = scipy.linalg.eig(covariance)
                vec = eig_vec[:, np.argmax(eig_vals)]
                events["OOBAGRD_pc1"] = oobagrd3 * vec[0] + oobagrd6 * vec[1]
            else:
                logger.debug(
                    f"OBSID={self.obs.obsid} unphysical telem covariance in get_events"
                )
                events["OOBAGRD_pc1"] = np.nan
        return events

    @stored_result("periscope_drift_sources", fmt="table", subdir="cache")
    def get_sources(self, apply_filter=True):
        src = self.obs.get_sources(version="gaussian_detect")

        if len(src) == 0:
            dtype = np.dtype(
                [
                    ("obsid", ">i8"),
                    ("id", ">i4"),
                    ("ra", ">f8"),
                    ("dec", ">f8"),
                    ("net_counts", ">f4"),
                    ("y_angle", ">f8"),
                    ("z_angle", ">f8"),
                    ("r_angle", ">f8"),
                    ("snr", ">f4"),
                    ("near_neighbor_dist", ">f8"),
                    ("psfratio", ">f4"),
                    ("pileup", ">f4"),
                    ("acis_streak", "?"),
                    ("caldb_version", "<U6"),
                ]
            )
            return table.Table(dtype=dtype)

        obs_info = self.obs.get_info()
        att = Quat([obs_info["ra_nom"], obs_info["dec_nom"], obs_info["roll_nom"]])

        src["yag"], src["zag"] = radec_to_yagzag(src["ra"], src["dec"], att)
        src.sort(["net_counts", "snr"], reverse=True)

        # determine pileup metrics for all sources
        pileup_img_file = list(self.obs.file_glob("**/*_pileup.img"))
        if pileup_img_file:
            hdus = fits.open(pileup_img_file[0])

            # for that, we add the pixel values to each source
            # maybe there is a way to do this in CIAO
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FITSFixedWarning)
                wcs = WCS(hdus[0].header)
            loc = SkyCoord(src["ra"] * u.deg, src["dec"] * u.deg)
            src["i"], src["j"] = np.round(wcs.world_to_pixel(loc)).astype(int)

            add_pileup_metric(src, hdus[0].data)
        else:
            add_pileup_metric(src, None)

        # add the maximum expected correction
        start = CxoTime(float(obs_info["tstart"]))
        stop = CxoTime(float(obs_info["tstop"]))
        telem = fetch_telemetry(start, stop)

        corr = self.get_expected_correction()

        if len(telem["time"]) > 0:
            src["d_OOBAGRD3"] = telem["OOBAGRD3"].max() - telem["OOBAGRD3"].min()
            src["d_OOBAGRD6"] = telem["OOBAGRD6"].max() - telem["OOBAGRD6"].min()
            src["OOBAGRD3_mean"] = np.mean(telem["OOBAGRD3"])
            src["OOBAGRD6_mean"] = np.mean(telem["OOBAGRD6"])

            src["max_yang_corr"] = np.max(np.abs(corr["ang_y_corr"]))
            src["max_zang_corr"] = np.max(np.abs(corr["ang_z_corr"]))
            ang_corr = np.sqrt(corr["ang_y_corr"] ** 2 + corr["ang_z_corr"] ** 2)
            src["max_ang_corr"] = np.max(np.abs(ang_corr))
        else:
            # having no telemetry is rare (obsid 28065), and these could be filtered beforehand
            # but we can be defensive and handle it properly
            src["d_OOBAGRD3"] = np.nan
            src["d_OOBAGRD6"] = np.nan
            src["OOBAGRD3_mean"] = np.nan
            src["OOBAGRD6_mean"] = np.nan

            src["max_yang_corr"] = np.nan
            src["max_zang_corr"] = np.nan
            src["max_ang_corr"] = np.nan

        # add observation duration and other small things
        src["obs_duration"] = CxoTime(stop).cxcsec - CxoTime(start).cxcsec
        src["acis"] = self.obs.is_acis
        src["hrc"] = self.obs.is_hrc

        src["theta"] = self.obs.get_off_axis_angle(ra=src["ra"], dec=src["dec"])

        formats = {
            "ra": ".5f",
            "dec": ".5f",
            "theta": ".2f",
            "y_angle": ".2f",
            "z_angle": ".2f",
            "r_angle": ".2f",
            "yag": ".2f",
            "zag": ".2f",
            "snr": ".2f",
            "psfratio": ".2f",
            "pileup": ".2f",
            "max_pileup": ".2f",
            "obs_duration": ".2f",
            "max_yang_corr": ".2f",
            "max_zang_corr": ".2f",
            "d_OOBAGRD3": ".2f",
            "d_OOBAGRD6": ".2f",
        }

        for col, fmt in formats.items():
            src[col].format = fmt

        if apply_filter:
            # this will be cached, so do not use excluded regions (they can change at any time)
            src = src[self.is_selected_source(src, exclude_regions=False)]

        return src

    @stored_result("expected_correction", fmt="pickle", subdir="cache")
    def get_expected_correction(self):
        obspar = self.obs.get_obspar()
        tstart = float(obspar["tstart"])
        tstop = float(obspar["tstop"])
        telem = fetch_telemetry(CxoTime(tstart), CxoTime(tstop))
        return _get_expected_correction(telem)

    @stored_result("periscope_drift_summary", fmt="pickle", subdir="cache")
    def get_summary(self):
        obspar = self.obs.get_obspar()
        correction = self.get_expected_correction()

        # maybe the corresponding stuff in get_sources should be removed
        info = {
            key: obspar[key]
            for key in [
                "obsid",
                "date_obs",
                "tstart",
                "tstop",
                "instrument",
                "grating",
                "ra_targ",
                "dec_targ",
                "ra_nom",
                "dec_nom",
                "roll_nom",
                "ra_pnt",
                "dec_pnt",
                "roll_pnt",
                "obs_mode",
            ]
        }
        info.update(
            {
                "obsid_selected": self.is_selected(),
                "OOBAGRD_corr_angle": correction["OOBAGRD_corr_angle"],
                "tstart": float(obspar["tstart"]),
                "tstop": float(obspar["tstop"]),
                "datamode": obspar.get("datamode", ""),
                "readmode": obspar.get("readmode", ""),
                "dtycycle": obspar.get("dtycycle", -1),
            }
        )
        return info

    @stored_result("source_data", fmt="pickle", subdir="cache")
    def get_source_data(self):
        src = self.get_sources(apply_filter=False)
        selected = self.is_pre_selected_source(src)
        src = src[selected]
        events = self.get_events()
        correction = self.get_expected_correction()

        # process all sources
        binned_data = {
            source["id"]: dat
            for source in src
            if (dat := process_source(self.obs, source, events, correction))
        }
        return binned_data

    def get_periscope_drift_data(self):
        data = self.get_source_data()
        src = self.get_sources(apply_filter=False)

        source_ids = list(set(data.keys()) & set(src["id"]))

        src = src[np.in1d(src["id"], source_ids)]
        data = {sid: data[sid] for sid in source_ids}
        return ObservationData(
            summary=self.get_summary(),
            sources=src,
            data=data,
            expected_correction=self.get_expected_correction(),
        )

    def fetch_telemetry(self):
        obs_info = self.obs.get_info()
        start = CxoTime(float(obs_info["tstart"]))
        stop = CxoTime(float(obs_info["tstop"]))
        return fetch_telemetry(start, stop)


class PeriscopeDriftDataProperty:
    def __get__(self, obj, objtype_):
        return PeriscopeDriftData(obj)


class Observation(astromon.observation.Observation):
    """
    Observation class for processing periscope drift data.

    This class is a subclass of astromon.observation.Observation.
    """

    def __init__(self, obsid, workdir=None, archive_dir=None):
        super().__init__(obsid, workdir=workdir, archive_dir=archive_dir)

    periscope_drift = PeriscopeDriftDataProperty()


def process_source(
    obs: Observation, source: int, events: table.Table, correction: dict, dr: float = 4
):
    """
    Do all the processing for a single source.

    This functions does the following:
    - get the events around the source (with a radius dr)
    - perform several fits (yag/zag v rel_time/OOBAGRD3/OOBAGRD6/OOBAGRD_pc1)
    - make one table with the results
    """
    matches = events[
        (np.abs(events["yag"] - source["yag"]) < dr)
        & (np.abs(events["zag"] - source["zag"]) < dr)
    ].copy()
    matches["residual_yag"] = matches["yag"] - source["yag"]
    matches["residual_zag"] = matches["zag"] - source["zag"]
    matches["src_id"] = source["id"]
    matches["obsid"] = obs.obsid

    # require at least 3 bins in each column
    # if (
    #     len(np.unique(matches[["OOBAGRD3"]])) < 3
    #     or len(np.unique(matches[["OOBAGRD6"]])) < 3
    # ):
    #     logger.debug(f"Not enough points in OBSID {obs.obsid} source {source['id']}")
    #     return {}

    fits = [
        fit(
            obs,
            source["id"],
            matches,
            bin_col="rel_time",
            target_col=target_col,
            extra_cols=["rel_time", "OOBAGRD3", "OOBAGRD6", "OOBAGRD_pc1"],
        )
        for target_col in ["yag", "zag"]
    ]

    line_fit = table.vstack([fit["fits_1d"] for fit in fits])
    line_fit["obsid"] = obs.obsid

    yag_fits = fits[0]["binned_data"]
    zag_fits = fits[1]["binned_data"]

    yag_fits.rename_columns(
        ["x_min", "x_max"],
        ["rel_time_min", "rel_time_max"],
    )
    yag_fits.remove_columns(["x_mean", "x_std", "target_col"])
    zag_fits.rename_columns(
        ["x_min", "x_max"],
        ["rel_time_min", "rel_time_max"],
    )
    zag_fits.remove_columns(
        [
            "x_mean",
            "x_std",
            "obsid",
            "src_id",
            "bin_col",
            "target_col",
            "rel_time_min",
            "rel_time_max",
            "rel_time_mean",
            "rel_time_std",
            "x_std",
            "OOBAGRD3_mean",
            "OOBAGRD3_std",
            "OOBAGRD6_mean",
            "OOBAGRD6_std",
            "OOBAGRD_pc1_mean",
            "OOBAGRD_pc1_std",
        ]
    )
    # this one might not be necessary
    # fits += [
    #     fit(obs, source["id"], matches, col, target_col, extra_cols=["OOBAGRD3", "OOBAGRD6"])
    #     for col in ["OOBAGRD3", "OOBAGRD6"]
    #     for target_col in ["yag", "zag"]
    # ]

    binned_data = table.hstack([yag_fits, zag_fits], table_names=["yag", "zag"])

    formats = dict.fromkeys(
        [
            "x_min",
            "x_max",
            "x_mean",
            "x_std",
            "yag",
            "zag",
            "rel_time_mean",
            "rel_time_std",
            "OOBAGRD3_mean",
            "OOBAGRD6_mean",
        ],
        ".2f",
    )
    formats = dict.fromkeys(
        [
            "OOBAGRD6_mean",
        ],
        ".3f",
    )
    formats.update(
        dict.fromkeys(
            [
                "OOBAGRD3_std",
                "OOBAGRD6_std",
                "d_yag",
                "d_zag",
            ],
            ".2",
        )
    )
    for col, fmt in formats.items():
        if col in binned_data.colnames:
            binned_data[col].format = fmt

    spline_fit = do_spline_fit(obs, source)

    ## Summary
    sel = (binned_data["bin_col"] == "rel_time") & (np.isfinite(binned_data["yag"]))
    yag_vs_time = get_smoothing_spline(
        binned_data["rel_time_mean"][sel], binned_data["yag"][sel]
    )
    sel = (binned_data["bin_col"] == "rel_time") & (np.isfinite(binned_data["zag"]))
    zag_vs_time = get_smoothing_spline(
        binned_data["rel_time_mean"][sel], binned_data["zag"][sel]
    )

    cols = [
        "rel_time",
        "OOBAGRD3",
        "OOBAGRD6",
        "OOBAGRD_pc1",
    ]
    y_cols = [
        "yag",
        "zag",
    ]

    x = np.linspace(matches["rel_time"].min(), matches["rel_time"].max(), 2000)
    yag = yag_vs_time(x)
    d_yag = np.max(yag) - np.min(yag)

    zag = zag_vs_time(x)
    d_zag = np.max(zag) - np.min(zag)

    d_r = np.max(np.sqrt((yag - np.min(yag)) ** 2 + (zag - np.min(zag)) ** 2))

    results = {
        "src_id": source["id"],
        "d_OOBAGRD3": source["d_OOBAGRD3"],
        "d_OOBAGRD6": source["d_OOBAGRD6"],
        "spline_fit": spline_fit,
    }
    for col in cols:
        for y_col in y_cols:
            result = _summarize_col_(binned_data, line_fit, col, y_col)
            results.update(result)

    duration = np.max(matches["time"]) - np.min(matches["time"])

    r_corr = np.sqrt(correction["ang_y_corr"] ** 2 + correction["ang_y_corr"] ** 2)
    results.update(
        {
            "duration": duration,
            "drift_yag_actual": d_yag,
            "drift_zag_actual": d_zag,
            "drift_actual": d_r,
            "drift_expected": np.max(r_corr) - np.min(r_corr),
            "drift_yag_expected": np.max(correction["ang_y_corr"])
            - np.min(correction["ang_y_corr"]),
            "drift_zag_expected": np.max(correction["ang_z_corr"])
            - np.min(correction["ang_z_corr"]),
            "drift_yag_fit": results["d_OOBAGRD3"] * results["OOBAGRD3_yag_slope"],
            "drift_zag_fit": results["d_OOBAGRD6"] * results["OOBAGRD6_zag_slope"],
        }
    )

    src_summary = dict(source)
    src_summary["OOBAGRD_corr_angle"] = correction["OOBAGRD_corr_angle"]
    src_summary["tstart"] = obs.get_info()["tstart"]
    src_summary["n_points"] = len(binned_data)
    src_summary.update(results)

    return SourceData(
        source=dict(source),
        summary=src_summary,
        events=matches,
        binned_data_1d=binned_data,
        fits_1d=line_fit,
        yag_vs_time=yag_vs_time,
        zag_vs_time=zag_vs_time,
    )


def get_smoothing_spline(x, y):
    if len(x) == 0:
        return BSpline([0, 1], [np.nan], k=0, extrapolate=True)
    elif len(x) == 1:
        # this should never happen, but I do not want to deal with this error ever.
        # returning a constant spline
        return BSpline([0, 1], y, k=0, extrapolate=True)
    elif len(x) < 5:
        # the smoothing spline call fails with too few points
        # will return a first degree spline that evaluates to a linear regression
        regress = scipy.stats.linregress(x, y)
        tmin = x[0]
        tmax = x[-1]
        return BSpline(
            [tmin - 1, tmin, tmax, tmax + 1],
            [
                regress.intercept + regress.slope * tmin,
                regress.intercept + regress.slope * tmax,
            ],
            k=1,
            extrapolate=True,
        )
    # the value of lambda might seem arbitrary, but it can be estimated from cross-validation
    # this number is just close enough
    return make_smoothing_spline(x, y, lam=3e9)


def round_to_uncertainty(x, err):
    x = np.asarray(x)
    scale = 10 ** np.floor(np.log10(err) - 1)
    return np.round(x / scale) * scale


def _summarize_col_(binned_data_1d, fits_1d, col, y_col, bin_col="rel_time"):
    sel = (binned_data_1d["bin_col"] == bin_col) & np.isfinite(binned_data_1d[y_col])
    binned_data = binned_data_1d[sel]

    if "x_col" in fits_1d.colnames and np.any(
        sel := (fits_1d["x_col"] == col) & (fits_1d["target_col"] == y_col)
    ):
        line_fit = dict(fits_1d[sel][0])
    else:
        line_fit = None

    y_vals = binned_data[y_col]
    x_vals = binned_data[f"{col}_mean"]
    sigma_vals = binned_data[f"d_{y_col}"]

    ndf_0 = len(binned_data[y_col])
    chi2_0 = np.sum(y_vals**2 / sigma_vals**2) / ndf_0
    p_value_0 = scipy.stats.chi2.sf(chi2_0 * ndf_0, ndf_0)

    if line_fit is not None and len(binned_data) > 0:
        ndf = len(binned_data[y_col]) - 2
        params = line_fit["parameters"]
        cov = line_fit["covariance"]
        slope = round_to_uncertainty(params[0], np.sqrt(cov[0, 0]))
        slope_err = round_to_uncertainty(np.sqrt(cov[0, 0]), np.sqrt(cov[0, 0]))
        chi2 = np.sum((y_vals - line(x_vals, *params)) ** 2 / sigma_vals**2) / ndf
        p_value = scipy.stats.chi2.sf(chi2 * ndf, ndf)

        scale = np.sqrt(chi2)

        chi2_0_corr = np.sum(y_vals**2 / (sigma_vals * scale) ** 2) / ndf_0
        p_value_0_corr = scipy.stats.chi2.sf(chi2_0_corr * ndf_0, ndf_0)
        chi2_corr = (
            np.sum((y_vals - line(x_vals, *params)) ** 2 / (sigma_vals * scale) ** 2)
            / ndf
        )
        p_value_corr = scipy.stats.chi2.sf(chi2_corr * ndf, ndf)
    else:
        ndf = 0
        slope = np.nan
        slope_err = np.nan
        chi2 = np.nan
        p_value = np.nan
        chi2_0_corr = np.nan
        p_value_0_corr = np.nan
        chi2_corr = np.nan
        p_value_corr = np.nan

    result = {
        f"{col}_{y_col}_slope": slope,
        f"{col}_{y_col}_slope_err": slope_err,
        f"{col}_{y_col}_null_chi2": chi2_0,
        f"{col}_{y_col}_null_ndf": ndf_0,
        f"{col}_{y_col}_chi2": chi2,
        f"{col}_{y_col}_ndf": ndf,
        f"{col}_{y_col}_null_p_value": p_value_0,
        f"{col}_{y_col}_p_value": p_value,
        f"{col}_{y_col}_null_chi2_corr": chi2_0_corr,
        f"{col}_{y_col}_null_ndf_corr": ndf_0,
        f"{col}_{y_col}_chi2_corr": chi2_corr,
        f"{col}_{y_col}_ndf_corr": ndf,
        f"{col}_{y_col}_null_p_value_corr": p_value_0_corr,
        f"{col}_{y_col}_p_value_corr": p_value_corr,
    }
    return result


def add_pileup_metric(sources, image, threshold=0.06):
    if image is None:
        sources["pileup_size"] = 0
        sources["max_pileup"] = 0
        return

    components = [
        np.argwhere(
            get_connected_component(image > threshold, source["j"], source["i"])
        )
        for source in sources
    ]
    sources["pileup_size"] = [idx.shape[0] for idx in components]
    sources["max_pileup"] = [
        np.max(image[idx.T[0], idx.T[1]])
        if idx.shape[0] > 0
        else image[source["j"], source["i"]]
        for idx, source in zip(components, sources, strict=True)
    ]


def prob(y, y0, sigma, snr):
    scale = np.max(y) - np.min(y)
    return (1 - snr) / scale + snr * np.exp(-0.5 * (y - y0) ** 2 / sigma**2) / (
        np.sqrt(2 * np.pi) * sigma
    )


def log_likelihood(y, p):
    y0, sigma, snr = p
    return -np.sum(np.log10(prob(y, y0, sigma, snr)))


def compute_hessian(f, x, dx=None):
    dx = np.finfo(float).eps if dx is None else dx
    if not np.shape(dx):
        dx = dx * np.ones(len(x))
    if np.shape(dx) != (len(x),):
        raise Exception(f"Wrong dx shape: {np.shape(dx)}")
    dx = np.atleast_1d(dx)
    H = np.zeros((len(x), len(x)))
    h = np.diag(dx)
    for i in range(len(x)):
        for j in range(len(x)):
            H[i, j] = (f(x + h[i] + h[j]) - f(x + h[i]) - f(x + h[j]) + f(x)) / (
                dx[i] * dx[j]
            )
    return H


def get_bins(obs, col):
    telem = fetch_telemetry(obs.get_obspar()["tstart"], obs.get_obspar()["tstop"])

    # choose the gradient binning
    # using the Freedman–Diaconis rule for the bin size
    telem_col = telem[col][np.isfinite(telem[col])]
    q1, q3 = np.percentile(telem_col, [25, 75])
    if q1 == q3:
        return [q1, q3]
    dx = 2 * (q3 - q1) * (telem_col.shape[0]) ** (-1 / 3)
    max_x = np.max(telem_col)
    min_x = np.min(telem_col)
    n = int(np.round((max_x - min_x) / dx + 1))
    grad_bins = np.linspace(min_x - dx / 2, max_x + dx / 2, n + 1)
    return grad_bins


def line(x, m, b):
    return b + m * x


def fit(obs, src_id, matches, bin_col, target_col, extra_cols=None):  # noqa: PLR0912, PLR0915
    if extra_cols is None:
        extra_cols = []
    if bin_col not in extra_cols:
        extra_cols.append(bin_col)
    if bin_col == "rel_time":
        n = np.min([100, int(len(matches) // 200)])
        grad_bins = np.linspace(
            np.min(matches[bin_col]), np.max(matches[bin_col]), n + 1
        )
    else:
        grad_bins = get_bins(obs, bin_col)

    # fit within each bin
    res = []
    for bin_idx, (xmin, xmax) in enumerate(
        zip(grad_bins[:-1], grad_bins[1:], strict=True)
    ):
        sel = (matches[bin_col] < xmax) & (matches[bin_col] >= xmin)
        if np.count_nonzero(sel) < 10:
            logger.debug(
                f"OBSID={obs.obsid}, {src_id=}:"
                f"Not enough points in bin {xmin} < {bin_col} < {xmax} ({np.count_nonzero})"
            )
            continue

        def fun(p, sel=sel):
            return log_likelihood(matches[f"residual_{target_col}"][sel], p)

        result = scipy.optimize.minimize(
            fun, [0, 0.5, 0.9], bounds=[(-3, 3), (0.2, 0.9), (0.1, 0.99)]
        )
        # result.hess_inv gives an estimate of the inverse of the Hessian
        # which is the covariance matrix. Still, I will compute it using a discretization step
        # based on the error estimates
        hess_inv = np.vstack(
            [
                result.hess_inv([1, 0, 0]),
                result.hess_inv([0, 1, 0]),
                result.hess_inv([0, 0, 1]),
            ]
        )
        dx = np.sqrt(np.diagonal(hess_inv)) * 1e-4
        H = compute_hessian(fun, result.x, dx=dx)

        ok = result.success and check_hessian(
            H,
            msg=f"OBSID={obs.obsid}, source_id={src_id} ({xmin} < {bin_col} < {xmax})",
        )

        bd = {
            "obsid": obs.obsid,
            "src_id": src_id,
            "bin_col": bin_col,
            "bin": bin_idx,
            "x_min": xmin,
            "x_max": xmax,
            "x_mean": np.mean(matches[bin_col][sel]),
            "x_std": np.std(matches[bin_col][sel]),
            "target_col": target_col,
            target_col: np.nan,
            f"d_{target_col}": np.nan,
            "params": [np.nan, np.nan, np.nan],
            "params_err": [np.nan, np.nan, np.nan],
            "hess_inverse": [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            "covariance": [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            "success": False,
        }

        for ec in extra_cols:
            bd.update(
                {
                    f"{ec}_mean": np.mean(matches[ec][sel]),
                    f"{ec}_std": np.std(matches[ec][sel]),
                }
            )

        if ok:
            covariance = scipy.linalg.inv(H)
            bd.update(
                {
                    target_col: result.x[0],
                    f"d_{target_col}": np.sqrt(covariance[0, 0]),
                    "params": result.x,
                    "params_err": np.sqrt(np.diagonal(covariance)),
                    "hess_inverse": hess_inv,
                    "covariance": covariance,
                    "success": result.success,
                }
            )

        res.append(bd)

    dtype = [
        ("x_mean", "<f8"),
        ("x_std", "<f8"),
        (target_col, "<f8"),
        (f"d_{target_col}", "<f8"),
        ("hess_inverse", "<f8", (3, 3)),
        ("covariance", "<f8", (3, 3)),
    ]
    dtype += [(f"{ec}_mean", "<f8") for ec in extra_cols]
    dtype += [(f"{ec}_std", "<f8") for ec in extra_cols]
    if res:
        t = table.Table(res)
    else:
        t = table.Table(np.array(res, dtype=dtype))

    line_fits = []
    for x_col in ["rel_time", "OOBAGRD3", "OOBAGRD6", "OOBAGRD_pc1"]:
        ok_rows = np.isfinite(t[target_col])
        if np.count_nonzero(ok_rows) > 2:
            line_fit = scipy.optimize.curve_fit(
                line,
                t[ok_rows][f"{x_col}_mean"],
                t[ok_rows][target_col],
                sigma=t[ok_rows][f"d_{target_col}"],
                p0=[0.0, np.mean(t[ok_rows][target_col])],
                absolute_sigma=True,
            )
            line_fit = {
                "bin_col": bin_col,
                "x_col": x_col,
                "target_col": target_col,
                "slope": line_fit[0][0],
                "intercept": line_fit[0][1],
                "parameters": line_fit[0],
                "errors": np.sqrt(np.diagonal(line_fit[1])),
                "covariance": line_fit[1],
            }
        else:
            line_fit = {
                "bin_col": bin_col,
                "target_col": target_col,
                "slope": np.nan,
                "intercept": np.nan,
                "parameters": [np.nan, np.nan],
                "errors": [np.nan, np.nan],
                "covariance": np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            }
        line_fits.append(line_fit)

    return {
        "x": bin_col,
        "y": target_col,
        # "gaussian_fits": results,
        "binned_data": t,
        "fits_1d": table.Table(line_fits),
    }


def get_connected_component(mask, i, j):
    component = np.zeros_like(mask, dtype=bool)
    component[i, j] = mask[i, j]
    steps = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    for _ in range(np.sum(mask.shape) - 2):
        idx2 = (
            (steps[:, None, ...] + np.argwhere(component)[None, ...]).reshape((-1, 2)).T
        )
        # clip at the edge of the array
        sel = (
            (idx2[0] >= 0)
            & (idx2[0] < mask.shape[0])
            & (idx2[1] >= 0)
            & (idx2[1] < mask.shape[1])
        )
        idx2 = idx2[:, sel]
        # remove the ones already in the result
        sel = ~component[idx2[0], idx2[1]]
        idx2 = idx2[:, sel]
        if len(idx2) == 0:
            break
        if not np.any(mask[idx2[0], idx2[1]]):
            # this is the case when no new neighbors are in the region
            break
        component[idx2[0], idx2[1]] = mask[idx2[0], idx2[1]]
    return component


@functools.cache
def _fetch_telemetry_(start, stop):
    """ """
    msids = (
        "OOBAGRD3",
        "OOBAGRD6",
        # "OHRTHR42",
        # "OHRTHR43",
        # "OOBTHR39",
        # "OHRTHR24",
        # "4RT702T",
        # "OHRTHR24",
        # "AACBPPT",
        # "AACH1T",
        # "AACBPRT",
        # "AACCCDPT",
    )
    telem = fetch.MSIDset(msids, start, stop)

    # replace the "bad" values with the nearest good value
    for msid in msids:
        ok = ~telem[msid].bads
        bad = telem[msid].bads
        fix_vals = ska_numpy.interpolate(
            telem[msid].vals[ok],
            telem[msid].times[ok],
            telem[msid].times[bad],
            method="nearest",
        )
        telem[msid].vals[bad] = fix_vals

    # smooth
    smooth_gradients = {
        msid: smooth(telem[msid].vals, window_len=152) for msid in msids
    }

    result = {
        "time": telem["OOBAGRD3"].times,
    }

    result.update({f"{msid}_smooth": smooth_gradients[msid] for msid in msids})
    result.update({f"{msid}_raw": telem[msid].vals for msid in msids})
    result.update({msid: result[f"{msid}_raw"] for msid in msids})

    return result


def fetch_telemetry(start, stop, times=None):
    """ """

    telem = _fetch_telemetry_(start, stop)

    if times is None:
        return telem

    result = {"time": times}

    for msid in telem:
        if len(telem[msid]) > 0:
            interpolate = scipy.interpolate.interp1d(
                telem["time"],
                telem[msid],
                bounds_error=False,
                fill_value=(telem[msid][0], telem[msid][-1]),
            )
            result[msid] = interpolate(times)
        else:
            # this happens in rare cases, at least if there is no telemetry (obsid 28065)
            result[msid] = np.nan * np.ones(len(times), dtype=telem[msid].dtype)

    result.update(
        {
            "tilt_axial": result["OOBAGRD3"],
            "tilt_diam": result["OOBAGRD6"],
        }
    )
    return result


def smooth(x, window_len=10, window="hanning"):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Example::

      t = linspace(-2, 2, 50)
      y = sin(t) + randn(len(t)) * 0.1
      ys = ska_numpy.smooth(y)
      plot(t, y, t, ys)

    See also::

      numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
      scipy.signal.lfilter

    :param x: input signal
    :param window_len: dimension of the smoothing window
    :param window: type of window ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')

    :rtype: smoothed signal
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    # if x.size < window_len:
    #    raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    # the array is padded at both ends with (window_len-1) entries
    # of those, at most len(x) entries are the signal reflected
    # the rest are the begin and end of the signal
    # examples with x = [1, 2, 3, 4, 5]:
    # window_len = 4: s = [4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2]
    # window_len = 6: s = [5, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1]
    padded_length = x.shape[0] + 2 * window_len - 2
    s = np.r_[
        x[window_len - 1 : 0 : -1],
        x,
        x[-2 : -window_len - 1 : -1],
    ]
    if x.shape[0] == 0:
        return np.zeros_like(x)

    if s.shape[0] < padded_length:
        extra = int((padded_length - s.shape[0]) // 2)
        ones = np.ones(extra)
        s = np.r_[
            ones * s[0],
            s,
            ones * s[-1],
        ]

    # Moving average
    if window == "flat":
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1 : -window_len + 1]


def get_knots(time, min_n=200, min_t=2000):
    # we generate a list of interval edges that each has at least min_n/4 points
    n0 = int(len(time) // (min_n / 4))
    edges = np.percentile(time, np.linspace(0, 100, n0 + 1))
    idx_range = np.arange(len(edges))  # an integer array to refer to entries in edges

    # we will ultimately select a subset of them to ensure each interval is at least min_t long
    # and has at least min_n points. We start by selecting good intervals from the beginning
    sel = np.zeros_like(edges, dtype=bool)
    sel[0] = True

    idx0 = 0
    for idx in idx_range[1:]:
        if edges[idx] - edges[idx0] > min_t and idx - idx0 >= 4:
            # this closes one interval and starts the next
            sel[idx] = True
            idx0 = idx

    indices = idx_range[sel]

    if not sel[-1] and np.count_nonzero(sel) > 1:
        # fix the trailing interval by splitting it over the previous ones
        n_intervals = indices.shape[0] - 1  # the number of resulting long intervals
        n_trail = (
            idx_range[-1] - indices[-1] - 1
        )  # the number of trailing short intervals
        n, m = n_trail // n_intervals, n_trail % n_intervals
        intervals = np.diff(edges[indices])
        shortest = np.argsort(intervals)[: m + 1]
        # increase all long intervals by n small intervals
        shift = n * np.ones_like(indices[1:])
        # and increase the smallest m long intervals by one short interval
        shift[shortest] += 1
        shift = np.cumsum(shift)
        indices[1:] += shift
        intervals = np.diff(edges[indices])

    edges = edges[indices]

    if len(edges) == 1:
        # print("Only one edge found!")
        edges = np.array([time[0], time[-1]])

    # assert edges[0] == time[0]
    # assert edges[-1] == time[-1]
    # this is actually not guaranteed, because intervals can be slightly smaller than min_t
    # assert np.all(np.diff(edges) >= min_t)

    edges = np.pad(edges, (3, 3), mode="edge")
    return edges


class Likelihood:
    def __init__(self, x, y, knots, box_size=4, alpha=0):
        self.degree = 3
        self.x = x
        self.y = y
        self.knots = knots
        self.box_size = box_size
        self.alpha = alpha
        self.dm = BSpline.design_matrix(self.x, self.knots, self.degree)
        self.n_points = self.dm.shape[1]
        self.count = 0

    def __call__(self, x):
        self.count += 1
        x0 = self.dm @ x[: self.n_points]
        x1 = self.dm @ x[self.n_points : 2 * self.n_points]
        sigma_1, sigma_2, rho, snr = x[2 * self.n_points :]

        if sigma_1 <= 0 or sigma_2 <= 0 or snr < 0:
            return np.inf
        s = snr / (1 + snr)
        b = 1 / (1 + snr)
        p = s * source_detection.normal_prob_2d(
            self.y, x0, x1, sigma_1, sigma_2, rho
        ) + b * source_detection.p_uniform(self.y, box_size=self.box_size)
        res = -source_detection.nan_log(p).sum() + self.alpha * np.sum(
            (x0 - self.y.T[0]) ** 2 + (x1 - self.y.T[1]) ** 2
        )
        return res


class SplineFitUncertainty:
    def __init__(self, spline=None, covariance=None):
        self.spline = spline
        self.covariance = covariance

    def __call__(self, x):
        if self.spline is None or self.covariance is None:
            return np.nan * np.ones_like(x)
        dm = self.spline.design_matrix(x, self.spline.t, self.spline.k).toarray()
        # np.sqrt(np.einsum("ij,jk,ik->i", dm, cov, dm)) is the same as
        # np.sqrt((dm @ cov @ dm.T).diagonal()) but takes less memory
        return np.sqrt(np.einsum("ij,jk,ik->i", dm, self.covariance, dm))


def do_spline_fit(obs, source_id):
    gaussian_sources = obs.get_sources(version="gaussian_detect", astromon_format=False)

    box_size = 4
    idx = np.argwhere(gaussian_sources["id"] == source_id).flatten()[0]
    source = gaussian_sources[idx]
    events = obs.periscope_drift.get_events()
    events.rename_columns(["yag", "zag"], ["y_angle", "z_angle"])
    events = events[
        (np.abs(events["y_angle"] - source["y_angle"]) < box_size)
        & (np.abs(events["z_angle"] - source["z_angle"]) < box_size)
    ]

    x0, x1, sigma_1, sigma_2, rho, snr = source["params"]

    knot_list = get_knots(events["rel_time"].data)

    likelihood = Likelihood(
        x=events["rel_time"],
        y=np.vstack([events["y_angle"], events["z_angle"]]).T,
        knots=knot_list,
        alpha=0,
        # alpha=1e25
    )

    initial_guess = np.concatenate(
        [
            np.full(likelihood.n_points, x0),
            np.full(likelihood.n_points, x1),
            [sigma_1, sigma_2, rho, snr],
        ]
    )

    bounds = []
    bounds += likelihood.n_points * [(source["y_angle"] - 10, source["y_angle"] + 10)]
    bounds += likelihood.n_points * [(source["z_angle"] - 10, source["z_angle"] + 10)]
    bounds += [(0.1, 20)]
    bounds += [(0.1, 20)]
    bounds += [(-np.pi / 2, np.pi / 2)]
    bounds += [(0.1, 1000)]

    fit_result = scipy.optimize.minimize(
        likelihood,
        x0=initial_guess,
        bounds=bounds,
        options={
            "maxfun": 50000,
            # "ftol": 1e-6,
        },
    )

    hess_inv = fit_result.hess_inv(np.eye(fit_result.x.shape[0]))
    dx = np.sqrt(np.diagonal(hess_inv)) * 1e-4
    H = compute_hessian(likelihood, fit_result.x, dx=dx)

    ok = fit_result.success and check_hessian(
        H, msg=f"OBSID={obs.obsid}, source_id={source_id} spline fit"
    )

    maxx = int(np.ceil(events["rel_time"].max()))
    x = np.linspace(likelihood.x.min(), likelihood.x.max(), maxx)
    dm = BSpline.design_matrix(x, likelihood.knots, likelihood.degree)
    idx = np.argmax(dm.toarray(), axis=0)
    spline_pos = x[idx]

    result = {
        "degree": likelihood.degree,
        "knots": likelihood.knots,
        "n_points": likelihood.n_points,  # the number of degrees of freedom per axis
        "spline_pos": spline_pos,
        "value": fit_result.fun,
        "success": ok,
        "message": fit_result.message,
        "params": fit_result.x,
        "params_err": [np.nan, np.nan, np.nan],
        "hess_inverse": hess_inv,
        "covariance": np.nan * np.ones(hess_inv.shape),
        "spline_yag": BSpline([0, 1], [np.nan], k=0, extrapolate=True),
        "spline_zag": BSpline([0, 1], [np.nan], k=0, extrapolate=True),
        "yag_error": SplineFitUncertainty(),
        "zag_error": SplineFitUncertainty(),
        "smooth_spline_yag": BSpline([0, 1], [np.nan], k=0, extrapolate=True),
        "smooth_spline_zag": BSpline([0, 1], [np.nan], k=0, extrapolate=True),
    }

    if ok:
        spline_yag = BSpline(likelihood.knots, fit_result.x[: likelihood.n_points], 3)
        spline_zag = BSpline(
            likelihood.knots,
            fit_result.x[likelihood.n_points : 2 * likelihood.n_points],
            3,
        )

        smspl_yag = get_smoothing_spline(spline_pos, spline_yag(spline_pos))
        smspl_zag = get_smoothing_spline(spline_pos, spline_zag(spline_pos))

        covariance = scipy.linalg.inv(H)
        n = likelihood.n_points

        result.update(
            {
                "covariance": covariance,
                "params_err": np.sqrt(np.diagonal(covariance)),
                "spline_yag": spline_yag,
                "spline_zag": spline_zag,
                "yag_error": SplineFitUncertainty(
                    spline=spline_yag, covariance=covariance[:n, :n]
                ),
                "zag_error": SplineFitUncertainty(
                    spline=spline_zag, covariance=covariance[n : 2 * n, n : 2 * n]
                ),
                "smooth_spline_yag": smspl_yag,
                "smooth_spline_zag": smspl_zag,
            }
        )

    return result


def check_hessian(H, msg=None):
    if msg is None:
        msg = "Hessian check"
    ok = True
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        logger.debug(f"{msg}: Hessian has Nan or inf entries")
        ok = False
    if scipy.linalg.det(H) == 0.0:
        logger.debug(f"{msg}: singular hessian matrix")
        ok = False
    if not scipy.linalg.issymmetric(H):
        logger.debug(f"{msg}: Hessian is not symmetric")
        ok = False

    if ok:
        covariance = scipy.linalg.inv(H)
        if np.any(np.diagonal(covariance) < 0):
            logger.debug(f"{msg}: Covariance has negative diagonal values")
            ok = False
    return ok
