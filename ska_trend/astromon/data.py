"""
Data access for the astromon trending reports.

This module wraps the ``astromon`` package data layer (cross-matches and the per-OBSID
catalog/x-ray source tables) and prepares the cropped sky-view image used by the per-source
report pages.

The image and marker logic is a static-rendering port of the per-OBSID logic in the kadi-apps
astromon flask blueprint (``kadi_apps/blueprints/astromon.py``).
"""

import functools
import logging
import warnings
from pathlib import Path

import numpy as np
from astromon import db, utils
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join
from astropy.wcs import WCS, FITSFixedWarning
from cxotime import CxoTime

logger = logging.getLogger("astromon")

__all__ = [
    "get_matches",
    "add_reprocessed_offsets",
    "binned_offsets",
    "simbad_url",
    "get_obsid_image",
    "get_source_markers",
    "crop_box",
    "get_source_figure_data",
]

# Plotly marker symbols cycled through for the vicinity catalog sources (one per catalog).
CATALOG_SYMBOLS = [
    "star-triangle-up",
    "star-triangle-down",
    "star-square",
    "star-diamond",
]

# Science categories excluded by the celmon CAL/MTA selections (they confound source detection).
CELMON_EXCLUDE_CATEGORIES = [
    "SN, SNR, and Isolated NS",
    "Solar System and Misc",
    "Clusters of Galaxies",
]

# Cross-match selections. "all" is unfiltered; "cal" and "mta" reproduce the celmon
# create_figures_cal / create_figures_mta selections (they differ only in the SNR threshold).
SELECTIONS = {
    "all": {},
    "cal": {
        "snr": 5,
        "exclude_bad_targets": True,
        "sim_z": 4,
        "exclude_categories": CELMON_EXCLUDE_CATEGORIES,
    },
    "mta": {
        "snr": 3,
        "exclude_bad_targets": True,
        "sim_z": 4,
        "exclude_categories": CELMON_EXCLUDE_CATEGORIES,
    },
}


# The two versions of the offsets the report can show. "archive" is the offsets as the
# observations were actually processed (what the archive holds today); "reprocessed" is the
# same offsets brought to the latest CALALIGN calibration (see add_reprocessed_offsets).
# The suffix is what the offset column name gets, e.g. "dy" and "dy_repro".
OFFSET_VERSIONS = {
    "reprocessed": {"suffix": "_repro", "label": "After reprocessing"},
    "archive": {"suffix": "", "label": "In the archive"},
}

DEFAULT_OFFSET_VERSION = "reprocessed"


def get_matches(selection="mta", dbfile=None):
    """
    Return a table of astrometric cross-matches, one row per (OBSID, x-ray source) pair.

    Parameters
    ----------
    selection : str
        One of "all", "cal", "mta" (see :data:`SELECTIONS`). "all" applies no filtering;
        "cal"/"mta" reproduce the celmon selections. Default: "mta".
    dbfile : str or Path, optional
        Path to the astromon HDF5 db. If None, the astromon default is used
        ($ASTROMON_FILE or $SKA/data/astromon/astromon.h5).

    Returns
    -------
    astropy.table.Table
    """
    matches = db.get_cross_matches(dbfile=dbfile, **SELECTIONS[selection])
    if selection != "all":
        # celmon drops observations processed without a caldb version
        matches = matches[matches["caldb_version"] != "0.0"]
    matches["idx"] = np.arange(len(matches))
    # date_obs is a string column; expose an ISO datetime usable as a plotly x-axis value.
    matches["date_iso"] = CxoTime(matches["date_obs"]).isot
    return matches


def add_reprocessed_offsets(matches, calalign_dir=None):
    """
    Add the offsets that reprocessing with the latest CALALIGN calibration would give.

    The offsets in the astromon db are the ones the observations were actually processed
    with, which is what the archive holds today. Each one carries the CALALIGN calibration
    that was current at the time, so the trend mixes several calibration epochs. Replacing
    that by a single reference calibration takes the calibration history out of the trend.
    This is the same correction celmon applies for its "offsets after reprocessing" figures.

    Adds the ``dy_repro``, ``dz_repro`` and ``dr_repro`` columns to ``matches`` in place.

    Parameters
    ----------
    matches : astropy.table.Table
        Cross-match table (needs ``obsid``, ``x_id``, ``detector``, ``time``,
        ``caldb_version``, ``dy`` and ``dz``).
    calalign_dir : str or Path, optional
        Directory holding the CALALIGN files. If None, the astromon default is used
        (/data/caldb/data/chandra/pcad/align).

    Raises
    ------
    ValueError
        If ``matches`` is a subset that ``astromon.utils.get_calalign_offsets`` cannot
        handle (see below), or if the CALALIGN offsets do not cover every match.
    """
    # Pass only the columns get_calalign_offsets needs, sorted by key: it joins the whole
    # input table on "detector", and parts of its output are assembled from the input row
    # order rather than from the keys.
    keys = matches[["obsid", "x_id", "detector", "time", "caldb_version"]].copy()
    keys.sort(["obsid", "x_id"])

    # get_calalign_offsets parses caldb_version into a column of lists of ints. When every
    # version in the input has the same number of components that column comes out as a 2-d
    # array instead, and indexing with it raises a confusing IndexError. Guard for it here:
    # the full cross-match table always mixes 3- and 4-component versions (e.g. "4.12.0" and
    # "4.11.0.1"), so this only bites on a short time slice.
    if len({len(version.split(".")) for version in keys["caldb_version"]}) < 2:
        raise ValueError(
            "cannot compute the reprocessed offsets: every caldb_version in this table has "
            "the same number of components, which astromon.utils.get_calalign_offsets does "
            "not handle. Pass the full cross-match table rather than a time slice."
        )

    calalign = utils.get_calalign_offsets(keys, calalign_dir=calalign_dir)
    _set_reprocessed_offsets(matches, calalign)


def _set_reprocessed_offsets(matches, calalign):
    """
    Set the reprocessed-offset columns from a table of CALALIGN offsets.

    ``calalign`` (the output of ``astromon.utils.get_calalign_offsets``) has one row per
    (obsid, x_id), but in join order rather than in the order of ``matches``, so the rows
    are aligned by key here and never by position.

    Parameters
    ----------
    matches : astropy.table.Table
        Cross-match table. Modified in place.
    calalign : astropy.table.Table
        Needs ``obsid``, ``x_id`` and the ``calalign_`` / ``ref_calalign_`` dy/dz columns.
    """
    keys = matches[["obsid", "x_id"]].copy()
    keys["_row"] = np.arange(len(matches))
    columns = [
        "obsid",
        "x_id",
        "calalign_dy",
        "calalign_dz",
        "ref_calalign_dy",
        "ref_calalign_dz",
    ]
    # an inner join keeps the columns unmasked; the length check catches both missing and
    # duplicated keys, either of which would misalign the result.
    aligned = join(keys, calalign[columns], keys=["obsid", "x_id"], join_type="inner")
    if len(aligned) != len(matches):
        raise ValueError(
            "expected one CALALIGN row per match, "
            f"got {len(aligned)} rows for {len(matches)} matches"
        )
    aligned.sort("_row")

    # the correction is the difference between the CALALIGN used to process the observation
    # and the reference (latest) one.
    matches["dy_repro"] = np.asarray(matches["dy"]) - np.asarray(
        aligned["calalign_dy"] - aligned["ref_calalign_dy"]
    )
    matches["dz_repro"] = np.asarray(matches["dz"]) - np.asarray(
        aligned["calalign_dz"] - aligned["ref_calalign_dz"]
    )
    matches["dr_repro"] = np.hypot(matches["dy_repro"], matches["dz_repro"])


def binned_offsets(matches, coord, bins_per_year=2):
    """
    Per-bin median and 1-sigma band of an offset column versus time.

    Replicates the celmon ``binned_median``: time is binned ``bins_per_year`` bins per calendar
    year; each bin reports the median and the 15.8 / 84.2 percentiles (the ±1-sigma band).

    Parameters
    ----------
    matches : astropy.table.Table
        Cross-match table (needs a CxoTime ``time`` column and the ``coord`` column).
    coord : str
        Offset column name ("dy" or "dz", or one of their "_repro" variants).
    bins_per_year : int
        Number of time bins per calendar year.

    Returns
    -------
    dict with array values ``start``, ``stop`` (bin edges as fractional years) and ``median``,
    ``sigma_minus``, ``sigma_plus`` (the offset statistics per bin). Empty arrays if no data.
    """
    if len(matches) == 0:
        keys = ("start", "stop", "median", "sigma_minus", "sigma_plus")
        return dict.fromkeys(keys, np.array([]))

    years = np.array(matches["time"].frac_year)
    ymin = np.floor(np.min(years / bins_per_year)) * bins_per_year
    ymax = np.ceil(np.max(years / bins_per_year)) * bins_per_year
    nbins = int((ymax - ymin) * bins_per_year)
    bins = np.linspace(ymin, ymax, nbins + 1)

    groups = Table(
        [np.digitize(years, bins), np.asarray(matches[coord])], names=["bin", "val"]
    ).group_by("bin")

    def q(quantile):
        return np.asarray(
            groups["val"].groups.aggregate(functools.partial(np.quantile, q=quantile))
        )

    bin_idx = np.asarray(groups.groups.keys["bin"])
    return {
        "start": bins[bin_idx - 1],
        "stop": bins[bin_idx],
        "median": q(0.5),
        "sigma_minus": q(0.158),
        "sigma_plus": q(0.842),
    }


def _images_dir(obsid, archive_dir):
    return (
        Path(archive_dir) / f"obs{int(obsid) // 1000:02d}" / str(int(obsid)) / "images"
    )


def _find_flux_image(obsid, archive_dir):
    subdir = _images_dir(obsid, archive_dir)
    images = list(subdir.glob("*broad_flux.img")) + list(subdir.glob("*wide_flux.img"))
    return images[0] if images else None


def get_obsid_image(obsid, archive_dir):
    """
    Load the flux image for an OBSID and return (log-scaled image, WCS).

    Parameters
    ----------
    obsid : int
    archive_dir : str or Path
        Astromon archive directory (e.g. $SKA/data/astromon/xray_observations).

    Returns
    -------
    tuple(np.ndarray, astropy.wcs.WCS) or (None, None)
        The log10-scaled image and its WCS, or (None, None) if no image is available.
    """
    filename = _find_flux_image(obsid, archive_dir)
    if filename is None:
        logger.info(
            f"No flux image for OBSID {obsid} in {_images_dir(obsid, archive_dir)}"
        )
        return None, None

    hdus = fits.open(filename)
    # FITS data is typically big-endian; use a native-endian float array so the result
    # can be JSON-serialized by plotly.
    data = hdus[0].data
    image = np.zeros(data.shape, dtype=np.float64)
    positive = data > 0
    if np.any(positive):
        n_min = np.min(data[positive])
        image[positive] = np.log10(data[positive])
        image[~positive] = np.log10(n_min) - 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        wcs = WCS(hdus[0].header)
    return image, wcs


def simbad_url(loc):
    ra = loc.ra.to_string(unit="hour")
    dec = loc.dec.to_string(unit="deg")
    return (
        "https://simbad.u-strasbg.fr/simbad/sim-coo?"
        f"Coord={ra}%2B{dec}&CooFrame=ICRS&CooEpoch=2000&Radius=2&Radius.unit=arcsec"
    )


def get_source_markers(obsid, wcs, dbfile=None):
    """
    Build the marker layers (in pixel coordinates) for the per-OBSID sky view.

    Parameters
    ----------
    obsid : int
    wcs : astropy.wcs.WCS
        WCS of the OBSID flux image (used to convert world -> pixel).
    dbfile : str or Path, optional

    Returns
    -------
    dict with keys:
        ``xray`` : dict(x, y, id) for the detected x-ray sources
        ``matches`` : list of dicts (matched catalog counterparts)
        ``vicinity`` : list of dicts (rough catalog sources in the field), each carrying
                       a ``symbol`` for plotting.
    """
    cat = db.get_table("astromon_cat_src", dbfile=dbfile)
    xray = db.get_table("astromon_xray_src", dbfile=dbfile)
    cat = cat[cat["obsid"] == obsid]
    xray = xray[xray["obsid"] == obsid]

    xray["loc"] = SkyCoord(xray["ra"] * u.deg, xray["dec"] * u.deg)
    xray_x, xray_y = wcs.world_to_pixel(xray["loc"])
    xray_markers = {
        "x": np.atleast_1d(xray_x).tolist(),
        "y": np.atleast_1d(xray_y).tolist(),
        "id": xray["id"].tolist(),
    }

    cat["loc"] = SkyCoord(cat["ra"] * u.deg, cat["dec"] * u.deg)
    cat_x, cat_y = wcs.world_to_pixel(cat["loc"])
    catalogs = sorted(set(cat["catalog"].tolist()))
    symbol_for = {
        name: CATALOG_SYMBOLS[i % len(CATALOG_SYMBOLS)]
        for i, name in enumerate(catalogs)
    }
    vicinity = []
    for row, x, y in zip(cat, np.atleast_1d(cat_x), np.atleast_1d(cat_y), strict=True):
        vicinity.append(
            {
                "id": row["id"],
                "x": float(x),
                "y": float(y),
                "name": str(row["name"]),
                "catalog": str(row["catalog"]),
                "symbol": symbol_for[str(row["catalog"])],
                "simbad_url": simbad_url(row["loc"]),
            }
        )

    return {"xray": xray_markers, "vicinity": vicinity}


def crop_box(image, cx, cy, half_size):
    """
    Crop a square window of ``image`` centered on pixel (cx, cy).

    Parameters
    ----------
    image : np.ndarray
        2D image array (indexed [y, x]).
    cx, cy : float
        Center pixel coordinates (column, row).
    half_size : int
        Half the window size in pixels.

    Returns
    -------
    tuple(np.ndarray, int, int)
        The cropped image and the (x0, y0) pixel offset of its lower-left corner. Marker pixel
        coordinates can be shifted into the cropped frame by subtracting (x0, y0).
    """
    ny, nx = image.shape
    cxi = int(round(cx))
    cyi = int(round(cy))
    x0 = max(0, cxi - half_size)
    x1 = min(nx, cxi + half_size + 1)
    y0 = max(0, cyi - half_size)
    y1 = min(ny, cyi + half_size + 1)
    return image[y0:y1, x0:x1], x0, y0


def get_source_figure_data(
    obsid, x_loc, archive_dir, c_loc=None, dbfile=None, half_size=40
):
    """
    Assemble everything needed to draw the cropped sky view for one source.

    Parameters
    ----------
    obsid : int
    x_loc : astropy.coordinates.SkyCoord
        Sky position of the x-ray source (the crop is centered here).
    archive_dir : str or Path
    c_loc : astropy.coordinates.SkyCoord, optional
        Sky position of the matched catalog counterpart (drawn as the highlighted match).
    dbfile : str or Path, optional
    half_size : int
        Half the crop window size in pixels.

    Returns
    -------
    dict or None
        None if the OBSID has no flux image. Otherwise a dict with the cropped ``image``,
        the crop offsets ``x0``/``y0``, the source center pixel (``cx``, ``cy``), the matched
        counterpart pixel (``match``), and the marker layers (``xray``, ``vicinity``) with
        pixel coordinates already shifted into the cropped frame.
    """
    image, wcs = get_obsid_image(obsid, archive_dir)
    if image is None:
        return None

    cx, cy = wcs.world_to_pixel(x_loc)
    cx = float(cx)
    cy = float(cy)
    cropped, x0, y0 = crop_box(image, cx, cy, half_size)

    markers = get_source_markers(obsid, wcs, dbfile=dbfile)

    ny, nx = cropped.shape

    def _inside(x, y):
        return 0 <= x < nx and 0 <= y < ny

    # shift marker pixel coordinates into the cropped frame, keeping only those that
    # actually fall inside the cropped region.
    xray_in = [
        (x - x0, y - y0, sid)
        for x, y, sid in zip(
            markers["xray"]["x"],
            markers["xray"]["y"],
            markers["xray"]["id"],
            strict=True,
        )
        if _inside(x - x0, y - y0)
    ]
    xray = {
        "x": [x for x, _, _ in xray_in],
        "y": [y for _, y, _ in xray_in],
        "id": [sid for _, _, sid in xray_in],
    }

    vicinity = []
    for m in markers["vicinity"]:
        m["x"] -= x0
        m["y"] -= y0
        if _inside(m["x"], m["y"]):
            vicinity.append(m)

    match = None
    if c_loc is not None:
        mx, my = wcs.world_to_pixel(c_loc)
        mx = float(mx) - x0
        my = float(my) - y0
        if _inside(mx, my):
            match = {"x": mx, "y": my, "simbad_url": simbad_url(c_loc)}

    return {
        "image": cropped,
        "x0": x0,
        "y0": y0,
        "cx": cx - x0,
        "cy": cy - y0,
        "match": match,
        "xray": xray,
        "vicinity": vicinity,
    }
