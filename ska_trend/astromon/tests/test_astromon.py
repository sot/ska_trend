import numpy as np
import pytest
from astropy.table import Column, Table
from cxotime import CxoTime

from ska_trend.astromon import data
from ska_trend.astromon import plotly as plots


def test_crop_box_interior():
    image = np.arange(100).reshape(10, 10)
    # crop a 3x3 window (half_size=1) centered on pixel (4, 5) -> column 4, row 5
    cropped, x0, y0 = data.crop_box(image, 4, 5, 1)
    assert (x0, y0) == (3, 4)
    assert cropped.shape == (3, 3)
    # the center pixel of the crop is image[5, 4]
    assert cropped[1, 1] == image[5, 4]


def test_crop_box_edge_is_clipped():
    image = np.arange(100).reshape(10, 10)
    # near the corner the window is clipped to the image bounds (no negative indices)
    cropped, x0, y0 = data.crop_box(image, 0, 0, 3)
    assert (x0, y0) == (0, 0)
    assert cropped.shape == (4, 4)
    assert cropped[0, 0] == image[0, 0]


# mid-year dates: binned_offsets bins by fractional year and cannot handle a date that falls
# exactly on the last bin edge (e.g. 2020:001).
DATES = CxoTime(["2015:100", "2020:200", "2018:150", "2012:050"])


def _matches_table():
    """A small cross-match table, deliberately not sorted by (obsid, x_id)."""
    return Table(
        {
            "obsid": [5, 1, 5, 3],
            "x_id": [2, 1, 1, 4],
            "time": DATES,
            "date_iso": DATES.isot,
            "dy": [1.0, 2.0, 3.0, 4.0],
            "dz": [0.5, -0.5, 1.5, -1.5],
            "detector": ["ACIS-S", "ACIS-I", "ACIS-S", "HRC-I"],
            # mixed component counts, as in the full cross-match table
            "caldb_version": ["4.12.0", "4.11.0.1", "4.12.0", "4.10.2"],
        }
    )


def _calalign_table():
    """CALALIGN offsets for _matches_table, in yet another row order."""
    return Table(
        {
            "obsid": [1, 3, 5, 5],
            "x_id": [1, 4, 1, 2],
            "calalign_dy": [0.1, 0.2, 0.3, 0.4],
            "calalign_dz": [0.01, 0.02, 0.03, 0.04],
            "ref_calalign_dy": [0.5, 0.5, 0.5, 0.5],
            "ref_calalign_dz": [0.05, 0.05, 0.05, 0.05],
        }
    )


def _expected_repro(matches, calalign, coord):
    """The reprocessed offsets computed row by row, looking each match up by key."""
    by_key = {(row["obsid"], row["x_id"]): row for row in calalign}
    return np.array(
        [
            row[coord]
            - (
                by_key[(row["obsid"], row["x_id"])][f"calalign_{coord}"]
                - by_key[(row["obsid"], row["x_id"])][f"ref_calalign_{coord}"]
            )
            for row in matches
        ]
    )


def test_set_reprocessed_offsets_aligns_by_key():
    matches = _matches_table()
    calalign = _calalign_table()
    data._set_reprocessed_offsets(matches, calalign)

    assert np.allclose(matches["dy_repro"], _expected_repro(matches, calalign, "dy"))
    assert np.allclose(matches["dz_repro"], _expected_repro(matches, calalign, "dz"))
    assert np.allclose(
        matches["dr_repro"], np.hypot(matches["dy_repro"], matches["dz_repro"])
    )


def test_set_reprocessed_offsets_is_order_invariant():
    # the report sorts the matches by time before the offsets are added, so the result must
    # not depend on the row order of either table.
    matches = _matches_table()
    data._set_reprocessed_offsets(matches, _calalign_table())

    by_time = _matches_table()
    by_time.sort("time")
    data._set_reprocessed_offsets(by_time, _calalign_table()[::-1])

    by_time.sort("dy")
    matches.sort("dy")
    assert np.allclose(matches["dy_repro"], by_time["dy_repro"])
    assert np.allclose(matches["dz_repro"], by_time["dz_repro"])


def test_set_reprocessed_offsets_missing_key_raises():
    matches = _matches_table()
    with pytest.raises(ValueError, match="one CALALIGN row per match"):
        data._set_reprocessed_offsets(matches, _calalign_table()[1:])


def test_reprocessed_offsets_are_unmasked_floats():
    # binned_offsets drops the mask and np.quantile poisons a whole bin with a single NaN,
    # so a masked column here would silently empty out the median/band traces.
    matches = _matches_table()
    data._set_reprocessed_offsets(matches, _calalign_table())
    for name in ("dy_repro", "dz_repro", "dr_repro"):
        assert type(matches[name]) is Column
        assert matches[name].dtype.kind == "f"


def test_binned_offsets_accepts_a_reprocessed_column():
    matches = _matches_table()
    data._set_reprocessed_offsets(matches, _calalign_table())
    binned = data.binned_offsets(matches, "dy_repro")
    assert len(binned["median"]) > 0


def test_offsets_history_figure_has_both_versions():
    matches = _matches_table()
    data._set_reprocessed_offsets(matches, _calalign_table())
    fig = plots.get_offsets_history_figure(matches, "dy")

    # three traces (points, band, median) per version, each tagged with its version
    versions = [trace.meta for trace in fig.data]
    assert sorted(set(versions)) == ["archive", "reprocessed"]
    assert versions.count("archive") == 3
    assert versions.count("reprocessed") == 3

    # only the default version is visible, and the two point traces carry the two columns
    for trace in fig.data:
        assert trace.visible == (trace.meta == data.DEFAULT_OFFSET_VERSION)
    points = {trace.meta: trace for trace in fig.data if trace.mode == "markers"}
    assert np.allclose(points["archive"].y, matches["dy"])
    assert np.allclose(points["reprocessed"].y, matches["dy_repro"])
    # the click handler needs customdata on whichever version is shown
    for trace in points.values():
        assert trace.customdata is not None


def test_offsets_history_figure_without_reprocessed_offsets():
    # the CALALIGN files may not be available; the report then shows the archive offsets only
    fig = plots.get_offsets_history_figure(_matches_table(), "dy")
    assert [trace.meta for trace in fig.data] == ["archive"] * 3
    assert all(trace.visible is None for trace in fig.data)


def test_offsets_history_figure_band_colors_match_the_coordinate():
    # "dz_repro" is not "dz", so a naive coord comparison would paint the reprocessed dZ band
    # with the dY color.
    matches = _matches_table()
    data._set_reprocessed_offsets(matches, _calalign_table())
    fig = plots.get_offsets_history_figure(matches, "dz")
    bands = [trace for trace in fig.data if trace.fill == "toself"]
    assert len(bands) == 2
    assert {band.fillcolor for band in bands} == {plots.COORD_INFO["dz"]["fillcolor"]}


def test_add_reprocessed_offsets_rejects_uniform_caldb_versions():
    # astromon.utils.get_calalign_offsets cannot handle a table in which every caldb_version
    # has the same number of components; that has to be a clear message, not an IndexError
    # from deep inside astromon. Nothing here touches CALDB.
    matches = _matches_table()
    matches["caldb_version"] = ["4.12.0"] * len(matches)
    with pytest.raises(ValueError, match="same number of components"):
        data.add_reprocessed_offsets(matches)
