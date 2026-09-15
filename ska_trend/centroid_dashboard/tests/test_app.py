import numpy as np
import pytest

import ska_trend.centroid_dashboard.app as cent_app

CentroidResidualsLite = cent_app.CentroidResidualsLite


@pytest.fixture
def crs() -> CentroidResidualsLite:
    return CentroidResidualsLite(
        dyags=np.array([1.0, 2.0, 3.0, 4.0]),
        dzags=np.array([10.0, 20.0, 30.0, 40.0]),
        yag_times=np.array([1000.0, 1200.0, 1400.0, 1600.0]),
        zag_times=np.array([1050.0, 1250.0, 1450.0, 1650.0]),
    )


def test_slice_time_window_from_start(crs: CentroidResidualsLite) -> None:
    out = crs[:500.0]

    np.testing.assert_array_equal(out.dyags, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(out.dzags, np.array([10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(out.yag_times, np.array([1000.0, 1200.0, 1400.0]))
    np.testing.assert_array_equal(out.zag_times, np.array([1050.0, 1250.0, 1450.0]))


def test_slice_time_window_from_end(crs: CentroidResidualsLite) -> None:
    out = crs[-500.0:]

    np.testing.assert_array_equal(out.dyags, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(out.dzags, np.array([20.0, 30.0, 40.0]))
    np.testing.assert_array_equal(out.yag_times, np.array([1200.0, 1400.0, 1600.0]))
    np.testing.assert_array_equal(out.zag_times, np.array([1250.0, 1450.0, 1650.0]))


def test_slice_by_index_still_works(crs: CentroidResidualsLite) -> None:
    out = crs[1:3]

    np.testing.assert_array_equal(out.dyags, np.array([2.0, 3.0]))
    np.testing.assert_array_equal(out.dzags, np.array([20.0, 30.0]))
    np.testing.assert_array_equal(out.yag_times, np.array([1200.0, 1400.0]))
    np.testing.assert_array_equal(out.zag_times, np.array([1250.0, 1450.0]))


def test_time_slice_with_step_raises(crs: CentroidResidualsLite) -> None:
    with pytest.raises(TypeError, match="time-based slicing does not support a step"):
        _ = crs[-500.0::2]


def test_get_centroid_resids_for_obsid_with_source() -> None:
    crs = cent_app.get_centroid_resids_for_obsid(29833, source="FEB0226A")

    assert list(crs.keys()) == [3, 4, 5, 6, 7]


def test_get_centroid_resids_for_obsid_without_source_raises_value_error() -> None:
    match = r"expected one observation matching the filter criteria but got 2"
    with pytest.raises(ValueError, match=match):
        cent_app.get_centroid_resids_for_obsid(29833)


def test_write_centroid_resids_does_not_bridge_no_track_gap(tmp_path) -> None:
    """NaN not-tracking samples stay NaN when interpolated onto the output grid.

    ``CentroidResiduals`` with ``set_no_track_to_nan=True`` gives a complete time base
    with NaN where the OBC was not tracking. The interpolation in
    ``write_centroid_resids`` must propagate that NaN rather than bridging the dropout
    with a smooth ramp of valid-looking residuals.
    """
    # Complete 1.025 sec time base with samples 10-19 not tracking.
    times = 1000.0 + np.arange(30) * 1.025
    dyags = np.ones(30)
    dzags = np.ones(30) * 2
    dyags[10:20] = np.nan
    dzags[10:20] = np.nan
    cr = CentroidResidualsLite(
        dyags=dyags, dzags=dzags, yag_times=times.copy(), zag_times=times.copy()
    )
    save_path = tmp_path / "centroid_resids.pkl"

    cent_app.write_centroid_resids({6: cr}, save_path)
    out = cent_app.get_centroid_resids_from_file(save_path)

    for attr in ["dyags", "dzags"]:
        vals = np.asarray(getattr(out[6], attr), dtype=np.float64)
        in_gap = (out[6].yag_times >= times[10]) & (out[6].yag_times <= times[19])
        assert np.all(np.isnan(vals[in_gap]))
        # Samples outside the dropout are unaffected.
        assert not np.any(np.isnan(vals[out[6].yag_times <= times[9]]))
