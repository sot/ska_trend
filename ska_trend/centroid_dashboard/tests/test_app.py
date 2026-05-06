
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

    assert (
        crs.keys()
        == {
            np.int64(3),
            np.int64(4),
            np.int64(5),
            np.int64(6),
            np.int64(7),
        }.keys()
    )


def test_get_centroid_resids_for_obsid_without_source_raises_value_error() -> None:
    match = r"expected one observation matching the filter criteria but got 2"
    with pytest.raises(ValueError, match=match):
        cent_app.get_centroid_resids_for_obsid(29833)
