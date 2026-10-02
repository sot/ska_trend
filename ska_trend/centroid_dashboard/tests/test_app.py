import argparse
import itertools
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from cxotime import CxoTime
from matplotlib import pyplot as plt

import ska_trend.centroid_dashboard.app as cent_app

# Non-interactive backend so the plot tests never touch a display.
matplotlib.use("agg")

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


@pytest.fixture
def crs_slots() -> dict[int, CentroidResidualsLite]:
    """Centroid residuals for slots 3-7 with distinguishable dyags."""
    times = 1000.0 + np.arange(5) * 1.025
    return {
        slot: CentroidResidualsLite(
            dyags=np.full(5, float(slot)),
            dzags=np.full(5, float(slot)),
            yag_times=times.copy(),
            zag_times=times.copy(),
        )
        for slot in range(3, 8)
    }


def test_select_crs_slots_default_is_unchanged(crs_slots) -> None:
    assert cent_app.select_crs_slots(crs_slots, None) is crs_slots


@pytest.mark.parametrize("slot", [6, np.int64(6)])
def test_select_crs_slots_int(crs_slots, slot) -> None:
    """A plain int (or numpy int) selects a single slot."""
    out = cent_app.select_crs_slots(crs_slots, slot)

    assert list(out) == [slot]
    assert out[slot] is crs_slots[6]


def test_select_crs_slots_list_keeps_order(crs_slots) -> None:
    out = cent_app.select_crs_slots(crs_slots, [7, 3])

    assert list(out) == [7, 3]
    # The input dict is not modified.
    assert list(crs_slots) == [3, 4, 5, 6, 7]


def test_select_crs_slots_missing_raises(crs_slots) -> None:
    """All missing slots are reported at once, not just the first."""
    with pytest.raises(
        ValueError, match=r"slots \[9, 11\] not in .* \[3, 4, 5, 6, 7\]"
    ):
        cent_app.select_crs_slots(crs_slots, [3, 9, 11])


def test_plot_crs_time_slots_is_keyword_only(crs_slots, tmp_path) -> None:
    with pytest.raises(TypeError, match="positional argument"):
        cent_app.plot_crs_time(crs_slots, tmp_path / "x.png", [6])


@pytest.mark.parametrize("slots", [6, [6], [6, 3], None])
def test_plot_crs_time_slots(crs_slots, tmp_path, slots) -> None:
    """Plotting works for a single slot as well as a subset or all slots."""
    save_path = tmp_path / "crs_time.png"

    cent_app.plot_crs_time(crs_slots, save_path, slots=slots)

    assert save_path.exists()


def test_shade_no_track_intervals() -> None:
    """One shaded span per NaN run, positioned relative to t_ref."""
    times = 1000.0 + np.arange(20) * 1.025
    dyags = np.ones(20)
    dyags[3:6] = np.nan
    dyags[15:18] = np.nan
    cr = CentroidResidualsLite(
        dyags=dyags,
        dzags=dyags.copy(),
        yag_times=times.copy(),
        zag_times=times.copy(),
    )
    _fig, ax = plt.subplots()

    cent_app.shade_no_track_intervals(ax, cr, times[0])

    # Two NaN runs in each of dyag and dzag, which are shaded independently.
    spans = sorted((patch.get_x(), patch.get_width()) for patch in ax.patches)
    assert len(spans) == 4
    # Spans bracket the NaN samples, relative to t_ref.
    for x, width in spans[:2]:
        assert times[2] - times[0] < x < times[3] - times[0]
        assert times[5] < x + width + times[0] < times[6]
    plt.close(_fig)


def test_shade_no_track_intervals_no_nan() -> None:
    """No NaN means no shading at all."""
    times = 1000.0 + np.arange(10) * 1.025
    cr = CentroidResidualsLite(
        dyags=np.ones(10),
        dzags=np.ones(10),
        yag_times=times.copy(),
        zag_times=times.copy(),
    )
    _fig, ax = plt.subplots()

    cent_app.shade_no_track_intervals(ax, cr, times[0])

    assert len(ax.patches) == 0
    plt.close(_fig)


def make_obs_stub(obsid, source, date_starcat, obs_start, obs_stop, manvrs):  # noqa: PLR0917
    """Minimal stand-in for Observation with the attrs resolve_duplicate_obs_key uses."""
    return SimpleNamespace(
        obsid=obsid,
        source=source,
        date_starcat=CxoTime(date_starcat),
        obs_start=CxoTime(obs_start),
        obs_stop=CxoTime(obs_stop),
        manvrs=manvrs,
    )


def test_resolve_duplicate_obs_key_distinct_keys() -> None:
    """Different obsids are left alone."""
    obs = make_obs_stub(
        26461,
        "SEP2622B",
        "2022:274:10:56:33.361",
        "2022:274:11:16:58.964",
        "2022:274:11:25:18.712",
        ["manvr1"],
    )
    obs_next = make_obs_stub(
        26093,
        "SEP2622B",
        "2022:274:16:04:24.712",
        "2022:274:16:29:32.111",
        "2022:275:00:20:31.859",
        ["manvr2"],
    )

    assert cent_app.resolve_duplicate_obs_key(obs, obs_next) is False
    assert obs_next.obs_start == CxoTime("2022:274:16:29:32.111")


def test_resolve_duplicate_obs_key_merges_split_dwell() -> None:
    """A dwell split by a mid-NPNT obsid change merges into one observation."""
    obs = make_obs_stub(
        26461,
        "SEP2622B",
        "2022:274:10:56:33.361",
        "2022:274:11:16:58.964",
        "2022:274:11:25:18.712",
        ["manvr_to_dwell"],
    )
    obs_next = make_obs_stub(
        26461,
        "SEP2622B",
        "2022:274:10:56:33.361",
        "2022:274:11:25:18.713",
        "2022:274:16:04:18.712",
        ["manvr_spanning_split"],
    )

    assert cent_app.resolve_duplicate_obs_key(obs, obs_next) is True
    # Merged observation spans the full dwell and keeps the real maneuver to it.
    assert obs_next.obs_start == CxoTime("2022:274:11:16:58.964")
    assert obs_next.obs_stop == CxoTime("2022:274:16:04:18.712")
    assert obs_next.manvrs == ["manvr_to_dwell"]


def test_resolve_duplicate_obs_key_drops_distinct_starcat() -> None:
    """Distinct observations sharing a report key: keep the later one only."""
    obs = make_obs_stub(
        25768,
        "MAR1323A",
        "2023:075:16:17:15.705",
        "2023:075:16:40:34.435",
        "2023:075:19:01:34.183",
        ["manvr1"],
    )
    obs_next = make_obs_stub(
        25768,
        "MAR1323A",
        "2023:075:19:01:40.183",
        "2023:075:19:32:55.411",
        "2023:075:23:42:15.159",
        ["manvr2"],
    )

    assert cent_app.resolve_duplicate_obs_key(obs, obs_next) is True
    # The surviving observation is untouched, unlike the merge case.
    assert obs_next.obs_start == CxoTime("2023:075:19:32:55.411")
    assert obs_next.manvrs == ["manvr2"]


@pytest.mark.parametrize(
    "start,stop,obsid,source",
    [
        # Dwell split in two by a mid-NPNT obsid change (OR -> perigee CTI obsid).
        ("2022:274:08:00:00", "2022:274:18:00:00", 26461, "SEP2622B"),
        # Distinct observations that kadi gave the same scheduled obsid after the
        # 2023:074 SCS-107 plus an unusually timed COAOSQID command event.
        ("2023:075:14:00:00", "2023:076:02:00:00", 25768, "MAR1323A"),
    ],
)
def test_resolve_duplicate_obs_key_on_real_observations(
    start, stop, obsid, source
) -> None:
    """kadi still yields these as an adjacent pair, and both are resolved away."""
    opt = argparse.Namespace(obsid=None, data_root="reports", remote_copy=False)
    obss = [obs for obs in cent_app.yield_observations(start, stop, opt) if obs]

    keys = [(obs.source, obs.obsid) for obs in obss]
    assert keys.count((source, obsid)) == 2

    idx = keys.index((source, obsid))
    assert idx + 1 == keys.index((source, obsid), idx + 1), "pair is not adjacent"
    assert cent_app.resolve_duplicate_obs_key(obss[idx], obss[idx + 1]) is True

    # No other adjacent pair in the window shares a report key.
    pairs = list(itertools.pairwise(obss))
    others = [pair for i, pair in enumerate(pairs) if i != idx]
    assert not any(cent_app.resolve_duplicate_obs_key(a, b) for a, b in others)
