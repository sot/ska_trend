from astropy import units as u
from cxotime import CxoTime

from ska_trend.time_utils import parse_time_arg


def test_parse_time_arg_stop_relative():
    stop = CxoTime("2026-02-02")

    # days and years back, relative to stop
    assert parse_time_arg("stop-90d", stop).date == (stop - 90 * u.day).date
    assert parse_time_arg("stop-5yr", stop).date == (stop - 5 * u.yr).date

    # forward offset, and whitespace-tolerant
    assert parse_time_arg("stop+30d", stop).date == (stop + 30 * u.day).date
    assert parse_time_arg("stop + 30 d", stop).date == (stop + 30 * u.day).date


def test_parse_time_arg_absolute():
    stop = CxoTime("2026-02-02")
    assert parse_time_arg("2020-01-01", stop).date == CxoTime("2020-01-01").date
