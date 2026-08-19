from astropy import units as u
from cxotime import CxoTime

from ska_trend.time_utils import parse_time_arg


def test_parse_time_arg_reference_relative():
    stop = CxoTime("2026-02-02")

    # days and years back, relative to the named reference
    assert parse_time_arg("stop-90d", stop=stop).date == (stop - 90 * u.day).date
    assert parse_time_arg("stop-5yr", stop=stop).date == (stop - 5 * u.yr).date

    # forward offset, and whitespace-tolerant
    assert parse_time_arg("stop+30d", stop=stop).date == (stop + 30 * u.day).date
    assert parse_time_arg("stop + 30 d", stop=stop).date == (stop + 30 * u.day).date

    # the reference name is agnostic: any keyword works
    start = CxoTime("2020-01-01")
    assert parse_time_arg("start+10d", start=start).date == (start + 10 * u.day).date

    # the bare reference name (no offset) returns that reference time
    assert parse_time_arg("stop", stop=stop).date == stop.date


def test_parse_time_arg_absolute():
    stop = CxoTime("2026-02-02")
    assert parse_time_arg("2020-01-01", stop=stop).date == CxoTime("2020-01-01").date
