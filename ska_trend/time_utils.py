"""
Small time-parsing helpers shared by the report CLI scripts.
"""

from cxotime import CxoTime, TimeDelta

__all__ = ["parse_time_arg"]


def parse_time_arg(value, **references):
    """
    Parse a report-time CLI argument.

    ``value`` is either:

    - a named reference followed by a ``TimeDelta`` offset, e.g. ``"stop-90d"`` or ``"start+10d"``.
      The reference name must match one of the keyword arguments; the offset (astropy
      ``quantity_str`` format, units ``yr d hr min s``) is added to that reference time. Using a
      name prefix (rather than a bare leading ``-``) keeps argparse from mistaking the value for an
      option flag. The reference name alone (no offset) returns that reference time.
    - anything else, passed to ``CxoTime``: an absolute date (e.g. ``"2026-02-02"``) or a
      now-relative offset (e.g. ``"-90d"``).

    Examples
    --------
    >>> parse_time_arg("stop-90d", stop=stop)      # stop - 90 days
    >>> parse_time_arg("start+10d", start=start)   # start + 10 days
    >>> parse_time_arg("2026-02-02", stop=stop)    # absolute date

    Parameters
    ----------
    value : str
        The CLI argument value.
    **references : CxoTime
        Named reference times that ``value`` may be expressed relative to.

    Returns
    -------
    CxoTime
    """
    value = str(value).strip()
    for name, ref in references.items():
        if value.startswith(name):
            delta = value[len(name) :].strip()
            if not delta:
                return CxoTime(ref)
            return CxoTime(ref) + TimeDelta(delta, format="quantity_str")
    return CxoTime(value)
