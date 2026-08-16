"""
Small time-parsing helpers shared by the report CLI scripts.
"""

from cxotime import CxoTime, TimeDelta

__all__ = ["parse_time_arg"]


def parse_time_arg(value, stop):
    """
    Parse a report-time CLI argument.

    Two forms are accepted:

    - ``"stop<delta>"`` is interpreted relative to ``stop``, where ``<delta>`` is passed straight
      to ``TimeDelta`` (astropy ``quantity_str`` format). Examples: ``"stop-90d"``, ``"stop-5yr"``,
      ``"stop+30d"`` (units: ``yr d hr min s``). Using ``stop`` as the prefix (rather than a bare
      leading ``-``) keeps argparse from mistaking the value for an option flag.
    - Anything else is passed to ``CxoTime``: an absolute date (e.g. ``"2026-02-02"``) or a
      now-relative offset (e.g. ``"-90d"``).

    Parameters
    ----------
    value : str
        The CLI argument value.
    stop : CxoTime
        The reference stop time for ``"stop<delta>"`` forms.

    Returns
    -------
    CxoTime
    """
    value = str(value).strip()
    if value.startswith("stop"):
        delta = value[len("stop") :].strip()
        return stop + TimeDelta(delta, format="quantity_str")
    return CxoTime(value)
