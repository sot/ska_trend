import argparse
import functools
from pathlib import Path

import astropy.units as u
import jinja2
import numpy as np
from acdc.common import send_mail
from astropy.table import Table, join, vstack
from cheta import fetch
from cheta.utils import logical_intervals
from cxotime import CxoTime, CxoTimeLike
from kadi import events
from kadi.commands.observations import get_starcats
from ska_helpers.logging import basic_logger
from ska_helpers.run_info import log_run_info

from ska_trend import __version__

DOC_ID = "1qHF6-3oz6MSJYd3goQTMdcteXWUjKyHycdWSMCd9S40"
GID = 0
url_start = "https://docs.google.com/spreadsheets/d"
GSHEET_URL = f"{url_start}/{DOC_ID}/export?format=csv&id={DOC_ID}&gid={GID}"
GSHEET_USER_URL = f"{url_start}/{DOC_ID}/edit?usp=sharing"

# Constants and file path definitions
FILE_DIR = Path(__file__).parent


def INDEX_TEMPLATE_PATH():
    return FILE_DIR / "index_template.html"


URL = "https://cxc.cfa.harvard.edu/mta/ASPECT/fid_drop_mon/index.html"
LOGGER = basic_logger(__name__, level="INFO")

fetch.data_source.set("cxc")


def get_opt():
    parser = argparse.ArgumentParser(description="Fid drop monitor")
    parser.add_argument("--start", help="Start date")
    parser.add_argument("--stop", help="Stop date")
    parser.add_argument(
        "--out-dir",
        default="./out",
        help="Output directory",
    )
    parser.add_argument(
        "--email",
        action="append",
        dest="emails",
        default=[],
        help="Email address for notification",
    )
    return parser


def get_scs107_intervals(start: CxoTimeLike, stop: CxoTimeLike) -> Table:
    """
    Get the SCS107 SOSA intervals from the telemetry.

    This is only valid after SOSA start in 2011.

    Parameters
    ----------
    start : CxoTimeLike
        Start time for the data to be fetched.
    stop : CxoTimeLike
        Stop time for the data to be fetched.

    Returns
    -------
    Table
        Table of SCS107 SOSA intervals.

    The returned table has the following columns:
        - datestart: Start time of the interval.
        - datestop: Stop time of the interval.
    """
    import kadi.commands as kc

    # Manually add some SCS107s before cmd event sheet
    scs107_dates = [
        "2017:254:07:51:39.000",
        "2017:250:02:41:28.000",
        "2015:173:22:40:00.000",
        "2015:076:04:34:30.000",
        "2014:357:11:32:23.000",
        "2014:356:04:52:35.000",
        "2014:255:11:51:18.000",
        "2014:007:20:39:16.000",
        "2013:076:12:30:11.336",
        "2012:201:11:44:57.000",
        "2012:196:21:07:00.000",
        "2012:194:19:59:42.088",
        "2012:138:02:18:00.000",
        "2012:073:22:41:25.000",
        "2012:067:05:29:47.000",
        "2012:058:03:24:21.000",
        "2012:027:15:15:02.056",
        "2012:023:06:00:38.000",
    ]

    sosa_patch = "2011:335"
    min_cmd_start = np.min([CxoTime(start), CxoTime(sosa_patch)])
    cmds = kc.get_cmds(start=min_cmd_start)
    ok = (cmds["tlmsid"] == "OORMPDS") & (cmds["source"] == "CMD_EVT")
    cmds_rmpds_evt = cmds[ok]
    ok2 = cmds_rmpds_evt["event"] == "SCS-107"
    cmd_dates_scs107 = [cmd["date"] for cmd in cmds_rmpds_evt[ok2]]
    ok_sci_cmds = (cmds["source"] != "CMD_EVT") & (
        np.in1d(cmds["scs"], [131, 132, 133])
    )

    scs107_dates.extend(cmd_dates_scs107)
    scs107_dates = sorted(set(scs107_dates))

    scs107_intervals = []

    for date in scs107_dates:
        # Get the date of the next command that has a source that isn't "CMD_EVT"
        idx = np.searchsorted(cmds[ok_sci_cmds]["date"], date)
        if idx == len(cmds[ok_sci_cmds]):
            datestop = stop
        else:
            datestop = cmds[ok_sci_cmds]["date"][idx]
        scs107_intervals.append(
            {
                "datestart": date,
                "datestop": datestop,
            }
        )
    return Table(scs107_intervals)


def get_fid_data(start: CxoTimeLike, stop: CxoTimeLike) -> Table:
    """
    Get the fid tracking data for dwells from time start.

    Parameters
    ----------
    start : CxoTimeLike
        Start time for the data to be fetched.
    stop : CxoTimeLike
        Stop time for the data to be fetched.

    Returns
    -------
    Table
        Table of fid tracking data.

    The returned table has the following columns:
        - kalman_start: Start time of the Kalman filter.
        - next_nman_start: Start time of the next NMAN.
        - starcat_date: Date of the star catalog.
        - obsid_starcat: Observation ID from star catalog / kadi starcat
        - slot: Slot number.
        - track_samples: Number of AOACFCT{n} samples.
        - track_ok: Number of AOACFCT samples with "TRAK" status.
        - track_fraction: Fraction of samples with "TRAK" status.
    """
    start = CxoTime(start)
    stop = CxoTime(stop)
    manvrs = events.manvrs.filter(
        kalman_start__gte=start.date, next_nman_start__lte=stop.date
    )
    starcats = get_starcats(start=start - 5 * u.day)
    dates = [starcat.date for starcat in starcats]

    scs107_intervals = get_scs107_intervals(start, stop)

    fid_data = []
    for i, manvr in enumerate(manvrs):
        LOGGER.info(
            f"Processing npnt intervals in manvr with kalman_start: {manvr.kalman_start} "
        )
        # get the index of the starcat before the kalman start
        idx = np.searchsorted(dates, manvr.kalman_start) - 1
        # get the starcat before the kalman start
        starcat = starcats[idx]
        fid_slots = list(starcat["slot"][(starcat["type"] == "FID")])
        fid_ids = list(starcat["id"][(starcat["type"] == "FID")])
        if len(fid_slots) == 0:
            continue
        # If the "last" starcat happened before the last manvr kalman start
        # skip, as something isn't right
        if i > 0 and starcat.date < manvrs[i - 1].kalman_start:
            continue

        # Break up the manvr steady interval into NPNT intervals if needed
        aopcadmd = fetch.Msid("AOPCADMD", manvr.kalman_start, manvr.next_nman_start)
        npnt = logical_intervals(aopcadmd.times, aopcadmd.vals == "NPNT")

        for row in npnt:
            npnt_start = row["datestart"]
            npnt_stop = row["datestop"]

            # If the start of the interval is within an SCS107 interval, skip this row
            if np.any(
                (CxoTime(npnt_start) >= CxoTime(scs107_intervals["datestart"]))
                & (CxoTime(npnt_start) <= CxoTime(scs107_intervals["datestop"]))
            ):
                continue

            # Check CORADMEN for SCS107 equivalent
            radmon = fetch.Msid("CORADMEN", npnt_start, npnt_stop)
            # If SCS107, truncate the range to process
            if np.any(radmon.vals == "DISA"):
                npnt_stop = CxoTime(
                    radmon.times[np.where(radmon.vals == "DISA")[0][0]]
                ).date

            # If down to less than a minute, skip
            if CxoTime(npnt_stop) - CxoTime(npnt_start) < 60 * u.s:
                continue

            dat = fetch.Msidset(
                ["AOACASEQ", "AOACFCT0", "AOACFCT1", "AOACFCT2"], npnt_start, npnt_stop
            )
            dat.interpolate(1.025)

            for slot, fid_id in zip(fid_slots, fid_ids, strict=True):
                track_msid = f"AOACFCT{slot}"
                track_telem = dat[track_msid]
                ok_kalm = dat["AOACASEQ"].vals == "KALM"
                track_samples = np.count_nonzero(ok_kalm)
                track_ok = np.count_nonzero(track_telem.vals[ok_kalm] == "TRAK")
                track_fraction = track_ok / track_samples
                fid_data.append(
                    {
                        "start": npnt_start,
                        "stop": npnt_stop,
                        "starcat_date": starcat.date,
                        "obsid_telem": starcat.obsid,
                        "slot": slot,
                        "fid_id": fid_id,
                        "track_samples": track_samples,
                        "track_ok": track_ok,
                        "track_fraction": track_fraction,
                    }
                )

    return Table(fid_data)


@functools.cache
def get_fid_notes(data_root) -> Table:
    """
    Get the high background notes from the Google Sheet.

    Parameters
    ----------
    data_root : str
        The root directory for the data
    Returns
    -------
    dat : astropy.table.Table
        Table of notes
    """
    LOGGER.info(f"Reading google sheet {GSHEET_URL}")
    dat = None
    try:
        dat = Table.read(GSHEET_URL, format="ascii.csv")
    except Exception as e:
        LOGGER.error(f"Failed to read {GSHEET_URL} with error: {e}")

    if dat is not None:
        dat.write(
            Path(data_root) / "notes.csv",
            format="ascii.csv",
            overwrite=True,
        )
    else:
        dat = Table.read(Path(data_root) / "notes.csv", format="ascii.csv")

    return dat


def main(args=None):
    """
    Main function to update the fid drop monitor data and web page.
    """
    opt = get_opt().parse_args(args)
    log_run_info(LOGGER.info, opt, version=__version__)

    fid_data_file = Path(opt.out_dir) / "fids_data.dat"
    Path(opt.out_dir).mkdir(parents=True, exist_ok=True)

    start = None
    stop = None
    fid_data_archive = None
    min_table_start = "2011:335"

    if Path(fid_data_file).exists():
        fid_data_archive = Table.read(fid_data_file, format="ascii.ecsv")
        start = CxoTime(fid_data_archive["start"][-1]) + 10 * u.s
    else:
        start = "2002:001"

    if opt.start is not None:
        # Override whatever we have in the archive
        start = CxoTime(opt.start)
        # And clear the archive after the supplied time
        if fid_data_archive is not None:
            ok = fid_data_archive["start"] < CxoTime(start).date
            fid_data_archive = fid_data_archive[ok]

    if opt.stop is not None:
        stop = CxoTime(opt.stop)
    else:
        # Get the time of the last AOPCADMD data in the cxc archive
        _, end_aopcadmd = fetch.get_time_range("AOPCADMD")
        stop = CxoTime(end_aopcadmd)

    if fid_data_archive is None:
        fid_data = get_fid_data(start, stop)
    else:
        new_fid_data = get_fid_data(start, stop)
        fid_data = vstack([fid_data_archive, new_fid_data])
        fid_data.sort("start")

    fid_data.write(fid_data_file, format="ascii.ecsv", overwrite=True)

    # Filter to only include drops since min_table_start
    ok = CxoTime(fid_data["start"]) > CxoTime(min_table_start)
    fid_data = fid_data[ok]

    # Fid drops just defined as less than 95% tracking
    drop_events = fid_data[(fid_data["track_fraction"] < 0.95)]

    # Get any info from the google sheet
    fid_notes = get_fid_notes(opt.out_dir)

    # Add the notes to the drop events
    if len(drop_events) > 0:
        drop_events = join(drop_events, fid_notes, keys="start", join_type="left")

    # Last year of events
    ok = CxoTime(drop_events["start"]) > CxoTime() - 365 * u.day
    drop_events_last = drop_events[ok]

    index_template_html = INDEX_TEMPLATE_PATH().read_text()
    template = jinja2.Template(index_template_html)
    out_html = template.render(
        obs_events=drop_events[::-1],
        obs_events_last=drop_events_last[::-1],
        sheet_url=GSHEET_USER_URL,
        start=min_table_start,
    )
    html_path = Path(opt.out_dir) / "index.html"
    LOGGER.info(f"Writing HTML to {html_path}")
    html_path.write_text(out_html)

    # If any of the drops are new, send emails
    ok = drop_events["start"] > CxoTime(start).date
    if np.any(ok):
        for row in drop_events[ok]:
            LOGGER.warning(f"Fid drop in dwell starting at {row['start']}")
            if len(opt.emails) > 0:
                send_mail(
                    LOGGER,
                    opt,
                    f"Fid drop: Obsid {row['obsid_telem']} {row['start']}",
                    f"Fid drop in the dwell that starts at {row['start']}\n"
                    f"Obsid {row['obsid_telem']} Fid slot {row['slot']} tracked "
                    f"for {row['track_fraction']:.3f} fraction.\n"
                    f"See {URL}",
                    __file__,
                )


if __name__ == "__main__":
    main()
