from pathlib import Path


def report_subdir(obsid) -> str:
    obsid_str = "{:05d}".format(obsid)
    return f"{obsid_str[:2]}/{obsid_str}"


def report_dir(obsid, data_root) -> Path:
    return Path(data_root) / "reports" / report_subdir(obsid)


def index_html(obsid, data_root) -> Path:
    return report_dir(obsid, data_root) / "index.html"


def info_json(obsid, data_root) -> Path:
    return report_dir(obsid, data_root) / "info.json"
