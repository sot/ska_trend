from pathlib import Path


def report_dir(obs, data_root) -> Path:
    return Path(data_root) / "reports" / obs.report_subdir


def index_html(obs, data_root) -> Path:
    return report_dir(obs, data_root) / "index.html"


def info_json(obs, data_root) -> Path:
    return report_dir(obs, data_root) / "info.json"
