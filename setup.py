# Licensed under a 3-clause BSD style license - see LICENSE
from setuptools import setup

entry_points = {
    "console_scripts": [
        "ska_trend_wrong_box_updates=ska_trend.wrong_box_acq.wrong_box_acq:main",
    ]
}

setup(
    name="ska_trend",
    author="Tom Aldcroft, Jean Connelly, Javier Gonzalez",
    description="Ska trending",
    author_email="jconnelly@cfa.harvard.edu",
    url="https://sot.github.io/ska_trend",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    zip_safe=False,
    license=(
        "New BSD/3-clause BSD License\nCopyright (c) 2023"
        " Smithsonian Astrophysical Observatory\nAll rights reserved."
    ),
    entry_points=entry_points,
    packages=["ska_trend", "ska_trend.wrong_box_acq"],
    package_data={
        "ska_trend": [
            "wrong_box_acq/index_template.html",
            "wrong_box_acq/task_schedule.cfg",
        ]
    },
)
