# Licensed under a 3-clause BSD style license - see LICENSE
from setuptools import setup

entry_points = {
    "console_scripts": [
        "ska_trend_wrong_box_updates=ska_trend.wrong_box_anom.wrong_box_anom:main",
        "ska_trend_bad_periscope=ska_trend.bad_periscope_gradient.periscope_update:main",
        "ska_trend_centroid_dashboard=ska_trend.centroid_dashboard.app:main",
        "ska_trend_fid_drop_mon_update=ska_trend.fid_drop_mon.update:main",
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
    packages=[
        "ska_trend",
        "ska_trend.wrong_box_anom",
        "ska_trend.bad_periscope_gradient",
        "ska_trend.centroid_dashboard",
        "ska_trend.fid_drop_mon",
    ],
    package_data={
        "ska_trend": [
            "wrong_box_anom/index_template.html",
            "wrong_box_anom/task_schedule.cfg",
            "bad_periscope_gradient/task_schedule.cfg",
            "centroid_dashboard/index_template.html",
            "centroid_dashboard/task_schedule.cfg",
            "fid_drop_mon/index_template.html",
            "fid_drop_mon/task_schedule.cfg",
        ]
    },
)
