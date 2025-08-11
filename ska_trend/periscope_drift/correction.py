"""
Periscope Drift Correction

This module contains the function and constants used to correct the periscope drift.
"""

import numpy as np
import scipy

__all__ = [
    "GRADIENTS",
    "get_expected_correction",
]

GRADIENTS = {
    "OOBAGRD3": {
        "yag": 6.98145650e-04,
        "zag": 9.51578351e-05,
    },
    "OOBAGRD6": {
        "yag": -1.67009240e-03,
        "zag": -2.79084775e-03,
    },
}


def get_expected_correction(telem):
    """
    Calculate the expected periscope drift correction.

    Parameters
    ----------
    telem : dict
        Dictionary containing the telemetry data.
        Required keys are "OOBAGRD3" and "OOBAGRD6".
    """
    corr = {
        "ang_y_corr": np.zeros_like(telem["OOBAGRD6"]),
        "ang_z_corr": np.zeros_like(telem["OOBAGRD6"]),
        "OOBAGRD_corr_angle": np.nan,
    }

    if len(telem["OOBAGRD3"]) < 4:
        return corr

    msids = [key for key in telem if key != "time"]
    for msid in msids:
        if msid not in GRADIENTS:
            continue
        mean_gradient = np.mean(telem[msid])
        corr["ang_y_corr"] -= (telem[msid] - mean_gradient) * GRADIENTS[msid]["yag"]
        corr["ang_z_corr"] -= (telem[msid] - mean_gradient) * GRADIENTS[msid]["zag"]

    corr["ang_y_corr"] *= 3600
    corr["ang_z_corr"] *= 3600

    covariance = np.cov([telem["OOBAGRD3"], telem["OOBAGRD6"]])
    eig_vals, eig_vec = scipy.linalg.eig(covariance)
    vec = eig_vec[:, np.argmax(eig_vals)]
    corr_angle = np.arctan2(vec[1], vec[0])
    corr["OOBAGRD_corr_angle"] = corr_angle

    return corr
