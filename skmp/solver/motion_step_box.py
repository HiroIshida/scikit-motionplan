from typing import List

import numpy as np


def interpolate_fractions(
    motion_step_box: np.ndarray, q1: np.ndarray, q2: np.ndarray, include_q1: bool
) -> List[float]:
    # Note that q1 is omitted as in the checkMotion in ompl
    assert motion_step_box is not None

    # determine the active axis idx
    diff = q2 - q1
    abs_scaled_diff = np.abs(diff) / motion_step_box
    active_idx = np.argmax(abs_scaled_diff)

    diff_active_axis = diff[active_idx]
    two_point_two_close = abs(diff_active_axis) < 1e-6
    if two_point_two_close:
        return []

    step_ratio = motion_step_box[active_idx] / abs(diff_active_axis)
    if step_ratio > 1.0:
        return [1.0]  # only the last one

    travel_rate = 0.0
    interp_fractions = []

    if include_q1:
        interp_fractions.append(travel_rate)

    while travel_rate + step_ratio < 1.0:
        travel_rate += step_ratio
        interp_fractions.append(travel_rate)
    if abs(interp_fractions[-1] - 1) > 1e-6:
        interp_fractions.append(1.0)

    return interp_fractions
