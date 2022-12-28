from typing import List, Union

import numpy as np

from skmp.constraint import AbstractIneqConst


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


def is_valid_motion_step(
    motion_step_box: Union[np.ndarray, float],
    q1: np.ndarray,
    q2: np.ndarray,
    ineq_const: AbstractIneqConst,
) -> bool:
    if isinstance(motion_step_box, float):
        motion_step_box = motion_step_box * np.ones(len(q1))

    fractions = interpolate_fractions(motion_step_box, q1, q2, True)
    for frac in fractions:
        q_test = q1 + (q2 - q1) * frac
        fs, _ = ineq_const.evaluate_single(q_test, False)
        is_valid = np.all(fs > 0.0)
        if not is_valid:
            return False
    return True
