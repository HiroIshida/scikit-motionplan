from typing import Tuple

from skmp.constraint import ConfigPointConst
from skmp.solver.interface import Problem
from skmp.solver.trajectory_constraint import (
    TrajectoryEqualityConstraint,
    TrajectoryInequalityConstraint,
)


def translate(
    problem: Problem, n_wp: int
) -> Tuple[TrajectoryEqualityConstraint, TrajectoryInequalityConstraint]:
    n_dof = len(problem.start)

    traj_eq_const = TrajectoryEqualityConstraint(n_dof, n_wp, {}, [])
    traj_eq_const.add(0, ConfigPointConst(problem.start))
    traj_eq_const.add_goal_constraint(problem.goal_const)

    traj_ineq_const = TrajectoryInequalityConstraint.create_homogeneous(
        n_wp, n_dof, problem.global_ineq_const
    )

    return traj_eq_const, traj_ineq_const
