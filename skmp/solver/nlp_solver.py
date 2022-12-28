import time
from dataclasses import dataclass, fields
from typing import Literal, Optional, Tuple, Union

import numpy as np

from skmp.constraint import ConfigPointConst
from skmp.solver.interface import AbstractSolver, Problem
from skmp.solver.osqp_sqp import OsqpSqpConfig, OsqpSqpResult, OsqpSqpSolver
from skmp.solver.trajectory_constraint import (
    MotionStepInequalityConstraint,
    TrajectoryEqualityConstraint,
    TrajectoryInequalityConstraint,
)
from skmp.trajectory import Trajectory


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


def smoothcost_fullmat(n_dof: int, n_wp: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute A of eq. (17) of IJRR-version (2013) of CHOMP"""

    def construct_smoothcost_mat(n_wp):
        # In CHOMP (2013), squared sum of velocity is computed.
        # In this implementation we compute squared sum of acceralation
        # if you set acc_block * 0.0, vel_block * 1.0, then the trajectory
        # cost is same as the CHOMP one.
        acc_block = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        vel_block = np.array([[1, -1], [-1, 1]])
        A_ = np.zeros((n_wp, n_wp))
        for i in [1 + i for i in range(n_wp - 2)]:
            A_[i - 1 : i + 2, i - 1 : i + 2] += acc_block * 1.0
            A_[i - 1 : i + 1, i - 1 : i + 1] += vel_block * 0.0  # do nothing
        return A_

    if weights is None:
        weights = np.ones(n_dof)

    w_mat = np.diag(weights)
    A_ = construct_smoothcost_mat(n_wp)
    A = np.kron(A_, w_mat**2)
    return A


@dataclass
class SQPBasedSolverConfig:
    """this is almost same as OsqpSqpConfig
    except for n_wp, motion_step_box, check_motion_step_finally

    I wanted to use OsqpSqpConfig as a super class but it's not possible
    because we need to put n_wp at the top of the attribute
    """

    n_wp: int
    ftol: float = 1e-3
    ctol_eq: float = 1e-4  # constraint tolerance
    ctol_ineq: float = 1e-3  # constraint tolerance
    n_max_eval: int = 30
    maxrelax: int = 10
    trust_box_init_size: float = 0.5
    osqp_verbose: bool = False
    verbose: bool = False
    relax_step_convex: float = 0.1
    motion_step_box: Union[np.ndarray, float] = 0.1
    motion_step_satisfaction: Literal["implicit", "explicit", "post"] = "implicit"

    def to_osqpsqp_config(self) -> OsqpSqpConfig:
        dic = {}
        for f in fields(OsqpSqpConfig):
            dic[f.name] = self.__dict__[f.name]
        return OsqpSqpConfig(**dic)


@dataclass
class SQPBasedSolver(AbstractSolver):
    traj_eq_const: TrajectoryEqualityConstraint
    traj_ineq_const: TrajectoryInequalityConstraint
    problem: Problem
    smooth_mat: np.ndarray
    config: SQPBasedSolverConfig

    @dataclass
    class Result:
        traj: Optional[Trajectory]
        time_elapsed: float
        osqpsqp_raw_result: OsqpSqpResult

    @classmethod
    def setup(cls, problem: Problem, config: SQPBasedSolverConfig) -> "SQPBasedSolver":  # type: ignore[override]
        traj_eq_const, traj_ineq_const = translate(problem, config.n_wp)
        n_dof = len(problem.start)
        mat = smoothcost_fullmat(n_dof, config.n_wp)
        return cls(traj_eq_const, traj_ineq_const, problem, mat, config)

    def solve(self, init_traj: Optional[Trajectory] = None) -> "SQPBasedSolver.Result":
        """solve
        Actually this function don't case init_traj
        """
        assert init_traj is not None  # TODO: remove this
        # TODO: add motion step constraint
        ts = time.time()

        x_init = init_traj.numpy().flatten()

        box_const = self.problem.box_const
        lb_stacked = np.tile(box_const.lb, self.config.n_wp)
        ub_stacked = np.tile(box_const.ub, self.config.n_wp)

        n_dof = self.traj_ineq_const.n_dof
        n_wp = self.traj_ineq_const.n_wp

        motion_step_box = self.config.motion_step_box
        if isinstance(motion_step_box, float):
            len(self.problem.start)
            motion_step_box = np.ones(n_dof) * motion_step_box

        if self.config.motion_step_satisfaction == "implicit":
            self.traj_ineq_const.motion_step_box = motion_step_box
        elif self.config.motion_step_satisfaction == "explicit":
            msconst = MotionStepInequalityConstraint(n_dof, n_wp, motion_step_box)
            self.traj_ineq_const.global_constraint_table.append(msconst)

        def ineq_tighten(x):
            # somehow, osqp-sqp result has some ineq error
            # thus to compensate that, we tighten the ineq constraint here
            f, jac = self.traj_ineq_const.evaluate(x)
            return f - self.config.ctol_ineq * 2, jac

        solver = OsqpSqpSolver(
            self.smooth_mat,
            lambda x: self.traj_eq_const.evaluate(x),
            ineq_tighten,
            lb_stacked,
            ub_stacked,
        )
        raw_result = solver.solve(x_init, config=self.config.to_osqpsqp_config())

        traj_solution: Optional[Trajectory] = None
        if raw_result.success:
            traj_solution = Trajectory(list(raw_result.x.reshape(self.config.n_wp, -1)))
        return self.Result(traj_solution, time.time() - ts, raw_result)
