import copy
import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

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
    init_const = ConfigPointConst(problem.start)
    init_const.reflect_skrobot_model(None)
    traj_eq_const.add(0, init_const)
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
    """
    motion_step_satisfaction: either of "implicit", "explicit", "post"

    NOTE: choice motion_step_satisfaction affects performance a lot.

    In general, the following inequality is observed.
    solvability: implicit > explicit >> post
    speed: post >> explicit > implicit
    when you choose good n_wp, the solvability order will be
    solvability: explicit > post ~ implicit
    """

    n_wp: int
    n_max_call: int = 30
    motion_step_satisfaction: Literal["implicit", "explicit", "post", "debug_ignore"] = "implicit"
    _osqpsqp_config: OsqpSqpConfig = OsqpSqpConfig()  # don't directly access this

    @property
    def osqpsqp_config(self) -> OsqpSqpConfig:
        osqpsqp_config = copy.deepcopy(self._osqpsqp_config)
        # for f in fields(OsqpSqpConfig):
        #     has_same_attribute = f.name in self.__dict__
        #     if has_same_attribute:
        #         osqpsqp_config.__dict__[f.name] = self.__dict__[f.name]
        osqpsqp_config.n_max_eval = self.n_max_call
        return osqpsqp_config


@dataclass
class SQPBasedSolverResult:
    traj: Optional[Trajectory]
    time_elapsed: float
    n_call: int
    osqpsqp_raw_result: OsqpSqpResult


@dataclass
class SQPBasedSolver(AbstractSolver[SQPBasedSolverConfig, SQPBasedSolverResult]):
    solver: OsqpSqpSolver
    config: SQPBasedSolverConfig
    post_motion_step_validator: Optional[MotionStepInequalityConstraint]

    @classmethod
    def setup(cls, problem: Problem, config: SQPBasedSolverConfig) -> "SQPBasedSolver":
        traj_eq_const, traj_ineq_const = translate(problem, config.n_wp)
        n_dof = len(problem.start)
        smooth_mat = smoothcost_fullmat(n_dof, config.n_wp)

        box_const = problem.box_const
        lb_stacked = np.tile(box_const.lb, config.n_wp)
        ub_stacked = np.tile(box_const.ub, config.n_wp)

        n_dof = traj_ineq_const.n_dof
        n_wp = traj_ineq_const.n_wp

        motion_step_box = problem.motion_step_box
        msconst = MotionStepInequalityConstraint(n_dof, n_wp, motion_step_box)

        if config.motion_step_satisfaction == "implicit":
            traj_ineq_const.motion_step_box = motion_step_box
            post_motion_step_validator = None
        elif config.motion_step_satisfaction == "explicit":
            traj_ineq_const.global_constraint_table.append(msconst)
            post_motion_step_validator = None
        elif config.motion_step_satisfaction in ["post", "debug_ignore"]:
            post_motion_step_validator = msconst
        else:
            assert False

        ctol_ineq = config.osqpsqp_config.ctol_ineq

        def ineq_tighten(x):
            # somehow, osqp-sqp result has some ineq error
            # thus to compensate that, we tighten the ineq constraint here
            f, jac = traj_ineq_const.evaluate(x)
            return f - ctol_ineq * 2, jac

        solver = OsqpSqpSolver(
            smooth_mat,
            lambda x: traj_eq_const.evaluate(x),
            ineq_tighten,
            lb_stacked,
            ub_stacked,
        )
        return cls(solver, config, post_motion_step_validator)

    def solve(self, init_traj: Optional[Trajectory] = None) -> "SQPBasedSolverResult":
        assert init_traj is not None  # TODO: remove this
        # TODO: add motion step constraint
        ts = time.time()

        x_init = init_traj.resample(self.config.n_wp).numpy().flatten()
        raw_result = self.solver.solve(x_init, config=self.config.osqpsqp_config)

        success = raw_result.success
        if success and self.post_motion_step_validator is not None:
            vals, _ = self.post_motion_step_validator.evaluate(raw_result.x)
            if np.any(vals < 0):
                if self.config.motion_step_satisfaction == "post":
                    success = False
                elif self.config.motion_step_satisfaction == "debug_ignore":
                    print("motion step constraint is not satisfied but ignore")
                else:
                    assert False

        traj_solution: Optional[Trajectory] = None
        if success:
            traj_solution = Trajectory(list(raw_result.x.reshape(self.config.n_wp, -1)))

        return SQPBasedSolverResult(traj_solution, time.time() - ts, raw_result.nit, raw_result)
