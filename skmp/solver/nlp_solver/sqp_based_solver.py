import copy
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Type

import numpy as np

from skmp.constraint import ConfigPointConst
from skmp.satisfy import SatisfactionResult, satisfy_by_optimization
from skmp.solver.interface import AbstractScratchSolver, Problem
from skmp.solver.nlp_solver.osqp_sqp import OsqpSqpConfig, OsqpSqpResult, OsqpSqpSolver
from skmp.solver.nlp_solver.trajectory_constraint import (
    MotionStepInequalityConstraint,
    TrajectoryEqualityConstraint,
    TrajectoryInequalityConstraint,
)
from skmp.trajectory import Trajectory


def translate(
    problem: Problem, n_wp: int
) -> Tuple[TrajectoryEqualityConstraint, TrajectoryInequalityConstraint]:
    n_dof = len(problem.start)

    # equality
    traj_eq_const = TrajectoryEqualityConstraint(n_dof, n_wp, {}, [])

    init_const = ConfigPointConst(problem.start)
    init_const.reflect_skrobot_model(None)

    traj_eq_const.add(0, init_const)
    traj_eq_const.add_goal_constraint(problem.goal_const)

    if problem.global_eq_const is not None:
        for i in range(n_wp):
            traj_eq_const.add(i, problem.global_eq_const)

    # inequality
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
    force_deterministic: bool = False
    osqp_verbose: bool = False
    verbose: bool = False
    n_max_satisfaction_trial: int = 100  # only used if init traj is not satisfied
    ctol_eq: float = 1e-4
    ctol_ineq: float = 1e-3
    ineq_tighten_coef: float = (
        2.0  # NOTE: in some large problem like humanoid planning, this value should be zero
    )
    step_box: Optional[np.ndarray] = None
    _osqpsqp_config: OsqpSqpConfig = OsqpSqpConfig()  # don't directly access this
    timeout: Optional[int] = None
    return_osqp_result: bool = False  # helpful for debugging but memory footprint is large

    @property
    def osqpsqp_config(self) -> OsqpSqpConfig:
        osqpsqp_config = copy.deepcopy(self._osqpsqp_config)
        osqpsqp_config.n_max_eval = self.n_max_call
        osqpsqp_config.force_deterministic = self.force_deterministic
        osqpsqp_config.verbose = self.verbose
        osqpsqp_config.osqp_verbose = self.osqp_verbose
        osqpsqp_config.ctol_eq = self.ctol_eq
        osqpsqp_config.ctol_ineq = self.ctol_ineq
        if self.step_box is not None:
            # self.step_box is for single waypont
            # thus we need to scale it to n_wp
            step_box_stacked = np.tile(self.step_box, self.n_wp)
            osqpsqp_config.step_box = step_box_stacked
        return osqpsqp_config


@dataclass
class SQPBasedSolverResult:
    traj: Optional[Trajectory]
    time_elapsed: Optional[float]
    n_call: int
    osqpsqp_raw_result: Optional[OsqpSqpResult]

    @classmethod
    def abnormal(cls) -> "SQPBasedSolverResult":
        return cls(None, None, -1, None)


@dataclass
class SQPBasedSolver(AbstractScratchSolver[SQPBasedSolverConfig, SQPBasedSolverResult]):
    config: SQPBasedSolverConfig
    solver: Optional[OsqpSqpSolver]
    problem: Optional[Problem]
    post_motion_step_validator: Optional[MotionStepInequalityConstraint]

    @classmethod
    def get_result_type(cls) -> Type[SQPBasedSolverResult]:
        return SQPBasedSolverResult

    @classmethod
    def init(
        cls, config: SQPBasedSolverConfig, data_path: Optional[Any] = None
    ) -> "SQPBasedSolver":
        return cls(config, None, None, None)

    def _setup(self, problem: Problem) -> None:
        config = self.config
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
            return f - ctol_ineq * config.ineq_tighten_coef, jac

        solver = OsqpSqpSolver(
            smooth_mat,
            lambda x: traj_eq_const.evaluate(x),
            ineq_tighten,
            lb_stacked,
            ub_stacked,
        )
        self.solver = solver
        self.post_motion_step_validator = post_motion_step_validator
        self.problem = problem

    def _solve(self, init_traj: Optional[Trajectory] = None) -> "SQPBasedSolverResult":
        assert self.solver is not None, "setup is not called yet"
        assert self.problem is not None
        # TODO: add motion step constraint

        if init_traj is None:

            satisfy_result: Optional[SatisfactionResult] = None
            for _ in range(self.config.n_max_satisfaction_trial):
                satisfy_result = satisfy_by_optimization(
                    self.problem.goal_const,
                    self.problem.box_const,
                    self.problem.global_ineq_const,
                    None,
                )
                if satisfy_result.success:
                    break
            assert satisfy_result is not None
            if not satisfy_result.success:
                return SQPBasedSolverResult(None, None, -1, None)

            q_goal = satisfy_result.q
            init_traj = Trajectory.from_two_points(self.problem.start, q_goal, self.config.n_wp)

        x_init = init_traj.resample(self.config.n_wp).numpy().flatten()
        raw_result = self.solver.solve(x_init, config=self.config.osqpsqp_config)

        success = raw_result.success
        if success and self.post_motion_step_validator is not None:
            vals, _ = self.post_motion_step_validator.evaluate(raw_result.x)
            if np.any(vals < 0):
                if self.config.motion_step_satisfaction == "post":
                    success = False
                elif self.config.motion_step_satisfaction == "debug_ignore":
                    pass
                else:
                    assert False

        traj_solution: Optional[Trajectory] = None
        if success:
            traj_solution = Trajectory(list(raw_result.x.reshape(self.config.n_wp, -1)))

        if self.config.return_osqp_result:
            return SQPBasedSolverResult(traj_solution, None, raw_result.nit, raw_result)
        else:
            return SQPBasedSolverResult(traj_solution, None, raw_result.nit, None)
