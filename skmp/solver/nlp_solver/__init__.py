# flake8: noqa

import importlib

from packaging import version

from skmp.solver.nlp_solver.sqp_based_solver import (
    SQPBasedSolver,
    SQPBasedSolverConfig,
    SQPBasedSolverResult,
)

GPY_INSTALLED = importlib.util.find_spec("GPy") is not None  # type: ignore[attr-defined]
if GPY_INSTALLED:
    import numpy

    _NUMPY_COMPATIBLE = version.parse(numpy.__version__) <= version.parse("1.23")
    assert _NUMPY_COMPATIBLE, "GPy works only with numpy <= 1.23"

    from skmp.solver.nlp_solver.memmo import (
        GprMemmoSolver,
        NnMemmoSolver,
        PcaGprMemmoSolver,
    )
