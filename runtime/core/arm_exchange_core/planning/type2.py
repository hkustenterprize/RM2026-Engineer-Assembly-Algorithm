from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def plan_joint_path_bitstar(
    start_q,
    goal_q,
    *,
    validity_fn: Callable[[np.ndarray], bool],
    joint_lower: Sequence[float],
    joint_upper: Sequence[float],
    timeout_s: float,
) -> np.ndarray:
    """
    Minimal OMPL BIT* point-to-point planner in 6D joint space.

    Returns sparse joint waypoints including start and goal. The caller owns all
    task semantics, smoothing, time parameterization, and diagnostics.
    """

    import ompl.base as ob
    import ompl.geometric as og

    start_q = np.asarray(start_q, dtype=float)
    goal_q = np.asarray(goal_q, dtype=float)
    lower = np.asarray(joint_lower, dtype=float)
    upper = np.asarray(joint_upper, dtype=float)
    if start_q.shape != goal_q.shape:
        raise ValueError(f"start_q and goal_q shape mismatch: {start_q.shape} vs {goal_q.shape}")
    if lower.shape != start_q.shape or upper.shape != start_q.shape:
        raise ValueError("joint bounds must match joint vector shape")
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")

    dof = int(start_q.shape[0])
    space = ob.RealVectorStateSpace(dof)
    bounds = ob.RealVectorBounds(dof)
    for i in range(dof):
        bounds.setLow(i, float(lower[i]))
        bounds.setHigh(i, float(upper[i]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)

    class ValidityChecker(ob.StateValidityChecker):
        def __init__(self, space_information):
            super().__init__(space_information)

        def isValid(self, state) -> bool:
            q = np.asarray([state[i] for i in range(dof)], dtype=float)
            return bool(validity_fn(q))

    si.setStateValidityChecker(ValidityChecker(si))
    si.setStateValidityCheckingResolution(0.003)
    si.setup()

    start_state = space.allocState()
    goal_state = space.allocState()
    for i in range(dof):
        start_state[i] = float(start_q[i])
        goal_state[i] = float(goal_q[i])

    problem = ob.ProblemDefinition(si)
    problem.setStartAndGoalStates(start_state, goal_state)
    problem.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    planner = og.BITstar(si)
    planner.setProblemDefinition(problem)
    planner.setup()

    solved = planner.solve(float(timeout_s))
    if not solved:
        raise RuntimeError("OMPL BIT* failed to find a Type II path")

    path = problem.getSolutionPath()
    simplifier = og.PathSimplifier(si)
    simplifier.reduceVertices(path, maxSteps=200)
    simplifier.collapseCloseVertices(path)

    waypoints = np.asarray(
        [[path.getState(i)[j] for j in range(dof)] for i in range(path.getStateCount())],
        dtype=float,
    )
    if waypoints.ndim != 2 or waypoints.shape[0] < 2:
        raise RuntimeError(f"OMPL returned invalid path shape {waypoints.shape}")
    return waypoints
