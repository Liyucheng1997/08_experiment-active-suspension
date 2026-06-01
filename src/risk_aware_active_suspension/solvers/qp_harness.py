from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp


@dataclass
class QPSolveResult:
    u: np.ndarray
    status: str
    iterations: int
    solve_time_s: float
    objective: float


class BoxQP:
    """Thin OSQP wrapper for QPs with a constant Hessian, varying linear term, and box bounds.

    Cost (OSQP convention): (1/2) u^T P u + q^T u
        s.t. lb <= u <= ub

    Hessian P is set once at setup() and reused; on each compute() only the linear
    term q is updated and the solver is warm-started from the previous solution.
    This is the "Parameter pattern" mentioned in the plan but implemented at the
    OSQP layer (no cvxpy overhead).
    """

    def __init__(
        self,
        hessian: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        eps_abs: float = 1e-8,
        eps_rel: float = 1e-8,
        max_iter: int = 4000,
        warm_starting: bool = True,
    ) -> None:
        n = hessian.shape[0]
        if hessian.shape != (n, n):
            raise ValueError(f"Hessian must be square, got {hessian.shape}.")
        if lower.shape != (n,) or upper.shape != (n,):
            raise ValueError("Bounds must have shape (n,).")
        self._n = n
        self._hessian = np.asarray(hessian, dtype=float)
        self._lower = np.asarray(lower, dtype=float)
        self._upper = np.asarray(upper, dtype=float)

        # OSQP uses upper-triangular P internally; pass the symmetric Hessian.
        P_sparse = sp.csc_matrix(self._hessian)
        A_sparse = sp.eye(n, format="csc")
        q_init = np.zeros(n)

        self._prob = osqp.OSQP()
        self._prob.setup(
            P=P_sparse,
            q=q_init,
            A=A_sparse,
            l=self._lower,
            u=self._upper,
            verbose=False,
            warm_starting=warm_starting,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            polishing=False,
        )

    @property
    def n(self) -> int:
        return self._n

    def update_linear(self, q: np.ndarray) -> None:
        self._prob.update(q=np.asarray(q, dtype=float))

    def update_bounds(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self._prob.update(l=np.asarray(lower, dtype=float), u=np.asarray(upper, dtype=float))

    def solve(self) -> QPSolveResult:
        t0 = time.perf_counter()
        res = self._prob.solve(raise_error=False)
        elapsed = time.perf_counter() - t0
        info = res.info
        status = str(getattr(info, "status", "unknown"))
        if status not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"OSQP failed: status={status}")
        u = np.asarray(res.x, dtype=float)
        return QPSolveResult(
            u=u,
            status=status,
            iterations=int(getattr(info, "iter", 0)),
            solve_time_s=float(elapsed),
            objective=float(getattr(info, "obj_val", np.nan)),
        )
