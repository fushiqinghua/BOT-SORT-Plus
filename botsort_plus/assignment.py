"""One-to-one assignment with real rejection choices and optional mutual margin."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_with_rejection(cost, max_cost, margin=0.0):
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or np.isnan(cost).any() or np.isneginf(cost).any():
        raise ValueError("Cost must be a 2D matrix; +inf denotes a forbidden edge")
    if np.any(cost < 0) or not np.isfinite(max_cost) or max_cost <= 0 or not np.isfinite(margin) or margin < 0:
        raise ValueError("Invalid assignment cost, threshold or margin")
    n, m = cost.shape
    if not n or not m:
        return [], list(range(n)), list(range(m))
    eligible = cost < max_cost
    if margin > 0:
        # Evaluate competitors before deleting any edge: ambiguity must not
        # disappear merely because a different ambiguous edge was removed.
        original = np.where(eligible, cost, np.inf)
        for i, j in zip(*np.nonzero(eligible.copy())):
            row_other = np.delete(original[i], j)
            col_other = np.delete(original[:, j], i)
            row_gap = (row_other.min() if row_other.size else np.inf) - cost[i,j]
            col_gap = (col_other.min() if col_other.size else np.inf) - cost[i,j]
            if row_gap < margin or col_gap < margin:
                eligible[i,j] = False
    forbidden = max_cost * (n + m + 2) + 1.0
    matrix = np.full((n, m+n), forbidden)
    matrix[:, :m] = np.where(eligible, cost, forbidden)
    matrix[np.arange(n), m + np.arange(n)] = max_cost
    rows, cols = linear_sum_assignment(matrix)
    pairs = [(int(i), int(j)) for i,j in zip(rows,cols) if j < m and eligible[i,j]]
    used_i, used_j = {i for i,j in pairs}, {j for i,j in pairs}
    return pairs, [i for i in range(n) if i not in used_i], [j for j in range(m) if j not in used_j]
