"""Paired tests for three prompts evaluated on the same questions."""

import itertools

import numpy as np
from scipy import stats


def holm(pvalues):
    """Holm step-down family-wise correction, preserving input order."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(pvalues))
    maximum = 0.0
    for position, index in enumerate(order):
        maximum = max(maximum, (len(pvalues) - position) * pvalues[index])
        adjusted[index] = min(1.0, maximum)
    return adjusted.tolist()


def blocked_permutation_pvalue(values, resamples=49999, seed=42):
    """Shuffle prompt labels within each question, not across questions.

    Sum-of-squared column totals is monotone in Cochran's Q for binary
    observations and in Friedman Q for within-question ranks. Constants
    and tie correction are unchanged by these within-block permutations.
    """
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=1, keepdims=True)
    observed = float(np.square(centered.sum(axis=0)).sum())
    if observed == 0 and np.all(centered == 0):
        return 1.0
    permutations = np.asarray(list(itertools.permutations(range(3))))
    lookup = np.stack([centered[:, permutation] for permutation in permutations], axis=1)
    rng = np.random.default_rng(seed)
    extreme = 0
    done = 0
    while done < resamples:
        size = min(1000, resamples - done)
        choices = rng.integers(0, 6, size=(size, len(values)))
        samples = lookup[np.arange(len(values))[None, :], choices, :]
        statistic = np.square(samples.sum(axis=1)).sum(axis=1)
        extreme += int(np.count_nonzero(statistic >= observed - 1e-10))
        done += size
    return (extreme + 1) / (resamples + 1)


def cochran_q(values):
    values = np.asarray(values, dtype=int)
    n, k = values.shape
    if k != 3 or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected an n x 3 binary matrix")
    columns = values.sum(axis=0)
    rows = values.sum(axis=1)
    total = values.sum()
    denominator = k * total - np.square(rows).sum()
    statistic = 0.0 if denominator == 0 else float(
        (k - 1) * (k * np.square(columns).sum() - total ** 2) / denominator)
    variable_rows = values[(rows > 0) & (rows < k)]
    if len(variable_rows) <= 10:
        # Each discordant binary row has three equiprobable placements of
        # its one success (or one failure). Enumerate those placements.
        signs = np.where(variable_rows.sum(axis=1) == 1, 1, -1)
        lookup = signs[:, None, None] * (np.eye(3) - 1 / 3)[None, :, :]
        observed = float(np.square((variable_rows - variable_rows.mean(
            axis=1, keepdims=True)).sum(axis=0)).sum())
        assignments = list(itertools.product(range(3), repeat=len(variable_rows)))
        choices = np.asarray(assignments, dtype=int).reshape(len(assignments), len(variable_rows))
        samples = lookup[np.arange(len(variable_rows))[None, :], choices, :]
        simulated = np.square(samples.sum(axis=1)).sum(axis=1)
        pvalue = float(np.mean(simulated >= observed - 1e-10))
        calibration = "exact within-question label permutation"
        resamples = len(assignments)
    else:
        pvalue = blocked_permutation_pvalue(values)
        calibration = "Monte Carlo within-question label permutation"
        resamples = 49999
    return {"test": "Cochran Q", "statistic": statistic, "df": k - 1,
            "asymptotic_p": float(stats.chi2.sf(statistic, k - 1)),
            "permutation_p": pvalue, "permutation_calibration": calibration,
            "resamples": resamples, "n": n,
            "discordant_questions": int(np.count_nonzero((rows > 0) & (rows < k)))}


def friedman(values, seed):
    values = np.asarray(values, dtype=float)
    ranks = stats.rankdata(values, axis=1)
    if np.all(ranks == ranks[:, :1]):
        statistic, asymptotic_p = 0.0, 1.0
    else:
        result = stats.friedmanchisquare(*values.T)
        statistic, asymptotic_p = float(result.statistic), float(result.pvalue)
    return {"test": "Friedman with within-question permutation calibration",
            "statistic": statistic, "df": 2, "asymptotic_p": asymptotic_p,
            "permutation_p": blocked_permutation_pvalue(ranks, seed=seed),
            "permutation_calibration": "Monte Carlo within-question label permutation",
            "resamples": 49999,
            "n": len(values)}


def paired_binary(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a_only = int(np.count_nonzero((a == 1) & (b == 0)))
    b_only = int(np.count_nonzero((a == 0) & (b == 1)))
    discordant = a_only + b_only
    pvalue = 1.0 if discordant == 0 else float(
        stats.binomtest(a_only, discordant, 0.5).pvalue)
    return {"test": "Exact McNemar", "a_wins": a_only, "b_wins": b_only,
            "ties": len(a) - discordant, "p": pvalue}


def paired_sign(a, b):
    # A sign test avoids assuming symmetric numerical differences.
    differences = np.round(np.asarray(a) - np.asarray(b), 12)
    a_wins = int(np.count_nonzero(differences > 0))
    b_wins = int(np.count_nonzero(differences < 0))
    nonzero = a_wins + b_wins
    pvalue = 1.0 if nonzero == 0 else float(stats.binomtest(a_wins, nonzero, 0.5).pvalue)
    return {"test": "Exact paired sign", "a_wins": a_wins, "b_wins": b_wins,
            "ties": len(differences) - nonzero, "p": pvalue}
