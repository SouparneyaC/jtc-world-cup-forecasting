"""Pricing count statistics: shots on target, corners, shots, offsides.

A count with expected value lam is modelled as Poisson. Three identities do most
of the work:

    P(X >= k)          = 1 - poisson.cdf(k - 1, lam)
    X + Y              ~ Poisson(lam_x + lam_y) for independent counts
    P(X > Y)           = skellam.sf(0, lam_x, lam_y)

Expected counts come from a Poisson regression of the statistic on the Elo
difference, a home flag and a data-source indicator, which lets rows from
different feeds share one fit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson, skellam


def poisson_at_least(lam: float, k: int) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    return float(1.0 - poisson.cdf(k - 1, lam))


def poisson_sum_at_least(lams: Sequence[float], k: int) -> float:
    """P(X1 + X2 + ... >= k) for independent Poisson counts."""
    return poisson_at_least(sum(lams), k)


def skellam_more_than(lam_a: float, lam_b: float) -> float:
    """P(A > B) for independent Poisson counts A and B."""
    return float(skellam.sf(0, lam_a, lam_b))


@dataclass(frozen=True)
class CountModel:
    intercept: float
    elo_diff_coef: float
    home_coef: float
    source_coefs: dict[str, float] = field(default_factory=dict)
    baseline_source: str | None = None

    def expected(self, elo_diff: float, is_home: int = 0, data_source: str | None = None) -> float:
        """Expected count for a team with the given Elo advantage."""
        source = data_source or self.baseline_source
        eta = self.intercept + self.elo_diff_coef * elo_diff + self.home_coef * is_home
        eta += self.source_coefs.get(source, 0.0)
        return math.exp(eta)


def fit_count_model(rows: Iterable[Mapping[str, Any]], stat: str) -> CountModel:
    """Fit stat ~ elo_diff_pre + is_home + data_source by Poisson maximum likelihood.

    Each row needs the keys named by stat, plus elo_diff_pre, is_home and
    data_source. Rows with a missing value are skipped. The alphabetically first
    data source is the baseline.
    """
    usable = [r for r in rows
              if all(r.get(k) not in (None, "", "NA") for k in (stat, "elo_diff_pre", "is_home"))]
    sources = sorted({str(r.get("data_source")) for r in usable})
    baseline, others = sources[0], sources[1:]

    y = np.array([float(r[stat]) for r in usable])
    X = np.array([
        [1.0, float(r["elo_diff_pre"]), float(int(r["is_home"]))]
        + [1.0 if str(r.get("data_source")) == s else 0.0 for s in others]
        for r in usable
    ])
    params = sm.GLM(y, X, family=sm.families.Poisson()).fit().params
    return CountModel(
        intercept=float(params[0]),
        elo_diff_coef=float(params[1]),
        home_coef=float(params[2]),
        source_coefs={s: float(p) for s, p in zip(others, params[3:], strict=True)},
        baseline_source=baseline,
    )
