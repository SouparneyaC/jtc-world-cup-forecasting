"""Ordered-logit model of match outcome.

The goal model derives win, draw and loss from a scoreline grid. This model is
fit directly on the three outcomes, ordered away win < draw < home win, so it
is calibrated against the quantity it predicts.

    eta = b_elo * elo_diff + b_home * home_adv

    P(away win) = sigmoid(c1 - eta)
    P(draw)     = sigmoid(c2 - eta) - sigmoid(c1 - eta)
    P(home win) = 1 - sigmoid(c2 - eta)

The cutpoints satisfy c1 < c2; the fit parametrises c2 = c1 + exp(delta) to keep
that order. Recency weights match the goal model.

Reference: Hvattum and Arntzen (2010), section 4.3.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from scipy.optimize import minimize

XI_DEFAULT = 0.0008


def _to_date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _is_neutral(value: Any) -> bool:
    return value is True or str(value).upper() == "TRUE"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def outcome_probs(elo_diff: float, home_adv: float, b_elo: float, b_home: float,
                  c1: float, c2: float) -> dict[str, float]:
    """P(away win), P(draw) and P(home win) for one fixture."""
    eta = b_elo * elo_diff + b_home * home_adv
    p_away = sigmoid(c1 - eta)
    p_le_draw = sigmoid(c2 - eta)
    return {"away_win": float(p_away), "draw": float(p_le_draw - p_away),
            "home_win": float(1.0 - p_le_draw)}


@dataclass(frozen=True)
class OrderedLogit:
    b_elo: float
    b_home: float
    c1: float
    c2: float
    xi_decay: float
    n_matches: int

    def probs(self, elo_home: float, elo_away: float, neutral: bool = True) -> dict[str, float]:
        """Outcome probabilities for a fixture, from the home side's view."""
        return outcome_probs(elo_home - elo_away, 0.0 if neutral else 1.0,
                             self.b_elo, self.b_home, self.c1, self.c2)


def _build_data(rows: list[Mapping[str, Any]], xi: float):
    latest = max(_to_date(r["date"]) for r in rows)
    elo_diff, home_adv, y, w = [], [], [], []
    for r in rows:
        weight = math.exp(-xi * (latest - _to_date(r["date"])).days)
        elo_diff.append(float(r["elo_home_pre"]) - float(r["elo_away_pre"]))
        home_adv.append(0.0 if _is_neutral(r.get("neutral")) else 1.0)
        home_score, away_score = int(r["home_score"]), int(r["away_score"])
        y.append(2 if home_score > away_score else 1 if home_score == away_score else 0)
        w.append(weight)
    return np.array(elo_diff), np.array(home_adv), np.array(y, dtype=int), np.array(w)


def _neg_log_likelihood(params, elo_diff, home_adv, y, w):
    b_elo, b_home, c1, delta = params
    c2 = c1 + math.exp(delta)
    eta = b_elo * elo_diff + b_home * home_adv
    p_away = sigmoid(c1 - eta)
    p_le_draw = sigmoid(c2 - eta)
    eps = 1e-12
    p_away_c = np.clip(p_away, eps, 1.0)
    p_draw_c = np.clip(p_le_draw - p_away, eps, 1.0)
    p_home_c = np.clip(1.0 - p_le_draw, eps, 1.0)
    ll = np.where(y == 0, np.log(p_away_c), np.where(y == 1, np.log(p_draw_c), np.log(p_home_c)))
    return -np.sum(w * ll)


def fit_ordered_logit(rows: Iterable[Mapping[str, Any]], xi: float = XI_DEFAULT) -> OrderedLogit:
    """Fit by weighted maximum likelihood (L-BFGS-B)."""
    played = [r for r in rows if r.get("home_score") not in ("", "NA", None)]
    elo_diff, home_adv, y, w = _build_data(played, xi)
    x0 = np.array([0.004, 0.3, -0.7, 0.0])
    result = minimize(_neg_log_likelihood, x0, args=(elo_diff, home_adv, y, w), method="L-BFGS-B")
    b_elo, b_home, c1, delta = (float(v) for v in result.x)
    return OrderedLogit(b_elo, b_home, c1, c1 + math.exp(delta), xi, len(played))
