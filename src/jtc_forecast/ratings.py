"""Point-in-time Elo ratings for national teams.

Implements the World Football Elo update rule:

    R_new = R_old + K * G * (W - We)

    K   match-importance weight, looked up by tournament name
    G   goal-difference multiplier: 1 for a margin of 0 or 1, 1.5 for 2,
        (11 + margin) / 8 for 3 or more
    W   result for the home team: 1 win, 0.5 draw, 0 loss
    We  expected result, 1 / (10 ** (-dr / 400) + 1)
    dr  rating difference, plus a home-advantage bonus unless the venue is neutral

Matches are processed in date order, so the rating a team carries into a match
depends only on earlier matches. That makes the series point-in-time by
construction, with no lookahead.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0
DEFAULT_K = 30.0

_MISSING_SCORES = ("", "NA", None)


def goal_diff_multiplier(goal_diff: int) -> float:
    """Scale an update by the winning margin."""
    margin = abs(goal_diff)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8.0


def expected_result(rating_diff: float) -> float:
    """Expected score for the side with the given rating advantage."""
    return 1.0 / (10 ** (-rating_diff / 400.0) + 1.0)


def _is_neutral(value: Any) -> bool:
    return value is True or str(value).upper() == "TRUE"


def compute_pit_ratings(
    matches: Iterable[Mapping[str, Any]],
    k_factors: Mapping[str, float] | None = None,
    *,
    initial: float = INITIAL_RATING,
    home_advantage: float = HOME_ADVANTAGE,
    default_k: float = DEFAULT_K,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run the Elo update over a match history.

    Each match is a mapping with the keys date (ISO string), home_team,
    away_team, home_score, away_score, tournament and neutral. A missing score
    (empty, "NA" or None) marks a future fixture: its pre-match ratings are
    recorded and nothing is updated.

    Returns the panel and the final ratings. The panel has one row per match
    with the ratings each team carried into it (elo_home_pre, elo_away_pre).
    """
    kmap = k_factors or {}
    ratings: dict[str, float] = {}
    panel: list[dict[str, Any]] = []

    for row in sorted(matches, key=lambda r: r["date"]):
        home, away = row["home_team"], row["away_team"]
        ratings.setdefault(home, initial)
        ratings.setdefault(away, initial)
        pre_home, pre_away = ratings[home], ratings[away]
        neutral = _is_neutral(row.get("neutral"))

        out = {
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "home_score": None,
            "away_score": None,
            "tournament": row.get("tournament"),
            "neutral": neutral,
            "k_factor": None,
            "elo_home_pre": pre_home,
            "elo_away_pre": pre_away,
        }

        home_score, away_score = row.get("home_score"), row.get("away_score")
        if home_score in _MISSING_SCORES or away_score in _MISSING_SCORES:
            panel.append(out)
            continue

        home_score, away_score = int(home_score), int(away_score)
        k = float(kmap.get(row.get("tournament"), default_k))
        if home_score > away_score:
            result = 1.0
        elif home_score == away_score:
            result = 0.5
        else:
            result = 0.0

        bonus = 0.0 if neutral else home_advantage
        expected = expected_result(pre_home - pre_away + bonus)
        delta = k * goal_diff_multiplier(home_score - away_score) * (result - expected)
        ratings[home] = pre_home + delta
        ratings[away] = pre_away - delta

        out.update(home_score=home_score, away_score=away_score, k_factor=k)
        panel.append(out)

    return panel, ratings
