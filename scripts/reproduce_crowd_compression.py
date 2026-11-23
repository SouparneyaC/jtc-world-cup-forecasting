"""Reproduce the crowd-compression fit from the committed results table.

Usage:
    python scripts/reproduce_crowd_compression.py

Fits crowd = a + b * (mine - 0.5) on every question in
results/questions_group_to_r16.csv that has both a submitted-style estimate and
a crowd probability, then prints the slope, intercept, correlation and a worked
example.
"""

from __future__ import annotations

import csv
from pathlib import Path

from jtc_forecast.scoring import expected_crowd, fit_crowd_compression

RESULTS = Path(__file__).resolve().parents[1] / "results" / "questions_group_to_r16.csv"


def main() -> None:
    mine, crowd = [], []
    with open(RESULTS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pair = (float(row["our_estimate"]), float(row["crowd_estimate"]))
            except ValueError:
                continue
            mine.append(pair[0])
            crowd.append(pair[1])

    fit = fit_crowd_compression(mine, crowd)
    print(f"questions   {fit['n']}")
    print(f"slope       {fit['slope']:.3f}")
    print(f"intercept   {fit['intercept']:.3f}  (crowd probability when ours is 50%)")
    print(f"r           {fit['r']:.3f}  (r squared {fit['r_squared']:.3f})")
    print(f"at 90%      crowd averages {expected_crowd(0.9, fit['slope'], fit['intercept']):.3f}")


if __name__ == "__main__":
    main()
