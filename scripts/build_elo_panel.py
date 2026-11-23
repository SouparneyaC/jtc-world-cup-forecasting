"""Build the point-in-time Elo panel from an international results file.

Usage:
    python scripts/build_elo_panel.py path/to/results.csv --out-dir out/

results.csv is the martj42 international results file (columns date, home_team,
away_team, home_score, away_score, tournament, neutral). The K-factor table
ships with the repo in data/tournament_kfactor_map.csv.

Writes elo_match_panel.csv (the rating each team carried into every match) and
elo_current_ratings.csv (each team's latest rating).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from jtc_forecast.ratings import compute_pit_ratings

ROOT = Path(__file__).resolve().parents[1]
PANEL_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament",
                 "neutral", "k_factor", "elo_home_pre", "elo_away_pre"]


def load_k_factors(path: Path) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["tournament"]: float(row["k_factor"]) for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="international results CSV")
    parser.add_argument("--k-factors", type=Path,
                        default=ROOT / "data" / "tournament_kfactor_map.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    args = parser.parse_args()

    with open(args.results, newline="", encoding="utf-8") as f:
        matches = list(csv.DictReader(f))
    panel, ratings = compute_pit_ratings(matches, load_k_factors(args.k_factors))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.out_dir / "elo_match_panel.csv"
    with open(panel_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PANEL_COLUMNS)
        writer.writeheader()
        for row in panel:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in PANEL_COLUMNS})

    ratings_path = args.out_dir / "elo_current_ratings.csv"
    with open(ratings_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team", "elo_rating"])
        for team, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
            writer.writerow([team, f"{rating:.4f}"])

    print(f"Wrote {panel_path} ({len(panel)} matches)")
    print(f"Wrote {ratings_path} ({len(ratings)} teams)")


if __name__ == "__main__":
    main()
