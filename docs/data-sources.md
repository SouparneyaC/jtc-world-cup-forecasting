# Data sources

This repository does not redistribute third-party data. It ships its own forecasts and results, plus the small K-factor table the Elo model needs. The sources below are what the research used and where to get them.

| Source | Used for | How to get it |
|---|---|---|
| International results, 1872 to date (martj42, CC BY 4.0) | Elo history | `results.csv` from the `martj42/international_results` repository on GitHub |
| ESPN public match summaries | Team box scores, rosters, goal, card and substitution timelines, referee | Unofficial endpoint `site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event=<id>` |
| Smarkets exchange | Quotes for results, totals, both teams to score, cards, corners, shots on target and player markets | Smarkets public API |
| StatsBomb open data | Event-level base rates from the 2018 and 2022 World Cups (128 matches) | `statsbomb/open-data` on GitHub, under StatsBomb's terms |
| Transfermarkt datasets | Squad value and age | `transfermarkt-datasets`, filtered to national teams |
| FIFA rankings, venue altitude, rest days | Context features | Public ranking files; altitude and rest days were compiled by hand |
| SportsPredict platform | Questions, the crowd's probabilities, outcomes | A free account on `play.sportspredict.com` |

## What the results tables contain

- `results/questions_group_to_r16.csv`: 943 question rows from 2026-06-11 to 2026-07-07, with the submitted-style estimate, the crowd's probability, the outcome and the RBP recorded by the platform.
- `results/questions_qf_sf.csv`: 87 rows for the quarterfinals and semifinals.

The crowd probabilities and outcomes come from the participant's own account on the platform. This repository is not affiliated with the platform, with Jump Trading or with any data provider, and anyone reusing the tables should check the platform's terms.

## Not published on purpose

Leaderboard pulls and per-user trade data on other participants were collected during the research. They contain usernames and identifiers of people who did not agree to be in a public dataset, so they are not part of this repository.
