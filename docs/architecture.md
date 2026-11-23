# Architecture

Every match opens a window of about fifteen questions, and every question gets a probability before the window closes. The work around one match is a short pipeline. This repository holds the models at the center of it. The fetchers, the per-match scripts and the review files stay in the private research lab (see the last section).

## From question to submitted probability

| Stage | What happens |
|---|---|
| Ingestion | Questions are pulled from the SportsPredict API and written to disk as one JSON line per match, with no in-memory buffer. Two early data-loss incidents led to that rule. |
| Classification | Each question is routed by type to a pricing tier (below). |
| Data acquisition | ESPN box scores, rosters and match timelines, and Smarkets exchange quotes, fetched for that match. |
| Pricing | The tier's method turns the data into a probability. Named situational rules then apply floors, ceilings and consistency checks. |
| Review | Every number is written to a review file and read back by hand before submission. This is the main defense against transcription errors. |
| Settlement | Results and per-question scores are recorded next to the estimate and the crowd's number. |
| Feedback | A postmortem traces costly misses to a cause in the data, a model or the process, and the fix becomes a rule or a decision record (`docs/decisions/`). |

## Pricing tiers

| Tier | Used when | Method |
|---|---|---|
| Direct market | An exchange quote answers the question (match result, goal totals, both teams to score, card and corner totals) | The vig-adjusted mid of the quote, with little or no adjustment |
| Player liquid | The player has a two-sided market | The mid, trusted near-raw |
| Team model | No market answers the question, but team-level rates can be recovered from liquid team markets | Poisson rates from the quotes, then Poisson and Skellam arithmetic for thresholds, totals and head-to-head questions |
| Model only | No usable market | The models in this package |

A liquid market takes priority over a model by default (`docs/decisions/0001-market-priority-over-domain-model.md`).

## What the package implements

| Module | Role |
|---|---|
| `ratings` | Point-in-time Elo ratings, the team-strength input to every model |
| `glm` | Poisson regression of goals on the Elo difference |
| `goals` | Dixon-Coles scoreline grid and the markets read from it |
| `outcome` | Ordered-logit model fit directly on win, draw and loss |
| `counts` | Thresholds, totals and head-to-head for shots, corners and offsides |
| `scoring` | Brier score and the crowd-compression fit |

## What is not in this repository

- The fetch scripts for the platform, ESPN and Smarkets.
- The per-match working folders and the hand-built situational rules.
- The hierarchical (partial-pooling) models for shots on target and corners, and the Monte Carlo simulation used for timing questions.
- Scraped data on other participants, which is deliberately not published.
