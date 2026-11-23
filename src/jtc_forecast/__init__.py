"""jtc_forecast: models and tools used to price World Cup forecasting questions.

Modules:
    ratings   point-in-time Elo ratings
    goals     Dixon-Coles scoreline grid and the markets derived from it
    glm       Poisson regression of goals on the Elo difference
    outcome   ordered-logit model of win, draw and loss
    counts    count-stat pricing: thresholds, totals and head-to-head
    scoring   Brier score and the crowd-compression fit
"""

__version__ = "0.1.0"
