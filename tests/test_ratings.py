import pytest

from jtc_forecast.ratings import (
    HOME_ADVANTAGE,
    INITIAL_RATING,
    compute_pit_ratings,
    expected_result,
    goal_diff_multiplier,
)


def match(date, home, away, hs, as_, tournament="Friendly", neutral="FALSE"):
    return {"date": date, "home_team": home, "away_team": away, "home_score": hs,
            "away_score": as_, "tournament": tournament, "neutral": neutral}


@pytest.mark.parametrize("diff, expected", [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5),
                                            (-2, 1.5), (3, 14 / 8), (5, 2.0)])
def test_goal_diff_multiplier(diff, expected):
    assert goal_diff_multiplier(diff) == pytest.approx(expected)


def test_expected_result_is_symmetric_and_centred():
    assert expected_result(0) == pytest.approx(0.5)
    for d in (50, 100, 400):
        assert expected_result(d) + expected_result(-d) == pytest.approx(1.0)
    assert expected_result(400) == pytest.approx(10 / 11)


def test_update_conserves_total_rating():
    matches = [match("2020-01-01", "A", "B", 2, 0), match("2020-02-01", "B", "C", 1, 1)]
    _, ratings = compute_pit_ratings(matches)
    assert sum(ratings.values()) == pytest.approx(3 * INITIAL_RATING)


def test_first_match_matches_hand_calculation():
    panel, ratings = compute_pit_ratings([match("2020-01-01", "A", "B", 1, 0)],
                                         {"Friendly": 20})
    expected = expected_result(HOME_ADVANTAGE)
    assert panel[0]["elo_home_pre"] == INITIAL_RATING
    assert ratings["A"] == pytest.approx(INITIAL_RATING + 20 * (1 - expected))
    assert ratings["B"] == pytest.approx(INITIAL_RATING - 20 * (1 - expected))


def test_neutral_venue_removes_home_advantage():
    _, home = compute_pit_ratings([match("2020-01-01", "A", "B", 1, 0)], {"Friendly": 20})
    _, neutral = compute_pit_ratings([match("2020-01-01", "A", "B", 1, 0, neutral="TRUE")],
                                     {"Friendly": 20})
    assert neutral["A"] > home["A"]  # a win is worth more when no bonus was expected


def test_pre_match_ratings_ignore_later_matches():
    history = [match("2020-01-01", "A", "B", 3, 0), match("2020-06-01", "A", "C", 0, 1),
               match("2021-01-01", "B", "C", 2, 2)]
    full, _ = compute_pit_ratings(history)
    truncated, _ = compute_pit_ratings(history[:2])
    for a, b in zip(truncated, full[:2], strict=True):
        assert a["elo_home_pre"] == pytest.approx(b["elo_home_pre"])
        assert a["elo_away_pre"] == pytest.approx(b["elo_away_pre"])


def test_future_fixture_records_ratings_without_updating():
    history = [match("2020-01-01", "A", "B", 1, 0), match("2020-06-01", "A", "B", "NA", "NA")]
    panel, ratings = compute_pit_ratings(history)
    _, after_first = compute_pit_ratings(history[:1])
    assert panel[1]["home_score"] is None
    assert panel[1]["elo_home_pre"] == pytest.approx(after_first["A"])
    assert ratings == after_first


def test_matches_are_processed_in_date_order():
    shuffled = [match("2021-01-01", "A", "B", 0, 1), match("2020-01-01", "A", "B", 2, 0)]
    panel, _ = compute_pit_ratings(shuffled)
    assert [r["date"] for r in panel] == ["2020-01-01", "2021-01-01"]
