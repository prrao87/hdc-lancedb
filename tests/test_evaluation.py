from __future__ import annotations

from evaluation.semantic import (
    EvaluationResult,
    people_by_city,
    rank_cities,
)


def test_person_paths_are_collapsed_to_city_ranking() -> None:
    ranking = rank_cities(
        [
            {"person": "Maya", "city": "Seattle", "score": 0.55},
            {"person": "Robby", "city": "Seattle", "score": 0.55},
            {"person": "Elena", "city": "Salt Lake City", "score": 0.48},
        ]
    )
    result = EvaluationResult(
        query="mountainous Pacific city",
        expected_city="Seattle",
        ranked_cities=ranking,
    )

    assert ranking == [
        ("Seattle", 0.55),
        ("Salt Lake City", 0.48),
    ]
    assert result.expected_rank == 1
    assert result.reciprocal_rank == 1.0
    assert result.passes


def test_visited_query_checks_exact_people_from_graph_expansion() -> None:
    paths = [
        {"person": "Maya", "city": "Seattle", "score": 0.41},
        {"person": "Robby", "city": "Seattle", "score": 0.41},
        {"person": "Elena", "city": "Salt Lake City", "score": 0.37},
    ]
    result = EvaluationResult(
        query="persons who visited cities on the pacific coast",
        expected_city="Seattle",
        ranked_cities=rank_cities(paths),
        expected_people=("Maya", "Robby"),
        people_by_city=people_by_city(paths),
    )

    assert result.people_by_city["Seattle"] == ("Maya", "Robby")
    assert result.visitor_set_matches
    assert result.passes
