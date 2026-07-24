from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from graph.paths import PROJECT_ROOT
from hdc.retrieve import hdc_fuzzy_paths


DEFAULT_QUERY_SET = PROJECT_ROOT / "data" / "evaluation" / "semantic_queries.csv"
DEFAULT_PATH_QUERY_SET = (
    PROJECT_ROOT / "data" / "evaluation" / "semantic_path_queries.csv"
)


@dataclass(frozen=True)
class EvaluationResult:
    query: str
    expected_city: str
    ranked_cities: list[tuple[str, float]]
    expected_people: tuple[str, ...] = ()
    people_by_city: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def expected_rank(self) -> int | None:
        for rank, (city, _) in enumerate(self.ranked_cities, start=1):
            if city == self.expected_city:
                return rank
        return None

    @property
    def reciprocal_rank(self) -> float:
        rank = self.expected_rank
        return 0.0 if rank is None else 1.0 / rank

    @property
    def visitor_set_matches(self) -> bool:
        if not self.expected_people:
            return True
        return set(self.people_by_city.get(self.expected_city, ())) == set(
            self.expected_people
        )

    @property
    def passes(self) -> bool:
        return self.expected_rank == 1 and self.visitor_set_matches


def load_queries(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(
        not row.get("query") or not row.get("expected_city") for row in rows
    ):
        raise ValueError(f"Evaluation query set is empty or malformed: {path}")
    return rows


def rank_cities(paths: list[dict[str, object]]) -> list[tuple[str, float]]:
    """Collapse duplicate person paths into one score per candidate city."""
    scores: dict[str, float] = {}
    for path in paths:
        city = str(path["city"])
        score = float(path["score"])
        scores[city] = max(score, scores.get(city, -1.0))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def people_by_city(
    paths: list[dict[str, object]],
) -> dict[str, tuple[str, ...]]:
    """Collect the exact graph-expanded visitor set for every candidate city."""
    grouped: dict[str, set[str]] = {}
    for path in paths:
        grouped.setdefault(str(path["city"]), set()).add(str(path["person"]))
    return {
        city: tuple(sorted(people))
        for city, people in grouped.items()
    }


def evaluate(
    query_path: Path,
    db_uri: Path,
    ollama_host: str,
    limit: int,
    *,
    require_expected_people: bool = False,
) -> list[EvaluationResult]:
    results = []
    for row in load_queries(query_path):
        if require_expected_people and not row.get("expected_people"):
            raise ValueError(
                f"Path query has no expected people: {row['query']!r}"
            )
        paths = hdc_fuzzy_paths(
            row["query"],
            db_uri,
            min_score=None,
            limit=limit,
            ollama_host=ollama_host,
        )
        results.append(
            EvaluationResult(
                query=row["query"],
                expected_city=row["expected_city"],
                ranked_cities=rank_cities(paths),
                expected_people=tuple(
                    person.strip()
                    for person in row.get("expected_people", "").split("|")
                    if person.strip()
                ),
                people_by_city=people_by_city(paths),
            )
        )
    return results
