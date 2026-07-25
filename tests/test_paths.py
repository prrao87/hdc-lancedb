from pathlib import Path

from graph.paths import PROJECT_ROOT, RAW_DATA_DIR, SRC_ROOT


def test_repository_paths_are_stable_after_package_import() -> None:
    expected_project_root = Path(__file__).resolve().parents[1]
    assert PROJECT_ROOT == expected_project_root
    assert SRC_ROOT == expected_project_root / "src"
    assert RAW_DATA_DIR == PROJECT_ROOT / "data"
    assert (RAW_DATA_DIR / "nodes" / "person.csv").is_file()
