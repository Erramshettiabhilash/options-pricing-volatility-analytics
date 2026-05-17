"""Basic smoke tests for the initial project scaffold."""

from pathlib import Path

from utils.types import OptionType
from utils.validation import validate_positive


def test_required_project_folders_exist() -> None:
    """The repository should expose the main folders used in the course roadmap."""
    project_root = Path(__file__).resolve().parents[1]
    expected_folders = {
        "data",
        "models",
        "analytics",
        "simulations",
        "visualization",
        "utils",
        "notebooks",
        "results",
        "tests",
    }

    missing = [folder for folder in expected_folders if not (project_root / folder).is_dir()]

    assert missing == []


def test_option_type_values_are_stable() -> None:
    """Option type labels should stay lowercase for clean market-data joins later."""
    assert OptionType.CALL == "call"
    assert OptionType.PUT == "put"


def test_validate_positive_accepts_positive_values() -> None:
    """Positive inputs should pass validation silently."""
    validate_positive(1.0, "spot")

