"""
BUG-15 regression tests — misleading print statement in all four compare_models scripts.

Old behaviour: print("\nResults saved to model_performance.json") was hardcoded
in all four scripts regardless of the actual output filename.

Fix: each script now uses an f-string with the actual output_file variable.
"""
import ast
import sys
from pathlib import Path

import pytest

SCRIPTS = [
    "compare_models_top1_quali.py",
    "compare_models_top1_pre_quali.py",
    "compare_models_top3_quali.py",
    "compare_models_top3_pre_quali.py",
]

PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "src" / "predictions"

EXPECTED_FILENAMES = {
    "compare_models_top1_quali.py":     "model_performance_top1_quali.json",
    "compare_models_top1_pre_quali.py": "model_performance_top1_pre_quali.json",
    "compare_models_top3_quali.py":     "model_performance_top3_quali.json",
    "compare_models_top3_pre_quali.py": "model_performance_top3_pre_quali.json",
}


def _collect_print_strings(source: str) -> list[str]:
    """Return every string constant that appears as a direct arg to print()."""
    tree = ast.parse(source)
    strings = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    strings.append(arg.value)
    return strings


@pytest.mark.parametrize("script", SCRIPTS)
def test_no_hardcoded_model_performance_json(script):
    """The hardcoded string 'model_performance.json' must not appear in any print() call."""
    source = (PREDICTIONS_DIR / script).read_text()
    print_strings = _collect_print_strings(source)
    for s in print_strings:
        assert "model_performance.json" not in s, (
            f"{script}: print() still contains hardcoded 'model_performance.json' (BUG-15)\n"
            f"  Found: {s!r}"
        )


@pytest.mark.parametrize("script", SCRIPTS)
def test_output_file_variable_used_in_print(script):
    """
    The print statement for 'Results saved' must use an f-string referencing
    output_file, not a plain string literal.
    """
    source = (PREDICTIONS_DIR / script).read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            for arg in node.args:
                # Accept JoinedStr (f-string) that contains a reference to output_file
                if isinstance(arg, ast.JoinedStr):
                    # Walk the f-string parts for a Name node referencing output_file
                    for part in ast.walk(arg):
                        if isinstance(part, ast.Name) and part.id == "output_file":
                            return  # Found the expected f-string — test passes

    pytest.fail(
        f"{script}: print('Results saved ...') must use an f-string with output_file (BUG-15)"
    )


@pytest.mark.parametrize("script,expected_filename", EXPECTED_FILENAMES.items())
def test_correct_filename_in_source(script, expected_filename):
    """Each script must reference its own correct JSON filename as a string literal."""
    source = (PREDICTIONS_DIR / script).read_text()
    assert expected_filename in source, (
        f"{script}: expected filename '{expected_filename}' not found in source (BUG-15)"
    )
