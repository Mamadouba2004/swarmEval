"""Style gates run with the test suite: the comment rules always, and ruff when it is installed."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_comments  # noqa: E402


def test_repo_follows_comment_rules(capsys):
    paths = [str(ROOT / p) for p in check_comments.DEFAULT_PATHS]
    assert check_comments.main(paths) == 0, capsys.readouterr().out


@pytest.mark.parametrize("source, expected", [
    ('"""Overview."""\nx = 1  # explains x\n', "comment not allowed"),
    ('"""Overview."""\n\n\ndef f():\n    return 1\n', "needs a 1-2 line docstring"),
    ('"""Overview."""\n\n\ndef f():\n    """One.\n    Two.\n    Three."""\n', "keep it to 2"),
    ('"""Overview."""\n\n\nclass C:\n    """Doc."""\n', "class C has a docstring"),
    ('x = 1\n', "overview docstring"),
    ('"""Overview."""\nx = 1\n"""Stray note."""\n', "string statement"),
])
def test_checker_flags_each_rule(source, expected):
    errors = check_comments.check_source(source, Path("swarmeval/example.py"))
    assert any(expected in message for _, message in errors), errors


def test_checker_allows_directives_dunders_and_test_functions():
    source = ('"""Overview."""\nimport os  # noqa: F401\n\n\nclass C:\n    def __init__(self):\n        pass\n\n\n'
              'def test_x():\n    pass\n')
    assert check_comments.check_source(source, Path("tests/test_example.py")) == []
    assert [m for _, m in check_comments.check_source(source, Path("swarmeval/example.py"))] == [
        "function test_x() needs a 1-2 line docstring summary"]


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed (pip install -e '.[dev]')")
def test_ruff_passes():
    result = subprocess.run(["ruff", "check", str(ROOT)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
