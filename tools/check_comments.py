"""Enforce the comment rules: a module docstring, 1-2 line function summaries, and nothing else.

Usage: python tools/check_comments.py [PATH ...]  (defaults to swarmeval, tests, examples, tools).
Rules: every file opens with an overview docstring; every module- or class-level function outside tests
has a docstring of at most two lines (dunder methods such as __init__ are exempt); no class docstrings,
stray string statements or `#` comments, except tool directives (noqa, type:, pragma:, fmt:, pyright:)
and a first-line shebang. Classes are described in their module's overview instead.
"""
from __future__ import annotations

import ast
import inspect
import io
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

DEFAULT_PATHS = ("swarmeval", "tests", "examples", "tools")
DIRECTIVES = ("noqa", "type:", "pragma:", "fmt:", "pyright:")
MAX_SUMMARY_LINES = 2
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def is_test_file(path: Path) -> bool:
    """True for files under tests/, where test names stand in for function summaries."""
    return "tests" in path.parts


def docstring_node(body: list[ast.stmt]) -> ast.Constant | None:
    """The docstring node at the start of ``body``, if any."""
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[0].value
    return None


def comment_errors(source: str) -> Iterator[tuple[int, str]]:
    """Yield ``(line, message)`` for each ``#`` comment that is not a tool directive or shebang."""
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT:
            continue
        text = tok.string.lstrip("#").strip()
        if tok.start[0] == 1 and tok.string.startswith("#!"):
            continue
        if text.startswith(DIRECTIVES):
            continue
        yield tok.start[0], f"comment not allowed: {tok.string.strip()[:60]}"


def check_function(fn: ast.FunctionDef | ast.AsyncFunctionDef, required: bool) -> Iterator[tuple[int, str]]:
    """Yield errors for a function: missing summary (when required) or one longer than two lines."""
    doc = docstring_node(fn.body)
    if doc is None:
        if required and not (fn.name.startswith("__") and fn.name.endswith("__")):
            yield fn.lineno, f"function {fn.name}() needs a 1-2 line docstring summary"
        return
    lines = inspect.cleandoc(doc.value).splitlines()
    if len(lines) > MAX_SUMMARY_LINES:
        yield doc.lineno, f"docstring of {fn.name}() is {len(lines)} lines; keep it to {MAX_SUMMARY_LINES}"


def string_statement_errors(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yield errors for string-literal statements that are not module or function docstrings."""
    allowed = set()
    module_doc = docstring_node(tree.body)
    if module_doc is not None:
        allowed.add(id(module_doc))
    for node in ast.walk(tree):
        if isinstance(node, FUNCTIONS):
            doc = docstring_node(node.body)
            if doc is not None:
                allowed.add(id(doc))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and docstring_node(node.body) is not None:
            yield node.lineno, f"class {node.name} has a docstring; describe it in the module overview instead"
            allowed.add(id(docstring_node(node.body)))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and id(node.value) not in allowed:
                yield node.lineno, "string statement used as a comment"


def check_source(source: str, path: Path) -> list[tuple[int, str]]:
    """Return all rule violations in one file's source, sorted by line."""
    tree = ast.parse(source)
    errors = list(comment_errors(source))
    if docstring_node(tree.body) is None:
        errors.append((1, "file needs an overview docstring at the top"))
    required = not is_test_file(path)
    for parent in [tree] + [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for node in parent.body:
            if isinstance(node, FUNCTIONS):
                errors.extend(check_function(node, required))
    nested = {id(n) for p in [tree] + [n for n in tree.body if isinstance(n, ast.ClassDef)]
              for n in p.body if isinstance(n, FUNCTIONS)}
    for node in ast.walk(tree):
        if isinstance(node, FUNCTIONS) and id(node) not in nested:
            errors.extend(check_function(node, required=False))
    errors.extend(string_statement_errors(tree))
    return sorted(errors)


def iter_files(paths: list[str]) -> Iterator[Path]:
    """Yield every ``.py`` file under the given files or directories."""
    for p in map(Path, paths):
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.py"))


def main(argv: list[str] | None = None) -> int:
    """Check the given paths, print violations as ``path:line: message``, and return 1 if any."""
    paths = list(argv if argv is not None else sys.argv[1:]) or list(DEFAULT_PATHS)
    count = 0
    for path in iter_files(paths):
        for line, message in check_source(path.read_text(encoding="utf-8"), path):
            print(f"{path}:{line}: {message}")
            count += 1
    if count:
        print(f"{count} comment-rule violation(s)")
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main())
