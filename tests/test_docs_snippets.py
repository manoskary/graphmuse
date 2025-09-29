from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DOC_PATHS: List[Path] = [
    PROJECT_ROOT / "docs" / "getting_started.rst",
    PROJECT_ROOT / "docs" / "tutorials" / "handling_data.rst",
    PROJECT_ROOT / "docs" / "tutorials" / "use_cases.rst",
]

PLACEHOLDER_REPLACEMENTS = {
    "path/to/score.musicxml": str((PROJECT_ROOT / "tests" / "samples" / "wtc1f01.musicxml").resolve()),
    "path_to_musicxml": str((PROJECT_ROOT / "tests" / "samples" / "wtc1f01.musicxml").resolve()),
}


def _extract_python_snippets(path: Path) -> Iterator[str]:
    """Yield Python code blocks from a reStructuredText document."""

    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(".. code-block:: python"):
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            snippet_lines: List[str] = []
            indent: int | None = None
            while i < len(lines):
                current = lines[i]
                if current.strip() == "":
                    snippet_lines.append("")
                    i += 1
                    continue
                current_indent = len(current) - len(current.lstrip())
                if indent is None:
                    if current_indent == 0:
                        break
                    indent = current_indent
                if current_indent >= indent:
                    snippet_lines.append(current[indent:])
                    i += 1
                else:
                    break
            if snippet_lines:
                yield "\n".join(snippet_lines).rstrip()
            continue
        i += 1


@pytest.mark.parametrize("doc_path", DOC_PATHS, ids=lambda p: p.name)
def test_documentation_snippets_execute(doc_path: Path):
    namespace: dict[str, object] = {}
    snippets = list(_extract_python_snippets(doc_path))
    assert snippets, f"No Python code snippets found in {doc_path}"

    for raw_snippet in snippets:
        snippet = raw_snippet
        for placeholder, actual in PLACEHOLDER_REPLACEMENTS.items():
            snippet = snippet.replace(placeholder, actual)

        lowered = snippet.lower()
        if "import partitura" in lowered or "from partitura" in lowered:
            pytest.importorskip("partitura")
        if "import sklearn" in lowered or "from sklearn" in lowered:
            pytest.importorskip("sklearn")

        try:
            exec(compile(snippet, f"{doc_path} snippet", "exec"), namespace)
        except ModuleNotFoundError as exc:  # pragma: no cover - optional deps missing
            pytest.skip(f"Snippet requires optional dependency {exc.name!r}")

