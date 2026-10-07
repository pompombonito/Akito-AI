"""Evaluate local retrieval against a JSONL set of expected source files.

This measures source recall only; it does not establish factual correctness of
the generated answer or electrical safety of a design.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant_core import load_runtime, query_local  # noqa: E402


def main():
    cases_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evaluation" / "retrieval_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise SystemExit("No evaluation cases found.")
    embedding_model, collection = load_runtime()
    passed = 0
    for case in cases:
        records = query_local(case["query"], embedding_model, collection, limit=8)
        expected = str(case.get("expected_source_contains", "")).lower()
        filenames = [str(item["metadata"].get("filename", "")) for item in records]
        hit = bool(expected) and any(expected in filename.lower() for filename in filenames)
        passed += int(hit)
        print(f"{'PASS' if hit else 'MISS'} {case.get('id', case['query'])}: {filenames[:3]}")
    print(f"Source recall@8: {passed}/{len(cases)} ({passed / len(cases):.0%})")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
