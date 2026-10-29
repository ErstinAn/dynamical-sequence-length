#!/usr/bin/env python3
"""Check selected text rules. This check does not prove STE compliance."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_file(path):
    problems = []
    inside_code = False
    paragraph = []

    def check_paragraph():
        text = " ".join(paragraph)
        text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"`[^`]+`", "IDENTIFIER", text)
        sentences = re.split(r"[.!?](?:\s|$)", text)
        for sentence in sentences:
            words = re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*", sentence)
            if len(words) > 25:
                problems.append(f"Long sentence ({len(words)} words): {sentence.strip()}")
        if len([s for s in sentences if s.strip()]) > 6:
            problems.append("The paragraph contains more than six sentences.")
        if re.search(
            r"\b(?:can't|don't|doesn't|isn't|won't|ensure|utilize|leverage)\b", text, re.I
        ):
            problems.append("Check the contraction or unapproved word.")
        paragraph.clear()

    for line in path.read_text().splitlines():
        if line.lstrip().startswith("```"):
            check_paragraph()
            inside_code = not inside_code
            continue
        if inside_code:
            continue
        if not line.isascii():
            problems.append("Use ASCII characters in project prose.")
        if not line.strip() or line.startswith(("#", "|")):
            check_paragraph()
            continue
        if re.match(r"\s*(?:\d+\.|-)\s", line):
            check_paragraph()
            paragraph.append(line)
            check_paragraph()
        else:
            paragraph.append(line)
    check_paragraph()
    return problems


def main():
    files = [ROOT / "README.md", ROOT / "NOTICE.md"]
    files += list((ROOT / "docs").glob("*.md"))
    files += list((ROOT / "cluster").rglob("*.md"))
    failures = []
    for path in files:
        failures.extend(f"{path.relative_to(ROOT)}: {problem}" for problem in check_file(path))
    if failures:
        raise SystemExit("\n".join(failures))
    print("The text checks passed. Dictionary and grammar review is separate.")


if __name__ == "__main__":
    main()
