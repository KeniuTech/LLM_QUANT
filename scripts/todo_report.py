#!/usr/bin/env python3
"""Generate a consolidated report of TODO/FIXME markers in the repository."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any

DEFAULT_PATTERNS = (
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bHACK\b",
    r"\bXXX\b",
)

EXCLUDE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".venv",
}


def iter_source_files(root: Path, extensions: Iterable[str] | None = None) -> Iterable[Path]:
    extensions = set(ext.lower() for ext in (extensions or []))
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDE_DIRS:
                # Prevent descending into excluded directories
                dirs = list(path.iterdir())
                for child in dirs:
                    if child.is_dir():
                        iter_source_files(child, extensions)
                continue
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        yield path


def scan_file(path: Path, patterns: List[re.Pattern[str]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return issues
    for idx, line in enumerate(text.splitlines(), start=1):
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                issues.append(
                    {
                        "file": str(path),
                        "line": idx,
                        "tag": match.group(0),
                        "text": line.strip(),
                    }
                )
                break
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".", help="Root directory to scan.")
    parser.add_argument(
        "--format",
        choices={"table", "json"},
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="Restrict scan to specific file extensions (e.g., --ext .py --ext .md).",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Additional regex pattern to match (case-insensitive).",
    )
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        return 1

    patterns = [re.compile(pat, re.IGNORECASE) for pat in DEFAULT_PATTERNS]
    if args.pattern:
        patterns.extend(re.compile(pat, re.IGNORECASE) for pat in args.pattern)

    issues: List[Dict[str, Any]] = []
    for file_path in iter_source_files(root, args.ext):
        issues.extend(scan_file(file_path, patterns))

    issues.sort(key=lambda item: (item["file"], item["line"]))

    if args.format == "json":
        json.dump(issues, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return 0

    # Default table output
    if not issues:
        print("No TODO/FIXME markers found.")
        return 0

    width_file = max(len(item["file"]) for item in issues)
    width_tag = max(len(item["tag"]) for item in issues)
    header = f"{'File'.ljust(width_file)}  {'Line':>5}  {'Tag'.ljust(width_tag)}  Text"
    print(header)
    print("-" * len(header))
    for item in issues:
        file_display = item["file"].ljust(width_file)
        tag_display = item["tag"].ljust(width_tag)
        print(f"{file_display}  {item['line']:>5}  {tag_display}  {item['text']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
