#!/usr/bin/env python3
"""
Line-by-line transform:
1) Replace the word 'capacity' with a hardcoded number.
2) If a line's first character is a space, remove just that one space.
Writes to a new (hardcoded) output file.
"""

import re
from pathlib import Path

# >>> EDIT THESE CONSTANTS <<<
INPUT_PATH = Path("/home/braun/Documents/Developer/cos888/instances/cfl/capc")
OUTPUT_PATH = Path("/home/braun/Documents/Developer/cos888/instances/cfl/capc4.txt")
REPLACEMENT_NUMBER = "7250"  # hardcoded replacement for the word 'capacity'

# Replace only the whole word 'capacity' (case-sensitive). Use (?i) for case-insensitive if needed.
CAPACITY_WORD = re.compile(r"\bcapacity\b")


def transform_line(line: str) -> str:
    line = CAPACITY_WORD.sub(REPLACEMENT_NUMBER, line)
    if line.startswith(" "):  # remove only the very first space if present
        line = line[1:]
    return line


def main() -> None:
    # Process large files safely (1.2MB is fine) without loading everything into memory.
    with INPUT_PATH.open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as fin, OUTPUT_PATH.open(
        "w", encoding="utf-8", errors="replace", newline=""
    ) as fout:
        for line in fin:
            fout.write(transform_line(line))


if __name__ == "__main__":
    main()
