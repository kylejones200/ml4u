#!/usr/bin/env python3
"""Helper script to find function/class line ranges in Python files.

Usage:
    python scripts/find_code_ranges.py content/c1/intro_to_ML.py
    python scripts/find_code_ranges.py content/c1/intro_to_ML.py --markdown
"""

import re
import sys
from pathlib import Path


def find_code_ranges(file_path):
    """Find line ranges for all functions and classes in a Python file."""
    with open(file_path) as f:
        lines = f.readlines()
    
    items = []
    for i, line in enumerate(lines, 1):
        # Match function and class definitions
        match = re.match(r'^(def|class)\s+(\w+)', line)
        if match:
            # Find the end (next top-level definition or end of file)
            start = i
            end = len(lines)
            for j in range(i + 1, len(lines)):
                # Check if this is a top-level definition (not indented)
                if lines[j].strip() and not lines[j].startswith((' ', '\t')):
                    if re.match(r'^(def|class|if __name__)', lines[j]):
                        end = j
                        break
            
            items.append({
                'type': match.group(1),
                'name': match.group(2),
                'start': start,
                'end': end,
                'lines': end - start
            })
    
    return items


def print_ranges(file_path, markdown=False):
    """Print code ranges in a useful format."""
    ranges = find_code_ranges(file_path)
    
    if markdown:
        print(f"## Code Structure: {file_path.name}\n")
        print("| Function/Class | Lines | Shortcode |")
        print("|----------------|-------|-----------|")
        for item in ranges:
            shortcode = f"`{{{{< pyfile file=\"{file_path.name}\" from=\"{item['start']}\" to=\"{item['end']}\" >}}}}`"
            print(f"| `{item['name']}` | {item['start']}-{item['end']} ({item['lines']} lines) | {shortcode} |")
    else:
        print(f"Code Structure: {file_path.name}")
        print("=" * 60)
        for item in ranges:
            print(f"{item['type'].upper()} {item['name']}:")
            print(f"  Lines {item['start']}-{item['end']} ({item['lines']} lines)")
            print(f"  Shortcode: {{{{< pyfile file=\"{file_path.name}\" from=\"{item['start']}\" to=\"{item['end']}\" >}}}}")
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/find_code_ranges.py <file.py> [--markdown]")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    markdown = "--markdown" in sys.argv
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print_ranges(file_path, markdown)

