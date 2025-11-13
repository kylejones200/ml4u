"""Shared pytest fixtures and utilities for testing book code."""

import sys
from pathlib import Path

# Add content directories to path so we can import chapter code
BOOK_ROOT = Path(__file__).parent.parent
CONTENT_DIR = BOOK_ROOT / "content"

# Add each chapter directory to sys.path for imports
for chapter_dir in sorted(CONTENT_DIR.glob("c*")):
    if chapter_dir.is_dir():
        sys.path.insert(0, str(chapter_dir))


def get_chapter_path(chapter_num: int) -> Path:
    """Get the path to a chapter directory."""
    return CONTENT_DIR / f"c{chapter_num}"


def import_chapter_module(chapter_num: int, module_name: str):
    """Import a module from a chapter directory.
    
    Args:
        chapter_num: Chapter number (e.g., 1 for c1)
        module_name: Name of Python file without .py extension
        
    Returns:
        Imported module
    """
    chapter_path = get_chapter_path(chapter_num)
    sys.path.insert(0, str(chapter_path))
    try:
        module = __import__(module_name)
        return module
    finally:
        if str(chapter_path) in sys.path:
            sys.path.remove(str(chapter_path))

