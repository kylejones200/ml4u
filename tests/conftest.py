"""Shared pytest fixtures and utilities for testing book code."""

import sys
from pathlib import Path

# Add code directory to path so we can import chapter code
BOOK_ROOT = Path(__file__).parent.parent
CODE_DIR = BOOK_ROOT / "code"

# Add code directory to sys.path for imports
sys.path.insert(0, str(CODE_DIR))


def get_chapter_path(chapter_num: int) -> Path:
    """Get the path to the code directory."""
    return CODE_DIR


def import_chapter_module(chapter_num: int, module_name: str):
    """Import a module from the code directory.
    
    Args:
        chapter_num: Chapter number (e.g., 1 for c1)
        module_name: Name of Python file without .py extension (e.g., c1_intro_to_ML)
        
    Returns:
        Imported module
    """
    sys.path.insert(0, str(CODE_DIR))
    try:
        module = __import__(module_name)
        return module
    finally:
        if str(CODE_DIR) in sys.path:
            sys.path.remove(str(CODE_DIR))

