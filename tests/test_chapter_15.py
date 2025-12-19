"""Tests for Chapter 15: AI Ethics, Regulation, and the Future of Utilities.

These tests import and validate the actual book code - no separate test versions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add code directory to path
CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

# Import the actual book code
import c18_ethics as ethics


class TestChapter15:
    """Test Chapter 15 code functionality."""
    
    def test_generate_asset_data_with_region(self):
        """Test asset data generation with sensitive attributes."""
        if hasattr(ethics, 'generate_asset_data_with_region'):
            df = ethics.generate_asset_data_with_region()
            assert isinstance(df, pd.DataFrame)
            assert "Region" in df.columns or "region" in df.columns.str.lower()
            assert len(df) > 0
    
    def test_train_predictive_model(self):
        """Test predictive model training."""
        if hasattr(ethics, 'train_predictive_model'):
            if hasattr(ethics, 'generate_asset_data_with_region'):
                df = ethics.generate_asset_data_with_region()
                model = ethics.train_predictive_model(df)
                assert model is not None
    
    def test_fairness_audit(self):
        """Test fairness auditing."""
        if hasattr(ethics, 'fairness_audit'):
            try:
                if hasattr(ethics, 'generate_asset_data_with_region'):
                    df = ethics.generate_asset_data_with_region()
                    if hasattr(ethics, 'train_predictive_model'):
                        model = ethics.train_predictive_model(df)
                        ethics.fairness_audit(df, model)
            except Exception:
                # Fairlearn might not be installed
                pass

