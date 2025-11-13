"""Tests for Chapter 6: Outage Prediction and Reliability Analytics.

These tests import and validate the actual book code - no separate test versions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Add chapter directory to path
CHAPTER_DIR = Path(__file__).parent.parent / "content" / "c6"
sys.path.insert(0, str(CHAPTER_DIR))

# Import the actual book code
import outage_prediction


class TestChapter06:
    """Test Chapter 6 code functionality."""
    
    def test_generate_storm_outage_data(self):
        """Test outage data generation."""
        df = outage_prediction.generate_storm_outage_data()
        
        assert isinstance(df, pd.DataFrame)
        assert "Outage" in df.columns
        assert len(df) > 0
        # Verify outage is binary
        assert set(df["Outage"].unique()).issubset({0, 1})
    
    def test_train_outage_model(self, tmp_path):
        """Test outage model training."""
        df = outage_prediction.generate_storm_outage_data()
        
        # Temporarily modify config
        original_output = outage_prediction.config["plotting"]["output_files"]["feature_importance"]
        outage_prediction.config["plotting"]["output_files"]["feature_importance"] = str(tmp_path / "test_features.png")
        
        try:
            outage_prediction.train_outage_model(df)
            assert Path(outage_prediction.config["plotting"]["output_files"]["feature_importance"]).exists()
        finally:
            outage_prediction.config["plotting"]["output_files"]["feature_importance"] = original_output

