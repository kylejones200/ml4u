"""Tests for Chapter 3: Machine Learning Fundamentals.

These tests import and validate the actual book code - no separate test versions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans

# Add chapter directory to path
CHAPTER_DIR = Path(__file__).parent.parent / "content" / "c3"
sys.path.insert(0, str(CHAPTER_DIR))

# Import the actual book code
import ML4U


class TestChapter03:
    """Test Chapter 3 code functionality."""
    
    def test_generate_regression_data(self):
        """Test regression data generation."""
        df = ML4U.generate_regression_data()
        
        assert isinstance(df, pd.DataFrame)
        assert "Temperature_C" in df.columns
        assert "Load_MW" in df.columns
        assert len(df) > 0
    
    def test_regression_example(self, tmp_path):
        """Test regression example runs successfully."""
        df = ML4U.generate_regression_data()
        
        # Temporarily modify config
        original_output = ML4U.config["plotting"]["output_files"]["regression"]
        ML4U.config["plotting"]["output_files"]["regression"] = str(tmp_path / "test_regression.png")
        
        try:
            ML4U.regression_example(df)
            assert Path(ML4U.config["plotting"]["output_files"]["regression"]).exists()
        finally:
            ML4U.config["plotting"]["output_files"]["regression"] = original_output
    
    def test_classification_example(self):
        """Test classification example runs successfully."""
        df = ML4U.generate_classification_data()
        
        assert isinstance(df, pd.DataFrame)
        assert "Failure" in df.columns or "Label" in df.columns
        ML4U.classification_example(df)  # Should run without errors
    
    def test_clustering_example(self, tmp_path):
        """Test clustering example runs successfully."""
        df = ML4U.generate_clustering_data()
        
        # Temporarily modify config if needed
        original_output = ML4U.config.get("plotting", {}).get("output_files", {}).get("clustering")
        if original_output:
            ML4U.config["plotting"]["output_files"]["clustering"] = str(tmp_path / "test_clustering.png")
        
        try:
            ML4U.clustering_example(df)
            if original_output:
                assert Path(ML4U.config["plotting"]["output_files"]["clustering"]).exists()
        finally:
            if original_output:
                ML4U.config["plotting"]["output_files"]["clustering"] = original_output

