"""Tests for Chapter 1: Introduction to Machine Learning in Power and Utilities.

These tests import and validate the actual book code - no separate test versions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Add code directory to path
CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

# Import the actual book code
import c1_intro_to_ML as intro_to_ML


class TestChapter01:
    """Test Chapter 1 code functionality."""
    
    def test_generate_synthetic_data(self):
        """Test that data generation function works and returns expected structure."""
        df = intro_to_ML.generate_synthetic_data()
        
        # Verify it returns a DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Verify expected columns
        assert "Date" in df.columns
        assert "Temperature_C" in df.columns
        assert "Load_MW" in df.columns
        
        # Verify data types
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])
        assert pd.api.types.is_numeric_dtype(df["Temperature_C"])
        assert pd.api.types.is_numeric_dtype(df["Load_MW"])
        
        # Verify reasonable data ranges (adjust based on your config)
        assert df["Temperature_C"].min() > -50  # Reasonable temperature range
        assert df["Temperature_C"].max() < 50
        assert df["Load_MW"].min() > 0  # Load should be positive
        
        # Verify data has rows
        assert len(df) > 0
    
    def test_plot_temperature_vs_load(self, tmp_path):
        """Test that plotting function runs without errors."""
        # Generate test data
        df = intro_to_ML.generate_synthetic_data()
        
        # Temporarily modify config to save to temp directory
        original_output = intro_to_ML.config["plotting"]["output_files"]["load_plot"]
        intro_to_ML.config["plotting"]["output_files"]["load_plot"] = str(tmp_path / "test_plot.png")
        
        try:
            # Should run without errors
            intro_to_ML.plot_temperature_vs_load(df)
            
            # Verify plot file was created
            assert Path(intro_to_ML.config["plotting"]["output_files"]["load_plot"]).exists()
        finally:
            # Restore original config
            intro_to_ML.config["plotting"]["output_files"]["load_plot"] = original_output
    
    def test_train_temperature_to_load_model(self, tmp_path):
        """Test that model training function works correctly."""
        # Generate test data
        df = intro_to_ML.generate_synthetic_data()
        
        # Temporarily modify config for output
        original_output = intro_to_ML.config["plotting"]["output_files"]["regression_trend"]
        intro_to_ML.config["plotting"]["output_files"]["regression_trend"] = str(tmp_path / "test_regression.png")
        
        try:
            # Train model
            model = intro_to_ML.train_temperature_to_load_model(df)
            
            # Verify model is a LinearRegression instance
            assert isinstance(model, LinearRegression)
            
            # Verify model has been fitted (has coefficients)
            assert hasattr(model, 'coef_')
            assert hasattr(model, 'intercept_')
            assert model.coef_ is not None
            
            # Verify plot was created
            assert Path(intro_to_ML.config["plotting"]["output_files"]["regression_trend"]).exists()
        finally:
            # Restore original config
            intro_to_ML.config["plotting"]["output_files"]["regression_trend"] = original_output
    
    def test_full_workflow(self, tmp_path):
        """Test that the complete workflow runs without errors."""
        # This tests the if __name__ == "__main__" block logic
        # We'll test it by calling the functions in sequence
        
        # Temporarily modify outputs
        original_load_plot = intro_to_ML.config["plotting"]["output_files"]["load_plot"]
        original_regression = intro_to_ML.config["plotting"]["output_files"]["regression_trend"]
        intro_to_ML.config["plotting"]["output_files"]["load_plot"] = str(tmp_path / "workflow_load.png")
        intro_to_ML.config["plotting"]["output_files"]["regression_trend"] = str(tmp_path / "workflow_regression.png")
        
        try:
            # Simulate the main workflow
            df = intro_to_ML.generate_synthetic_data()
            assert len(df) > 0
            
            intro_to_ML.plot_temperature_vs_load(df)
            assert Path(intro_to_ML.config["plotting"]["output_files"]["load_plot"]).exists()
            
            model = intro_to_ML.train_temperature_to_load_model(df)
            assert isinstance(model, LinearRegression)
            assert Path(intro_to_ML.config["plotting"]["output_files"]["regression_trend"]).exists()
        finally:
            # Restore original config
            intro_to_ML.config["plotting"]["output_files"]["load_plot"] = original_load_plot
            intro_to_ML.config["plotting"]["output_files"]["regression_trend"] = original_regression
    
    def test_model_predictions_are_reasonable(self):
        """Test that model predictions are within reasonable ranges."""
        df = intro_to_ML.generate_synthetic_data()
        model = intro_to_ML.train_temperature_to_load_model(df)
        
        # Make predictions
        X = df[["Temperature_C"]].values
        predictions = model.predict(X)
        
        # Verify predictions are numeric
        assert all(np.isfinite(predictions))
        
        # Verify predictions are in reasonable range (adjust based on your data)
        # Load should be positive and within a reasonable range
        assert predictions.min() > 0
        assert predictions.max() < 10000  # Adjust based on your typical load values

