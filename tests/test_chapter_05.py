"""Tests for Chapter 5: Predictive Maintenance for Grid Assets.

These tests import and validate the actual book code - no separate test versions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# Add chapter directory to path
CHAPTER_DIR = Path(__file__).parent.parent / "content" / "c5"
sys.path.insert(0, str(CHAPTER_DIR))

# Import the actual book code
import pdm_for_grid


class TestChapter05:
    """Test Chapter 5 code functionality."""
    
    def test_generate_synthetic_scada_data(self):
        """Test that SCADA data generation works."""
        df = pdm_for_grid.generate_synthetic_scada_data()
        
        # Verify structure
        assert isinstance(df, pd.DataFrame)
        assert "Temperature_C" in df.columns
        assert "Vibration_g" in df.columns
        assert "OilPressure_psi" in df.columns
        assert "Load_kVA" in df.columns
        assert "Failure" in df.columns
        
        # Verify data types
        assert pd.api.types.is_numeric_dtype(df["Temperature_C"])
        assert pd.api.types.is_numeric_dtype(df["Failure"])
        
        # Verify failure is binary
        assert set(df["Failure"].unique()).issubset({0, 1})
        
        # Verify reasonable ranges
        assert df["Temperature_C"].min() > 0
        assert len(df) > 0
    
    def test_plot_sensor_trends(self, tmp_path):
        """Test that plotting function runs without errors."""
        df = pdm_for_grid.generate_synthetic_scada_data()
        
        # Temporarily modify config
        original_output = pdm_for_grid.config["plotting"]["output_files"]["trends"]
        pdm_for_grid.config["plotting"]["output_files"]["trends"] = str(tmp_path / "test_trends.png")
        
        try:
            pdm_for_grid.plot_sensor_trends(df)
            assert Path(pdm_for_grid.config["plotting"]["output_files"]["trends"]).exists()
        finally:
            pdm_for_grid.config["plotting"]["output_files"]["trends"] = original_output
    
    def test_anomaly_detection(self, tmp_path):
        """Test that anomaly detection runs successfully."""
        df = pdm_for_grid.generate_synthetic_scada_data()
        
        # Temporarily modify config
        original_output = pdm_for_grid.config["plotting"]["output_files"]["anomaly"]
        pdm_for_grid.config["plotting"]["output_files"]["anomaly"] = str(tmp_path / "test_anomaly.png")
        
        try:
            pdm_for_grid.anomaly_detection(df)
            assert Path(pdm_for_grid.config["plotting"]["output_files"]["anomaly"]).exists()
        finally:
            pdm_for_grid.config["plotting"]["output_files"]["anomaly"] = original_output
    
    def test_failure_prediction(self):
        """Test that failure prediction model trains successfully."""
        df = pdm_for_grid.generate_synthetic_scada_data()
        
        # This function prints output but doesn't return anything
        # We just verify it runs without errors
        pdm_for_grid.failure_prediction(df)
        
        # If we wanted to test the model, we'd need to refactor slightly
        # For now, we just verify no exceptions are raised
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow runs without errors."""
        # Temporarily modify all outputs
        original_trends = pdm_for_grid.config["plotting"]["output_files"]["trends"]
        original_anomaly = pdm_for_grid.config["plotting"]["output_files"]["anomaly"]
        pdm_for_grid.config["plotting"]["output_files"]["trends"] = str(tmp_path / "workflow_trends.png")
        pdm_for_grid.config["plotting"]["output_files"]["anomaly"] = str(tmp_path / "workflow_anomaly.png")
        
        try:
            # Simulate the main workflow
            df = pdm_for_grid.generate_synthetic_scada_data()
            assert len(df) > 0
            
            pdm_for_grid.plot_sensor_trends(df)
            pdm_for_grid.anomaly_detection(df)
            pdm_for_grid.failure_prediction(df)
        finally:
            # Restore config
            pdm_for_grid.config["plotting"]["output_files"]["trends"] = original_trends
            pdm_for_grid.config["plotting"]["output_files"]["anomaly"] = original_anomaly

