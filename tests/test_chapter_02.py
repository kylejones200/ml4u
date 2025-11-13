"""Tests for Chapter 2: Data in Power and Utilities.

These tests import and validate the actual book code - no separate test versions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add chapter directory to path
CHAPTER_DIR = Path(__file__).parent.parent / "content" / "c2"
sys.path.insert(0, str(CHAPTER_DIR))

# Import the actual book code
import data_for_power_and_utilities


class TestChapter02:
    """Test Chapter 2 code functionality."""
    
    def test_generate_synthetic_scada_data(self):
        """Test that SCADA data generation works."""
        df = data_for_power_and_utilities.generate_synthetic_scada_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Frequency_Hz" in df.columns or "frequency" in df.columns.lower()
    
    def test_plot_consumption(self, tmp_path):
        """Test that consumption plotting works."""
        # Create sample data
        dates = pd.date_range("2022-01-01", periods=100, freq="H")
        df = pd.DataFrame({
            "timestamp": dates,
            "Consumption_kWh": np.random.normal(10, 2, 100)
        })
        df = df.set_index("timestamp")
        
        # Temporarily modify config
        original_output = data_for_power_and_utilities.config["plotting"]["output_files"]["smart_meter"]
        data_for_power_and_utilities.config["plotting"]["output_files"]["smart_meter"] = str(tmp_path / "test_consumption.png")
        
        try:
            data_for_power_and_utilities.plot_consumption(df)
            assert Path(data_for_power_and_utilities.config["plotting"]["output_files"]["smart_meter"]).exists()
        finally:
            data_for_power_and_utilities.config["plotting"]["output_files"]["smart_meter"] = original_output
    
    def test_clean_and_resample(self):
        """Test that data cleaning and resampling works."""
        dates = pd.date_range("2022-01-01", periods=100, freq="15min")
        df = pd.DataFrame({
            "timestamp": dates,
            "Consumption_kWh": np.random.normal(10, 2, 100)
        })
        
        result = data_for_power_and_utilities.clean_and_resample(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

