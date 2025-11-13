"""Tests for Chapter 9: Customer Analytics and Demand Response.

These tests import and validate the actual book code - no separate test versions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Add chapter directory to path
CHAPTER_DIR = Path(__file__).parent.parent / "content" / "c9"
sys.path.insert(0, str(CHAPTER_DIR))

# Import the actual book code
import demand_response


class TestChapter09:
    """Test Chapter 9 code functionality."""
    
    def test_generate_smart_meter_data(self):
        """Test smart meter data generation."""
        df = demand_response.generate_smart_meter_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Should have customer and consumption columns
        assert "Consumption_kWh" in df.columns or "consumption" in df.columns.str.lower()
    
    def test_customer_segmentation(self, tmp_path):
        """Test customer segmentation."""
        df = demand_response.generate_smart_meter_data()
        
        # Temporarily modify config
        original_output = demand_response.config.get("plotting", {}).get("output_files", {}).get("clusters")
        if original_output:
            demand_response.config["plotting"]["output_files"]["clusters"] = str(tmp_path / "test_clusters.png")
        
        try:
            if hasattr(demand_response, 'customer_segmentation'):
                demand_response.customer_segmentation(df)
                if original_output:
                    assert Path(demand_response.config["plotting"]["output_files"]["clusters"]).exists()
        finally:
            if original_output:
                demand_response.config["plotting"]["output_files"]["clusters"] = original_output

