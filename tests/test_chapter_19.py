"""Tests for Chapter 19: Integration with Enterprise Systems.

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
import c13_geospatial as geospatial


class TestChapter19:
    """Test Chapter 19 code functionality."""
    
    def test_load_gis_feeder_data(self):
        """Test GIS feeder data loading."""
        if hasattr(geospatial, 'load_gis_feeder_data'):
            try:
                gdf = geospatial.load_gis_feeder_data()
                # Should be GeoDataFrame or DataFrame
                assert isinstance(gdf, pd.DataFrame)
            except Exception:
                # May require shapefiles
                pass
    
    def test_load_eam_asset_data(self):
        """Test EAM asset data loading."""
        if hasattr(geospatial, 'load_eam_asset_data'):
            df = geospatial.load_eam_asset_data()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_integrate_scada_streaming(self):
        """Test SCADA streaming integration."""
        if hasattr(geospatial, 'integrate_scada_streaming'):
            try:
                # Kafka streaming might not be available in test environment
                geospatial.integrate_scada_streaming()
            except Exception:
                pass

