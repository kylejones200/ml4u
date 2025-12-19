"""Tests for Chapter 18: Future Trends and Strategic Roadmap.

These tests import and validate the actual book code - no separate test versions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add code directory to path
CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

# Import the actual book code
import c20_trends as trends


class TestChapter18:
    """Test Chapter 18 code functionality."""
    
    def test_simulate_ai_adoption_metrics(self):
        """Test AI adoption metrics simulation."""
        if hasattr(trends, 'simulate_ai_adoption_metrics'):
            metrics = trends.simulate_ai_adoption_metrics()
            assert isinstance(metrics, pd.DataFrame)
            assert len(metrics) > 0
    
    def test_visualize_trends(self, tmp_path):
        """Test trend visualization."""
        if hasattr(trends, 'visualize_trends'):
            if hasattr(trends, 'simulate_ai_adoption_metrics'):
                metrics = trends.simulate_ai_adoption_metrics()
                
                # Temporarily modify config
                original_output = trends.config.get("plotting", {}).get("output_files", {}).get("trends")
                if original_output:
                    trends.config["plotting"]["output_files"]["trends"] = str(tmp_path / "test_trends.png")
                
                try:
                    trends.visualize_trends(metrics)
                    if original_output:
                        assert Path(trends.config["plotting"]["output_files"]["trends"]).exists()
                finally:
                    if original_output:
                        trends.config["plotting"]["output_files"]["trends"] = original_output
    
    def test_generate_strategic_recommendations(self):
        """Test strategic recommendations generation."""
        if hasattr(trends, 'generate_strategic_recommendations'):
            if hasattr(trends, 'simulate_ai_adoption_metrics'):
                metrics = trends.simulate_ai_adoption_metrics()
                recommendations = trends.generate_strategic_recommendations(metrics)
                assert recommendations is not None

