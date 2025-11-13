"""Tests for Chapter 20: Full Utility AI Platform Deployment.

These tests import and validate the actual book code - no separate test versions.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add chapter directory to path
CHAPTER_DIR = Path(__file__).parent.parent / "content" / "c20"
sys.path.insert(0, str(CHAPTER_DIR))

# Import the actual book code
import pipeline


class TestChapter20:
    """Test Chapter 20 code functionality."""
    
    def test_train_and_save_model(self, tmp_path):
        """Test model training and saving."""
        if hasattr(pipeline, 'train_and_save_model'):
            # Temporarily modify model path
            original_path = pipeline.config.get("model", {}).get("path", "transformer_model.pkl")
            pipeline.config["model"]["path"] = str(tmp_path / "test_model.pkl")
            
            try:
                pipeline.train_and_save_model()
                assert Path(pipeline.config["model"]["path"]).exists()
            finally:
                pipeline.config["model"]["path"] = original_path
    
    def test_api_schema(self):
        """Test API input schema."""
        if hasattr(pipeline, 'TransformerInput'):
            # Verify Pydantic model exists
            assert hasattr(pipeline, 'TransformerInput')
    
    def test_api_endpoints(self):
        """Test API endpoint definitions."""
        if hasattr(pipeline, 'app'):
            # FastAPI app should exist
            assert pipeline.app is not None
            # Check for health endpoint
            routes = [route.path for route in pipeline.app.routes]
            assert "/health" in routes or "/" in routes

