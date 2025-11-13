# Testing Strategy for ML4U

## Philosophy: Single Source of Truth

**The code in the book IS the code we test.** We don't maintain separate "test" versions or "book" versions. The Python files in `content/c*/` are the canonical source.

## Testing Approach

### 1. Import-Based Testing
Tests import functions directly from the book's code files and validate they work correctly.

### 2. Smoke Tests
Verify that code runs without errors and produces expected outputs.

### 3. Output Validation
Check that functions return expected data types, shapes, and reasonable values.

### 4. Integration Tests
Test that the full workflow (data generation → model training → prediction) completes successfully.

## Test Structure

```
tests/
├── README.md            # This file - testing philosophy
├── conftest.py          # Shared fixtures and utilities
├── test_chapter_01.py   # Tests for Chapter 1: Introduction
├── test_chapter_02.py   # Tests for Chapter 2: Data in Power and Utilities
├── test_chapter_03.py   # Tests for Chapter 3: ML Fundamentals
├── test_chapter_04.py   # Tests for Chapter 4: Load Forecasting
├── test_chapter_05.py   # Tests for Chapter 5: Predictive Maintenance
├── test_chapter_06.py   # Tests for Chapter 6: Outage Prediction
├── test_chapter_07.py   # Tests for Chapter 7: Grid Optimization
├── test_chapter_08.py   # Tests for Chapter 8: DER Forecasting
├── test_chapter_09.py   # Tests for Chapter 9: Customer Analytics
├── test_chapter_10.py   # Tests for Chapter 10: Computer Vision
├── test_chapter_11.py   # Tests for Chapter 11: NLP
├── test_chapter_12.py   # Tests for Chapter 12: MLOps
├── test_chapter_13.py   # Tests for Chapter 13: Cybersecurity
├── test_chapter_14.py   # Tests for Chapter 14: Integrated Pipelines
├── test_chapter_15.py   # Tests for Chapter 15: AI Ethics
├── test_chapter_16.py   # Tests for Chapter 16: [Placeholder]
├── test_chapter_17.py   # Tests for Chapter 17: LLMs
├── test_chapter_18.py   # Tests for Chapter 18: Future Trends
├── test_chapter_19.py   # Tests for Chapter 19: Enterprise Integration
└── test_chapter_20.py   # Tests for Chapter 20: Platform Deployment
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run tests for a specific chapter
pytest tests/test_chapter_01.py

# Run with verbose output
pytest tests/ -v

# Run and show print statements
pytest tests/ -v -s
```

## What We Test

✅ **Code runs without errors** - No syntax errors, import errors, or runtime crashes
✅ **Functions return expected types** - DataFrames, models, predictions have correct structure
✅ **Models train successfully** - Training completes and produces valid models
✅ **Outputs are reasonable** - Predictions, metrics, plots are within expected ranges
✅ **Config files load correctly** - YAML configs are valid and accessible

## What We DON'T Test

❌ **Exact numerical values** - Synthetic data has randomness, so exact values vary
❌ **Model performance thresholds** - We verify models train, not that they achieve specific accuracy
❌ **Plot aesthetics** - We verify plots are created, not their visual appearance

## Benefits

- **No code duplication** - Book code is the only version
- **Confidence** - Know that published code actually works
- **Regression detection** - Catch breaking changes when updating code
- **Documentation** - Tests serve as usage examples

