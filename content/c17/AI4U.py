"""
Chapter 17: Advanced AI for Utilities (LLMs and Multimodal Models)
Uses LLMs to analyze incident reports and combine insights with structured sensor data.
"""

import pandas as pd
import numpy as np
from openai import OpenAI

# --- Synthetic Data ---
def generate_incident_logs():
    logs = [
        "Breaker trip occurred at Substation A after storm. Reset required.",
        "Overheating detected on Transformer T-102. High oil temperature noted.",
        "Vegetation contact on feeder line caused temporary outage.",
        "Routine inspection completed with no anomalies detected.",
        "Cybersecurity alert: suspicious login attempts flagged at control center."
    ]
    return logs

def generate_sensor_data():
    np.random.seed(42)
    return pd.DataFrame({
        "TransformerID": ["T-101", "T-102", "T-103", "T-104", "T-105"],
        "Temp_C": np.random.normal(65, 5, 5),
        "Vibration_g": np.random.normal(0.22, 0.03, 5),
        "OilQuality_Index": np.random.uniform(60, 80, 5)
    })

# --- LLM Analysis ---
def analyze_incidents_with_llm(logs):
    """
    Summarize maintenance incident logs and extract root causes using an LLM.
    """
    client = OpenAI()
    prompt = "Analyze these incident logs and summarize root causes:\n\n" + "\n".join(logs)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an expert in utility operations."},
                  {"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

# --- Multimodal Context Fusion ---
def fuse_structured_unstructured(sensor_df, logs_summary):
    """
    Combine sensor context with LLM-driven incident log summary.
    """
    print("\n--- Sensor Data Snapshot ---")
    print(sensor_df)
    print("\n--- Incident Log Analysis ---")
    print(logs_summary)

if __name__ == "__main__":
    logs = generate_incident_logs()
    sensors = generate_sensor_data()
    summary = analyze_incidents_with_llm(logs)
    fuse_structured_unstructured(sensors, summary)
