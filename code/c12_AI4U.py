"""Chapter 12: Large Language Models and Multimodal AI for Utilities."""

import logging
import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

np.random.seed(42)


def generate_incident_logs():
    """Generate synthetic incident logs."""
    return [
        "Breaker trip occurred at Substation A after storm. Reset required.",
        "Overheating detected on Transformer T-102. High oil temperature noted.",
        "Vegetation contact on feeder line caused temporary outage.",
        "Routine inspection completed with no anomalies detected.",
        "Cybersecurity alert: suspicious login attempts flagged at control center."
    ]


def generate_sensor_data():
    """Generate synthetic sensor data."""
    return pd.DataFrame({
        "TransformerID": ["T-101", "T-102", "T-103", "T-104", "T-105"],
        "Temp_C": np.random.normal(65, 5, 5),
        "Vibration_g": np.random.normal(0.22, 0.03, 5),
        "OilQuality_Index": np.random.uniform(60, 80, 5)
    })


def analyze_incidents_with_llm(logs):
    """Summarize maintenance incident logs using an LLM."""
    if not os.environ.get("OPENAI_API_KEY"):
        issues = [
            ("breaker", any("breaker" in l.lower() for l in logs)),
            (
                "overheating",
                any(
                    "overheating" in l.lower() or "temperature" in l.lower()
                    for l in logs
                )
            ),
            ("vegetation", any("vegetation" in l.lower() for l in logs)),
            ("inspection", any("inspection" in l.lower() for l in logs)),
            (
                "cybersecurity",
                any(
                    "cyber" in l.lower() or "login" in l.lower()
                    for l in logs
                )
            ),
        ]
        present = [k for k, v in issues if v]
        summary = (
            "Incident log review indicates recurring operational themes: "
            + ", ".join(present) + ". "
            "Recommended actions: prioritize thermal checks on flagged transformers, "
            "trim vegetation on affected feeders, and validate access "
            "controls at control center systems."
        )
        return summary

    client = OpenAI()
    prompt = (
        "Analyze these incident logs and summarize root causes:\n\n" +
        "\n".join(logs)
    )

    response = client.chat.completions.create(
        model=config["llm"]["model"],
        messages=[
            {"role": "system", "content": "You are an expert in utility operations."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config["llm"]["max_tokens"]
    )
    return response.choices[0].message.content


def fuse_structured_unstructured(sensor_df, logs_summary):
    """Combine sensor context with LLM-driven incident log summary."""
    logger.debug(f"Sensor data:\n{sensor_df}")
    logger.debug(f"Logs summary:\n{logs_summary}")


if __name__ == "__main__":
    logs = generate_incident_logs()
    sensors = generate_sensor_data()
    summary = analyze_incidents_with_llm(logs)
    fuse_structured_unstructured(sensors, summary)
