"""
Computer Vision for Solar Panel Inspection using Databricks
Defect detection pipeline with YOLOv8, MLflow model registry, and Delta Lake.
"""

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from pyspark.sql import SparkSession
from PIL import Image

# --- Spark Initialization (Databricks runtime) ---
spark = SparkSession.builder.getOrCreate()

# --- Paths ---
DATA_DIR = "/dbfs/FileStore/solar_images/"
DELTA_TABLE = "solar_inspections.delta"
MODEL_NAME = "solar-panel-defect-detector"

# --- Load Training Data ---
def load_images(image_dir):
    """
    List image files from DBFS.
    """
    return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]

# --- YOLO Training ---
def train_yolo_model(yaml_path, epochs=50):
    """
    Train YOLOv8 for defect detection and log to MLflow.
    """
    mlflow.set_experiment("/Shared/solar_inspection_experiment")
    with mlflow.start_run():
        model = YOLO("yolov8n.pt")  # nano version for speed
        model.train(data=yaml_path, epochs=epochs, imgsz=640)
        mlflow.log_param("epochs", epochs)
        mlflow.pyfunc.log_model(
            artifact_path="yolo_model",
            python_model=YOLOWrapper(model)
        )
        model_path = f"models:/{MODEL_NAME}/1"
        mlflow.register_model(model_path, MODEL_NAME)
    return model

class YOLOWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for YOLO inference.
    """
    def __init__(self, yolo_model):
        self.model = yolo_model

    def predict(self, context, model_input):
        results = []
        for img_path in model_input["image_path"]:
            result = self.model(img_path)
            results.append(result[0].boxes.xyxy.cpu().numpy())
        return results

# --- Inference and Annotation Storage ---
def run_inference(model_uri, image_paths):
    """
    Run inference on solar panel images and save annotations to Delta.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    df_input = pd.DataFrame({"image_path": image_paths})
    predictions = model.predict(df_input)

    records = []
    for img_path, preds in zip(image_paths, predictions):
        for p in preds:
            x_min, y_min, x_max, y_max, conf, cls = p
            records.append({
                "image_path": img_path,
                "x_min": float(x_min),
                "y_min": float(y_min),
                "x_max": float(x_max),
                "y_max": float(y_max),
                "confidence": float(conf),
                "class_id": int(cls)
            })

    spark.createDataFrame(pd.DataFrame(records)).write.format("delta").mode("append").saveAsTable(DELTA_TABLE)
    print(f"Saved {len(records)} detections to Delta table: {DELTA_TABLE}")

# --- Visualization ---
def visualize_detection(image_path, predictions):
    """
    Visualize YOLO predictions on solar panel image.
    """
    img = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    ax = plt.gca()
    for box in predictions:
        x_min, y_min, x_max, y_max, conf, cls = box
        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)
        ax.text(x_min, y_min-5, f"Defect {cls} ({conf:.2f})", color="red", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- Main Pipeline ---
if __name__ == "__main__":
    # Example workflow
    yaml_path = "/dbfs/FileStore/solar_dataset/solar_defect.yaml"
    model = train_yolo_model(yaml_path, epochs=30)

    image_paths = load_images(DATA_DIR)
    run_inference(f"models:/{MODEL_NAME}/1", image_paths)

    # Example visualization
    sample_image = image_paths[0]
    sample_predictions = YOLO(model.model).predict(sample_image)[0].boxes.xyxy.cpu().numpy()
    visualize_detection(sample_image, sample_predictions)
