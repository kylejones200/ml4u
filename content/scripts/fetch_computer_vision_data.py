"""
Computer Vision for Utilities (Chapter 10)
Automated power line detection using the IEEE Power Line Dataset (PLD) and YOLOv8.
"""

import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def download_pld_dataset(data_dir="data/PLD"):
    """
    Instructions to download and prepare IEEE PLD dataset.
    The dataset is available from: https://github.com/Robotics-UTN/Power-Line-Detection-Dataset

    Args:
        data_dir (str): Path to dataset folder.
    """
    print(f"Please manually download the IEEE PLD dataset and extract it into: {data_dir}")
    print("The dataset contains 'images' and 'labels' folders in YOLO format.")

def train_yolo_model(data_yaml: str, epochs: int = 50, img_size: int = 640):
    """
    Train a YOLOv8 model on the IEEE PLD dataset.
    
    Args:
        data_yaml (str): Path to YAML file describing dataset splits.
        epochs (int): Number of training epochs.
        img_size (int): Image size for YOLO model.
    Returns:
        YOLO model object.
    """
    print("Training YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # Start from YOLOv8 Nano weights
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size, batch=16)
    return model

def evaluate_model(model: YOLO):
    """
    Evaluate the trained YOLO model.
    """
    print("Evaluating YOLOv8 model...")
    results = model.val()
    print(results)

def run_inference(model: YOLO, image_path: str, output_path: str = "output.jpg"):
    """
    Run inference on a single image and save the result.
    
    Args:
        model (YOLO): Trained YOLO model.
        image_path (str): Path to test image.
        output_path (str): Output image path with detections.
    """
    print(f"Running inference on image: {image_path}")
    results = model(image_path)
    results[0].save(filename=output_path)
    print(f"Output saved to {output_path}")

    # Display result
    img = plt.imread(output_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Detected Power Lines")
    plt.show()

if __name__ == "__main__":
    # Define paths
    dataset_dir = "data/PLD"
    yaml_file = os.path.join(dataset_dir, "pld.yaml")  # This YAML defines train/val/test splits
    
    # Step 1: Download dataset manually
    download_pld_dataset(dataset_dir)
    
    # Step 2: Train YOLO model (ensure GPU available)
    if not torch.cuda.is_available():
        print("Warning: Training YOLO without GPU will be slow.")
    model = train_yolo_model(yaml_file, epochs=50)
    
    # Step 3: Evaluate model
    evaluate_model(model)
    
    # Step 4: Test inference on a sample image
    sample_image = os.path.join(dataset_dir, "images", "test", "sample.jpg")
    run_inference(model, sample_image, output_path="powerline_detection.jpg")
