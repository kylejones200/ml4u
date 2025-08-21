"""
Chapter 10: Computer Vision for Utilities
Power line detection using YOLOv8 with the IEEE PLD dataset.
"""

import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def setup_yolo_model():
    """
    Initialize YOLOv8 model (nano version for faster training).
    """
    return YOLO("yolov8n.pt")

def train_model(yaml_path, epochs=20):
    """
    Train YOLOv8 on PLD dataset.
    Args:
        yaml_path (str): Path to dataset YAML file (train/val split defined).
    """
    model = setup_yolo_model()
    print("Training YOLOv8 model...")
    model.train(data=yaml_path, epochs=epochs, imgsz=640, batch=16)
    return model

def evaluate_model(model):
    """
    Validate trained YOLOv8 model.
    """
    print("Evaluating YOLOv8 model...")
    model.val()

def run_inference(model, image_path, output_path="detected_output.jpg"):
    """
    Run inference on a sample image.
    """
    results = model(image_path)
    results[0].save(filename=output_path)

    img = plt.imread(output_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Power Line Detection Result")
    plt.tight_layout()
    plt.savefig("chapter10_inference_result.png")
    plt.show()

if __name__ == "__main__":
    # Path to dataset YAML (must be prepared in YOLO format)
    dataset_yaml = "data/PLD/pld.yaml"

    # Train and evaluate model
    model = train_model(dataset_yaml, epochs=20)
    evaluate_model(model)

    # Test inference on a sample image
    sample_image = "data/PLD/images/test/sample.jpg"
    run_inference(model, sample_image)
