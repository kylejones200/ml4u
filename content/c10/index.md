---
title: "Computer Vision for Utilities"
description: "Automated inspections using object detection and image analytics."
weight: 10
draft: false
pyfile: "computervision.py"
---

## What You'll Learn

By the end of this chapter, you will understand how computer vision automates infrastructure inspections and reduces field costs. You'll learn the basics of object detection using YOLO, which stands for You Only Look Once. You'll see how to train models on utility-specific imagery such as power lines and vegetation. You'll apply geospatial mapping to locate detected issues on utility maps, and you'll recognize the data requirements, computational needs, and deployment considerations for computer vision systems.

---

## The Business Problem: Automating Inspections and Reducing Field Costs

Utilities manage enormous physical infrastructure spread across thousands of square miles, including poles, wires, substations, transformers, and solar farms. Inspecting and maintaining this infrastructure has traditionally relied on field crews conducting manual patrols or scheduled visits. These inspections are expensive, labor-intensive, and slow, making it difficult to identify problems before they escalate into failures or safety hazards.

Utilities spend millions on inspection programs that still miss critical issues because crews can't cover everything. The problem isn't lack of effort—it's lack of scale.

Aging infrastructure, increasing regulatory scrutiny, and the growing scale of distributed assets—such as rooftop solar and grid-scale renewables—make manual inspection unsustainable. Vegetation encroachment near lines can lead to faults or fires. Damaged insulators, cracked panels, or corroded connectors can go unnoticed until they fail. Utilities need scalable ways to monitor assets continuously and detect issues early.

---

## The Analytics Solution: Using Computer Vision for Asset Monitoring

Computer vision automates inspection by using machine learning models to analyze images or video captured from drones, fixed cameras, or mobile devices. Models trained on labeled examples can detect defects such as broken insulators, damaged conductors, panel cracks, or vegetation encroachment.

These systems allow utilities to monitor infrastructure without dispatching crews for every inspection cycle. High-resolution imagery can be collected via drones after storms, during routine patrols, or even continuously from fixed cameras in substations or along rights-of-way. Automated analysis identifies anomalies and prioritizes those requiring human review or immediate action.

Integration with enterprise systems makes the process even more powerful. Detected defects can trigger work orders in asset management platforms or feed into risk scoring models for predictive maintenance. This reduces the time from detection to resolution and improves overall asset reliability.

---

## Understanding Object Detection

Object detection is a computer vision task that identifies and locates objects in images. Unlike classification, which only says what an object is, detection also says where by drawing bounding boxes around detected objects.

YOLO, which stands for You Only Look Once, is a popular object detection architecture that processes entire images in a single pass for fast inference, predicts bounding boxes and class probabilities simultaneously, can detect multiple object types in one image, and works well for real-time applications.

YOLO models are trained on labeled datasets where humans draw boxes around objects and assign class labels such as power line, vegetation, or insulator. The model learns to recognize these patterns and generalize to new images.

Data requirements include training data, where you typically need hundreds to thousands of labeled images per object class, labeling, which is manual annotation that is time-consuming but essential for quality, diversity, where images should cover different lighting, weather, angles, and backgrounds, and quality, where high-resolution images improve detection accuracy.

---

## Getting Started with Drone Imagery

Drone-based inspection is becoming standard for utilities. Key considerations include flight planning, where automated flight paths ensure complete coverage, image overlap of 60-80% between images that enables stitching and 3D reconstruction, resolution, where higher resolution improves defect detection but increases storage and compute requirements, GPS metadata, where embedding GPS coordinates enables geospatial mapping of findings, and weather conditions, where clear skies improve image quality and detection accuracy.

The code assumes you have a dataset with labeled images. In practice, utilities often start with vendor-provided datasets or partner with drone service providers who handle data collection and initial labeling.

The scale challenge here is enormous. EPRI has been working on using computer vision and drones to reduce wildfire risks from electrical infrastructure. In the United States, we're talking about 200,000 miles of transmission lines and 5.5 million miles of distribution lines. Traditional helicopter inspections have safety issues—about one fatality per year from collisions with transmission lines—and they can't scale to cover everything.

EPRI's approach uses autonomous drone flights that capture inspection imagery in about two minutes per structure, getting around 48 photos per flight. The goal is to automate both data capture and interpretation. But here's the problem: building these models requires labeled datasets that didn't exist. As EPRI's team noted, if you go to ImageNet or Google's open images and search for "transformer" or "insulator" or "pole," you get nothing. These datasets haven't existed.

So EPRI labeled close to 50,000 images for transmission and distribution infrastructure and put them on Kaggle. They partnered with Labelbox to manage the annotation workflow, using about 3,500 non-experts under guidance from utility subject matter experts, plus 200 hours of management time. The imagery they're providing is the kind that's "near impossible to go and get by yourself"—specialized power line inspection imagery that requires flying drones around energized infrastructure. By making these datasets public, they're trying to accelerate innovation across the industry. That's the kind of collaboration we need more of.

---

## Operational Benefits

Automating inspections through computer vision reduces costs and speeds detection of issues that could cause outages or safety incidents. Drones can survey large areas quickly and safely, reducing the need for crews to climb poles or drive remote circuits. Early detection of vegetation encroachment allows proactive trimming, lowering fire risk and improving reliability during storms.

For renewable energy operators, computer vision can scan solar panel arrays for cracks, hotspots, or soiling, improving performance and reducing energy losses. On transmission lines, models can detect conductor sag or tower corrosion, supporting targeted maintenance before failures occur.

By scaling inspections with machine learning, utilities shift from infrequent, labor-heavy inspections to near-continuous monitoring that improves both reliability and safety.

---

## Managing False Positives

Computer vision models can produce false positives, detecting issues that aren't actually problems. This is especially challenging for utilities because false alarms waste crew time and erode trust in the system, missing real issues, or false negatives, can cause outages or hazards, and investigating false positives consumes resources.

This kills projects. A utility deployed a computer vision system that flagged everything—trees, shadows, birds—as vegetation risks. Crews got so many false alarms that they stopped trusting the system. The model was technically working, but it wasn't useful.

Strategies to manage false positives include confidence thresholds, where you only flag detections above a certain confidence score such as 0.7, human review, where you route low-confidence detections to human reviewers before creating work orders, ensemble methods, where you combine multiple models and only flag issues detected by multiple models, and feedback loops, where you track false positive rates and retrain models with corrected labels. The key is finding the right balance—too high a threshold and you miss real issues, too low and you flood crews with false alarms.

---

## Building Computer Vision Inspection Systems

Let's walk through a complete computer vision pipeline for utility inspections: training a YOLO model to detect power lines and vegetation, running inference on new images, and mapping detected issues geospatially. This workflow is representative of production systems used by utilities today.

First, we set up and train the YOLO model:

{{< pyfile file="computervision.py" from="19" to="35" >}}

The code loads a pre-trained YOLOv8 nano model (lightweight, suitable for edge deployment) and trains it on utility-specific data with labeled power lines and vegetation. Training typically takes hours to days depending on dataset size and hardware. In practice, you'd start with a pre-trained model and fine-tune it on your specific imagery—this is much faster than training from scratch.

Next, we run inference on new images:

{{< pyfile file="computervision.py" from="43" to="57" >}}

The model outputs bounding boxes and confidence scores for each detection. The visualization shows the input image with bounding boxes drawn around detected objects. Different colors indicate different classes (e.g., blue for power lines, green for vegetation). High-confidence detections (e.g., >0.8) are more reliable. I've found that confidence scores above 0.7 are usually worth investigating, but you'll need to tune this based on your false positive tolerance.

We then detect vegetation risk and map it geospatially:

{{< pyfile file="computervision.py" from="59" to="87" >}}

This analyzes detections to identify vegetation too close to power lines and converts pixel coordinates to geographic coordinates using GPS metadata from drone imagery. This is where the rubber meets the road—detections are only useful if you can tell crews where to go.

Finally, we visualize the geospatial risk points:

{{< pyfile file="computervision.py" from="89" to="103" >}}

The map plots vegetation risk points (red markers) where vegetation is too close to power lines. This helps dispatch crews to specific locations and plan vegetation management activities. Utilities use maps like this to prioritize trimming work, cutting costs by 30% while improving reliability.

The complete, runnable script is at `content/c10/computervision.py`. Note: This requires the `ultralytics` package for YOLO and `geopandas` for geospatial operations.

---

## What I Want You to Remember

Computer vision scales inspections. Drones combined with machine learning can inspect thousands of miles of infrastructure faster and more safely than manual methods. Data quality determines model quality. Accurate labeling is essential. Invest in high-quality training data—it pays off in production accuracy. Teams spend weeks labeling data, and it's worth every hour.

False positive management is critical. Computer vision models will make mistakes. Design workflows that filter low-confidence detections and enable human review. Projects fail when false positives erode trust—manage this from day one. Geospatial integration adds value. Mapping detections to locations enables targeted responses and feeds into broader asset management systems.

Start simple, scale thoughtfully. Begin with high-value use cases like vegetation encroachment before expanding to more complex defects. Build trust through accurate, actionable results. The technology is impressive, but it only works if operators trust it.

---

## What's Next

In Chapter 11, we'll explore natural language processing—using NLP to extract insights from maintenance logs, inspection reports, and regulatory documents that are typically analyzed manually. It's a different use case, but the principles are the same.
