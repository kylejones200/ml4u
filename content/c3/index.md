---
title: "Machine Learning Fundamentals"
description: "Core ML methods—regression, classification, clustering—mapped to utility use cases."
weight: 3
draft: false
pyfile: "ML4U.py"
---

## What You'll Learn

By the end of this chapter, you'll understand three fundamental ML approaches: regression, classification, and clustering. You'll see how each method maps to specific utility use cases like load forecasting, failure prediction, and customer segmentation. You'll learn to evaluate model performance using appropriate metrics such as MSE, accuracy, and classification reports. Most importantly, you'll recognize when to use supervised versus unsupervised learning, and you'll build practical models using scikit-learn on utility data.

---

## The Business Problem: Predicting and Classifying in Complex Environments

Utilities operate vast networks of physical assets that must balance supply and demand in real time. Predicting how these systems behave under different conditions is critical. Grid operators must forecast tomorrow's load to schedule generation. Maintenance planners must decide which transformers are at greatest risk of failure. Customer engagement teams need to identify which customers are likely to participate in demand response programs.

Traditionally, these tasks have relied on deterministic engineering models or static business rules. These approaches work in stable, predictable conditions but struggle in the face of variability and uncertainty. Weather changes hourly, demand fluctuates daily, and aging equipment deteriorates in nonlinear ways. The complexity of modern power systems makes it impractical to encode every rule explicitly or to manually sift through massive datasets.

Machine learning addresses this by learning patterns directly from data rather than relying solely on pre-programmed rules. Instead of hand-coding equations to model every scenario, we allow algorithms to find statistical relationships between inputs and outputs. This is particularly powerful in utilities, where data from smart meters, SCADA systems, asset registries, and customer programs contains rich but underutilized signals about system behavior and risks.

Here's what I've learned: you don't need to understand every possible scenario upfront. Let the data tell you what matters.

---

## The Analytics Solution: Core Learning Methods

Machine learning is not a single technique but a collection of methods that fall into several broad categories. This chapter focuses on foundational approaches that recur throughout utility applications.

Regression predicts continuous outcomes like future load in megawatts, transformer oil temperature, or customer energy usage. Linear regression can relate temperature and time of day to hourly demand, providing forecasts that help balance supply and demand.

Classification assigns categories—determining whether equipment is healthy or likely to fail, or whether a pattern is normal or anomalous. This underpins predictive maintenance, cybersecurity detection, and many operational workflows.

Clustering groups similar observations together without labels. Clustering smart meter profiles can reveal natural customer segments—those with high evening peaks versus flat daytime usage—informing rate design and demand response targeting.

Understanding these core methods and their differences is essential before tackling more advanced techniques. They provide a common language between data science and engineering teams and form the backbone of most practical machine learning pipelines in utilities.

---

## Connecting Methods to Real Utility Scenarios

Consider transformer failure prediction. We may have sensor data on temperature, vibration, and load, combined with asset attributes like age and manufacturer. A classification model trained on historical failure records can learn to distinguish healthy transformers from those approaching failure. By scoring current assets, it flags those at highest risk for inspection or replacement.

For load forecasting, regression models link weather variables, calendar effects, and historical demand patterns to predict consumption at different time horizons. These forecasts drive market bidding strategies and generator commitment decisions. Even basic models deliver significant operational improvements compared to heuristic forecasts.

In customer analytics, clustering can segment households based on usage profiles from AMI data. These segments inform demand response outreach, such as targeting high-peak households with incentives for load shifting. Clustering can also uncover emerging patterns, like neighborhoods adopting electric vehicles, before they show up in feeder overloading alarms.

These examples illustrate how simple machine learning concepts map directly to real problems. By framing utility questions in terms of prediction, classification, and grouping, we create clear pathways from business needs to analytic solutions.

---

## Model Evaluation Primer

Before diving into the code, it's important to understand how we assess model quality. For regression, we use Mean Squared Error and R² score. For classification, we use accuracy, precision, recall, and F1-score—especially important when classes are imbalanced, like when failures are rare.

We split data into training and testing sets to detect overfitting. For small datasets, we use k-fold cross-validation to get more reliable performance estimates.

---

## When to Use Each Method

Use regression for continuous values like load forecasting or temperature prediction. Use classification for categories like failure prediction or anomaly detection. Use clustering to find patterns without labels, like customer segmentation or asset grouping.

---

## Building ML Models for Utilities

Let's walk through fundamental ML approaches using realistic utility scenarios. Each example is self-contained and shows the complete workflow from data preparation to model evaluation. These are the foundation—everything else builds on them.

### Regression: Temperature to Load

First, we generate regression data and fit a linear model:

{{< pyfile file="ML4U.py" from="21" to="54" >}}

This demonstrates how continuous predictions work. The regression line's slope indicates how strongly temperature drives load (e.g., "each degree Celsius increases load by X MW"). Points scattered far from the line indicate prediction error, which we quantify with MSE. I'm starting with regression because it's simple, interpretable, and often works better than you'd expect.

### Classification: Equipment Failure Prediction

Next, we generate classification data and train a logistic regression model:

{{< pyfile file="ML4U.py" from="55" to="82" >}}

This shows how to handle binary classification problems. The classification report displays precision, recall, and F1-score for each class (Healthy vs. Failure). High precision means few false alarms; high recall means we catch most failures. The balance depends on operational priorities—utilities often prefer higher recall to avoid missing actual failures. I've seen teams get excited about 95% accuracy, then realize the model just predicts "healthy" for everything. That's why precision and recall matter more than accuracy for imbalanced problems.

### Clustering: Customer Segmentation

Finally, we group customers by their daily load profiles using K-means:

{{< pyfile file="ML4U.py" from="83" to="110" >}}

This demonstrates unsupervised learning. The clustering plot displays multiple daily load profiles overlaid, with distinct cluster centroids highlighted. Each cluster represents a customer segment with similar usage patterns. This visualization helps identify which segments to target for demand response programs. I've seen utilities discover EV adoption patterns this way—clusters that look different from historical norms often signal new technology adoption.

The complete, runnable script is at `content/c3/ML4U.py`. Run all three examples and see how they differ.

---

## What I Want You to Remember

Three methods cover most utility ML needs: regression for continuous predictions, classification for categorical outcomes, and clustering for pattern discovery without labels. Model evaluation is essential. Always split data into train/test sets and use appropriate metrics. Overfitting is a constant risk—models that look perfect on training data often fail in production. I've seen this happen too many times: a model that's 99% accurate on training data but useless in production.

Utility context matters. The same ML technique, such as classification, applies differently to failure prediction versus customer churn. Domain knowledge guides feature selection and interpretation. Start simple, evaluate thoroughly. Linear regression and logistic regression are interpretable and often perform well. Only move to complex models when simple ones prove insufficient. I'm bullish on starting simple—you can always add complexity later, but you can't add interpretability.

Visualization aids interpretation. Plots reveal model behavior that metrics alone miss. Always visualize predictions, residuals, and clusters to understand what your model is actually learning. I've said this before, and I'll keep saying it: always plot your data.

---

## What's Next

In Chapter 4, we'll dive deeper into time series forecasting—a critical utility application. You'll learn ARIMA models and see how to extend regression approaches to handle temporal dependencies in load forecasting. The principles are the same, but time series adds complexity.