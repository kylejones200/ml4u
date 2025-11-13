---
title: "MLOps for Utilities"
description: "From pilot to production: tracking, deployment, and monitoring of models."
weight: 14
draft: false
pyfile: "mlflow_demo.py"
---

## What You'll Learn

By the end of this chapter, you will understand the gap between ML experimentation and production deployment. You'll learn to use MLflow for experiment tracking and model versioning. You'll see how to deploy models as APIs for real-time integration with operational systems. You'll recognize the importance of model monitoring and automated retraining, and you'll build complete MLOps workflows that meet regulatory requirements for auditability.

---

## The Business Problem: Scaling Machine Learning Beyond Pilots

Utilities increasingly run pilot projects to apply machine learning to forecasting, maintenance, or outage prediction. While these pilots often show promise, few make it into production. Models remain in notebooks or are run manually by analysts. There is no automated way to retrain them, deploy them reliably, or monitor their performance over time.

Great models that never make it to production. The technical work is done, but the operational work isn't. That's where MLOps comes in.

This gap between experimentation and production limits impact. Without a clear path to operational deployment, even accurate models fail to inform real-time decision-making. Meanwhile, regulatory requirements for auditability and reliability add complexity to deploying analytics in critical infrastructure environments.

Utilities need a structured approach to operationalize machine learning, ensuring models are reproducible, monitored, and integrated with existing IT and OT systems. This is the "pilot purgatory" problem I mentioned earlier—technically sound models that never deliver value because they're not connected to operations.

---

## The Analytics Solution: Machine Learning Operations (MLOps)

MLOps brings the discipline of software operations to machine learning workflows. It provides frameworks and processes for model versioning, automated training pipelines, deployment, and monitoring.

In practice, this means using tools like MLflow to track model experiments and register the best-performing versions in a centralized repository. Deployment frameworks then package models as APIs, allowing integration with control room dashboards, SCADA feeds, or enterprise systems. Continuous monitoring tracks model drift, retraining models automatically when performance declines.

By formalizing these workflows, utilities move from ad hoc analytics to repeatable, scalable machine learning. MLOps also supports governance and compliance by recording exactly which data, code, and parameters produced each model, ensuring transparency and auditability.

---

## Understanding MLflow

MLflow is an open-source platform for managing ML lifecycle. Key components include tracking, which logs experiments, parameters, metrics, and artifacts like models and plots, models, which provide a standardized format for packaging models with dependencies, the Model Registry, which is a centralized repository for model versions with stages like Staging, Production, and Archived, and projects, which enable reproducible ML code with environment specifications.

The MLflow workflow involves training a model in an experiment and logging parameters and metrics, registering the best model to the Model Registry, promoting the model through stages from Staging to Production, deploying the model as an API or loading it for batch scoring, and monitoring performance and retraining when needed.

The code demonstrates this workflow for a transformer failure prediction model.

---

## Model Monitoring and Drift Detection

Deployed models can degrade over time due to data drift, where input data distribution changes such as new equipment types or different operating conditions, concept drift, where the relationship between inputs and outputs changes such as when failure patterns evolve, and performance degradation, where model accuracy declines on new data.

Monitoring strategies include input monitoring to track feature distributions and detect outliers, output monitoring to monitor prediction distributions and flag unusual patterns, performance monitoring to compare predictions to actuals when available and track metrics over time, and automated alerts that trigger retraining when drift exceeds thresholds.

For utilities, model drift is especially critical because grid conditions evolve with DER penetration, load growth, and aging assets. Models trained on historical data may become less accurate over time.

---

## A/B Testing for Model Deployment

Before replacing a production model, utilities often run A/B tests. The A group, or control, uses the current model. The B group, or treatment, uses the new model candidate. Traffic is split, routing some predictions to each model. Performance is compared using metrics, operational outcomes, and user feedback. Gradual rollout increases B's traffic if it performs better.

This reduces risk of deploying models that perform worse in production than in testing.

---

## Operational Benefits

Implementing MLOps accelerates the time from proof-of-concept to production, allowing utilities to realize value faster. Predictive maintenance models can be deployed as APIs that score assets continuously from incoming SCADA or IoT data. Load forecasting models can feed directly into market scheduling systems. Outage prediction models can be run automatically when storm warnings are issued, producing real-time crew staging plans.

MLOps also improves reliability. Automated retraining ensures models remain accurate as system conditions evolve, such as load growth or changes in DER penetration. Alerting and monitoring help detect anomalies, preventing models from silently degrading. This approach aligns machine learning with the rigor expected of operational tools in critical infrastructure.

Xcel Energy's journey with MLOps is a good example. When they first acquired Databricks, the platform sat unused for a year—that's a common pattern in utilities where technology gets purchased with good intentions but lacks execution strategy.

The turning point came when they partnered with Nousot to modernize their approach. They established a unified data platform strategy and started building MLOps capabilities systematically. One of their first projects was an MLOps "Kickstarter" that helped their data science team modernize model development. The previous process was slow—data scientists would create models, but getting them into production required extensive manual work. They established MLOps templates with Git and CI/CD, which let data scientists spend more time on models versus managing automation.

The impact has been real: they've completed three models with six more in progress. The MLOps processes have cut the time from model development to deployment, enabling faster iteration and more reliable production models. Tom Moore, their Technology Analytics and AI Leader, told me they went from "a lot of different siloed solutions, a lot of old tech, a lot of really difficult ways to get to your insights" to "really great innovative use cases" and "excellent progress." He said they've done a complete 180, and it's changing how they interact with the business. That's the kind of transformation that matters.

---

## Building MLOps Workflows

Let's walk through a complete MLOps workflow using MLflow: training a model, logging it with full versioning, registering it for production, and deploying it as an API. It also shows how to integrate with streaming data (Kafka) for real-time scoring—a common pattern in utility operations.

First, we generate synthetic asset data and train a model:

{{< pyfile file="mlflow_demo.py" from="28" to="78" >}}

This creates transformer health data (temperature, vibration, oil quality, age) with failure labels, trains a Random Forest classifier, and logs everything to MLflow. The logging includes model artifacts, parameters, and metrics—creating an audit trail essential for regulatory compliance. I'm bullish on MLflow because it solves the "which model version is in production?" problem that causes confusion.

Next, we load the production model from the registry:

{{< pyfile file="mlflow_demo.py" from="79" to="88" >}}

The Model Registry manages model versions and stages. Production models are clearly identified, and rollback is straightforward if new versions underperform.

Finally, we deploy the model as an API:

{{< pyfile file="mlflow_demo.py" from="89" to="106" >}}

FastAPI creates a REST endpoint that accepts transformer data and returns failure risk predictions. This enables integration with dashboards, SCADA systems, and mobile apps. The API format (JSON) is standard and easy to integrate.

The complete, runnable script with imports, configuration, Kafka streaming integration, and main execution is available at `content/c12/mlflow_demo.py` in the repository.

---

## What I Want You to Remember

MLOps bridges experimentation and production. Without MLOps, models remain in notebooks and fail to deliver operational value. MLflow provides essential capabilities. Experiment tracking, model registry, and deployment tools are the foundation of MLOps. Start here before building custom solutions.

Model versioning is critical. Production systems need clear model versions for rollback, debugging, and compliance. Model registries manage this systematically. APIs enable real-time integration. REST endpoints allow models to integrate with operational systems like dashboards, SCADA, and mobile apps without complex infrastructure.

Monitoring prevents silent degradation. Models drift over time. Automated monitoring and retraining keep models accurate as conditions evolve. Models that worked great for six months can start failing because the data distribution changed. Without monitoring, you don't know until it's too late.

Here's the bottom line: MLOps isn't optional. If you want models in production, you need versioning, monitoring, and deployment. Do it right from the start, or you'll pay for it later.

---

## What's Next

In Chapter 15, we'll explore orchestration—bringing together multiple ML use cases into integrated pipelines that show how predictive maintenance, outage prediction, and load forecasting work together to create systemwide operational intelligence.
