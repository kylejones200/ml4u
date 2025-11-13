| **Chapter** | **Datasets Used**  | **Purpose / Machine Learning Techniques** |
| -------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **1. Introduction to ML in Utilities** | EIA-930 (regional demand), Synthetic SCADA snapshots| Simple regression and visualization of utility data trends  |
| **2. Utility Data Foundations**  | Synthetic SCADA, AMI smart meter data, GIS feeder layouts | Data cleaning, resampling, joining telemetry and asset layers  |
| **3. ML Fundamentals for Grid Applications** | Synthetic load curves (AMI), SCADA readings| Regression (load prediction), classification (asset health labels), clustering (AMI load profiles) |
| **4. Load Forecasting and Demand Analytics** | EIA-930 demand data, NOAA weather data  | Time series forecasting (SARIMA), regression (weather vs. load)|
| **5. Predictive Maintenance for Grid Assets**| SCADA telemetry (temperature, current), Synthetic EAM records| Classification (failure prediction), anomaly detection (early degradation signals)  |
| **6. Outage Prediction and Reliability Analytics** | NOAA storm data, PUC outage reports, GIS feeder maps| Classification (storm outage risk), geospatial joins for feeder exposure|
| **7. Grid Operations Optimization** | SCADA voltage/reactive power data (synthetic IEEE feeders)| Reinforcement learning (voltage control), optimization for VAr dispatch |
| **8. Renewable Integration and DER Forecasting**| NREL NSRDB solar irradiance, PV generation profiles | Regression (PV power modeling), time series forecasting (PV output)  |
| **9. Customer Analytics and Demand Response**| AMI smart meter data, synthetic DR event logs | Clustering (customer segments), classification (participation prediction)  |
| **10. Computer Vision for Inspections**| Synthetic drone imagery of lines/substations, NDVI vegetation maps | CNN-based defect detection, segmentation (vegetation encroachment)|
| **11. NLP for Maintenance and Compliance**| Synthetic inspection logs, NERC CIP text| NLP: text classification (routine vs. failure), entity extraction (assets, failure modes) |
| **12. MLOps for Utilities**| Predictive maintenance model outputs (Chapter 5) | Model versioning, automated retraining workflows (MLflow)|
| **13. Cybersecurity Analytics**  | CICIDS2017 network traffic dataset, synthetic SCADA logs  | Anomaly detection (unsupervised), intrusion classification (supervised ML) |
| **14. Integrated Analytics Pipelines** | SCADA, outage risk models (Ch. 6), maintenance scores (Ch. 5)| Orchestration (Prefect pipelines) combining outputs for operational dashboards|
| **15. AI Ethics and Governance** | Predictive maintenance risk outputs (urban vs. rural segmentation) | Fairness audits (performance parity), explainability (SHAP analysis) |
| **16. Workflow Orchestration**| Combined datasets (load, outage, maintenance) | Automated scheduling of multi-model analytics|
| **17. Large Language Models and Multimodal AI** | Maintenance logs (NLP) + SCADA telemetry + drone imagery  | Multimodal AI combining text, structured data, and image insights |
| **18. AI Roadmap for Utilities** | Aggregate datasets from prior chapters  | Scenario modeling of AI maturity impacts (cost savings, SAIDI/SAIFI improvement) |
| **19. Enterprise Integration (SCADA/GIS/EAM)**  | SCADA telemetry, GIS asset layers, EAM maintenance records| Unified data pipelines linking IT/OT for analytics consumption |
| **20. AI Platform Deployment**| Predictive maintenance and outage prediction models | Real-time model deployment (API endpoints, streaming inference)|
