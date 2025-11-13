---
title: "Cybersecurity Analytics for Critical Infrastructure"
description: "Anomaly detection and intrusion classification tailored to OT/IT."
weight: 17
draft: false
pyfile: "cybersecurity.py"
---

## What You'll Learn

By the end of this chapter, you will understand the unique cybersecurity challenges utilities face in OT, or operational technology, environments. You'll learn to apply anomaly detection to identify unusual network patterns that may indicate intrusions. You'll see how supervised learning can classify known attack types from network traffic data. You'll recognize the critical importance of false positive management in security operations, and you'll appreciate the need for explainable models that security analysts can interpret.

---

## The Business Problem: Protecting the Grid from Evolving Threats

Utilities face increasing cybersecurity threats targeting both their IT systems and operational technology. Attacks on critical infrastructure can cause service disruptions, damage equipment, and even compromise public safety. High-profile incidents worldwide have demonstrated how vulnerable control systems and field devices can be to cyber intrusions.

Unlike traditional corporate IT environments, utilities operate industrial control systems with unique constraints. SCADA networks and field devices often run on legacy protocols with limited security features, making them attractive targets. Additionally, operational environments cannot tolerate frequent downtime, complicating the deployment of traditional IT security tools.

Utilities struggle with this. The security team wants to patch everything, but operations can't tolerate downtime. It's a constant tension.

The consequences of a successful attack are severe. Malicious actors could manipulate breaker controls, disable protective relays, or disrupt market operations. The interconnected nature of modern grids amplifies the risk, as an attack on one utility can cascade to others. Regulators have responded with standards such as NERC CIP, but compliance alone is insufficient in an era of fast-moving and sophisticated threats.

---

## The Analytics Solution: Data-Driven Intrusion Detection

Cybersecurity analytics uses machine learning to identify unusual network traffic, unauthorized access attempts, and other anomalies that may indicate an intrusion. Traditional signature-based detection systems struggle against novel attacks or insider threats that do not match known patterns.

Anomaly detection models are well suited to critical infrastructure environments. These models learn what normal operational behavior looks like, then flag deviations in real time. For example, they might detect unusual traffic patterns on SCADA networks or abnormal sequences of commands sent to field devices.

Supervised learning can also be applied where labeled attack data is available. Datasets such as CICIDS2017 provide examples of intrusion behaviors that can be used to train classification models capable of distinguishing legitimate activity from malicious actions. Combined with real-time monitoring, these models strengthen defenses against evolving threats.

---

## Accessing the CICIDS2017 Dataset

The CICIDS2017 dataset is a widely used benchmark for intrusion detection research. It contains labeled network traffic data with various attack types such as DDoS, port scan, and brute force, along with benign traffic.

To access the dataset, visit the University of New Brunswick's website or research data repositories, download the full dataset, which is large at approximately 50GB, or use pre-processed samples. The code includes a sample CSV for demonstration.

For production use, use your own network traffic data in anonymized form, label known attacks from security incident logs, work with security teams to create training datasets, and consider synthetic data generation for rare attack types.

Xcel Energy identified cybersecurity as one of their four biggest risks, alongside nuclear, gas pipeline, and wildfire risks. As part of their enterprise data strategy, they're working on projects that ingest log and user data to enable cyber insider threat detection and anomaly detection capabilities. The goal is to get ahead of any type of threats to their systems, which requires analyzing vast amounts of log data from IT and OT systems to identify unusual patterns that might indicate malicious activity.

Their approach leverages their unified data platform, which allows them to ingest, process, and analyze security logs alongside operational data. This integrated approach enables them to correlate security events with operational impacts, providing a more complete picture of potential threats. The system uses anomaly detection models to identify unusual network traffic patterns, unauthorized access attempts, and other behaviors that may indicate an intrusion. By learning what normal operational behavior looks like, the models can flag deviations in real time.

This shows how utilities can extend their data analytics capabilities to include cybersecurity, using the same platforms and tools they use for operational analytics. The unified approach enables better threat detection while maintaining the operational reliability required for critical infrastructure.

Duke Energy is using AI and satellite technology for a different kind of monitoring—methane emissions from their natural gas distribution assets. They set an ambitious net-zero methane goal for 2030, which required going beyond current EPA regulations. They built an end-to-end Azure-based cloud platform that uses satellite monitoring, analytics, and AI to quantify and prioritize methane leaks. The platform provides near real-time leak detection with pinpoint geolocation, so workers can find leaks in minutes instead of days. The solution has the potential to identify system vulnerabilities and prevent future leaks, not just detect current ones. Once scaled across all asset types and jurisdictions, it will help them achieve their net-zero methane goals. The key insight: sometimes the best way to monitor infrastructure is from above, not from the ground.

The code handles missing datasets gracefully, generating synthetic data if needed for demonstration.

---

## False Positive Management: Critical for Security Operations

False positives are a major challenge in cybersecurity analytics. Security operations centers can be overwhelmed by alerts, most of which are false alarms. This leads to alert fatigue, where analysts ignore alerts and miss real threats, resource waste from time spent investigating non-threats, and loss of trust as operators lose confidence in detection systems.

Strategies to reduce false positives include confidence thresholds, where you only alert on high-confidence detections, whitelisting to exclude known-good traffic patterns, context enrichment that combines multiple signals before alerting, feedback loops that track false positive rates and retrain models, and tiered alerting that routes low-confidence alerts to automated review and high-confidence alerts to analysts.

For utilities, false positives are especially costly because security teams are often small and operational impact is high. Models must be tuned to balance detection, meaning catching attacks, with precision, meaning avoiding false alarms.

---

## Adversarial Machine Learning Considerations

ML models used for cybersecurity can themselves be attacked. Evasion attacks occur when adversaries craft inputs that fool models, such as traffic that looks benign but is malicious. Poisoning attacks happen when training data is corrupted to make models miss specific attacks. Model extraction occurs when attackers probe models to understand their logic.

Defense strategies include adversarial training, where you include adversarial examples in training data, ensemble methods that combine multiple models to reduce vulnerability, input validation that sanitizes and validates inputs before model processing, and model monitoring that detects unusual model behavior that may indicate attacks.

---

## Explainability for Security Analysts

Security analysts need to understand why models flag certain traffic as suspicious. Black-box models like deep neural networks are difficult to interpret, making it hard to investigate alerts because analysts can't determine what triggered the alert, tune thresholds because it's unclear which features drive detections, and build trust because analysts may ignore alerts they don't understand.

Explainability techniques include feature importance, which shows which network features like packet size and frequency drive detections, SHAP values that explain individual predictions and why specific traffic was flagged, rule extraction that converts models to interpretable rules, and visualization that plots feature values for flagged traffic versus normal traffic.

The code focuses on detection, but production systems should include explainability to support analyst workflows.

---

## Operational Benefits

Integrating analytics into cybersecurity provides several benefits. It enables faster detection of threats that evade conventional tools, reducing dwell time and limiting potential damage. By focusing alerts on high-risk anomalies, it reduces false positives and eases the burden on security operations centers.

These capabilities are especially valuable for utilities adopting more digital and distributed technologies. As advanced metering, DER integration, and remote monitoring expand the attack surface, analytics offers scalable ways to manage risk without overwhelming human analysts.

---

## Building Cybersecurity Detection Models

Let's walk through two complementary approaches to cybersecurity analytics: anomaly detection (unsupervised) for finding unusual patterns without attack labels, and supervised classification for detecting known attack types. Both methods are essential in production security systems.

First, we load and preprocess network traffic data:

{{< pyfile file="cybersecurity.py" from="21" to="66" >}}

This reads the CICIDS2017 dataset (or synthetic fallback) containing network flow features and attack labels. The preprocessing handles missing values, scales features, and encodes labels. If the dataset has only one class, the code synthesizes a second class for demonstration. In practice, you'd use your own network traffic data, but the principles are the same.

Next, we apply anomaly detection:

{{< pyfile file="cybersecurity.py" from="67" to="78" >}}

Isolation Forest is an unsupervised method that learns normal behavior and flags outliers. It's useful when attack labels aren't available or for detecting novel attacks. High accuracy means the model effectively identifies unusual patterns, though not all anomalies are attacks. This can catch insider threats that signature-based systems miss.

Finally, we train a supervised detection model:

{{< pyfile file="cybersecurity.py" from="79" to="97" >}}

The Random Forest classifier learns to distinguish known attack types from benign traffic. This approach is more accurate for known attacks but requires labeled training data. The classification report shows precision, recall, and ROC AUC. Values above 0.9 indicate excellent detection. But remember: false positives are the enemy here. A model that flags everything isn't useful.

The complete, runnable script is at `content/c13/cybersecurity.py`. Run it, but remember: security models need careful tuning to balance detection with false positive rates.

---

## What I Want You to Remember

Cybersecurity analytics complements signature-based detection. ML models catch novel attacks and insider threats that traditional tools miss. Two approaches serve different purposes. Anomaly detection finds unusual patterns without labels. Supervised learning detects known attacks more accurately.

False positive management is critical. Security teams can't handle thousands of false alarms. Models must be tuned for precision, and workflows must filter low-confidence alerts. Security teams ignore alerts when there are too many false positives—that defeats the purpose. Explainability supports analysts. Security analysts need to understand why models flag traffic. Use feature importance, SHAP values, and visualization to explain detections.

Adversarial ML is a real concern. Models themselves can be attacked. Use adversarial training, input validation, and monitoring to defend against model attacks. This is advanced, but it's worth thinking about.

---

## What's Next

In Chapter 18, we'll explore AI ethics and governance—ensuring that ML models are fair, explainable, and compliant with regulatory requirements. This is essential for building trust and meeting utility industry standards.
