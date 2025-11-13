---
title: "AI Ethics, Regulation, and the Future of Utilities"
description: "Fairness, explainability, and governance for regulated utility AI."
weight: 18
draft: false
pyfile: "ethics.py"
---

## What You'll Learn

By the end of this chapter, you will understand why ethics and governance are critical for utility AI deployment. You'll learn to perform fairness audits to detect bias across different subgroups. You'll see how explainability techniques like SHAP help interpret model decisions. You'll recognize regulatory frameworks such as NERC and state PUCs that govern utility AI, and you'll build governance processes that ensure models are auditable and compliant.

---

## The Business Problem: Balancing Innovation with Responsibility

As utilities adopt artificial intelligence and machine learning, they face a growing need to ensure that these technologies are deployed responsibly. Decisions driven by models can influence grid investments, maintenance priorities, pricing strategies, and even customer interactions. If these models are opaque or biased, they risk undermining trust, attracting regulatory scrutiny, or producing inequitable outcomes.

For example, predictive maintenance models might inadvertently deprioritize assets in rural areas if those areas lack historical sensor data. Customer segmentation algorithms could unintentionally reinforce inequities by targeting certain neighborhoods more aggressively for programs or rate changes. In a regulated industry where transparency and fairness are paramount, such risks must be managed deliberately.

A utility deployed a customer segmentation model that inadvertently targeted certain neighborhoods more than others, not because of bias in the algorithm, but because of bias in the training data. The model learned patterns that reflected historical inequities. That's why fairness audits matter.

SDG&E is taking this a step further with their Community Impact Platform. They're using digital twin technology, AI, and advanced analytics to promote decarbonization in vulnerable communities, which are often disproportionately affected by pollution and climate change. The platform combines community data with operational data to bring community impact front and center in decision-making. They have an innovation framework—learn, prove, scale—where they test 10-15 emerging technologies every year, hoping one or two will fundamentally transform how they operate. The learn phase is six to eight weeks: does the technology solve the business problem? How does it fit within architecture and culture? The prove phase is straightforward: what's the cost and what's the value? But the value isn't just financial—it's about maximizing decarbonization efforts and increasing equity. That's how you build systems that serve everyone, not just the communities that historically had better data or more resources.

Utilities also operate under strict compliance frameworks. Regulatory agencies demand explainability and auditability for operational tools, particularly those affecting system reliability or customer billing. The integration of AI raises questions about how to verify model decisions and ensure alignment with established standards.

---

## The Analytics Solution: Building Trustworthy AI

Ethical and regulatory considerations must be built into utility analytics from the start. This involves incorporating fairness checks, explainability tools, and governance mechanisms into machine learning workflows.

Fairness audits evaluate how models perform across different subgroups. For example, a predictive maintenance model can be assessed to ensure it treats urban and rural assets consistently, regardless of differences in data availability. Explainability methods, such as SHAP values, help engineers and regulators understand why a model made a particular prediction, fostering confidence in its outputs.

Governance frameworks document the data, code, and training process behind every model version, creating a clear lineage for audits. Integrated monitoring ensures that deployed models are continually evaluated for drift or performance degradation. These practices align with regulatory demands while also building internal confidence that AI-driven tools are reliable and equitable.

---

## Understanding SHAP (SHapley Additive exPlanations)

SHAP, which stands for SHapley Additive exPlanations, is a method for explaining individual model predictions. It answers which features contributed most to a specific prediction and by how much.

How SHAP works: it calculates the contribution of each feature to a prediction, uses game theory through Shapley values to fairly allocate contributions, and provides both global explanations showing overall feature importance and local explanations for each prediction.

For utilities, engineers can see why a transformer was flagged, such as temperature contributing +0.3 to failure risk. Regulators can verify that models use appropriate factors. Operators can trust predictions when they understand the reasoning.

The code focuses on fairness, but production systems should also include SHAP or similar explainability methods.

---

## Regulatory Frameworks for Utility AI

Utilities must comply with multiple regulatory requirements. NERC, the North American Electric Reliability Corporation, sets reliability standards that may require documentation of operational tools. State PUCs, or Public Utility Commissions, govern rate-setting and service quality regulations. FERC, the Federal Energy Regulatory Commission, regulates market operations and transmission. Data privacy laws include GDPR, CCPA, and state privacy laws for customer data.

AI-specific considerations include explainability, where regulators may require explanations for decisions affecting customers or reliability, auditability, which requires complete records of model versions, training data, and decisions, fairness, ensuring models don't discriminate against protected groups, and transparency, which involves public disclosure of how AI is used in rate-setting or service delivery.

---

## Model Documentation Requirements

For regulatory compliance, utilities should document the model purpose, explaining what problem it solves, training data including sources, time periods, and preprocessing steps, model architecture including algorithm, hyperparameters, and version, performance metrics such as accuracy, precision, and recall on test data, limitations including known issues, edge cases, and assumptions, deployment details covering where it's used, how often it's retrained, and the monitoring approach, and a change log that records the history of model updates and rationale.

This documentation supports audits and demonstrates due diligence in model development and deployment.

---

## Benefits for Utilities and Stakeholders

Embedding ethics and governance in AI deployment strengthens regulatory compliance and reduces reputational risk. It also helps utilities navigate public and political scrutiny, particularly in areas like rate design, service prioritization, and resource allocation. Transparent models are easier to explain to regulators, boards, and customers, reducing friction and accelerating adoption.

Responsible AI also supports long-term operational resilience. Monitoring for fairness and drift ensures that models remain valid as conditions evolve, from changing grid architectures to new customer technologies like electric vehicles and distributed storage. By proactively addressing ethical and regulatory considerations, utilities can confidently scale analytics into core operations.

---

## Building Fairness and Explainability into Models

Let's perform fairness audits and explainability analysis on predictive maintenance models. I'm showing you how to detect bias across different subgroups (urban vs. rural assets) and understand what drives individual predictions. This isn't optional in a regulated industry.

First, we generate asset data with a sensitive attribute (region):

{{< pyfile file="ethics.py" from="20" to="42" >}}

This creates transformer data with region labels (urban vs. rural). The data includes sensor readings and failure outcomes. In practice, this would come from SCADA and asset management systems.

Next, we train a predictive model:

{{< pyfile file="ethics.py" from="43" to="61" >}}

The Random Forest classifier predicts failures from sensor data. The model doesn't explicitly use region as a feature, but region may correlate with other factors (e.g., rural assets might have different operating conditions). This is the tricky part—bias can creep in even when you don't explicitly use sensitive attributes.

Finally, we perform a fairness audit:

{{< pyfile file="ethics.py" from="62" to="78" >}}

Fairlearn calculates fairness metrics (selection rate, false negative rate, false positive rate) separately for urban and rural assets. This reveals whether the model treats both groups equitably. Higher false negatives in one group means that group gets less protection—a serious fairness issue. Models that perform well overall can fail fairness audits because they treat different groups differently.

The complete, runnable script is at `content/c15/ethics.py`. Run it and see if your models pass fairness audits.

---

## What I Want You to Remember

Ethics and governance are not optional. In regulated industries, AI must be fair, explainable, and auditable. Building these into workflows from the start is easier than retrofitting. Fairness audits reveal hidden bias. Models can discriminate even when sensitive attributes aren't explicit features. Regular audits detect and mitigate bias before it causes harm.

Explainability builds trust. Engineers, operators, and regulators need to understand model decisions. SHAP and similar methods provide interpretable explanations. Regulatory compliance requires documentation. Complete model documentation including data, code, and performance supports audits and demonstrates due diligence.

Governance enables scaling. Systematic processes for model development, deployment, and monitoring allow utilities to scale AI responsibly. Utilities get in trouble with regulators when they can't explain how their models worked—don't let that be you.

Here's the bottom line: ethics isn't a nice-to-have. In a regulated industry, it's a requirement. Build it in from the start, or you'll pay for it later.

---

## What's Next

In Chapter 19, we'll explore cost optimization and ROI measurement—learning how to justify ML investments, calculate financial returns, and build business cases that demonstrate value to stakeholders. This is where the rubber meets the road.
