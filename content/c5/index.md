---
title: "Predictive Maintenance for Grid Assets"
description: "Condition-based maintenance with classification and anomaly detection for assets."
weight: 5
draft: false
pyfile: "pdm_for_grid.py"
---

## What You'll Learn

By the end of this chapter, you will understand the business case for shifting from reactive to predictive maintenance. You'll learn to use classification models to predict equipment failure risk from sensor data. You'll apply anomaly detection to identify equipment deviating from normal behavior. You'll recognize the challenge of class imbalance, where failures are rare, and learn how to address it. Finally, you'll build risk scoring systems that prioritize maintenance resources effectively.

---

## The Business Problem: The Cost of Unplanned Failures

Utility infrastructure is aging. Many transformers, breakers, and other critical components have been in service for decades, operating far beyond their original design lifespans. Routine maintenance schedules and periodic inspections have historically been the backbone of asset management. Crews replace components on fixed timelines or after visible deterioration, and major failures are often addressed reactively when they occur.

This approach is costly and inefficient. Unplanned failures disrupt service, trigger expensive emergency repairs, and can cascade into larger outages affecting thousands of customers. Each transformer failure or feeder trip carries not only repair costs but penalties for reliability metrics like SAIDI and SAIFI, as well as reputational damage. Worse, utilities often replace equipment too early, discarding assets that still have years of useful life because their age meets replacement criteria.

I learned this lesson working on drilling automation at ExxonMobil. We were trying to figure out how to automate more of the process of adding pipe sections as you drill deeper. The first step was measuring each step of the process. We broke down all the different parts that have to happen to add a piece of pipe and spin them together—they have threading on them, just like a peanut butter jar. We found we could cut about 5 seconds from every time they had to do this process. I thought, "5 seconds? What does that matter?" But they do it so many times that it ended up saving ExxonMobil millions of dollars per well. Improving quality just a little bit, when you do it so often, had a huge outcome value.

The challenge lies in predicting failures before they happen and targeting maintenance only where it is needed. Doing so requires leveraging data from sensors, inspection logs, and operational histories to assess asset health dynamically, rather than relying on static schedules. This is where predictive maintenance comes in.

---

## The Analytics Solution: From Reactive to Predictive Maintenance

Predictive maintenance uses data-driven analytics to estimate an asset’s remaining useful life and detect signs of impending failure. Rather than replacing equipment based purely on age or fixed intervals, utilities can prioritize interventions based on condition. This minimizes both premature replacements and catastrophic failures.

SCADA systems already collect equipment-level telemetry such as transformer loading, oil temperature, and breaker operations. Modern deployments add IoT sensors measuring vibration, dissolved gas analysis (DGA), or partial discharge activity, each providing a window into asset health. Combined with asset registries tracking manufacturer, installation date, and maintenance history, these data sources feed machine learning models that predict failure risk.

One common approach is classification modeling: assets are labeled as failed or healthy based on historical outcomes, and the model learns to distinguish between them. Another is anomaly detection, flagging equipment whose sensor readings deviate from normal operating patterns. Time series models can also be used to identify slow degradation trends or predict remaining useful life.

These techniques transform maintenance from a calendar-driven process into a data-driven one, aligning scarce field resources with the equipment most in need of attention.

---

## Handling Class Imbalance in Failure Prediction

A critical challenge in predictive maintenance is class imbalance: failures are rare events. In a dataset of 2000 transformer readings, you might have only 50 failures, representing 2.5% of the data. This imbalance causes problems. Models tend to predict healthy for everything to achieve high accuracy. A model that's 97% accurate sounds great until you realize it's just predicting "healthy" for everything.

Precision and recall become more important than overall accuracy. Techniques like stratified sampling, class weighting, or SMOTE help balance the dataset. The code uses stratified train/test splits to ensure both classes are represented. In production, utilities often use ensemble methods or adjust decision thresholds based on operational priorities, such as preferring false alarms over missed failures.

Here's the thing: utilities usually prefer false alarms over missed failures. A false alarm means you send a crew to check something that's fine. A missed failure means an outage. The cost trade-off is usually clear.

---

## Labeling Historical Failures

To train classification models, you need labeled data: which assets failed and when. Common approaches include work order systems that link maintenance records to asset IDs and failure dates, SCADA alarms that use equipment trip events as failure indicators, inspection reports that flag assets marked for replacement due to condition, and expert labeling where engineers review sensor trends and label pre-failure periods.

The synthetic data in this chapter simulates this process. In practice, labeling requires domain expertise to distinguish true failures from maintenance events or false alarms.

---

## Real-World Impact

The business value of predictive maintenance is tangible. Reducing unplanned outages improves reliability metrics and customer satisfaction while lowering regulatory penalties. Targeted interventions reduce maintenance costs by avoiding unnecessary replacements and emergency overtime. Predicting transformer failures before they occur prevents feeder trips and associated cascading effects.

Moreover, predictive maintenance supports capital planning. Utilities can use risk scores to defer non-critical replacements, extending asset lifespans safely and freeing up budget for more urgent needs. Insights also inform procurement, as failure patterns may reveal manufacturer-specific issues or environmental factors affecting equipment longevity.

In practice, predictive maintenance often operates alongside preventive strategies. Routine inspections remain necessary, but they are augmented by analytics that direct crews toward equipment showing early warning signs. This hybrid approach maximizes the value of existing programs while gradually modernizing asset management.

The scale challenge is real. Utilities struggle with this—you have decades-old assets with diverse systems built over different periods. Shell faced this exact problem and built something interesting: the Real-Time Data Ingestion Platform (RTDIP), which they've open-sourced. The challenge they were solving is familiar to anyone in energy: how do you proactively monitor assets at scale when they're running on legacy systems, modern systems, and everything in between?

RTDIP integrates all of Shell's legacy and current assets into a common data platform that handles both historical and real-time data. They're pulling time series data from historians, DCS systems, alarms, videos, and lab data—basically everything. The scale is impressive: over 80 energy sites globally, about 5 million sensors flowing through daily, monitoring more than 18,000 pieces of equipment. Dan Jeavons, Shell's Chief Digital Technology Adviser, told me they've integrated time series data from historians and DCS systems, added alarms and videos, and even lab data—all with very short lag times for real-time processing.

They use Delta Lake for storage and built time series functions for resampling, interpolation, and time-weighted averages. The platform makes data accessible for Python developers, BI analysts, digital twins, and data science applications. What I like about this is that they open-sourced it through the Linux Foundation for Energy, so other companies can use it. That's the kind of collaboration the industry needs.

Duke Energy shows another approach. Their Monitoring and Diagnostics center uses five analysts to monitor generating assets across seven states, covering over 87% of their generating fleet with more than 11,000 models and 500,000 data points. The center uses predictive analytics to catch problems early—they saved over $34 million in a single early catch event in 2016. The key is having analysts with 20-30 years of experience who understand what the system is trying to tell them. It's not just about the technology—it's about the people who can interpret the signals.

Dominion Energy took a different angle, focusing on pole decay prediction. They developed a predictive maintenance model that factors in pole age, wood species, soil type, weather events, and inspection history to predict when poles need inspection. The model reduced overall pole failure rates from 8% to less than 3%, and reduced manual inspections by over 60%. That's the kind of targeted impact that matters—preventing failures that could disrupt electricity supply.

Enel, the world's largest private renewable energy operator, uses predictive analytics across their global fleet. Their remote predictive diagnostic center has prevented 461 failures and avoided an estimated €47M in losses. They've also reduced emissions—410,000 tCO2e over 24 months from thermal fleet catches, equivalent to taking 95,635 gas-powered vehicles off the road for a year. The platform uses edge data capture to avoid overloading their network infrastructure, which is critical when you're managing assets across 31 countries.

---

## Setting Risk Score Thresholds

Predictive maintenance models output risk scores, such as a 0-1 probability of failure. Utilities must set thresholds to decide when to act. A high threshold, such as 0.8, only flags highest-risk assets, resulting in fewer false alarms but potentially missing some failures. A low threshold, such as 0.3, flags more assets, catching more failures but increasing false alarms and maintenance costs.

Threshold selection depends on the cost of false alarms, the cost of missed failures including outages and emergency repairs, available maintenance resources, and regulatory requirements for reliability. Utilities often use ROC curves to visualize the precision-recall trade-off and select thresholds that balance operational priorities.

---

## Building Predictive Maintenance Models

Let's walk through a complete predictive maintenance workflow using two complementary approaches: classification for failure prediction and anomaly detection for identifying unusual behavior patterns. This dual approach is common in production systems—I've seen it work well because you get both the "what's likely to fail" signal from classification and the "something's weird here" signal from anomaly detection.

First, we generate synthetic SCADA sensor data:

{{< pyfile file="pdm_for_grid.py" from="21" to="44" >}}

This creates realistic sensor readings (temperature, vibration, oil pressure, load) for transformers. The data includes correlations between sensor values and failure probability, simulating real SCADA systems. In practice, you'd pull this from your SCADA historian, but the patterns are the same.

Next, let's visualize what normal operation looks like:

{{< pyfile file="pdm_for_grid.py" from="45" to="61" >}}

Normal operation appears as relatively stable values with small fluctuations. Sudden spikes or sustained deviations indicate potential problems. This visualization helps operators spot issues before models flag them. I've seen control rooms where operators watch these plots and catch problems the models miss—that's why you want both human expertise and ML working together.

We then apply anomaly detection to identify unusual patterns:

{{< pyfile file="pdm_for_grid.py" from="62" to="88" >}}

Isolation Forest identifies equipment with unusual sensor patterns, even without explicit failure labels. This is valuable when historical failure data is limited. Not all anomalies are failures, but they warrant investigation. In my experience, about 20% of anomalies turn out to be real problems—not great odds, but better than random inspection.

Finally, we train a classification model to predict failures:

{{< pyfile file="pdm_for_grid.py" from="89" to="109" >}}

The Random Forest classifier learns which combinations of conditions indicate high failure risk. The classification report shows precision, recall, and F1-score. High precision means few false alarms; high recall means we catch most actual failures. ROC AUC scores above 0.7 are generally considered good for imbalanced problems, but I've seen production models with AUCs in the 0.85-0.90 range when you have good data and features.

The complete, runnable script is at `content/c5/pdm_for_grid.py`. Run it, experiment with different thresholds, and see how precision and recall trade off.

---

## What I Want You to Remember

Predictive maintenance reduces costs and improves reliability. By targeting maintenance where it's needed, utilities avoid both premature replacements and catastrophic failures. Two approaches complement each other. Classification models predict failure risk when historical data exists. Anomaly detection flags unusual behavior even without failure labels.

Class imbalance is a constant challenge. Failures are rare, so accuracy alone is misleading. I've seen models that are 97% accurate but useless because they just predict "healthy" for everything. Focus on precision, recall, and ROC AUC to assess model quality. Risk scores need operational context. Model outputs are probabilities, not certainties. Thresholds must balance false alarms against missed failures based on operational priorities.

Sensor data quality matters. Garbage in, garbage out. Faulty sensors produce misleading predictions. Always validate sensor readings before feeding them to models. I've seen projects where the model was technically correct, but the sensors were drifting—the model learned the drift, not the actual failure patterns.

The key insight from my drilling automation work: small improvements, when you do them many times, compound into huge value. The same principle applies here. If you can predict just 20% of failures before they happen, that's still a massive win.

---

## What's Next

In Chapter 6, we'll apply similar predictive techniques to outage prediction—using weather, vegetation, and asset data to anticipate storm-related failures and optimize crew staging. The principles are the same, but the data sources and use cases are different.