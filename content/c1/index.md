---
title: "Introduction to Machine Learning in Power and Utilities"
description: "Why ML matters for utilities, key pressures on the grid, and a simple weather-to-load demo to build intuition."
weight: 1
draft: false
pyfile: "intro_to_ML.py"
---

## What You'll Learn

By the end of this chapter, you'll understand why utilities are turning to machine learning—not because it's trendy, but because the old ways are breaking. You'll see how simple models like linear regression can deliver immediate value, and you'll build a temperature-to-load model that forms the foundation for everything that follows. Most importantly, you'll learn to think about ML as a tool that augments engineering judgment, not replaces it.

---

## The Three Forces Disrupting Utilities

I was in a control room in 2019 when a transformer failure cascaded into a feeder trip, leaving 15,000 customers without power. The operator's dashboard showed dozens of alarms, but no system had predicted this failure. That moment crystallized why machine learning matters for utilities: we have the data, but we're not using it to prevent problems before they happen.

Electric utilities have powered economies for over a century. Their networks of substations, transformers, transmission lines, and distribution feeders form one of the most complex engineering systems ever built. Historically, these systems ran on deterministic models, engineered tolerances, and manual oversight. Operators managed generation schedules, dispatching units to meet expected demand. Engineers maintained equipment through scheduled inspections and planned replacements. Regulatory frameworks incentivized stability and predictability, not experimentation.

But this model is under strain. Three forces are converging to disrupt the old equilibrium: aging infrastructure, changing load patterns, and decarbonization pressures.

First, aging assets. Much of the grid was built decades ago. Transformers installed in the 1970s still run in many substations. Lines sag under summer heat as thermal limits are tested. Components last long past their original design life, but failures become more frequent and harder to predict.

Second, changing load patterns. Electrification of transport and heating introduces new demand peaks. Electric vehicles can double evening residential loads in neighborhoods. Distributed energy resources, like rooftop solar, inject variable generation into the distribution grid, inverting power flows designed for one-way delivery.

Third, decarbonization and renewables. Wind and solar provide clean energy but fluctuate with weather. Their variability erodes the predictability of grid balancing and forces utilities to operate closer to their technical and economic limits.

These pressures stretch human operators and rule-based tools. A single grid operator might watch dozens of dashboards. Maintenance engineers still rely on paper logs, periodic inspections, and basic SCADA alarms. Forecasting teams use econometric models tuned to historical demand curves that no longer hold. When failures happen—like that transformer outage I saw—utilities scramble reactively. The costs ripple outward: reliability penalties, reputational damage, and customer dissatisfaction.

The problem isn't that utilities lack data. It's that the data sits in silos, and the tools we've used for decades can't handle the complexity and variability of modern grids.

---

## Why Machine Learning Matters

Machine learning addresses these cracks by shifting utilities from reactive to predictive and adaptive operations. Instead of fixed schedules and post-event diagnosis, ML can continuously analyze data streams, detect emerging risks, forecast outcomes, and suggest interventions in real time.

But here's what I've learned: ML isn't magic. It's not going to solve every problem. What it does is give you probabilistic insights that you layer on top of engineering expertise. The engineer still makes the final call—but now they have better information to work with.

---

## Data as a Strategic Asset (That We're Not Using)

Utilities generate staggering volumes of data, much of it unused. Advanced Metering Infrastructure (AMI) captures hourly or sub-hourly consumption for millions of customers. Supervisory Control and Data Acquisition (SCADA) systems record substation voltages and feeder currents every few seconds. Phasor Measurement Units (PMUs) monitor grid oscillations at 60 samples per second. Asset management systems track equipment nameplate data, inspections, and work orders. Weather feeds, vegetation encroachment maps, and market signals add external context.

The problem? Historically, this data sat in silos: AMI in customer systems, SCADA in control rooms, EAM in separate IT stacks. It's like having all the ingredients for a meal but keeping them in different kitchens. ML thrives on integrating these silos, finding correlations invisible to manual inspection.

Here's what I've seen work: Load forecasting predicts demand hours or days ahead to plan generation, scheduling, and market participation. Predictive maintenance uses vibration and temperature sensor trends to anticipate transformer failures before catastrophic faults. Outage prediction combines weather forecasts, feeder topology, and vegetation maps to pre-stage crews. DER integration forecasts solar output to manage net load and voltage excursions in distribution grids.

Even simple models—like linear regression relating temperature to load—deliver value. I'm bullish on starting simple. These methods underpin dynamic pricing, demand-side management, and operational planning. As models advance (e.g., neural networks for DER forecasting), they unlock new efficiencies and resilience. But you don't need to start there.

One thing I want to be clear about: ML augments, not replaces, engineering expertise. Utility engineers know the physics, safety limits, and regulatory constraints of their grids. ML provides probabilistic insights layered atop this expertise. For example, an anomaly detection algorithm might flag a transformer for inspection based on SCADA patterns; engineers decide whether to dispatch a crew, guided by their knowledge of asset criticality and system risk. The model says "this looks unusual." The engineer says "is it worth a crew callout?" That's the right division of labor.



---

## Bridging IT and OT: The Gap Where Pilots Die

Adoption often stumbles at the interface between information technology and operational technology. IT encompasses data, models, and analytics platforms, while OT includes grid control, SCADA, and field crews. I've seen too many utilities run pilots in IT sandboxes disconnected from OT workflows. Models sit in notebooks, producing reports but not real-time alerts. Without integration, ML insights fail to reach control rooms or dispatch centers where they matter.

This is the "pilot purgatory" problem: technically sound models that never make it to production because they're not connected to the systems that actually run the grid. The fix isn't just better technology—it's better integration. Modern platforms close this gap. Cloud-native architectures unify data storage (e.g., Delta Lake for time series), support model training (MLflow for experiment tracking), and connect downstream systems (Kafka for SCADA streaming). They enable utilities to run ML workflows on live operational data with governance and audit trails required for regulated industries.

But here's the thing: you can have the best platform in the world, and it won't help if operators don't trust the outputs. That's why starting with simple, interpretable models matters. If an operator can't understand why the model flagged a transformer, they won't act on it.

---

## A Simple Starting Point: Temperature-to-Load Modeling

To ground these concepts, we start with a basic but powerful demonstration: modeling the relationship between temperature and load. Weather is the dominant driver of electricity demand in most regions, as heating and cooling account for large fractions of consumption. Understanding this relationship is essential for short-term forecasting, where operators need accurate day-ahead and intra-day load forecasts to bid into energy markets and schedule generation. It's also critical for planning, where planners model load growth and climate impacts to size substations and feeders. And it supports demand response programs, where knowing how demand reacts to temperature helps design peak reduction strategies.

In this chapter, we'll generate synthetic temperature and load data that mirrors real-world patterns. We'll then fit a linear regression model to capture the temperature-to-load relationship. This exercise introduces time-indexed data handling common in utility analytics, feature-target relationships like how temperature drives consumption, and basic model evaluation to assess prediction quality.

While simplistic, this forms the foundation for everything that follows: ARIMA time series in Chapter 4, predictive maintenance in Chapter 5, and reinforcement learning for grid control in Chapter 7.

---

## Prerequisites

Before running the code in this chapter, ensure you have Python 3.8 or higher installed, along with the required packages: numpy, pandas, matplotlib, scikit-learn, and pyyaml. All dependencies can be installed via `pip install -r requirements.txt`.

---

## Building the Temperature-to-Load Model

Let's start with something simple: modeling the relationship between temperature and electricity load. I'm starting here because it's the foundation for everything else, and because I've seen utilities get real value from even basic models like this.

The code generates synthetic data that captures what utilities actually see: higher temperatures drive increased cooling demand, while very cold temperatures increase heating demand. It's a U-shaped relationship—demand goes up at both extremes.

Here's how we generate the data:

{{< pyfile file="intro_to_ML.py" from="24" to="59" >}}

The temperature follows a seasonal pattern (sinusoidal) with some noise, and load depends on temperature in that U-shaped way I mentioned. In practice, you'd pull this from your SCADA or AMI systems, but this synthetic data lets us focus on the modeling without getting bogged down in data quality issues.

Before we model anything, let's look at what we're working with:

{{< pyfile file="intro_to_ML.py" from="61" to="79" >}}

Always plot your data first. This scatter plot shows the relationship and helps you spot outliers or weird patterns. I've seen too many projects fail because someone skipped this step and fed garbage data into a model.

Now let's fit the model:

{{< pyfile file="intro_to_ML.py" from="81" to="127" >}}

This trains a linear regression model, makes predictions, and calculates how well it performs (R² score and mean squared error). The visualization shows how well the model captures the relationship. You'll notice it's not perfect—that's fine. Real-world models aren't perfect either.

The complete, runnable script is at `content/c1/intro_to_ML.py`. You can run it with `python intro_to_ML.py`. I've set it up so you can see the whole workflow from data generation to evaluation.

---

## What I Want You to Remember

Machine learning isn't magic—it's a tool that helps you solve real problems. The three forces I mentioned (aging infrastructure, changing load patterns, decarbonization) are creating challenges that traditional methods can't handle. That's why utilities are turning to ML.

Start simple. I'm bullish on starting with basic models like linear regression. They're interpretable, they build trust, and they often deliver real value. I've seen utilities get excited about a fancy neural network, only to realize later that a simple regression would have done the job just fine.

Data integration matters. ML works best when you combine data from multiple sources—weather, SCADA, AMI—that used to sit in silos. But here's the thing: integration is hard. It's not just a technical problem; it's an organizational one. I'll show you how to handle that in the next chapter.

Most importantly: ML augments engineering expertise, it doesn't replace it. The model says "this transformer looks unusual." The engineer decides whether to dispatch a crew. That's the right division of labor.

---

## What's Next

In Chapter 2, we'll tackle the data preparation challenges that kill most utility ML projects before they even start. You'll learn how to clean, resample, and integrate data from multiple sources—the unglamorous but essential work that makes everything else possible.