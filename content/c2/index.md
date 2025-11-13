---
title: "Data in Power and Utilities"
description: "Data sources, cleaning, resampling, and integration for utility analytics."
weight: 2
draft: false
pyfile: "data_for_power_and_utilities.py"
---

## What You'll Learn

By the end of this chapter, you'll understand why data preparation is where most utility ML projects die—not because the models are wrong, but because the data isn't ready. You'll master three fundamental tasks: cleaning, resampling, and integration. You'll learn to handle the data quality issues I've seen kill projects: missing values, timezone mismatches, unit inconsistencies. Most importantly, you'll see how data preparation quality directly impacts model performance. Garbage in, garbage out—it's that simple.

---

## The Business Problem: Data Without Insight

Utilities are data-rich but insight-poor. Every second, power grids generate vast amounts of telemetry from sensors, meters, and control systems. Smart meters report household consumption in 15-minute intervals. SCADA systems collect voltages and currents across substations and feeders. Phasor Measurement Units stream high-resolution synchrophasor data. Enterprise Asset Management platforms house detailed records of transformers, breakers, and other field equipment. Yet despite these torrents of data, many utilities still rely on manual processes, siloed systems, and static reports.

Utilities have all the data they need, but it's locked in silos. The root issue is fragmentation. Operational Technology (OT) systems like SCADA are often isolated from Information Technology (IT) environments that host enterprise and market data. AMI data may reside in separate customer information systems. Maintenance records might be buried in work order logs. Integrating these disparate streams is cumbersome, often requiring bespoke ETL pipelines. As a result, much of the data sits unused, limiting its value for analytics, machine learning, and decision support.

This creates tangible business problems. Maintenance crews lack predictive insights because equipment health data remains disconnected from condition monitoring sensors. Grid operators cannot fully leverage weather and demand data together to anticipate loading risks. Regulatory compliance reporting is tedious because data for audits is scattered across incompatible formats. The cost of inefficiency is high: missed opportunities to optimize investments, reduce outages, and improve customer satisfaction.

Here's what I've learned: the problem isn't that utilities lack data. It's that the data lives in different systems that don't talk to each other, and nobody has time to build the bridges.

---

## The Analytics Solution: Preparing Data for Machine Learning

Analytics begins with data readiness. To make machine learning work for utilities, data must be accessible, reliable, and modeled in ways that reflect grid realities. This chapter focuses on the mechanics of preparing utility data for analysis.

Data cleaning is where projects stall. Utility data is often noisy, containing gaps, duplicates, or faulty readings. Sensors malfunction, meters fail, and logs contain inconsistent timestamps. Cleaning requires handling missing values, removing erroneous spikes, and reconciling mismatched units or formats. Teams spend weeks just figuring out that one system reports in kW while another uses MW—simple unit mismatches that break everything downstream.

Resampling and alignment matter because utility datasets operate at different granularities. AMI data may be every 15 minutes, SCADA readings every 4 seconds, and weather data hourly. Aligning these time series to common intervals allows joint analysis. This often involves aggregation (summing sub-minute SCADA readings to hourly values) or interpolation (filling short gaps in time series). The key is choosing the right granularity for your use case—too fine and you're drowning in data, too coarse and you lose signal.

Feature integration is where meaningful analytics emerges. Weather impacts demand, asset age influences failure rates, vegetation encroachment correlates with storm outages. Joining these datasets requires careful handling of time zones, coordinate systems for geospatial joins, and equipment identifiers across systems. This is where the organizational silos really show up—different teams use different naming conventions, and you spend more time mapping IDs than building models.

Modern utility analytics platforms often use columnar storage formats like Parquet or Delta Lake for time series data. These formats provide efficient compression, schema evolution, and time travel capabilities. I'm bullish on Delta Lake for production systems—the time travel feature alone has saved me multiple times when we needed to debug what data a model actually saw.

While this chapter focuses on data preparation workflows, utilities should consider these storage formats when building production analytics platforms, which we cover in Chapter 20.

The scale challenge is real. Hydro-Québec, the world's third largest hydroelectric utility, runs an application called Octave that about 1,000 technicians and engineers use daily to analyze smart meter data from 4.5 million meters across Quebec. As adoption grew, they hit a wall—the backend needed to handle over 1 trillion data points while keeping interactive analytics responsive. They migrated to a modern data platform to handle the scale, governance, and security requirements.

Here's what I find interesting about this: Quebec is massive, about the size of Alaska, and their hydroelectric generation sites are often far from population centers, so they need very long transmission lines. Hydro-Québec actually pioneered 735 kV AC transmission lines back in the 1960s. The migration enabled them to support performant interactive analytics at scale while running complex ETL processes in parallel. The key factors that made it work were subject matter expertise, operational autonomy, code quality for maintainability, and proactive vendor support. When you're dealing with 4.5 million meters and 1,000 daily users, you need both scalable storage and compute—there's no way around it.

By addressing these steps systematically, utilities can unlock the full value of their data. Properly prepared datasets feed into machine learning models that predict failures, forecast load, and support data-driven investment planning. But here's the thing: this work is unglamorous. Nobody gets excited about data cleaning. Do it anyway—it's the foundation everything else sits on.

---

## From Raw Records to Actionable Signals

A typical example is transformer monitoring. SCADA data may include transformer load and oil temperature, while EAM holds the installation date and maintenance history. By joining these, we can calculate load-to-age stress factors, compare them across similar units, and flag transformers at higher risk of failure. Without integrated data, such insights remain invisible.

Another example is storm readiness. Outage records stored in OMS systems can be combined with feeder vegetation data and historical weather records. By cleaning and aligning these datasets, we can train models that predict which circuits are most likely to fail during high winds. This directly informs crew staging and vegetation management priorities.

These cases highlight a recurring theme: data silos hide patterns that cross organizational boundaries. Preparing data for analytics is as much about breaking down silos as it is about technical preprocessing.

---

## Common Data Quality Issues in Utilities

Before diving into the code, let me tell you about the data quality problems that kill projects. SCADA systems may drop readings during communication failures, creating missing timestamps. AMI data might use local time while SCADA uses UTC, causing timezone inconsistencies. Some systems report in kW while others use MW, creating unit mismatches. Over time, sensors may report values that drift from true measurements. ETL processes sometimes create duplicate entries. Faulty sensors can produce extreme values that skew analysis.

Teams spend weeks debugging models, only to discover that the "anomalies" they were detecting were actually timezone mismatches. One project I worked on had SCADA data in UTC and AMI data in local time, and nobody noticed until we tried to join them. The model was technically correct—it was just learning from misaligned data.

The code below demonstrates how to identify and handle these issues systematically. Trust me, it's worth the upfront effort.

---

## Data Preparation Workflow

Let's walk through a complete data preparation workflow for utility data. I'm showing you two common data sources: smart meter (AMI) data and SCADA telemetry. This is the kind of work that happens before the exciting ML modeling—but it's what makes or breaks your project.

First, we load and clean smart meter data:

{{< pyfile file="data_for_power_and_utilities.py" from="16" to="31" >}}

The `load_smart_meter_data()` function reads CSV files with consumption data, while `clean_and_resample()` handles missing values and resamples to hourly intervals for consistency. This is essential because different utility systems operate at different frequencies. In practice, you might have AMI data every 15 minutes, SCADA every 4 seconds, and weather data hourly. You need to align them to a common interval, and hourly is usually a good starting point.

Next, let's visualize what we're working with:

{{< pyfile file="data_for_power_and_utilities.py" from="32" to="43" >}}

Always plot your data first. This plot reveals daily cycles (higher during day, lower at night) and helps identify data quality issues like missing periods or outliers. Too many projects skip this step and feed garbage data into models. Visual inspection is critical before building ML models—it's not optional.

For SCADA data, we generate synthetic grid frequency telemetry:

{{< pyfile file="data_for_power_and_utilities.py" from="44" to="55" >}}

SCADA systems collect real-time grid measurements. This function simulates frequency data around the nominal 60 Hz, with small variations that indicate grid stability. In a real system, you'd pull this from your SCADA historian, but the principles are the same.

Finally, we visualize the SCADA frequency data:

{{< pyfile file="data_for_power_and_utilities.py" from="56" to="67" >}}

Frequency deviations indicate grid stress—values consistently above or below 60 Hz suggest supply-demand imbalances that operators must address. This visualization helps operators monitor grid health in real-time. Control room operators watch these plots constantly, looking for patterns that signal problems.

The complete, runnable script is at `content/c2/data_for_power_and_utilities.py`. Run it, modify it, break it—that's how you learn.

---

## What I Want You to Remember

Data fragmentation is the primary barrier. Utilities have abundant data, but it's scattered across IT and OT systems that don't communicate effectively. Cleaning handles missing and erroneous data, resampling aligns time series, and integration joins multiple sources. These form the foundation of utility analytics.

Visual inspection is essential. Always plot your data before modeling. I've said this before, and I'll say it again: visualizations reveal patterns, outliers, and quality issues that summary statistics miss. Time alignment matters because different utility systems operate at different frequencies, from seconds to minutes to hours. Resampling to common intervals enables joint analysis.

Data quality determines model quality. Garbage in, garbage out. It's that simple. Teams spend months building sophisticated models, only to realize later that their data had timezone mismatches or unit inconsistencies. Investing time in data preparation pays dividends in model accuracy and operational trust.

Here's the thing: this work is boring. Nobody gets excited about data cleaning. But it's the foundation everything else sits on. Do it right, and your models will work. Skip it, and you'll spend weeks debugging problems that could have been caught in an afternoon.

---

## What's Next

In Chapter 3, we'll apply machine learning fundamentals—regression, classification, and clustering—to utility use cases. You'll see how clean, well-prepared data enables these techniques to deliver actionable insights. The models are the fun part, but they only work if the data is ready.