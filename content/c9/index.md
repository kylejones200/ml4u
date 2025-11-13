---
title: "Customer Analytics and Demand Response"
description: "AMI-driven segmentation and targeting for effective demand response."
weight: 9
draft: false
pyfile: "demand_response.py"
---

## What You'll Learn

By the end of this chapter, you will understand how customer segmentation improves demand response program effectiveness. You'll learn to use K-means clustering to group customers by load profiles. You'll see how clustering reveals distinct consumption patterns from smart meter data. You'll recognize how segmentation informs program design and targeting strategies, and you'll apply clustering results to identify high-value customers for demand response.

---

## The Business Problem: Engaging Customers to Manage Demand

Electric grids must be built to meet peak demand even though those peaks occur only a few times each year. This leads to expensive infrastructure that sits underutilized for most hours. Demand response programs aim to address this by encouraging customers to shift or reduce consumption during critical periods, flattening peaks and easing stress on the grid.

However, not all customers respond the same way. Some households have highly flexible loads they can shift easily, while others cannot reduce usage without disruption. Programs that treat all customers alike often achieve disappointing results because incentives and messaging fail to match customer behavior.

Utilities spend millions on demand response programs that barely moved the needle because they treated everyone the same. The problem isn't the technology—it's the targeting.

Utilities need ways to identify which customers are best suited for demand response and to design tailored programs that maximize participation. Without this insight, they risk low enrollment, poor event compliance, and limited grid impact, undermining the value of demand-side resources.

---

## The Analytics Solution: Using Data to Target and Segment Customers

Customer analytics uses data from advanced metering infrastructure, billing systems, and program participation records to understand consumption patterns and identify load flexibility. Smart meters provide granular consumption data that can reveal daily, weekly, and seasonal usage profiles for each household.

Clustering techniques can segment customers into groups based on similar load shapes. For example, some customers may show high evening peaks tied to cooking and HVAC usage, while others have flatter profiles or midday spikes associated with daytime occupancy. These segments inform which customers are most likely to reduce load in response to incentives or pricing signals.

Classification models can also predict participation likelihood, drawing on historical demand response enrollment and demographic data. This helps utilities prioritize outreach to those most likely to engage while avoiding costly campaigns aimed at customers unlikely to respond.

By combining behavioral segmentation with predictive modeling, utilities can refine demand response strategies, increase event performance, and avoid overbuilding supply-side resources.

---

## Understanding K-Means Clustering

K-means is an unsupervised learning algorithm that groups similar observations into clusters. For customer segmentation, the input is daily load profiles with 24 hourly values for each customer. The process involves the algorithm finding K cluster centers, or centroids, that minimize within-cluster variance. The output assigns each customer to one of K clusters based on similarity.

Choosing the number of clusters K involves several common methods. The elbow method plots within-cluster variance versus K and looks for an elbow where improvement plateaus. Domain knowledge allows choosing K based on business needs, such as 3-5 segments for program design. The silhouette score measures how well customers fit their clusters, with higher scores being better.

The code uses K=3, which typically reveals high-peak customers with evening spikes, flat-profile customers with consistent usage, and mid-peak customers with moderate variation.

---

## Temporal Patterns in Customer Behavior

Customer load profiles change over time. Daily patterns show morning peaks from breakfast and getting ready, and evening peaks from cooking and heating or cooling. Weekly patterns reveal weekday versus weekend differences. Seasonal patterns show summer cooling versus winter heating.

The code focuses on daily patterns, but production systems often analyze weekly or seasonal profiles. Advanced approaches use time series clustering, such as Dynamic Time Warping, to handle temporal variations.

---

## Business Impact

Targeted demand response programs reduce peak load, defer costly infrastructure upgrades, and lower wholesale energy procurement during high-price periods. They also support integration of renewables by shifting load to times of abundant solar or wind generation.

Southern Company, which serves 9 million customers across electric and gas utilities in Alabama, Georgia, and Mississippi, demonstrates this approach. Their customer analytics strategy focuses on three things: affordability, customer satisfaction, and the value they provide to customers. They've built a unified data platform that enables dynamic, near real-time insights rather than just retrospective reports.

Here's a concrete example: they have robust social media monitoring tools that track customer sentiment and issues. In one case, a frustrated customer posted on social media but didn't identify themselves or their location. The power delivery team used data analytics to identify the customer, find their location, analyze their outage history, and confirm they were indeed experiencing frequent outages—sometimes weather-related, sometimes vegetation management issues. The team took action and did work that improved that customer's reliability. That's how customer analytics transforms reactive service into proactive problem-solving.

Their approach emphasizes meeting customers where they are—laptop, toughbook in the field, or phone. The platform enables customer 360 views that combine billing, usage, outage history, and program participation to provide a complete picture of each customer's relationship with the utility. Nick Whatley, their Director of Enterprise Data and Customer Analytics, told me they keep the customer at the center of everything. As they deploy insights internally, they make sure it's going to work day-to-day, leading to better satisfied customers and improved affordability, safety, and reliability.

Customers benefit from lower bills through participation incentives or time-based rates that reward off-peak consumption. Effective segmentation ensures these programs feel relevant and fair, avoiding customer dissatisfaction or program fatigue.

In competitive markets, successful demand response strategies can also provide revenue streams by aggregating flexible load into virtual power plants that bid into wholesale markets. Analytics makes this aggregation more precise and dependable.

---

## Building Customer Segmentation Models

Let's segment customers using K-means clustering on smart meter data. The approach transforms raw consumption data into actionable customer insights that inform demand response program design and targeting. Utilities double their demand response participation rates by using segmentation like this.

First, we generate synthetic smart meter data:

{{< pyfile file="demand_response.py" from="19" to="32" >}}

This creates hourly consumption data for multiple customers over several days. Each customer has a unique load profile with daily patterns (morning and evening peaks) plus random variation, simulating real AMI data utilities collect. In practice, you'd pull this from your AMI system, but the patterns are the same.

Next, we cluster customers by their daily load profiles:

{{< pyfile file="demand_response.py" from="33" to="58" >}}

The code extracts daily profiles (averages across days), standardizes them (focusing on shape rather than magnitude), and applies K-means clustering. The visualization shows multiple daily load profiles overlaid for each cluster, with cluster centroids highlighted. Each cluster represents a distinct customer segment with similar usage patterns. Utilities discover EV adoption patterns this way—clusters that look different from historical norms often signal new technology adoption.

Finally, we identify demand response targets:

{{< pyfile file="demand_response.py" from="59" to="65" >}}

Customers in the highest-load cluster are prime targets for demand response—they have high impact and likely flexibility (e.g., can shift EV charging, defer laundry). The key is targeting the right customers, not just the ones with the highest bills.

The complete, runnable script is at `content/c9/demand_response.py`. Run it and see what segments emerge from your data.

---

## What I Want You to Remember

Segmentation improves program effectiveness. Treating all customers the same leads to low participation. Clustering reveals distinct segments that need different approaches. K-means is simple but effective. For daily load profiles, K-means often captures meaningful segments. More complex methods like hierarchical clustering or time series clustering may be needed for weekly or seasonal patterns.

Standardization matters. Clustering on raw consumption values groups by total usage. Standardizing focuses on shape, meaning when peaks occur, which is more relevant for demand response targeting. Visualization is essential. Plots reveal whether clusters capture meaningful patterns or are just statistical artifacts. Always visualize cluster results before using them operationally. Utilities use clusters that don't make operational sense when they don't visualize them first.

Segmentation enables personalization. Different customer segments need different programs, messaging, and incentives. Analytics makes this personalization scalable. The key is matching the program to the segment—high-peak customers might respond to different incentives than flat-profile customers.

---

## What's Next

In Chapter 10, we'll explore computer vision—using object detection models to automate infrastructure inspections from drone imagery, reducing field costs while improving coverage and safety. It's a different use case, but the principles are the same.
