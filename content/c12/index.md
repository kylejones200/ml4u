---
title: "Large Language Models and Multimodal AI for Utilities"
description: "LLMs for text plus multimodal integration with images and sensors."
weight: 12
draft: false
pyfile: "AI4U.py"
---

## What You'll Learn

By the end of this chapter, you will understand how LLMs can extract insights from unstructured utility text like logs, reports, and compliance documents. You'll learn to use LLMs for summarization and entity extraction from maintenance logs. You'll see how multimodal AI combines text, images, and sensor data for comprehensive analysis. You'll recognize the costs, limitations, and deployment considerations for LLM applications, and you'll appreciate the experimental nature of LLMs in utility operations and when to use them.

---

## The Business Problem: Making Sense of Complex, Unstructured Data

Utilities operate in environments that generate a constant mix of structured and unstructured data. Structured data—such as SCADA readings, AMI meter data, and asset inventories—is well understood and widely used in analytics. However, an equally important layer of information lives in unstructured formats: maintenance logs, inspection notes, incident reports, regulatory filings, and even field images or drone footage.

Historically, these two worlds have remained separate. Engineers interpret logs and inspection notes manually, while structured data feeds into automated reports and dashboards. This split leads to slow, fragmented decision-making. For example, if a transformer fails, SCADA data may show an overload condition, but the underlying issue—like a minor oil leak noted in a maintenance report six months earlier—is buried in text. Without tools to unify these sources, valuable context is lost.

A transformer failed, and the SCADA data showed the overload, but the maintenance log from six months earlier mentioned an oil leak that nobody connected. That's the problem—structured and unstructured data don't talk to each other.

This challenge grows as the grid digitizes. Distributed energy resources, smart devices, and IoT sensors introduce new data streams. Field inspections increasingly involve photos or video from drones. Call center transcripts capture customer complaints that may indicate emerging reliability issues. Utilities need ways to synthesize these diverse data types to support faster, better-informed decisions.

---

## The Analytics Solution: LLMs and Multimodal Integration

Large language models (LLMs) excel at interpreting unstructured text, summarizing documents, and extracting key entities or themes. When combined with structured data, they can create a more complete picture of utility operations. For example, LLMs can read maintenance logs, extract references to specific equipment, and pair those findings with SCADA measurements to create context-rich asset profiles.

Multimodal AI takes this further by handling text, images, and sensor data together. A single model can analyze drone photos for defects, summarize technician notes, and correlate findings with temperature or load history to generate prioritized maintenance recommendations. This integration enables workflows that previously required manual coordination between teams.

---

## Understanding Large Language Models

LLMs like GPT-4, Claude, and Llama are neural networks trained on vast text corpora. They can summarize by condensing long documents into key points, extract specific information like entities, dates, and equipment IDs, classify text into categories such as routine versus failure or compliance versus operational, and generate reports, responses, or documentation.

How LLMs work: they are trained to predict next words in sequences, learn patterns, grammar, and domain knowledge from training data, can be fine-tuned on utility-specific text for better performance, and use attention mechanisms to understand context.

LLM options for utilities include OpenAI GPT models, which offer high performance through API-based, pay-per-use access, open-source models like Llama and Mistral, which are self-hosted with lower cost and more control, and fine-tuned models that are trained on utility text for domain-specific tasks.

---

## Costs and Limitations of LLMs

Cost considerations include API costs, where GPT-4 can cost $0.03-0.06 per 1K tokens for input and output combined, and processing thousands of logs daily adds up quickly. Infrastructure costs arise from self-hosting open-source models, which requires GPUs and significant compute. Development costs include prompt engineering, fine-tuning, and integration, which require expertise.

Limitations include hallucination, where LLMs can generate plausible but incorrect information, context windows that are limited in input length though improving, latency from API calls that add delay and may not be suitable for real-time applications, privacy concerns from sending sensitive data to external APIs, and regulatory uncertainty where LLM outputs may not meet explainability requirements.

Use LLMs for text summarization and extraction, which offer high value with lower risk, compliance document review to augment human review, and customer service chatbots with human oversight. Avoid LLMs for real-time control decisions because they are too slow and unreliable, critical safety decisions due to hallucination risk, and highly regulated decisions requiring explainability.

---

## Prompt Engineering for Utilities

Effective prompts are critical for LLM performance. Be specific, such as asking to summarize a maintenance log and extract equipment IDs rather than just analyzing text. Provide context by including domain knowledge, such as explaining that in utility operations, TX-101 refers to a transformer. Use examples, as few-shot learning improves performance. Specify format by requesting structured output like JSON or tables for easier integration.

An example prompt structure might instruct the LLM to act as an expert in utility operations, analyze maintenance logs, summarize key issues, extract equipment IDs in a specific format like TX-XXX or SUB-XXX, and flag any failure-related language.

The code shows basic LLM usage, but production systems should include prompt templates, validation, and error handling.

---

## Practical Applications

Utilities can use LLMs to automate the review of compliance documents, quickly identifying sections relevant to specific standards. Field inspection notes can be summarized automatically, highlighting critical observations and linking them to asset IDs. Multimodal models can screen solar panel images for cracks and pair results with inverter performance data, pinpointing which defects are reducing output.

In control centers, operators could query LLM-powered systems in natural language to retrieve SCADA trends, past incidents, and maintenance history for any given feeder or substation. This reduces the friction of navigating multiple systems and databases manually.

The energy sector demonstrates this pattern, even though it's not strictly utilities. There's a concept called "Agents4Energy" that shows how AI agents can democratize analytics for domain experts. Here's the problem it solves: a petroleum engineer might need to analyze 200 wells, understand decline curves, and recommend actions before tomorrow morning's production meeting. Traditional BI dashboards show KPIs but don't answer deeper questions like "decline by completion type" or "what-if scenarios for reservoir properties." Custom analysis requests to data science teams can take two to three weeks with multiple back-and-forth clarifications, and the analysts might not understand the domain terminology or physics-based constraints.

AI agents act as "digital co-workers" that hold context, have multi-turn conversations, and orchestrate specialized tools as needed. These aren't simple chatbots—they pick the right tool for the job and iterate on outputs to solve nuanced domain challenges. The impact is dramatic: what used to be a multi-day process with multiple handoffs becomes minutes. One engineer's workflow went from "days waiting for DS support, multiple handoffs, uncertain results" to "analysis in minutes, reliable and reproducible methodology, scalable across business units."

This pattern applies directly to utilities. Grid engineers could use agents to analyze transformer performance across hundreds of assets, forecast load impacts of new DER installations, or investigate outage patterns—all through natural language queries. The agent translates the question into the right data analysis and visualization. That's the future of how domain experts will interact with data.

NextEra Energy has been one of the early adopters of Gen AI in utilities. While others were experimenting, they took it seriously from the start. They built what they call a "Gen AI factory" with four layers: governance and risk management, platform capabilities, patterns and accelerators, and solution development. The governance layer is critical—they have approved usage policies, risk management templates, and guardrails that filter content and redact PII. The platform layer includes responsible AI services, observability, monitoring, and access controls. They've learned that prompt engineering matters—they use prompt versioning and model evaluation to ensure prompts actually deliver value. The key insight: start with low-risk applications like summarization, prove value, then scale. They're using Gen AI to enhance customer value, improve employee productivity, and optimize energy operations across the enterprise.

---

## Building LLM and Multimodal AI Systems

Let's use LLMs to analyze maintenance logs and combine those insights with structured sensor data. This creates a richer picture of asset health than either data source alone. I'm showing you this because LLMs are promising, but they're also risky if you don't use them carefully.

First, we generate incident logs and sensor data:

{{< pyfile file="AI4U.py" from="18" to="38" >}}

This creates synthetic maintenance and incident reports (unstructured text) and sensor readings (structured data) for transformers. These represent the types of data utilities collect from field crews and SCADA systems.

Next, we analyze logs with an LLM:

{{< pyfile file="AI4U.py" from="39" to="71" >}}

This uses OpenAI's GPT-4 (if API key available) or a deterministic fallback to analyze incident logs. The LLM summarizes key issues, extracts equipment references, and identifies recurring themes. The summary helps operators quickly understand patterns across multiple incidents. LLMs do a good job summarizing logs, but they can also hallucinate—always verify the output.

Finally, we fuse text and sensor data:

{{< pyfile file="AI4U.py" from="72" to="79" >}}

This combines LLM-extracted insights from text with structured sensor data, creating a comprehensive asset profile. For example, if logs mention "overheating on T-102" and sensors show high temperature for T-102, this confirms and quantifies the issue. This is where the real value is—connecting structured and unstructured data.

Important note: LLMs in utilities are still experimental. Always review LLM outputs before taking action, and start with low-risk applications like summarization before critical decisions. Teams get excited about LLMs, then realize they're too slow or unreliable for real-time operations. Start simple.

The complete, runnable script is at `content/c17/AI4U.py`. Note: This requires an OpenAI API key, which is optional, for LLM functionality.

---

## What I Want You to Remember

LLMs unlock value in unstructured text. Maintenance logs, inspection reports, and compliance documents contain insights that are difficult to extract with traditional NLP. Costs and limitations are real. LLM APIs are expensive at scale, and models can hallucinate. Use LLMs for augmentation, not replacement of human judgment.

Prompt engineering matters. Well-crafted prompts dramatically improve LLM performance. Invest in prompt templates and validation. Multimodal AI is emerging. Combining text, images, and sensors creates comprehensive insights, but the technology is still experimental for utilities.

Start with low-risk applications. Use LLMs for summarization and extraction before deploying for critical decisions. Build trust gradually. Teams get excited about LLMs, then realize they're too slow or unreliable for real-time operations. Start simple, prove value, then scale.

---

## What's Next

In Chapter 13, we'll explore enterprise integration—connecting ML models with GIS, SCADA, and EAM systems to create unified operational intelligence. This is where the rubber meets the road: making models actually work in production systems.
