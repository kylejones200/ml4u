# ML4U: Machine Learning for Power & Utilities

A practical, chapter-based guide with runnable Python for applying Machine Learning across the power and utilities lifecycle: forecasting, predictive maintenance, outage prediction, DER integration, NLP for compliance, orchestration, and more.

- Live site: https://kylejones200.github.io/ml4u/

## Who this is for

- Utility leaders and practitioners exploring ML’s practical value.
- Data scientists and engineers seeking domain-grounded examples.
- Students learning applied ML with real grid-oriented scenarios.

## What you’ll learn

- How core ML methods map to utility problems (load, reliability, maintenance, customer analytics).
- How to operationalize models (MLOps, orchestration, governance, ethics).
- How to integrate multimodal data (text, images, sensors) for real decisions.

## What’s inside

- Chapters `c1` … `c20`, each with:
  - A short narrative framing the business problem and analytics approach.
  - An accompanying Python file with runnable code examples.
  - Clean URLs (e.g., `content/c10/index.md` → https://kylejones200.github.io/ml4u/c10/).

Topics include: data readiness, ML fundamentals, load forecasting, predictive maintenance, outage prediction, grid optimization, DER forecasting, customer analytics, computer vision, NLP, MLOps, orchestration, cybersecurity, ethics, enterprise integration, and full-platform deployment.

## How the code examples render

- Each chapter auto-embeds a sibling Python file as a highlighted code block at the end of the page.
- You can override which file appears via the chapter front matter field `pyfile`, e.g.:
  ```yaml
  ---
  title: "Outage Prediction and Reliability Analytics"
  pyfile: "outage_prediction.py"
  ---
  ```
- Implementation lives in `layouts/_default/single.html` and the `pyfile` shortcode.

## Roadmap

- Expand line-range controls for code embeds per chapter.
- Add lightweight datasets and visualizations per topic.
- Publish notebooks that mirror the embedded scripts.
- Enrich chapter cross-links (e.g., how maintenance scores feed outage risk).

## Using the site

- Read chapters online at the live site; copy the embedded Python into your environment and run.
- Each chapter’s code is also stored alongside its content under `content/<chapter>/`.

## Contributing

- Open a PR against `main`.
- Keep chapters self-contained and focused on business impact + runnable code.
- Prefer `index.md` per chapter for clean URLs and add a concise `description` in front matter.

## License

Content © its authors.
