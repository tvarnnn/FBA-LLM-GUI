Local LLM–powered decision support for Amazon FBA product viability

This project is a local, offline Amazon FBA analysis tool that uses a large language model (LLM) to generate grounded, non-hallucinated business insights from structured product metrics and unstructured customer reviews.

----Key Features----

Local LLM inference (no cloud APIs, no data leakage)
Metrics ingestion from CSV (price, ratings, reviews, categories, weight)
Review analysis from raw customer text
Strict grounding guards
Prevents hallucinated numbers
Blocks unsupported business claims
Unified “facts block” pipeline for structured + unstructured data
Tkinter GUI for interactive analysis
CLI runner for scripted experiments and grading
No model files tracked (HuggingFace cache ignored via .gitignore)

----Design Philosophy----

This project intentionally avoids:
LLM-generated percentages or fabricated statistics
Over-engineered output schemas
Cloud-based dependencies

Instead, it prioritizes:
Transparency
Explainability
Guarded reasoning
Realistic decision support

----Typical Use Case----

Upload product metrics (CSV)
Upload customer reviews (TXT)
Run local inference
Receive a grounded advisory summary on:
Market competitiveness
Risk factors
Data limitations
Product viability direction

----Tech Stack----
Python
HuggingFace Transformers
PyTorch
Tkinter (GUI)
Local LLaMA-based models

Note: Model weights are intentionally excluded from this repository.
