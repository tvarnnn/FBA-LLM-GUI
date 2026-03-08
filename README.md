# FBA-LLM: Evidence-Based Amazon FBA Research Copilot

FBA-LLM is an AI-assisted research tool designed to evaluate Amazon FBA product opportunities using structured product metrics, customer reviews, and optional visual evidence. The system generates grounded, evidence-based insights while preventing unsupported claims or hallucinated statistics.

The goal is to provide **transparent decision support**, not automated investment advice.

---

# Key Features

## Evidence-Grounded Analysis

The system constructs a unified **facts block** from multiple data sources and restricts the LLM to reasoning only from that evidence.

This prevents:

- hallucinated numbers  
- invented business metrics  
- unsupported claims  

---

## Multi-Source Data Ingestion

The system supports:

- Product metrics (CSV)  
- Customer reviews (TXT)  
- Visual evidence (PNG screenshots such as charts or analytics)

These inputs are combined into a structured analysis context.

---

## Review Theme Extraction

Customer reviews are analyzed to extract common signals such as:

- recurring complaints  
- product strengths  
- quality issues  
- durability concerns  
- feature feedback  

These insights help identify **product improvement opportunities**.

---

## Interactive Research Workflow

Instead of generating a single static report, the system supports **follow-up questions** such as:

- “What are the biggest risks in this category?”
- “How could a new product differentiate?”
- “What complaints are easiest to fix?”

Each question is answered using the same grounded evidence.

---

## Guarded Reasoning System

The project includes strict safeguards to prevent unreliable output:

- Number hallucination prevention  
- Unsupported business claim detection  
- Schema validation for structured responses  
- Evidence citation validation  

This keeps outputs **traceable to the input data**.

---

## Research Copilot GUI

A Tkinter interface allows users to:

- upload datasets  
- configure assumptions  
- run analysis  
- ask follow-up questions  
- inspect the underlying evidence block  

---

# Typical Workflow

1. Upload product metrics CSV  
2. Upload review dataset TXT  
3. *(Optional)* Upload visual evidence PNGs  
4. Generate a screening analysis  
5. Ask follow-up research questions  

The system produces evidence-grounded insights about:

- market competitiveness  
- product quality signals  
- risk factors  
- missing data  
- potential differentiation strategies  

---

# Tech Stack

- Python  
- LangChain  
- Anthropic / Groq LLM APIs  
- ChromaDB (vector storage)  
- Sentence Transformers (embeddings)  
- Tkinter (GUI)  
- PyTorch

---

# Design Philosophy

This project prioritizes:

- Evidence-grounded reasoning  
- Transparency  
- Guardrails against hallucination  
- Modular architecture  
- Practical decision support  

It intentionally avoids generating unsupported statistics or speculative financial projections.
