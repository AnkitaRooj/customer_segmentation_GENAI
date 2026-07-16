# 🛒 FMCG GenAI Segmentation & Hyper-Localization Engine

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange?logo=langchain)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-red)](https://github.com/facebookresearch/faiss)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**Stop spraying and praying your marketing budget.**

This project simulates a **NielsenIQ-style Consumer Panel Analytics** engine, leveraging **Generative AI** to solve the biggest problem in the **FMCG (Fast-Moving Consumer Goods)** industry: **Identifying who to sell, what to sell, and how to talk to them.**

By combining **LangChain**, **RAG (Retrieval-Augmented Generation)**, and **FAISS vector search**, this engine ingests raw consumer purchase data (Spends on Snacks, Beverages, Household Care, Personal Care) and automatically generates:

1. **Hyper-Personalized Customer Segments** (e.g., *"Metro Millennial Snackers"*, *"Kirana Value Hunters"*).
2. **RAG-Powered Campaign Retrieval** (fetches past winning FMCG promotions).
3. **AIDA Marketing Copy** (Localized in English & Hinglish for Tier 2/3 markets).

---

## 🧠 Problem Statement

> *"An FMCG giant spends crores on mass-media advertising, yet 60% of their promotions fail to resonate with local Kirana store owners and metropolitan millennials simultaneously. They lack a real-time system to segment consumers based on actual purchase behavior and generate culturally relevant positioning at scale."*

**This project bridges that gap** by acting as an AI co-pilot for Brand Managers, helping them launch hyper-localized campaigns in minutes instead of weeks.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Raw FMCG Consumer Data<br>Age, Income, Category Spends, Channel] --> B(Sentence-Transformers)
    B --> C[FAISS Vector Index<br>Customer Segmentation]
    A --> D[Historical Campaigns Corpus]
    D --> E[LangChain + FAISS<br>RAG Vector Store]
    C --> F[FastAPI Endpoint]
    E --> F
    F --> G[LangChain Prompt Chain<br>AIDA Framework]
    G --> H[Output:<br>Segment Label + Campaign Context + Ad Copy]