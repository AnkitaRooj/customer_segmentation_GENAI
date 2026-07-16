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

# FMCG GenAI Positioning & Segmentation Engine 🚀

An end-to-end, production-ready Generative AI application designed for the FMCG (Fast-Moving Consumer Goods) sector. This engine bridges the gap between traditional consumer panel data and modern generative marketing copy. It uses vector-based similarity search to cluster raw customer profiles, retrieves relevant historical winning strategies via a Retrieval-Augmented Generation (RAG) pipeline, and utilizes LangChain prompt chaining to generate hyper-localized, AIDA-structured marketing copy tailored for diverse Indian retail audiences.

---

## 🧠 Step-by-Step Logic

1. **Embedding & Clustering:** Raw customer profiles are converted into dense vector embeddings using `all-MiniLM-L6-v2`. FAISS performs similarity search to assign the incoming profile to its nearest behavioral segment cluster.
2. **RAG Retrieval:** Based on the identified segment, the system queries a vector database populated with historical FMCG campaigns to fetch the most relevant past winning strategies, significantly reducing LLM hallucination.
3. **Prompt Chaining:** LangChain synthesizes the segment profile and the retrieved past campaign context to generate a new, hyper-localized AIDA (Attention-Interest-Desire-Action) marketing strategy.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🔍 **Vector-Based Segmentation** | Uses FAISS to cluster customers based on deep purchase behavior patterns rather than just basic demographics. |
| 🧠 **RAG-Powered Context** | Fetches real past FMCG campaigns (inspired by Kellogg's, Unilever, and P&G frameworks) to ground LLM outputs. |
| 📝 **AIDA Copywriting** | Automatically structures generated marketing copy into Attention, Interest, Desire, and Action components. |
| 🌏 **Hyper-Localization** | Explicitly supports Hinglish (Hindi + English) messaging tailored for Tier 2/3 Kirana store audiences. |
| ⚡ **FastAPI Backend** | Production-ready RESTful API architecture complete with automatic Swagger UI documentation. |
| 🐳 **Dockerized** | Clean containerization allowing one-command deployments to any cloud environment. |

---

## 🗂️ Dataset Details (Synthetic but Realistic)

This project simulates real-world Indian FMCG Consumer Panel Data through a custom-built dataset:

| Feature | Description | Values (Examples) |
| :--- | :--- | :--- |
| **City Tier** | Market classification | Metro, Tier 1, Tier 2, Tier 3 |
| **Primary Channel** | Purchase location | Kirana Store, Supermarket, Online |
| **Category Spends** | Monthly FMCG expenditure (₹) | Snacks, Beverages, Household, Personal Care |
| **Behavioral** | Promo affinity & Organic preference | Yes / No |
| **Demographics** | Age, Income (LPA), Family Size | 18-65 yrs, 2-25 LPA |

The `historical_campaigns.txt` file contains 6 distinct Consumer Packaged Goods (CPG) campaigns (e.g., Bulk Staples, Premium Snacks, Sachet-rural penetration) which serve as the knowledge base for the RAG pipeline.

---

## 🚀 Tech Stack

* **Core Language:** Python 3.9+
* **LLM Frameworks:** LangChain, LangGraph (Conceptual), HuggingFace Transformers
* **Vector Search:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Backend API:** FastAPI, Uvicorn (REST Architecture)
* **Containerization:** Docker
* **Database (Conceptual):** PostgreSQL / MySQL (for metadata mapping)
* **Cloud (Design):** AWS (S3 for artifact storage, Lambda for serverless inference)

---

## 📦 Installation & Setup

Follow these steps to spin up the engine locally:

### 1. Clone the repository
```bash
git clone [https://github.com/ankita-ai/fmcg-genai-engine.git](https://github.com/ankita-ai/fmcg-genai-engine.git)
cd fmcg-genai-engine

### 2. Create a virtual environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

### 3. Install dependencies
Bash
pip install -r requirements.txt

### 4. Set up Environment Variables
Create a .env file in the root directory and add your API key:

Code snippet
OPENAI_API_KEY=your_openai_api_key_here

### 5. Run the FastAPI Server
Bash
uvicorn api.app:app --reload
The API will be live at http://localhost:8000.