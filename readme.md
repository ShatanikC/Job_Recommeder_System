# 💼 Hybrid Job Recommender System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Sentence--BERT-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

An intelligent, multi-engine Job Recommender application built using **Streamlit**. The system blends traditional keyword-matching (`TF-IDF`) with deep learning semantic understanding (`Sentence-BERT`) to deliver highly accurate resume-to-job recommendations, fully filtered by experience requirements.

---

## 🚀 Features

* **Four Advanced Retrieval Modes:**
    * **TF-IDF:** High-precision, exact keyword matching (ideal for spotting specific programming languages/tools).
    * **Semantic Search:** Context-aware meaning mapping using `all-MiniLM-L6-v2`. (Matches "Software Engineer" to "Fullstack Developer").
    * **Hybrid Search:** Linear weighting combination of Keyword and Semantic models with customizable weights.
    * **Reciprocal Rank Fusion (RRF):** Modern, scale-free rank aggregation method for clean system blending.
* **Dynamic UI Side Panel:** Seamlessly filter profiles by **Job Title**, **Technical Skills**, and **Minimum Years of Experience** via sliding parameters.
* **Asset Pre-computation & Caching:** Designed for production speed. Heavy TF-IDF fitting and Transformer matrices are pre-computed, pickled, and cached instantly via `@st.cache_resource`.

---

## 🛠️ Architecture & Retrieval Logic

### 1. Hybrid Search
Combines both semantic distance and term frequency vector weights mathematically:
$$\text{Hybrid Score} = (\alpha \times \text{TFIDF Score}) + (\beta \times \text{Transformer Score})$$

### 2. Reciprocal Rank Fusion (RRF)
Evaluates position ranks rather than arbitrary similarity percentages to avoid scoring bias across models:
$$RRF\_Score = \frac{1}{k + \text{Rank}_{\text{tfidf}}} + \frac{1}{k + \text{Rank}_{\text{transformer}}}$$

---
