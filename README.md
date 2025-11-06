# 🧠 Document Intelligence System Architecture

### *Enterprise AI System for Oracle-AI Modernization Workloads*

---

## Overview

This repository presents the **Document Intelligence System**, an **enterprise-grade architecture** engineered for large-scale document understanding, reasoning, and explainable automation.

The system applies **Oracle AI modernization principles** — modularity, traceability, and governance — while leveraging open-source technologies for flexibility and transparency.

It operationalizes **Retrieval-Augmented Generation (RAG)** and **Graph-Augmented Reasoning (Graph RAG)** to enable **connected, explainable insights** across documents, tables, and OCR sources.

---

## ⚙️ System Capabilities

| Capability                      | Description                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| **Hybrid Retrieval**            | Combines semantic (vector) and relational (graph) retrieval.                               |
| **Explainable Reasoning**       | Generates answers with citations and confidence scores.                                    |
| **Multi-Format Ingestion**      | Handles text, tables, emails, and OCR-based documents.                                     |
| **Oracle-Aligned Architecture** | Mirrors Oracle AI modernization patterns for governance, explainability, and auditability. |
| **Modular & Scalable**          | FastAPI orchestration, Redis caching, and containerized deployment.                        |

---

## 🧩 System Architecture

The design is structured into **six modular layers**, following enterprise AI modernization principles:

1. **Clients Layer** — UI (Streamlit), APIs, and external integrations.
2. **FastAPI Layer** — async orchestration for document and reasoning pipelines.
3. **Core Services** — RAG engine, vector store (ChromaDB), and processing logic.
4. **Extractors Layer** — modular ingestion (PDF, Excel, Word, Email, OCR).
5. **Storage Layer** — ChromaDB (embeddings), Redis (cache), file persistence.
6. **External Services** — LLMs (Claude/OpenAI) and Graph DB (Neo4j) for reasoning.

📘 *See diagram for system flow and component relationships.*

---

## 🧠 Technology Stack

| Layer                   | Technology                     | Purpose                                       |
| ----------------------- | ------------------------------ | --------------------------------------------- |
| **API & Orchestration** | FastAPI                        | High-performance, async architecture          |
| **Vector Retrieval**    | ChromaDB                       | Embedding storage and semantic search         |
| **Graph Reasoning**     | Neo4j                          | Relationship and lineage analysis             |
| **LLM Interface**       | Claude / OpenAI                | Contextual reasoning and synthesis            |
| **Cache Layer**         | Redis                          | Response caching and concurrency optimization |
| **Extraction**          | spaCy / pdfplumber / Tesseract | Document parsing and NER                      |
| **Deployment**          | Docker                         | Portable and cloud-ready                      |

---

## 🧭 Alignment with Oracle AI Modernization

This system mirrors **Oracle’s AI modernization blueprint**:

* **Data lineage & traceability** → Each answer is source-linked and explainable.
* **Composable architecture** → Independent, replaceable layers for scalability.
* **Hybrid reasoning** → Combines semantic retrieval with graph causality.
* **Cloud readiness** → Deployable across OCI or hybrid environments.

It represents the **AI modernization layer** that connects **Oracle data estates** with retrieval-augmented and graph-aware intelligence.

---

## 🚀 Quick Start

This system can be deployed as a standard FastAPI app:

```bash
git clone https://github.com/yourusername/document-intelligence-system.git
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Access interactive API docs at:
👉 `http://localhost:8000/docs`

*(Requires valid API keys for LLM providers.)*

---

## 🧩 Example Use Cases

* M&A due diligence (risk dependencies, document contradictions).
* Financial data validation and compliance traceability.
* Legal discovery and contract intelligence.
* Enterprise document retrieval and contextual Q&A.

---

## 📘 Next Steps

* Integrate **Oracle Vector Store / OCI AI Services**.
* Expand **Graph RAG reasoning** for temporal and causal chains.
* Add **multi-tenant orchestration** for enterprise environments.

---

## 🏛️ Author

Rania Bin Taleb
AI Systems Architect | Oracle AI Modernization Specialist
🔗 https://www.linkedin.com/in/rania-bin-taleb-93199137b/ • 🧩 https://github.com/raniabt1978

---

### 💡 Summary

> A production-grade AI system engineered for **Oracle AI modernization workloads** — uniting retrieval, reasoning, and explainability into one enterprise-ready framework.

