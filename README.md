# Retrieval-Augmented Generation (RAG) Pipeline with ChromaDB

## 📌 Project Overview

This project implements a **baseline Retrieval-Augmented Generation (RAG) system** that retrieves semantically relevant document chunks from a vector database and answers user queries **only when the information is grounded in the retrieved context**.

The primary goal of this repository is to demonstrate a **clear, correct, and explainable RAG foundation**, while continuously extending it with more advanced RAG concepts as learning progresses.

---

## 🧠 Key Concepts Demonstrated

- Document ingestion and preprocessing  
- Text chunking for semantic retrieval  
- Vector embeddings using OpenAI  
- Vector similarity search using cosine similarity  
- Context-grounded question answering  
- Awareness and handling of hallucination scenarios  

---

## 🏗️ High-Level Architecture

```
Documents (.txt)
      ↓
Text Loading (DirectoryLoader)
      ↓
Chunking (CharacterTextSplitter)
      ↓
Embeddings (OpenAI text-embedding-3-small)
      ↓
ChromaDB (Vector Store)
      ↓
Cosine Similarity Search (Top-k)
      ↓
Grounded Answer Generation
```

---

## 📂 Project Structure

```
rag-pipeline/
│
├── docs/                     # Knowledge base documents (.txt)
│
├── ingestion_pipeline.py     # Document loading, chunking, embedding, storage
├── retrieval_pipeline.py     # Query embedding and similarity-based retrieval
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

> Note: Local artifacts such as `venv/`, persisted vector databases, and `.env` files are intentionally excluded from version control.

---

## 🔍 Retrieval Strategy

- **Embedding Model:** `text-embedding-3-small`
- **Vector Store:** ChromaDB  
- **Similarity Metric:** Cosine similarity  
- **Top-k Retrieval:** Configurable (default: top 3 chunks)

The **same embedding model** is used for both document chunks and user queries to ensure embedding-space consistency.

---

## 🔁 Retrieval & Answer Generation Enhancements

- Implemented **LLM-based answer generation strictly grounded in retrieved context**, preventing the model from introducing external knowledge.
- Explored and integrated multiple retrieval strategies to improve answer reliability and recall:
  - **MMR (Maximal Marginal Relevance):** diversity-aware retrieval to reduce redundancy in retrieved chunks
  - **Similarity score thresholding:** rejection of low-confidence context to avoid unsupported answers
  - **Multi-query retrieval:** improved recall for ambiguous or underspecified user queries

---

## ⚠️ Hallucination Awareness

If the retrieved context **does not explicitly contain the answer**, the system responds with:

> *The information is not available in the provided documents.*

This behavior aligns with real-world RAG safety best practices.

---

## ✅ Example Behavior

| User Query | Retrieved Context | System Response |
|-----------|------------------|-----------------|
| Tesla Roadster production year | Present | Correct answer returned |
| SpaceX Pacific island lease | Not present | Not available in provided documents |

---

## 🚧 Project Status

**Ongoing / Actively Evolving**

This repository starts with a baseline RAG implementation and will be progressively extended with advanced concepts such as:
- Retrieval quality tuning  
- Similarity score thresholds  
- Re-ranking strategies  
- Evaluation pipelines  
- Advanced RAG architectures  

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/rag-pipeline-chromadb-openai.git
cd rag-pipeline-chromadb-openai
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables
Create a `.env` file based on `.env.example`:

```
OPENAI_API_KEY=your_api_key_here
```

---


## 📌 Author

Abhishek Kumar  
MSc Data Science & Analytics  

