# 📄 Retrieve: AI-Powered Document Assistant (RAG)

An AI-powered web application that allows users to upload PDF documents and ask questions based on their content using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

* 📂 Upload PDF documents
* ✂️ Automatic text chunking & preprocessing
* 🔍 Semantic search using vector embeddings (FAISS)
* 🤖 LLM-based answer generation (via local model using Ollama)
* 📊 Confidence scoring for retrieved answers
* ⚠️ Hallucination detection (rule-based)
* 📈 Semantic relevance evaluation
* 📑 Source chunk display for answer transparency
* 💬 Chat-based interface with session history

---

## 🧠 How It Works

1. **Document Upload**

   * PDFs are parsed and split into smaller chunks.

2. **Embedding & Storage**

   * Text chunks are converted into vector embeddings using HuggingFace models.
   * Stored locally using FAISS vector database.

3. **Query Processing**

   * User query is embedded and matched against stored vectors.
   * Top relevant chunks are retrieved.

4. **Answer Generation**

   * Retrieved context is passed to a local LLM (Mistral via Ollama).
   * Model generates a context-aware answer.

5. **Evaluation Layer**

   * Confidence score (softmax over similarity scores)
   * Hallucination detection (keyword + numeric checks)
   * Semantic relevance (embedding similarity)

---

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI

### Frontend

* Streamlit

### AI / ML

* RAG (Retrieval-Augmented Generation)
* HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* FAISS (Vector Database)
* Ollama (Local LLM - Mistral)

---

## 📦 Project Structure

```bash
.
├── main.py             # FastAPI backend
├── app.py              # Streamlit frontend
├── rag_pipeline.py     # Core RAG logic
├── vector_store/       # FAISS index
├── data/               # Uploaded PDFs
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
```

---

### 3️⃣ Install Dependencies

```bash
pip install fastapi uvicorn streamlit langchain faiss-cpu pypdf sentence-transformers requests
```

---

### 4️⃣ Run Ollama (Local LLM)

Make sure Ollama is installed and running:

```bash
ollama run mistral
```

---

### 5️⃣ Start Backend

```bash
uvicorn main:app --reload
```

---

### 6️⃣ Start Frontend

```bash
streamlit run app.py
```

---


## 📊 Example Output

<img width="1600" height="755" alt="image" src="https://github.com/user-attachments/assets/3360f7fa-d9f3-4211-8a50-31419986fbfc" />

---

## 🎯 NOTE:

* Implemented end-to-end RAG pipeline
* Used local LLM (no paid APIs)
* Includes evaluation metrics 
* Provides explainability via source chunks

---



