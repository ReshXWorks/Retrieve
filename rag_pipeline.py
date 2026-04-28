import os
import re
import requests
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_PATH = "vector_store"

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)

    words = text.split()
    if len(words) > 10 and all(len(w) == 1 for w in words[:15]):
        text = "".join(words)

    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    return text.strip()


def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_PATH)

    return len(chunks)


def load_db():
    if not os.path.exists(DB_PATH):
        raise Exception("Vector DB not found. Upload a PDF first.")

    return FAISS.load_local(
        DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )


def has_numbers(text):
    return bool(re.search(r'\d', text))


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / denom)


def query_rag(query):
    try:
        db = load_db()
        docs = db.similarity_search_with_score(query, k=4)

        if not docs:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "confidence": 0.0,
                "hallucination": True,
                "relevance": 0.0
            }

        top_docs = docs[:2]
        context = "\n\n".join([doc[0].page_content for doc in top_docs])

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"""
Answer ONLY from the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
""",
                "stream": False
            },
            timeout=180
        )

        data = response.json()
        answer = data.get("response", "").strip()

        if not answer:
            answer = "No response generated."

        # 🔥 Hallucination Detection (unchanged)
        if "@" in answer or has_numbers(answer):
            hallucination_flag = False
        else:
            context_words = set(context.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(context_words & answer_words) / (len(answer_words) + 1)
            hallucination_flag = overlap < 0.25

        # 🔥 Confidence Score
        scores = np.array([float(score) for _, score in top_docs])
        weights = np.exp(-scores)
        confidence = float(round(weights[0] / weights.sum(), 2))

        # 🔥 ✅ SEMANTIC RELEVANCE (FINAL FIX)
        answer_vec = embedding_model.embed_query(answer)
        context_vec = embedding_model.embed_query(context)

        relevance = cosine_similarity(answer_vec, context_vec)
        relevance = float(round(relevance, 2))

        # 🔥 Sources
        sources = []
        seen = set()

        for doc, _ in top_docs:
            text = doc.page_content[:300]
            if text not in seen:
                sources.append(text + "...")
                seen.add(text)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "hallucination": hallucination_flag,
            "relevance": relevance
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "hallucination": True,
            "relevance": 0.0
        }