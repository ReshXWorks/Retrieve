from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from rag_pipeline import process_pdf, query_rag

app = FastAPI()

# 🔥 Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = process_pdf(file_path)

        return {
            "status": "success",
            "message": f"Processed {chunks} chunks"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/query")
def ask_question(q: str):
    try:
        result = query_rag(q)
        return result

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "hallucination": True,
            "relevance": 0.0
        }