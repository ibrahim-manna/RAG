import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vector_store.faiss_index import FAISSIndex
from rag.rag_model import RAGModel

# ── Configuration ──────────────────────────────────────────────────────────────
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "DocStore")
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "VectorDB", "nomic", "webMD")

QUESTIONS = [
    "What is the premier league?",
    "Who has the most wins in the premier league?",
    "Who has won the games at most different stadiums?",
]


def main():
    faiss_index = FAISSIndex(VECTOR_DB_DIR)

    if faiss_index.index_exists():
        db = faiss_index.load()
    else:
        db = faiss_index.build_from_pdfs(PDF_DIR)

    rag = RAGModel(db)

    for question in QUESTIONS:
        print(f"\nQuestion: {question}")
        print("Answer:", rag.ask(question))


if __name__ == "__main__":
    main()