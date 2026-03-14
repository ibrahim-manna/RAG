import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import TokenTextSplitter

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 0


class FAISSIndex:
    """Handles building, saving, and loading a LangChain FAISS vector store."""

    def __init__(self, vector_db_dir: str):
        self.vector_db_dir = vector_db_dir
        self.db: FAISS | None = None

    def build_from_pdfs(self, pdf_dir: str) -> FAISS:
        """Load PDFs, split into chunks, embed, save and return the FAISS store."""
        print(f"Loading PDFs from: {pdf_dir}")
        loader = DirectoryLoader(path=pdf_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} page(s) from PDF(s).")

        splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunk(s).")

        print("Generating embeddings and building FAISS index...")
        self.db = FAISS.from_documents(docs, OllamaEmbeddings(model=EMBED_MODEL))

        os.makedirs(self.vector_db_dir, exist_ok=True)
        self.db.save_local(self.vector_db_dir)
        print(f"FAISS index saved to: {self.vector_db_dir}")
        return self.db

    def load(self) -> FAISS:
        """Load a previously saved FAISS index from disk."""
        print(f"Loading existing FAISS index from: {self.vector_db_dir}")
        self.db = FAISS.load_local(
            self.vector_db_dir,
            OllamaEmbeddings(model=EMBED_MODEL),
            allow_dangerous_deserialization=True,
        )
        print("Index loaded successfully.")
        return self.db

    def index_exists(self) -> bool:
        """Return True if a saved index already exists on disk."""
        return os.path.exists(os.path.join(self.vector_db_dir, "index.faiss"))