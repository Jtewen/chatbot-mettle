from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class DocumentLoader:
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=embedding_model)
    
    def load_pdf(self, pdf_path: Path) -> list:
        loader = PyPDFLoader(str(pdf_path))
        return loader.load()
    
    def split_documents(self, documents: list, chunk_size: int = 1024, 
                       chunk_overlap: int = 128) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)
    
    def create_vectorstore(self, documents: list, save_path: Path) -> FAISS:
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_path))
        return vectorstore
    
    @staticmethod
    def load_vectorstore(path: Path, embeddings) -> Optional[FAISS]:
        if path.exists():
            return FAISS.load_local(str(path), embeddings, 
                                  allow_dangerous_deserialization=True)
        return None

def check_and_create_embeddings(pdf_path: Path, faiss_path: Path) -> None:
    """Check if FAISS store exists, create if it doesn't."""
    loader = DocumentLoader()
    
    if not faiss_path.exists():
        documents = loader.load_pdf(pdf_path)
        split_docs = loader.split_documents(documents)
        loader.create_vectorstore(split_docs, faiss_path)