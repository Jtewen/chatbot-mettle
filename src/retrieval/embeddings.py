from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class EmbeddingManager:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = config['embeddings']['model']
        self.chunk_size = config['embeddings']['chunk_size']
        self.chunk_overlap = config['embeddings']['chunk_overlap']
    
    def create_embeddings(self, force: bool = False) -> Optional[str]:
        """Create embeddings if they don't exist or if force is True."""
        store_path = Path(self.config['retriever']['faiss_store_path'])
        pdf_path = Path(self.config['paths']['pdf_path'])
        
        if store_path.exists() and not force:
            return None
            
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_documents = text_splitter.split_documents(documents)
        
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        vectorstore = FAISS.from_documents(split_documents, embeddings)
        
        store_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(store_path))
        
        return str(store_path)