from pathlib import Path
from typing import Optional
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class USCISRetriever:
    def __init__(self, embedding_model: str = "nomic-embed-text", 
                 store_path: str = "data/faiss_store",
                 llm = None):
        self.store_path = Path(store_path)
        self.embedding_model = embedding_model
        self.llm = llm
        self._embeddings = None
        self._vectorstore = None
    
    @property
    @lru_cache(maxsize=1)
    def embeddings(self):
        if not self._embeddings:
            self._embeddings = OllamaEmbeddings(model=self.embedding_model)
        return self._embeddings
    
    @property
    def vectorstore(self):
        if not self._vectorstore:
            if not self.store_path.exists():
                raise FileNotFoundError(
                    f"FAISS store not found at {self.store_path}. "
                    "Please run embeddings creation first."
                )
            self._vectorstore = FAISS.load_local(
                str(self.store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return self._vectorstore
    
    def get_retriever(self, use_compression: bool = True, 
                     k: int = 3, fetch_k: int = 5):
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "maximal_marginal_relevance": True
            }
        )
        
        if use_compression and self.llm:
            compressor = LLMChainExtractor.from_llm(self.llm)
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        
        return base_retriever

def initialize_retriever(config: dict, llm = None):
    retriever = USCISRetriever(
        embedding_model=config.get('embeddings', {}).get('model', "nomic-embed-text"),
        store_path=config.get('retriever', {}).get('faiss_store_path', "data/faiss_store"),
        llm=llm
    )
    return retriever.get_retriever(
        use_compression=config.get('retriever', {}).get('use_compression', True),
        k=config.get('retriever', {}).get('num_documents', 3),
        fetch_k=config.get('retriever', {}).get('fetch_k', 5)
    )
