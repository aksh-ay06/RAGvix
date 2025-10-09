"""End-to-end RAG pipeline (stub implementation)."""

from typing import Dict, List, Optional
from pathlib import Path

from ragvix.retriever.retriever import Retriever
from ragvix.utils.logging import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline (Week-1 stub)."""
    
    def __init__(self, index_path: Optional[Path] = None):
        """Initialize RAG pipeline.
        
        Args:
            index_path: Path to FAISS index directory
        """
        self.retriever = Retriever(index_path)
        self.retriever.load_index()
        logger.info("RAG pipeline initialized (retrieval-only mode)")
    
    def answer(self, query: str, k: int = 5) -> Dict:
        """Answer a query using RAG (currently retrieval-only).
        
        Args:
            query: User question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with retrieval results (no generation yet)
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents
        search_results = self.retriever.search_with_context(query, k=k)
        
        # TODO: Add generation step in Week-2
        # For now, return retrieval results with a stub answer
        response = {
            "query": query,
            "answer": "⚠️ Generation not implemented yet (Week-1 scope). See retrieved documents below.",
            "sources": search_results["results"],
            "retrieval_stats": search_results["index_stats"],
            "mode": "retrieval_only",
        }
        
        logger.info(f"Retrieved {len(search_results['results'])} relevant documents")
        return response
    
    def batch_answer(self, queries: List[str], k: int = 5) -> List[Dict]:
        """Answer multiple queries.
        
        Args:
            queries: List of questions
            k: Number of documents to retrieve per query
            
        Returns:
            List of answer dictionaries
        """
        return [self.answer(query, k=k) for query in queries]


def answer(query: str, index_path: Optional[Path] = None, k: int = 5) -> Dict:
    """Convenience function for single query RAG.
    
    Args:
        query: User question
        index_path: Optional custom index path
        k: Number of documents to retrieve
        
    Returns:
        Answer dictionary
    """
    pipeline = RAGPipeline(index_path)
    return pipeline.answer(query, k=k)