"""Retrieval interface for RAGvix."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import typer

from ragvix.config import settings
from ragvix.index.faiss_store import FAISSStore, load_index
from ragvix.utils.logging import get_logger

app = typer.Typer(help="Vector search and retrieval")
logger = get_logger(__name__)


class Retriever:
    """Main retrieval interface."""
    
    def __init__(self, index_path: Optional[Path] = None):
        """Initialize retriever with FAISS store.
        
        Args:
            index_path: Path to FAISS index directory
        """
        self.index_path = index_path or settings.index_dir
        self.store: Optional[FAISSStore] = None
    
    def load_index(self) -> None:
        """Load the FAISS index."""
        if not (self.index_path / "config.json").exists():
            raise FileNotFoundError(f"No index found at {self.index_path}")
        
        self.store = load_index(self.index_path)
        logger.info(f"Loaded index with {len(self.store.metadata)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if self.store is None:
            self.load_index()
        
        results = self.store.search(query, k=k)
        
        # Enhance results with formatted output
        for result in results:
            meta = result["metadata"]
            result["formatted"] = {
                "title": meta.get("title", "Unknown Title"),
                "arxiv_id": meta.get("arxiv_id", "unknown"),
                "section": meta.get("section", "unknown"),
                "score": f"{result['score']:.4f}",
            }
        
        return results
    
    def search_with_context(self, query: str, k: int = 5) -> Dict:
        """Search with additional context information.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            Dictionary with results and metadata
        """
        results = self.search(query, k=k)
        
        return {
            "query": query,
            "num_results": len(results),
            "results": results,
            "index_stats": {
                "total_documents": len(self.store.metadata) if self.store else 0,
                "model": settings.embedding_model,
            }
        }


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, help="Number of results to return"),
    index_path: Optional[str] = typer.Option(None, help="Path to index directory"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search for relevant documents."""
    try:
        # Initialize retriever
        retriever = Retriever(Path(index_path) if index_path else None)
        
        # Perform search
        if json_output:
            results = retriever.search_with_context(query, k=k)
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            results = retriever.search(query, k=k)
            
            print(f"\n🔍 Search: '{query}'\n")
            
            if not results:
                print("No results found.")
                return
            
            for i, result in enumerate(results, 1):
                fmt = result["formatted"]
                print(f"{i}. {fmt['title']}")
                print(f"   arXiv: {fmt['arxiv_id']} | Section: {fmt['section']} | Score: {fmt['score']}")
                
                # Show snippet of text
                text = result["metadata"].get("text", "")[:200]
                if len(text) > 0:
                    print(f"   {text}{'...' if len(result['metadata'].get('text', '')) > 200 else ''}")
                print()
        
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Build an index first using: ragvix-build-index build")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()