"""FAISS vector store for similarity search."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import typer
from sentence_transformers import SentenceTransformer

from ragvix.config import settings
from ragvix.utils.io import read_jsonl, write_jsonl
from ragvix.utils.logging import get_logger

app = typer.Typer(help="FAISS index management")
logger = get_logger(__name__)


class FAISSStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize FAISS store with embedding model.
        
        Args:
            model_name: Name of sentence-transformers model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def build_index(self, chunks: List[Dict]) -> None:
        """Build FAISS index from text chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return
        
        logger.info(f"Building FAISS index from {len(chunks)} chunks")
        logger.info(f"Using model: {self.model_name} (dim={self.dimension})")
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)
        
        # Create FAISS index (Inner Product for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata = [chunk["metadata"] for chunk in chunks]
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Embed query
        query_embedding = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                result = {
                    "score": float(score),
                    "metadata": self.metadata[idx],
                }
                results.append(result)
        
        return results
    
    def save(self, index_path: Path) -> None:
        """Save index and metadata to disk.
        
        Args:
            index_path: Directory to save index files
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_file = index_path / "faiss.index"
        faiss.write_index(self.index, str(faiss_file))
        
        # Save metadata
        meta_file = index_path / "meta.jsonl"
        write_jsonl(self.metadata, meta_file)
        
        # Save config
        config = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "num_vectors": self.index.ntotal,
        }
        config_file = index_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Index saved to {index_path}")
    
    def load(self, index_path: Path) -> None:
        """Load index and metadata from disk.
        
        Args:
            index_path: Directory containing index files
        """
        # Load config
        config_file = index_path / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Verify model compatibility
        if config["model_name"] != self.model_name:
            logger.warning(f"Model mismatch: {config['model_name']} vs {self.model_name}")
        
        # Load FAISS index
        faiss_file = index_path / "faiss.index"
        self.index = faiss.read_index(str(faiss_file))
        
        # Load metadata
        meta_file = index_path / "meta.jsonl"
        self.metadata = read_jsonl(meta_file)
        
        logger.info(f"Index loaded: {len(self.metadata)} vectors")


def load_index(index_path: Optional[Path] = None) -> FAISSStore:
    """Load FAISS index from standard location.
    
    Args:
        index_path: Optional custom index path
        
    Returns:
        Loaded FAISS store
    """
    if index_path is None:
        index_path = settings.index_dir
    
    store = FAISSStore(settings.embedding_model)
    store.load(index_path)
    return store


@app.command()
def build(
    chunks: str = typer.Option("data/processed/chunks.jsonl", help="Input chunks file"),
    out: str = typer.Option("data/index", help="Output directory for index"),
) -> None:
    """Build FAISS index from chunks."""
    chunks_path = Path(chunks)
    out_path = Path(out)
    
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.info("Create chunks first or use a different path")
        return
    
    try:
        # Load chunks
        chunk_data = read_jsonl(chunks_path)
        logger.info(f"Loaded {len(chunk_data)} chunks")
        
        # Build index
        store = FAISSStore(settings.embedding_model)
        store.build_index(chunk_data)
        
        # Save index
        store.save(out_path)
        logger.info(f"Index saved to {out_path}")
        
    except Exception as e:
        logger.error(f"Failed to build index: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()