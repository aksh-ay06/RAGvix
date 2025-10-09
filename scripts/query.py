#!/usr/bin/env python3
"""Query script for RAGvix."""

import subprocess
import sys
from pathlib import Path

from ragvix.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run sample query against the index."""
    logger.info("Running sample query...")
    
    try:
                # Use the actual vector search
        from ragvix.retriever.retriever import Retriever
        
        # Check if index exists
        index_file = Path("data/index/config.json")
        if not index_file.exists():
            logger.error("❌ FAISS index not found. Run build_index.py first.")
            return
        
        # Initialize retriever and perform search
        retriever = Retriever()
        query = "diffusion models"
        results = retriever.search(query, k=5)
        
        print(f"\n🔍 Vector Search: '{query}'\n")
        
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
        
        logger.info("✅ Vector search completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Query failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()