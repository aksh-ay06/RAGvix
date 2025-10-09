"""Text chunking utilities."""

from typing import Dict, List

from ragvix.utils.logging import get_logger

logger = get_logger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 120,
    arxiv_id: str = "",
    title: str = "",
    section: str = "main",
) -> List[Dict]:
    """Chunk text using naive sliding window with character count.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Overlap characters between chunks
        arxiv_id: arXiv ID for metadata
        title: Paper title for metadata
        section: Section name for metadata
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if not text.strip():
        logger.warning("Empty text provided for chunking")
        return []
    
    text = text.strip()
    chunks = []
    
    # Calculate step size (chunk_size - overlap)
    step = max(chunk_size - overlap, 1)
    
    for i in range(0, len(text), step):
        chunk_text = text[i:i + chunk_size]
        
        # Skip very small chunks at the end
        if len(chunk_text) < chunk_size * 0.1:
            break
            
        chunk_data = {
            "text": chunk_text,
            "metadata": {
                "arxiv_id": arxiv_id,
                "title": title,
                "section": section,
                "chunk_index": len(chunks),
                "char_start": i,
                "char_end": i + len(chunk_text),
                "chunk_size": len(chunk_text),
            }
        }
        chunks.append(chunk_data)
    
    logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
    return chunks


def chunk_papers_from_metadata(
    papers_metadata: List[Dict],
    chunk_abstracts: bool = True,
    chunk_size: int = 1200,
    overlap: int = 120,
) -> List[Dict]:
    """Chunk paper abstracts from metadata.
    
    Args:
        papers_metadata: List of paper metadata dictionaries
        chunk_abstracts: Whether to chunk abstracts (True) or use whole abstracts
        chunk_size: Maximum characters per chunk
        overlap: Overlap characters between chunks
        
    Returns:
        List of chunks with metadata
    """
    all_chunks = []
    
    for paper in papers_metadata:
        if not paper.get("abstract"):
            continue
            
        if chunk_abstracts and len(paper["abstract"]) > chunk_size:
            # Chunk long abstracts
            chunks = chunk_text(
                text=paper["abstract"],
                chunk_size=chunk_size,
                overlap=overlap,
                arxiv_id=paper.get("arxiv_id", ""),
                title=paper.get("title", ""),
                section="abstract",
            )
            all_chunks.extend(chunks)
        else:
            # Use whole abstract as single chunk
            chunk_data = {
                "text": paper["abstract"],
                "metadata": {
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "title": paper.get("title", ""),
                    "section": "abstract",
                    "chunk_index": 0,
                    "char_start": 0,
                    "char_end": len(paper["abstract"]),
                    "chunk_size": len(paper["abstract"]),
                    "authors": paper.get("authors", []),
                    "categories": paper.get("categories", []),
                    "published": paper.get("published", ""),
                }
            }
            all_chunks.append(chunk_data)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(papers_metadata)} papers")
    return all_chunks