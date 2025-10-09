"""arXiv API client for fetching paper metadata."""

from pathlib import Path
from typing import Dict, List, Optional

import arxiv
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from ragvix.config import settings
from ragvix.utils.io import write_jsonl
from ragvix.utils.logging import get_logger

app = typer.Typer(help="arXiv metadata ingestion")
logger = get_logger(__name__)


def fetch_arxiv_metadata(
    category: str = "cs.CL",
    max_papers: int = 100,
) -> List[Dict]:
    """Fetch metadata from arXiv API.
    
    Args:
        category: arXiv category to search
        max_papers: Maximum number of papers to fetch
        
    Returns:
        List of paper metadata dictionaries
    """
    logger.info(f"Fetching up to {max_papers} papers from category: {category}")
    
    # Create search query
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    papers = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Fetching papers...", total=None)
        
        for result in search.results():
            paper_data = {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "published": result.published.isoformat(),
            }
            papers.append(paper_data)
            
            if len(papers) % 10 == 0:
                progress.update(task, description=f"Fetched {len(papers)} papers...")
    
    logger.info(f"Successfully fetched {len(papers)} papers")
    return papers


def download_pdf_stub(pdf_url: str, output_path: Path) -> bool:
    """Stub function for PDF download (non-fatal).
    
    Args:
        pdf_url: URL to PDF file
        output_path: Local path to save PDF
        
    Returns:
        Success status (currently always False - stub)
    """
    # TODO: Implement actual PDF download
    logger.info(f"PDF download stub called for: {pdf_url}")
    logger.info(f"Would save to: {output_path}")
    return False


@app.command()
def fetch(
    category: str = typer.Option("cs.CL", help="arXiv category to search"),
    max_papers: int = typer.Option(50, help="Maximum number of papers to fetch"),
    out: Optional[str] = typer.Option(None, help="Output file path"),
) -> None:
    """Fetch arXiv metadata and save to JSONL."""
    if out is None:
        out = settings.raw_dir / "metadata.jsonl"
    else:
        out = Path(out)
    
    # Ensure output directory exists
    settings.ensure_dirs()
    
    # Fetch metadata
    try:
        papers = fetch_arxiv_metadata(category=category, max_papers=max_papers)
        
        # Write to file
        write_jsonl(papers, out)
        logger.info(f"Saved {len(papers)} papers to {out}")
        
        # Log summary statistics
        categories = set()
        for paper in papers:
            categories.update(paper["categories"])
        
        logger.info(f"Categories found: {', '.join(sorted(categories))}")
        logger.info(f"Date range: {papers[-1]['published'][:10]} to {papers[0]['published'][:10]}")
        
    except Exception as e:
        logger.error(f"Failed to fetch papers: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()