"""Configuration management for RAGvix."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables."""
    
    # arXiv settings
    arxiv_category: str = os.getenv("ARXIV_CATEGORY", "cs.CL")
    max_papers: int = int(os.getenv("MAX_PAPERS", "100"))
    
    # API keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Data directories
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/index")
    
    # Processing settings
    chunk_size: int = 1200
    chunk_overlap: int = 120
    embedding_model: str = "all-MiniLM-L6-v2"
    
    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        for dir_path in [self.raw_dir, self.interim_dir, self.processed_dir, self.index_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()