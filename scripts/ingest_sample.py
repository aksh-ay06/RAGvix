#!/usr/bin/env python3
"""Sample ingest script for RAGvix."""

import subprocess
import sys

from ragvix.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run sample arXiv metadata ingestion."""
    logger.info("Running sample arXiv ingestion...")
    
    try:
        # Run the ingestion command
        cmd = [
            "ragvix-ingest",
            "--category", "cs.CL",
            "--max-papers", "50",
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        logger.info("✅ Sample ingestion completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ingestion failed: {e}")
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        logger.error("❌ ragvix-ingest command not found. Install the package first:")
        logger.info("   uv pip install -e . (or pip install -e .)")
        sys.exit(1)


if __name__ == "__main__":
    main()