"""I/O utilities for RAGvix."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def write_jsonl(data: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
    """Write data to JSONL format.
    
    Args:
        data: List of dictionaries to write
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read data from JSONL format.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of dictionaries from JSONL file
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def safe_path_creation(path: Union[str, Path]) -> Path:
    """Safely create directory path if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path