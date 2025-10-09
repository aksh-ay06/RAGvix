# RAGvix

**RAGvix – a tiny end-to-end RAG over arXiv (Week-1 MVP)**

A minimal Retrieval-Augmented Generation system for searching and retrieving relevant academic papers from arXiv. This Week-1 implementation focuses on building a solid foundation with clean architecture and working retrieval capabilities.

## Architecture

```
arXiv API → Metadata → Parse PDFs → Chunk Text → Embed → FAISS Index → Retrieve → (Generation - Week 2)
     ↓           ↓          ↓           ↓         ↓          ↓           ↓
  ingest/   parsing/   index/     index/    index/    retriever/     rag/
```

## Quickstart

```bash
# Setup environment
uv venv && uv pip install -e .
# OR fallback: python -m venv venv && source venv/bin/activate && pip install -e .

# Run sample workflows
python scripts/ingest_sample.py    # Fetch arXiv metadata
python scripts/build_index.py      # Build FAISS index (if chunks exist)
python scripts/query.py           # Search "diffusion models"
```

## What Works Now (Day-1 Scope)

- ✅ **Metadata Ingestion**: Fetch arXiv paper metadata via API
- ✅ **Text Chunking**: Naive sliding window chunker for text processing
- ✅ **Vector Embeddings**: sentence-transformers (MiniLM-L6-v2) 
- ✅ **FAISS Indexing**: Fast similarity search with persistence
- ✅ **Retrieval**: Query → embed → search → rank results
- ✅ **CLI Tools**: Typer-based commands for each stage
- 🔧 **PDF Parsing**: Stub implementation (PyMuPDF ready)
- 🔧 **Generation**: Pipeline stub (retrieval-only for now)

## Project Structure

```
RAGvix/
├── src/ragvix/           # Core package
│   ├── ingest/          # arXiv API client
│   ├── parsing/         # PDF → text (stub)
│   ├── index/           # Chunking + FAISS
│   ├── retriever/       # Search interface
│   ├── rag/             # End-to-end pipeline (stub)
│   └── eval/            # Retrieval evaluation
├── scripts/             # Workflow scripts
├── data/               # Raw → processed → index
└── notebooks/          # Exploration
```

## CLI Commands

```bash
# Individual steps
ragvix-ingest fetch --category cs.CL --max-papers 50
ragvix-build-index build --chunks data/processed/chunks.jsonl
ragvix-query search --query "attention mechanisms" --k 5

# Development
make setup lint test
```

## Next Steps (Week-2)

- [ ] Wire PDF parsing into full pipeline
- [ ] Add LLM generation (OpenAI/local)
- [ ] Implement proper evaluation metrics
- [ ] Add more sophisticated chunking strategies
- [ ] Web interface or better CLI UX
- [ ] Production deployment configs

## License

MIT License - see [LICENSE](LICENSE) for details.