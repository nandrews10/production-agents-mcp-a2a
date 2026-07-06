from pathlib import Path

# Project root:
# production-agents-mcp-a2a/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw enterprise documents live here
DATA_DIR = PROJECT_ROOT / "data"

# Persistent Chroma DB will live here
CHROMA_DIR = PROJECT_ROOT / "vectorstore" / "chroma"

# Collection name inside Chroma
COLLECTION_NAME = "enterprise_docs"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150