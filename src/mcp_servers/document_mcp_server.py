"""
document_mcp_server.py

Baby Step 1 MCP server for our production MCP + A2A project.

What this server does:
1. Exposes document tools through MCP.
2. Lets an MCP client, such as MCP Inspector, list available enterprise documents.
3. Lets an MCP client read a specific document.

Important mental model:

MCP Host / Client
    Example: MCP Inspector, Claude Desktop, Cursor, VS Code Agent
        |
        | MCP protocol
        v
MCP Server
    This file: document_mcp_server.py
        |
        | Python file system access
        v
data/
    insurance/
    aws/
    security/
    mcp_a2a/

This is intentionally simple first.
No RAG yet.
No embeddings yet.
No A2A yet.

We first prove:
- server starts
- tools appear
- tools can read our corpus
"""

from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

from pypdf import PdfReader

# ---------------------------------------------------------------------
# 1. Create the MCP server object
# ---------------------------------------------------------------------
# This is the same pattern as our DataCamp PR MCP and yfinance MCP.
#
# The string name is what MCP Inspector will show as the server name.
# Think of this object as the "tool registry."
# Every @mcp.tool() function below gets registered into this server.
# ---------------------------------------------------------------------

mcp = FastMCP("production-document-mcp-server")


# ---------------------------------------------------------------------
# 2. Define where our enterprise documents live
# ---------------------------------------------------------------------
# This file is located at:
#
#   src/mcp_servers/document_mcp_server.py
#
# The project root is two folders up:
#
#   production-agents-mcp-a2a/
#
# So we use Path(__file__).resolve().parents[2]
# to reliably find the project root no matter where we run from.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------
# 3. Helper function: collect documents
# ---------------------------------------------------------------------
# This is a normal Python helper.
# It is NOT exposed directly as an MCP tool.
#
# Why?
# Because we separate:
#
#   core helper logic
#       from
#   MCP tool wrapper
#
# This avoids the problem we saw in RAG-MCP where one decorated tool
# called another decorated tool and caused weird behavior.
# ---------------------------------------------------------------------

def collect_documents() -> List[Dict]:
    """
    Walk through the data folder and collect supported documents.

    Returns a list of dictionaries like:

    {
        "name": "policy_summary_auto.pdf",
        "relative_path": "insurance/policy_summary_auto.pdf",
        "category": "insurance",
        "suffix": ".pdf",
        "size_bytes": 12345
    }
    """

    supported_suffixes = {".txt", ".md", ".pdf", ".html"}

    documents = []

    for path in DATA_DIR.rglob("*"):
        # Skip folders. We only want files.
        if not path.is_file():
            continue

        # Skip unsupported file types.
        if path.suffix.lower() not in supported_suffixes:
            continue

        # Get path relative to data/
        relative_path = path.relative_to(DATA_DIR)

        # The first folder under data/ is the category:
        # insurance, aws, security, mcp_a2a, architecture, etc.
        parts = relative_path.parts
        category = parts[0] if len(parts) > 1 else "root"

        documents.append(
            {
                "name": path.name,
                "relative_path": str(relative_path),
                "category": category,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )

    return documents


# ---------------------------------------------------------------------
# 4. MCP Tool 1: list_documents
# ---------------------------------------------------------------------
# This decorator is the key MCP concept.
#
# Without @mcp.tool():
#   list_documents is just a normal Python function.
#
# With @mcp.tool():
#   MCP Inspector / Claude Desktop / Cursor can discover and call it.
# ---------------------------------------------------------------------

#@mcp.tool()
#def list_documents() -> List[Dict]:
@mcp.tool()
def list_documents(include_sizes: bool = True) -> List[Dict]:
    """
    List all enterprise documents available in the local data corpus.

    Use this when you want to see what documents the system can access.
    """

    return collect_documents()


# ---------------------------------------------------------------------
# 5. Helper function: safely resolve a document path
# ---------------------------------------------------------------------
# We do NOT want tools reading arbitrary files on your computer.
#
# Bad:
#   read_document("../../.env")
#
# Good:
#   read_document("insurance/policy_summary_auto.pdf")
#
# This helper guarantees that requested files stay inside data/.
# ---------------------------------------------------------------------

def resolve_document_path(relative_path: str) -> Path:
    """
    Convert a user-provided relative path into a safe full path.

    Example:
        "insurance/policy_summary_auto.pdf"

    becomes:
        C:/.../production-agents-mcp-a2a/data/insurance/policy_summary_auto.pdf
    """

    requested_path = (DATA_DIR / relative_path).resolve()

    # Security check:
    # Make sure the resolved path is still inside DATA_DIR.
    # This prevents path traversal attacks.
    if not str(requested_path).startswith(str(DATA_DIR.resolve())):
        raise ValueError("Invalid path. Access outside data directory is not allowed.")

    if not requested_path.exists():
        raise FileNotFoundError(f"Document not found: {relative_path}")

    if not requested_path.is_file():
        raise ValueError(f"Path is not a file: {relative_path}")

    return requested_path


# ---------------------------------------------------------------------
# 6. MCP Tool 2: read_text_document
# ---------------------------------------------------------------------
# First baby version reads text-like files:
# .txt, .md, .html
#
# We will add PDF parsing in the next baby step.
# ---------------------------------------------------------------------

@mcp.tool()
def read_text_document(relative_path: str, max_chars: int = 4000) -> Dict:
    """
    Read a text, markdown, or HTML document from the data corpus.

    Arguments:
        relative_path:
            Path relative to the data folder.
            Example: "mcp_a2a/bhatti_mcp_a2a_production_agents.html"

        max_chars:
            Maximum number of characters to return.
            This prevents accidentally dumping a huge document into the client.

    Returns:
        A dictionary with document metadata and content preview.
    """

    path = resolve_document_path(relative_path)

    if path.suffix.lower() not in {".txt", ".md", ".html"}:
        return {
            "status": "unsupported_type",
            "message": "This tool only reads .txt, .md, and .html files for now. PDF support comes next.",
            "relative_path": relative_path,
            "suffix": path.suffix.lower(),
        }

    text = path.read_text(encoding="utf-8", errors="ignore")

    return {
        "status": "success",
        "relative_path": relative_path,
        "name": path.name,
        "suffix": path.suffix.lower(),
        "total_chars": len(text),
        "returned_chars": min(len(text), max_chars),
        "content_preview": text[:max_chars],
    }

@mcp.tool()
def read_pdf_document(
    relative_path: str,
    max_pages: int = 3,
    max_chars: int = 6000,
) -> Dict:
    """
    Read text from a PDF document inside the data corpus.

    Baby-step purpose:
    - We already proved MCP can list documents.
    - Now we prove MCP can extract text from realistic PDFs.
    - This becomes the foundation for RAG ingestion later.

    Arguments:
        relative_path:
            Path relative to the data folder.
            Example:
                insurance/policy_summary_auto.pdf

        max_pages:
            Maximum number of PDF pages to read.
            We keep this small so Inspector does not get overloaded.

        max_chars:
            Maximum number of characters returned in content_preview.
            This prevents huge outputs from freezing the UI.
    """

    # Reuse our safe path resolver.
    # This prevents reading files outside data/.
    path = resolve_document_path(relative_path)

    # Only allow PDFs in this tool.
    if path.suffix.lower() != ".pdf":
        return {
            "status": "unsupported_type",
            "message": "read_pdf_document only supports .pdf files.",
            "relative_path": relative_path,
            "suffix": path.suffix.lower(),
        }

    # Open the PDF.
    reader = PdfReader(str(path))

    total_pages = len(reader.pages)

    # Never read more pages than the PDF actually has.
    pages_to_read = min(max_pages, total_pages)

    extracted_pages = []
    combined_text = ""

    # Extract text page by page.
    for page_index in range(pages_to_read):
        page = reader.pages[page_index]

        # pypdf returns None if it cannot extract text.
        page_text = page.extract_text() or ""

        page_number = page_index + 1

        extracted_pages.append(
            {
                "page_number": page_number,
                "chars": len(page_text),
                "text_preview": page_text[:max_chars],
            }
        )

        combined_text += f"\n\n--- PAGE {page_number} ---\n\n{page_text}"

    return {
        "status": "success",
        "relative_path": relative_path,
        "name": path.name,
        "total_pages": total_pages,
        "pages_read": pages_to_read,
        "total_extracted_chars": len(combined_text),
        "returned_chars": min(len(combined_text), max_chars),
        "content_preview": combined_text[:max_chars],
        "pages": extracted_pages,
    }
# ---------------------------------------------------------------------
# 7. MCP Tool 3: get_corpus_summary
# ---------------------------------------------------------------------
# This gives us a quick overview:
# How many docs are in insurance, aws, security, etc.
# Very useful for testing and for demos.
# ---------------------------------------------------------------------

#@mcp.tool()
#def get_corpus_summary() -> Dict:
@mcp.tool()
def get_corpus_summary(include_details: bool = True) -> Dict:
    """
    Summarize the enterprise document corpus by category and file type.
    """

    docs = collect_documents()

    by_category = {}
    by_suffix = {}

    for doc in docs:
        category = doc["category"]
        suffix = doc["suffix"]

        by_category[category] = by_category.get(category, 0) + 1
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1

    return {
        "data_dir": str(DATA_DIR),
        "total_documents": len(docs),
        "by_category": by_category,
        "by_suffix": by_suffix,
    }


# ---------------------------------------------------------------------
# 8. Start the MCP server
# ---------------------------------------------------------------------
# When you run:
#
#   python src\mcp_servers\document_mcp_server.py
#
# this starts a stdio MCP server.
#
# It may look like it is hanging.
# That is normal.
# It is waiting for an MCP client.
#
# We usually test with:
#
#   mcp dev src\mcp_servers\document_mcp_server.py
# ---------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()