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
import contextlib
import os
import chromadb

import json
import subprocess
import sys

from pathlib import Path
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

from pypdf import PdfReader
from bs4 import BeautifulSoup

#from src.rag.retrieve_chroma import search_chroma


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

import sys

sys.path.append(str(PROJECT_ROOT))

from src.rag.retrieve_chroma import search_chroma
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
# 4. Helper function: extract_text_from_file
# ---------------------------------------------------------------------
# Extract text from a supported document.

#    Supports:
#    - .txt
#   - .md
#   - .html
#   - .pdf

# This is plain helper logic, not an MCP tool.
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
def extract_text_from_file(path: Path, max_pages: int = 10) -> str:
    """
    Extract text from a supported document.

    Supports:
    - .txt
    - .md
    - .html
    - .pdf

    This is plain helper logic, not an MCP tool.
    """

    suffix = path.suffix.lower()

    # if suffix in {".txt", ".md", ".html"}:
    #     return path.read_text(encoding="utf-8", errors="ignore")
    # Plain text and markdown
    if suffix in {".txt", ".md"}:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    # HTML needs cleaning before returning
    if suffix == ".html":

        raw_html = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        return clean_html_text(raw_html)

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages_to_read = min(max_pages, len(reader.pages))

        text_parts = []

        for page_index in range(pages_to_read):
            page_text = reader.pages[page_index].extract_text() or ""
            text_parts.append(f"\n--- PAGE {page_index + 1} ---\n{page_text}")

        return "\n".join(text_parts)

    return ""


def simple_keyword_score(text: str, query: str) -> int:
    """
    Very simple keyword score.

    Counts how many query words appear in the document text.
    This is NOT semantic search yet.
    It is our baby-step baseline.
    """

    text_lower = text.lower()
    query_terms = [t.strip().lower() for t in query.split() if t.strip()]

    score = 0

    for term in query_terms:
        if term in text_lower:
            score += text_lower.count(term)

    return score

def keyword_score_all_terms(text: str, query: str) -> int:
    """
    Better keyword scoring.

    Requires ALL query words to appear.
    So "collision deductible" only matches docs containing both words.
    """

    text_lower = text.lower()
    query_terms = [t.strip().lower() for t in query.split() if t.strip()]

    if not query_terms:
        return 0

    # Require every query term to appear at least once.
    if not all(term in text_lower for term in query_terms):
        return 0

    # Score by total frequency once all terms are present.
    return sum(text_lower.count(term) for term in query_terms)

def clean_html_text(raw_html: str) -> str:
    """
    Convert raw HTML into readable article text.

    Removes:
    - scripts
    - styles
    - HTML tags
    - extra whitespace
    """

    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)
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
    
# @mcp.tool()
# def search_documents(
#     query: str,
#     top_k: int = 5,
#     max_pages_per_pdf: int = 10,
#     preview_chars: int = 800,
# ) -> List[Dict]:
#     """
#     Search all documents in the enterprise corpus using simple keyword matching.

#     Baby-step purpose:
#     - This is our baseline search before embeddings.
#     - Later we compare this with vector search and hybrid retrieval.
#     """

#     docs = collect_documents()
#     results = []

#     for doc in docs:
#         relative_path = doc["relative_path"]
#         path = resolve_document_path(relative_path)

#         text = extract_text_from_file(path, max_pages=max_pages_per_pdf)

#         if not text.strip():
#             continue

#         score = simple_keyword_score(text, query)

#         if score <= 0:
#             continue

#         query_lower = query.lower()
#         text_lower = text.lower()

#         first_term = query.split()[0].lower() if query.split() else ""
#         hit_index = text_lower.find(first_term) if first_term else 0

#         if hit_index < 0:
#             hit_index = 0

#         start = max(hit_index - 200, 0)
#         end = min(start + preview_chars, len(text))

#         results.append(
#             {
#                 "relative_path": relative_path,
#                 "category": doc["category"],
#                 "suffix": doc["suffix"],
#                 "score": score,
#                 "preview": text[start:end],
#             }
#         )

#     results = sorted(results, key=lambda x: x["score"], reverse=True)

#     return results[:top_k]

@mcp.tool()
def search_documents(
    query: str,
    top_k: int = 5,
    max_pages_per_pdf: int = 5,
    preview_chars: int = 800,
    max_text_chars_per_doc: int = 20000,
) -> List[Dict]:
    """
    Search all documents using simple keyword matching.

    Production lesson:
    One bad document should NOT crash the whole server.
    So every document is processed inside try/except.
    """

    docs = collect_documents()
    results = []

    for doc in docs:
        relative_path = doc["relative_path"]

        try:
            path = resolve_document_path(relative_path)

            text = extract_text_from_file(
                path,
                max_pages=max_pages_per_pdf,
            )

            # Important safety limit:
            # HTML files can be huge because they contain scripts/styles.
            text = text[:max_text_chars_per_doc]

            if not text.strip():
                continue

            #score = simple_keyword_score(text, query)
            score = keyword_score_all_terms(text, query)

            if score <= 0:
                continue

            query_terms = [t.lower() for t in query.split() if t.strip()]
            text_lower = text.lower()

            hit_index = -1
            for term in query_terms:
                hit_index = text_lower.find(term)
                if hit_index >= 0:
                    break

            if hit_index < 0:
                hit_index = 0

            start = max(hit_index - 200, 0)
            end = min(start + preview_chars, len(text))

            results.append(
                {
                    "relative_path": relative_path,
                    "category": doc["category"],
                    "suffix": doc["suffix"],
                    "score": score,
                    "preview": text[start:end],
                }
            )

        # except Exception as e:
        #     # Do not crash MCP server.
        #     results.append(
        #         {
        #             "relative_path": relative_path,
        #             "category": doc.get("category"),
        #             "suffix": doc.get("suffix"),
        #             "score": -1,
        #             "error": str(e),
        #             "preview": "",
        #         }
        #     )
        except Exception as e:
             # Do not crash MCP server.
             # But also do NOT include broken documents in normal search results.
             # Later we will add a separate diagnostics tool for bad/corrupt files.
             continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]

# @mcp.tool()
# def semantic_search_documents(
#     query: str,
#     top_k: int = 5,
#     preview_chars: int = 800,
# ) -> List[Dict]:
#     """
#     Search enterprise documents using Chroma semantic vector search.

#     This searches meaning, not just exact keywords.
#     """

#     results = search_chroma(
#         query=query,
#         top_k=top_k,
#     )

#     documents = results["documents"][0]
#     metadatas = results["metadatas"][0]
#     distances = results["distances"][0]

#     output = []

#     for i, document in enumerate(documents):
#         metadata = metadatas[i]
#         distance = distances[i]

#         output.append(
#             {
#                 "rank": i + 1,
#                 "relative_path": metadata["source"],
#                 "category": metadata["category"],
#                 "chunk": metadata["chunk"],
#                 "distance": distance,
#                 "preview": document[:preview_chars],
#             }
#         )

#     return output
# @mcp.tool()
# def semantic_search_documents(
#     query: str,
#     top_k: int = 5,
#     preview_chars: int = 800,
# ) -> List[Dict]:
#     """
#     Temporary smoke test.
#     """

#     return [
#         {
#             "status": "entered_tool",
#             "query": query,
#             "top_k": top_k,
#         }
#     ]

# 

@mcp.tool()
def semantic_search_documents(
    query: str,
    top_k: int = 5,
    preview_chars: int = 800,
) -> List[Dict]:
    """
    Search enterprise documents using semantic vector search.

    Important architecture:
    - We do NOT run HuggingFace/Chroma semantic query directly inside MCP.
    - That was timing out in MCP stdio.
    - Instead, MCP launches a separate Python subprocess.
    - The subprocess runs semantic_search_cli.py.
    - The subprocess prints JSON.
    - MCP reads that JSON and returns it.
    """
    output_file = PROJECT_ROOT / "vectorstore" / "semantic_search_result.json"
    command = [
        sys.executable,
        "-m",
        "src.rag.semantic_search_cli",
        "--query",
        query,
        "--top-k",
        str(top_k),
        "--preview-chars",
        str(preview_chars),
        "--output-file",
        str(output_file),
    ]

    # completed = subprocess.run(
    #     command,
    #     cwd=str(PROJECT_ROOT),
    #     capture_output=True,
    #     text=True,
    #     timeout=60,
    # )
    
    env = os.environ.copy()

    # Force HuggingFace / Transformers to use local cache.
    # This avoids slow network checks inside the MCP subprocess.
    #env["HF_HUB_OFFLINE"] = "1"
    #env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["NO_COLOR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # completed = subprocess.run(
    #     command,
    #     cwd=str(PROJECT_ROOT),
    #     capture_output=True,
    #     text=True,
    #     timeout=180,
    #     env=env,
    # ) 
    
    completed = subprocess.run(
    command,
    cwd=str(PROJECT_ROOT),
    capture_output=True,
    text=True,
    timeout=180,
    env=env,
    stdin=subprocess.DEVNULL,
)  
    
    # ------------------------------------------------------------
    # Make stdout/stderr safe.
    #
    # Sometimes subprocess returns None instead of an empty string.
    # We convert None into "" so we can safely search it.
    # ------------------------------------------------------------

    if completed.returncode != 0:
        return [
            {
                "status": "error",
                "message": "Semantic search subprocess failed.",
                "stderr": completed.stderr or "",
                "stdout": completed.stdout or "",
            }
        ]

    if not output_file.exists():
        return [
            {
                "status": "error",
                "message": "Semantic search output file was not created.",
                "stderr": completed.stderr or "",
                "stdout": completed.stdout or "",
            }
        ]

    json_text = output_file.read_text(
        encoding="utf-8",
    )

    return json.loads(json_text)

@mcp.tool()
def hybrid_search_documents(
    query: str,
    top_k: int = 5,
    preview_chars: int = 800,
) -> List[Dict]:
    """
    Hybrid search = keyword search + semantic search.

    Why hybrid?
    - Keyword search is good for exact terms:
        policy numbers, names, codes, IDs, acronyms.
    - Semantic search is good for meaning:
        "what should we do during an incident?"
        "how are agents deployed?"
    - Hybrid combines both.
    """

    keyword_results = search_documents(
        query=query,
        top_k=top_k,
        preview_chars=preview_chars,
    )

    semantic_results = semantic_search_documents(
        query=query,
        top_k=top_k,
        preview_chars=preview_chars,
    )

    merged = {}

    for item in keyword_results:
        key = item["relative_path"]

        merged[key] = {
            "relative_path": item["relative_path"],
            "category": item.get("category"),
            "preview": item.get("preview", ""),
            "keyword_score": item.get("score", 0),
            "semantic_distance": None,
            "semantic_rank": None,
            "sources": ["keyword"],
        }

    for item in semantic_results:
        key = item["relative_path"]

        if key not in merged:
            merged[key] = {
                "relative_path": item["relative_path"],
                "category": item.get("category"),
                "preview": item.get("preview", ""),
                "keyword_score": 0,
                "semantic_distance": item.get("distance"),
                "semantic_rank": item.get("rank"),
                "sources": ["semantic"],
            }
        else:
            merged[key]["semantic_distance"] = item.get("distance")
            merged[key]["semantic_rank"] = item.get("rank")
            merged[key]["sources"].append("semantic")

    final_results = []

    for item in merged.values():
        keyword_score = item["keyword_score"]
        semantic_rank = item["semantic_rank"]

        keyword_component = min(keyword_score, 10) / 10

        if semantic_rank is None:
            semantic_component = 0
        else:
            semantic_component = 1 / semantic_rank

        hybrid_score = keyword_component + semantic_component

        item["hybrid_score"] = hybrid_score

        final_results.append(item)

    final_results = sorted(
        final_results,
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    return final_results[:top_k]

@mcp.tool()
def reranked_hybrid_search_documents(
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
    preview_chars: int = 1200,
) -> List[Dict]:
    """
    Hybrid search + reranking.

    Production RAG pattern:

    1. Retrieve broad candidates using hybrid search.
    2. Save those candidates to a JSON file.
    3. Launch reranker as a subprocess.
    4. Reranker scores query/chunk pairs.
    5. MCP reads reranked JSON and returns best results.

    Why candidate_k > top_k?
    - Retrieval should cast a wider net.
    - Reranking then selects the best final chunks.
    """

    candidates = hybrid_search_documents(
        query=query,
        top_k=candidate_k,
        preview_chars=preview_chars,
    )

    input_file = PROJECT_ROOT / "vectorstore" / "rerank_candidates.json"
    output_file = PROJECT_ROOT / "vectorstore" / "reranked_result.json"

    input_file.write_text(
        json.dumps(
            candidates,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        "src.rag.rerank_cli",
        "--query",
        query,
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--top-k",
        str(top_k),
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=240,
        stdin=subprocess.DEVNULL,
    )

    if completed.returncode != 0:
        return [
            {
                "status": "error",
                "message": "Reranker subprocess failed.",
                "stderr": completed.stderr or "",
                "stdout": completed.stdout or "",
            }
        ]

    if not output_file.exists():
        return [
            {
                "status": "error",
                "message": "Reranker output file was not created.",
                "stderr": completed.stderr or "",
                "stdout": completed.stdout or "",
            }
        ]

    json_text = output_file.read_text(
        encoding="utf-8",
    )

    return json.loads(json_text)

#Check if MCP subprocess works generically
@mcp.tool()
def test_subprocess_python() -> Dict:
    """
    Diagnostic tool.

    Checks whether the MCP server can launch a simple Python subprocess.
    """

    command = [
        sys.executable,
        "-c",
        "print('hello from child python')",
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    
@mcp.tool()
def test_subprocess_import_semantic_cli() -> Dict:
    """
    Diagnostic tool.

    Checks whether a child Python process launched by MCP
    can import our semantic_search_cli module.
    """

    command = [
        sys.executable,
        "-c",
        "import src.rag.semantic_search_cli; print('semantic cli import ok')",
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    
@mcp.tool()
def test_subprocess_embedding() -> Dict:
    """
    Diagnostic tool.

    Checks whether a child Python process launched by MCP
    can load the embedding model and create one query embedding.
    """

    command = [
        sys.executable,
        "-m",
        "src.rag.debug_embedding",
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=90,
        stdin=subprocess.DEVNULL,
    )

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    
    
@mcp.tool()
def test_subprocess_query_by_vector() -> Dict:
    """
    Diagnostic tool.

    Runs the full query-by-vector test in a child Python process.
    This separates:
    - embedding creation
    - Chroma opening
    - Chroma query
    """

    command = [
        sys.executable,
        "-m",
        "src.rag.debug_query_by_vector",
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

@mcp.tool()
def get_vectorstore_count() -> Dict:
    """
    Diagnostic tool.

    Opens Chroma without running semantic search.
    This checks whether MCP can access the vectorstore at all.
    """

    client = chromadb.PersistentClient(
        path=str(PROJECT_ROOT / "vectorstore" / "chroma")
    )

    collection = client.get_collection(
        name="enterprise_docs"
    )

    return {
        "collection": "enterprise_docs",
        "count": collection.count(),
    }
    
@mcp.tool()
def peek_vectorstore_chunks(limit: int = 3) -> Dict:
    """
    Diagnostic tool.

    Reads stored Chroma chunks without semantic search.
    Returns only JSON-safe fields.
    """

    client = chromadb.PersistentClient(
        path=str(PROJECT_ROOT / "vectorstore" / "chroma")
    )

    collection = client.get_collection(
        name="enterprise_docs"
    )

    results = collection.peek(limit=limit)

    safe_items = []

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    for i in range(len(ids)):
        safe_items.append(
            {
                "id": ids[i],
                "metadata": metadatas[i],
                "preview": documents[i][:500],
            }
        )

    return {
        "count": collection.count(),
        "items": safe_items,
    }
    
    

@mcp.tool()
def check_document_health(
    max_pages_per_pdf: int = 3,
    preview_chars: int = 300,
) -> List[Dict]:
    """
    Check whether each document can be safely read.

    Why this matters:
    - In production, one bad PDF should not crash the MCP server.
    - Before vector ingestion, we want to know which files are healthy.
    - Bad files will be skipped later during Chroma ingestion.
    """

    docs = collect_documents()
    health_report = []

    for doc in docs:
        relative_path = doc["relative_path"]

        item = {
            "relative_path": relative_path,
            "status": "unknown",
            "file_type": doc.get("file_type"),
            "text_preview": "",
            "error": None,
        }

        try:
            path = resolve_document_path(relative_path)

            text = extract_text_from_file(
                path,
                max_pages=max_pages_per_pdf,
            )

            if text and text.strip():
                item["status"] = "healthy"
                item["text_preview"] = text[:preview_chars]
            else:
                item["status"] = "empty_text"
                item["error"] = "File was readable, but no text was extracted."

        except Exception as e:
            item["status"] = "failed"
            item["error"] = str(e)

        health_report.append(item)

    return health_report
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