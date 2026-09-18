import os
import logging
from pathlib import Path
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_file(file_path: str | Path) -> str:
    """Extracts raw text from .txt, .md, or .pdf files using pure standard tools."""
    path = Path(file_path)
    if not path.is_file():
        logger.warning(f"File does not exist: {path}")
        return ""

    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {path}: {e}")
            return ""

    elif ext == ".pdf":
        try:
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except Exception as e:
            logger.error(f"Error extracting PDF text from {path}: {e}")
            return ""

    logger.debug(f"Skipping unsupported file extension: {ext} ({path.name})")
    return ""


def load_documents_from_folder(folder_path: str | Path) -> list[dict]:
    """
    Reads every supported document in a directory.
    Returns:
        [{"source": "filename.pdf", "text": "...", "path": "/path/to/file"}, ...]
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        logger.warning(f"Folder not found: {folder}")
        return []

    documents = []
    # Recursively discover documents in root folder and all date subfolders
    for entry in folder.rglob("*"):
        if entry.is_file() and entry.suffix.lower() in (".txt", ".md", ".pdf"):
            text = load_file(entry)
            if text.strip():
                # Provide relative path or name as source
                rel_path = entry.relative_to(folder)
                documents.append({
                    "source": str(rel_path).replace("\\", "/"),
                    "text": text,
                    "path": str(entry.resolve())
                })
    return documents
