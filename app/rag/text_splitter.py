import re

class RecursiveTextSplitter:
    """
    Pure Python recursive character text splitter.
    Attempts to split on paragraphs, then newlines, then sentence boundaries,
    then spaces, preserving natural language structure.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        final_chunks = []
        # Find the primary separator for this text
        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = ""
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        
        good_splits = []
        for s in splits:
            if s.strip() or separator == "":
                good_splits.append(s)

        # Merge splits into chunks respecting chunk_size and chunk_overlap
        current_chunk = []
        current_length = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            
            if piece_len > self.chunk_size and new_separators:
                # Recurse on excessively large piece
                if current_chunk:
                    joined = separator.join(current_chunk).strip()
                    if joined:
                        final_chunks.append(joined)
                    current_chunk = []
                    current_length = 0
                sub_chunks = self._split(piece, new_separators)
                final_chunks.extend(sub_chunks)
                continue

            if current_length + piece_len > self.chunk_size and current_chunk:
                joined = separator.join(current_chunk).strip()
                if joined:
                    final_chunks.append(joined)
                
                # Keep overlap from previous items
                overlap_items = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    item_cost = len(item) + len(separator)
                    if overlap_len + item_cost <= self.chunk_overlap:
                        overlap_items.insert(0, item)
                        overlap_len += item_cost
                    else:
                        break
                current_chunk = overlap_items
                current_length = overlap_len

            current_chunk.append(piece)
            current_length += piece_len

        if current_chunk:
            joined = separator.join(current_chunk).strip()
            if joined:
                final_chunks.append(joined)

        return final_chunks


def split_document_into_chunks(
    doc: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[dict]:
    """
    Takes a doc dictionary: {"source": ..., "text": ...}
    Returns a list of chunk dicts:
        [{"id": "doc.pdf_0", "text": "...", "source": "doc.pdf"}, ...]
    """
    splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = splitter.split_text(doc["text"])

    chunks = []
    source = doc.get("source", "unknown")
    for idx, text in enumerate(text_chunks):
        chunks.append({
            "id": f"{source}_{idx}",
            "text": text,
            "source": source,
            "chunk_index": idx,
        })
    return chunks
