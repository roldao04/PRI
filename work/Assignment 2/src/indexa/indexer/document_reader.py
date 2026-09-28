"""Document reader for handling Arrow and text file formats.

This module provides efficient document reading for Arrow and TXT formats.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import chardet

logger = logging.getLogger(__name__)


class DocumentReader(ABC):
    """Abstract base class for document readers.

    This design allows us to easily add support for new formats
    by creating new reader subclasses.
    """

    @abstractmethod
    def read(self, file_path: Path) -> dict[str, Any]:
        """Read a document and return its content and metadata.

        Returns:
            Dictionary with keys:
            - 'content': The text content of the document
            - 'title': The document title
            - 'metadata': Additional metadata (optional)
        """
        pass

    @abstractmethod
    def can_read(self, file_path: Path) -> bool:
        """Check if this reader can handle the given file."""
        pass


class TextDocumentReader(DocumentReader):
    """Reader for plain text documents.

    This reader handles various text encodings intelligently,
    attempting to detect the correct encoding if not specified.
    """

    def __init__(self, encoding: Optional[str] = None, detect_encoding: bool = True):
        """Initialize text reader.

        Args:
            encoding: Force specific encoding (e.g., 'utf-8')
            detect_encoding: Automatically detect encoding if True
        """
        self.encoding = encoding
        self.detect_encoding = detect_encoding

    def can_read(self, file_path: Path) -> bool:
        """Check if file is a text document."""
        text_extensions = {".txt", ".text", ".md", ".markdown", ".rst"}
        return file_path.suffix.lower() in text_extensions

    def read(self, file_path: Path) -> dict[str, Any]:
        """Read text document with intelligent encoding detection."""
        content = None

        if self.encoding:
            # Use specified encoding
            try:
                with open(file_path, encoding=self.encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                logger.warning(f"Failed to read {file_path} with encoding {self.encoding}")

        if content is None and self.detect_encoding:
            # Attempt to detect encoding
            try:
                with open(file_path, "rb") as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected["encoding"] or "utf-8"
                    confidence = detected.get("confidence", 0)

                    if confidence > 0.7:  # Only use detection if confident
                        content = raw_data.decode(encoding)
                        logger.debug(
                            f"Detected encoding {encoding} for {file_path} "
                            f"(confidence: {confidence:.2f})"
                        )
            except Exception as e:
                logger.warning(f"Encoding detection failed for {file_path}: {e}")

        if content is None:
            # Fall back to UTF-8 with error handling
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    logger.warning(f"Read {file_path} with UTF-8 fallback (may contain errors)")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                content = ""

        # Extract title from filename or first line
        title = self._extract_title(file_path, content)

        return {
            "content": content,
            "title": title,
            "metadata": {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "encoding": self.encoding or "auto-detected",
            },
        }

    def _extract_title(self, file_path: Path, content: str) -> str:
        """Extract a meaningful title from the document.

        This tries several strategies:
        1. Use filename without extension
        2. Use first non-empty line if it looks like a title
        3. Use beginning of content
        """
        # Try filename first
        title = file_path.stem

        # Try to find a better title in the content
        lines = content.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) < 200:  # Reasonable title length
                # Check if it looks like a title (e.g., starts with #, uppercase, etc.)
                if line.startswith("#"):  # Markdown header
                    title = line.lstrip("#").strip()
                    break
                elif line.isupper() and len(line) < 100:  # All caps title
                    title = line
                    break

        return title


class ArrowDocumentReader(DocumentReader):
    """Reader for Apache Arrow files.

    This reader handles large-scale datasets stored in Arrow format,
    yielding documents one at a time to avoid loading entire file into memory.
    """

    def __init__(
        self,
        content_field: str = "text",
        title_field: str = "title",
        filter_redirects: bool = True,
        redirect_field: str = "redirect",
    ):
        """Initialize Arrow reader.

        Args:
            content_field: Column containing the main text
            title_field: Column containing the title
            filter_redirects: Skip redirect documents if True
            redirect_field: Column indicating if document is a redirect
        """
        self.content_field = content_field
        self.title_field = title_field
        self.filter_redirects = filter_redirects
        self.redirect_field = redirect_field

    def can_read(self, file_path: Path) -> bool:
        """Check if file is an Arrow document."""
        return file_path.suffix.lower() in {".arrow", ".feather"}

    def read(self, file_path: Path) -> dict[str, Any]:
        """Read Arrow file - returns metadata about the file.

        Note: Use read_documents() to iterate through individual documents.
        """
        try:
            from pyarrow import ipc

            with ipc.open_file(file_path) as reader:  # type: ignore[arg-type]
                num_batches = reader.num_record_batches

                # Count total rows
                total_rows = sum(reader.get_batch(i).num_rows for i in range(num_batches))

                return {
                    "content": f"Arrow file with {total_rows} documents",
                    "title": file_path.stem,
                    "metadata": {
                        "file_path": str(file_path),
                        "format": "arrow",
                        "num_batches": num_batches,
                        "total_documents": total_rows,
                    },
                }
        except Exception as e:
            logger.error(f"Failed to read Arrow file {file_path}: {e}")
            return {"content": "", "title": file_path.stem, "metadata": {"error": str(e)}}

    def prescan_non_redirect_indices(self, file_path: Path) -> list[int]:
        """Fast pre-scan to identify non-redirect document row indices.

        Only reads the redirect column for maximum speed (columnar access).
        Returns list of global row indices that are actual documents.

        Args:
            file_path: Path to Arrow file

        Returns:
            List of row indices (global, across all batches) for non-redirect documents
        """
        try:
            from pyarrow import ipc

            non_redirect_indices: list[int] = []
            global_row_idx = 0

            logger.info(f"Pre-scanning Arrow file to identify non-redirect documents: {file_path}")

            with ipc.open_file(file_path) as reader:  # type: ignore[arg-type]
                for batch_idx in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_idx)

                    # Only read redirect column (fast!)
                    redirect_col = (
                        batch.column(self.redirect_field)
                        if self.filter_redirects and self.redirect_field in batch.schema.names
                        else None
                    )

                    for row_idx in range(batch.num_rows):
                        # Check if this row is NOT a redirect
                        if redirect_col is None or not redirect_col[row_idx].as_py():
                            non_redirect_indices.append(global_row_idx)
                        global_row_idx += 1

                    # Progress logging every 100 batches (~99K rows)
                    if (batch_idx + 1) % 100 == 0:
                        batch_info = f"{batch_idx + 1}/{reader.num_record_batches}"
                        logger.info(
                            f"Pre-scan progress: {batch_info} batches, "
                            f"found {len(non_redirect_indices)} non-redirect docs"
                        )

            logger.info(
                f"Pre-scan complete: found {len(non_redirect_indices)} non-redirect documents "
                f"out of {global_row_idx} total rows"
            )
            return non_redirect_indices

        except Exception as e:
            logger.error(f"Failed to pre-scan Arrow file {file_path}: {e}")
            return []

    def read_documents(
        self, file_path: Path, row_indices: list[int] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Iterate through documents in the Arrow file.

        This is memory-efficient as it yields one document at a time.
        Optimized to use columnar access instead of converting to dict.

        Args:
            file_path: Path to Arrow file
            row_indices: Optional list of specific row indices to process.
                        If provided, only these rows will be yielded.
                        If None, all rows are processed (with redirect filtering).
        """
        try:
            from pyarrow import ipc

            # Convert row_indices to set for O(1) lookup if provided
            row_indices_set: set[int] | None = set(row_indices) if row_indices is not None else None
            global_row_idx = 0

            with ipc.open_file(file_path) as reader:  # type: ignore[arg-type]
                for batch_idx in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_idx)

                    # Direct columnar access - much faster than to_pydict()
                    content_col = (
                        batch.column(self.content_field)
                        if self.content_field in batch.schema.names
                        else None
                    )
                    title_col = (
                        batch.column(self.title_field)
                        if self.title_field in batch.schema.names
                        else None
                    )
                    redirect_col = (
                        batch.column(self.redirect_field)
                        if self.filter_redirects and self.redirect_field in batch.schema.names
                        else None
                    )

                    for row_idx in range(batch.num_rows):
                        # If row_indices specified, check if this row should be processed
                        if row_indices_set is not None and global_row_idx not in row_indices_set:
                            global_row_idx += 1
                            continue

                        # Check if we should skip redirects (when not using row_indices)
                        if redirect_col is not None and redirect_col[row_idx].as_py():
                            global_row_idx += 1
                            continue

                        # Extract content and title using columnar access
                        content = content_col[row_idx].as_py() if content_col is not None else ""
                        content = content or ""

                        title = (
                            title_col[row_idx].as_py()
                            if title_col is not None
                            else f"Document {row_idx}"
                        )
                        title = title or f"Document {row_idx}"

                        yield {
                            "content": content,
                            "title": title,
                            "metadata": {
                                "file_path": str(file_path),
                                "batch_idx": batch_idx,
                                "row_idx": row_idx,
                                "global_row_idx": global_row_idx,
                            },
                        }
                        global_row_idx += 1
        except Exception as e:
            logger.error(f"Failed to iterate Arrow file {file_path}: {e}")

    def write_filtered_arrow_file(
        self, source_path: Path, output_path: Path, row_indices: list[int]
    ) -> bool:
        """Create a new Arrow file containing only specified rows.

        This creates a filtered Arrow file with only the indexed documents,
        making the output self-contained.

        Args:
            source_path: Path to source Arrow file
            output_path: Path where filtered Arrow file will be written
            row_indices: List of row indices to include (in order)

        Returns:
            True if successful, False otherwise
        """
        try:
            import pyarrow as pa
            from pyarrow import ipc

            logger.info(f"Creating filtered Arrow file with {len(row_indices)} documents")

            # Convert to set for O(1) lookup
            row_indices_set = set(row_indices)
            global_row_idx = 0

            # Collect filtered batches
            filtered_batches = []

            with ipc.open_file(source_path) as reader:  # type: ignore[arg-type]
                schema = reader.schema

                for batch_idx in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_idx)

                    # Find which rows from this batch should be included
                    rows_to_keep = []
                    for row_idx in range(batch.num_rows):
                        if global_row_idx in row_indices_set:
                            rows_to_keep.append(row_idx)
                        global_row_idx += 1

                    # If this batch has rows to keep, filter it
                    if rows_to_keep:
                        # Use take to select specific rows
                        filtered_batch = batch.take(rows_to_keep)
                        filtered_batches.append(filtered_batch)

                    if (batch_idx + 1) % 100 == 0:
                        logger.info(
                            f"Filtered {batch_idx + 1}/{reader.num_record_batches} batches, "
                            f"collected {sum(b.num_rows for b in filtered_batches)} documents"
                        )

            # Combine all filtered batches into a table
            if filtered_batches:
                table = pa.Table.from_batches(filtered_batches, schema=schema)

                # Write to output file
                with ipc.new_file(output_path, schema) as writer:  # type: ignore[arg-type]
                    writer.write_table(table)

                logger.info(
                    f"Successfully wrote filtered Arrow file: {output_path} "
                    f"({table.num_rows} documents)"
                )
                return True
            else:
                logger.warning("No documents to write to filtered Arrow file")
                return False

        except Exception as e:
            logger.error(f"Failed to create filtered Arrow file: {e}")
            return False


class DocumentReaderFactory:
    """Factory for creating appropriate document readers.

    This factory pattern allows the system to automatically
    select the right reader based on file type.
    """

    def __init__(self):
        """Initialize factory with default reader."""
        self.readers: list[DocumentReader] = [TextDocumentReader()]

    def add_reader(self, reader: DocumentReader):
        """Add a custom reader to the factory."""
        self.readers.append(reader)

    def get_reader(self, file_path: Path) -> Optional[DocumentReader]:
        """Get appropriate reader for the file."""
        for reader in self.readers:
            if reader.can_read(file_path):
                return reader

        # Default to text reader for unknown types
        logger.warning(f"No specific reader for {file_path.suffix}, using text reader")
        return TextDocumentReader()

    def read_document(self, file_path: Path) -> dict[str, Any]:
        """Read a document using the appropriate reader."""
        reader = self.get_reader(file_path)
        if reader:
            return reader.read(file_path)

        logger.error(f"No reader available for {file_path}")
        return {
            "content": "",
            "title": file_path.stem,
            "metadata": {"error": "No reader available"},
        }
