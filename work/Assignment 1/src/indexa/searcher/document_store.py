"""Document content retrieval from SQLite database.

This module provides efficient retrieval of full document content
using SQLite as a forward index for O(1) lookups by document ID.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DocumentStore:
    """Retrieves full document content from SQLite database.

    Uses SQLite as a forward index for fast O(1) document lookups
    by doc_id. Much faster than reading from Arrow files.
    """

    def __init__(self, db_path: Path):
        """Initialize document store.

        Args:
            db_path: Path to the SQLite database (documents.db)
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row  # Access columns by name
        self.cursor = self.connection.cursor()

        # In-memory cache for frequently accessed documents
        self._cache: dict[int, dict[str, Any]] = {}
        self.MAX_CACHE_SIZE = 1000

        logger.info(f"Initialized DocumentStore with SQLite: {db_path}")

    def get_document(self, doc_id: int) -> dict[str, Any] | None:
        """Retrieve full document content by document ID.

        Args:
            doc_id: The document ID from the index

        Returns:
            Dictionary with 'title', 'content', and 'url' or None if not found
        """
        # Check cache first
        if doc_id in self._cache:
            return self._cache[doc_id]

        try:
            # Fast primary key lookup in SQLite
            self.cursor.execute(
                "SELECT title, content FROM documents WHERE doc_id = ?",
                (doc_id,),
            )
            row = self.cursor.fetchone()

            if not row:
                logger.warning(f"Document {doc_id} not found in database")
                return None

            title = row["title"]
            content = row["content"]

            # Build Wikipedia URL
            url = ""
            if title:
                encoded_title = title.replace(" ", "_")
                url = f"https://pt.wikipedia.org/wiki/{encoded_title}"

            doc = {
                "title": title or f"Document {doc_id}",
                "content": content or "",
                "url": url,
            }

            # Cache for future requests (with size limit)
            if len(self._cache) < self.MAX_CACHE_SIZE:
                self._cache[doc_id] = doc

            return doc

        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None

    def get_document_preview(self, doc_id: int, max_length: int = 500) -> str:
        """Get a preview of document content.

        Args:
            doc_id: The document ID
            max_length: Maximum preview length in characters

        Returns:
            Preview string
        """
        doc = self.get_document(doc_id)
        if not doc:
            return ""

        content = doc["content"]
        if len(content) <= max_length:
            return content

        # Return first max_length characters with ellipsis
        return content[:max_length] + "..."

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()

    def __del__(self):
        """Ensure database connection is closed."""
        self.close()
