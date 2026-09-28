"""Document content retrieval from SQLite database.

This module provides efficient retrieval of full document content
using SQLite as a forward index for O(1) lookups by document ID.
"""

import logging
import sqlite3
import threading
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

        # Use thread-local storage for SQLite connections (thread-safety for async/FastAPI)
        # Each thread gets its own connection to avoid "SQLite objects created in a thread
        # can only be used in that same thread" errors
        self._thread_local = threading.local()

        # In-memory cache for frequently accessed documents (shared across threads)
        # This is safe because we only read from the cache, not write concurrently
        self._cache: dict[int, dict[str, Any]] = {}
        self.MAX_CACHE_SIZE = 1000

        logger.info(f"Initialized DocumentStore with SQLite (thread-safe): {db_path}")

    def _get_connection(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        """Get thread-local SQLite connection and cursor.

        Creates a new connection for each thread on first access.
        Subsequent calls in the same thread reuse the connection.

        Returns:
            Tuple of (connection, cursor) for the current thread
        """
        if not hasattr(self._thread_local, "connection"):
            # Create new connection for this thread
            self._thread_local.connection = sqlite3.connect(str(self.db_path))
            self._thread_local.connection.row_factory = sqlite3.Row
            self._thread_local.cursor = self._thread_local.connection.cursor()
            thread_name = threading.current_thread().name
            logger.debug(f"Created new SQLite connection for thread {thread_name}")

        return self._thread_local.connection, self._thread_local.cursor

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
            # Get thread-local connection and cursor
            _, cursor = self._get_connection()

            # Fast primary key lookup in SQLite
            cursor.execute(
                "SELECT title, content FROM documents WHERE doc_id = ?",
                (doc_id,),
            )
            row = cursor.fetchone()

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
        """Close all thread-local database connections."""
        # Close thread-local connection if it exists for the current thread
        if hasattr(self._thread_local, "connection"):
            try:
                self._thread_local.connection.close()
                thread_name = threading.current_thread().name
                logger.debug(f"Closed SQLite connection for thread {thread_name}")
            except Exception as e:
                thread_name = threading.current_thread().name
                logger.warning(f"Error closing connection for thread {thread_name}: {e}")

    def __del__(self):
        """Ensure database connections are closed."""
        self.close()
