"""CouchDB handler for the DoCA project.

Provides a :class:`DBHandler` that connects to a local CouchDB instance,
ensures the target database exists, and exposes small CRUD helpers for
storing and retrieving document metadata.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import couchdb
from couchdb.http import ResourceConflict, ResourceNotFound, ServerError

logger = logging.getLogger(__name__)


def _load_couchdb_url() -> str:
    """Build the CouchDB connection URL from environment variables.

    Reads COUCHDB_URL, COUCHDB_USER, and COUCHDB_PASSWORD from the
    environment (or a ``.env`` file via python-dotenv).  Falls back to
    ``http://127.0.0.1:5984`` with no credentials when unset.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional; env vars still work

    base_url = os.environ.get("COUCHDB_URL", "http://127.0.0.1:5984")
    user = os.environ.get("COUCHDB_USER", "")
    password = os.environ.get("COUCHDB_PASSWORD", "")

    if user and password:
        # Inject credentials into the URL scheme://user:pass@host:port
        scheme, rest = base_url.split("://", 1)
        return f"{scheme}://{user}:{password}@{rest}"
    return base_url


DEFAULT_DB_NAME = os.environ.get("COUCHDB_DB_NAME", "doca_db")


class DBHandlerError(Exception):
    """Raised when the DBHandler cannot complete an operation."""


class DBHandler:
    """Thin wrapper around :mod:`couchdb` for DoCA's metadata store.

    Parameters
    ----------
    url:
        Full CouchDB URL.  When ``None``, built automatically from the
        ``COUCHDB_URL``, ``COUCHDB_USER``, and ``COUCHDB_PASSWORD``
        environment variables (or a ``.env`` file).
    db_name:
        Name of the database to use. Created automatically if missing.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> None:
        self.url = url or _load_couchdb_url()
        self.db_name = db_name or DEFAULT_DB_NAME
        self._server: Optional[couchdb.Server] = None
        self._db: Optional[couchdb.Database] = None
        self._connect()

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        """Open a connection and ensure the target database exists."""
        try:
            self._server = couchdb.Server(self.url)
            # Touch the server to fail fast if it's unreachable.
            self._server.version()
        except (ConnectionError, OSError, ServerError) as exc:
            logger.error("Could not reach CouchDB at %s: %s", self.url, exc)
            raise DBHandlerError(
                f"Unable to connect to CouchDB at {self.url}"
            ) from exc

        self._db = self._get_or_create_db(self.db_name)
        logger.info("Connected to CouchDB database '%s'", self.db_name)

    def _get_or_create_db(self, name: str) -> couchdb.Database:
        """Return the database, creating it if it does not yet exist."""
        assert self._server is not None  # for type checkers
        if name in self._server:
            return self._server[name]
        try:
            logger.info("Database '%s' not found. Creating it.", name)
            return self._server.create(name)
        except (ResourceConflict, ServerError) as exc:
            # Another process may have created it in the meantime.
            if name in self._server:
                return self._server[name]
            raise DBHandlerError(
                f"Unable to create database '{name}'"
            ) from exc

    @property
    def db(self) -> couchdb.Database:
        """Return the active database, raising if not connected."""
        if self._db is None:
            raise DBHandlerError("Database connection is not initialized.")
        return self._db

    # ------------------------------------------------------------------ #
    # CRUD helpers
    # ------------------------------------------------------------------ #
    def save_document(self, doc_data: dict[str, Any]) -> str:
        """Persist a metadata document and return its CouchDB ``_id``.

        If ``doc_data`` already contains an ``_id`` that points to an
        existing record, the record is updated in place.
        """
        if not isinstance(doc_data, dict):
            raise TypeError("doc_data must be a dict")

        payload = dict(doc_data)  # avoid mutating the caller's dict

        try:
            doc_id = payload.get("_id")
            if doc_id and doc_id in self.db:
                existing = self.db[doc_id]
                payload["_rev"] = existing["_rev"]
                self.db[doc_id] = payload
                return doc_id

            new_id, _rev = self.db.save(payload)
            return new_id
        except ResourceConflict as exc:
            logger.warning("Conflict while saving document: %s", exc)
            raise DBHandlerError(f"Document update conflict: {exc}") from exc
        except (ConnectionError, OSError, ServerError) as exc:
            logger.error("Failed to save document: %s", exc)
            raise DBHandlerError(f"Failed to save document: {exc}") from exc

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Return the document with ``doc_id`` or ``None`` if not found."""
        if not doc_id:
            raise ValueError("doc_id must be a non-empty string")

        try:
            doc = self.db.get(doc_id)
            return dict(doc) if doc is not None else None
        except ResourceNotFound:
            return None
        except (ConnectionError, OSError, ServerError) as exc:
            logger.error("Failed to fetch document '%s': %s", doc_id, exc)
            raise DBHandlerError(
                f"Failed to fetch document '{doc_id}'"
            ) from exc