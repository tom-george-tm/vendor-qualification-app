"""
Azure Blob Storage helper module.

All credentials are loaded from environment variables via app.config.settings.
Use the functions here to upload, download, and list blobs under claim prefixes.
"""

import os
import logging
from typing import List

from azure.storage.blob import BlobServiceClient, ContainerClient

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal: lazy-initialised clients
# ---------------------------------------------------------------------------

_blob_service_client: BlobServiceClient | None = None
_container_client: ContainerClient | None = None


def _get_container_client() -> ContainerClient:
    """Return a cached ContainerClient, creating it on first call."""
    global _blob_service_client, _container_client
    if _container_client is None:
        conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
        container = settings.AZURE_STORAGE_CONTAINER_NAME
        if not conn_str or not container:
            raise RuntimeError(
                "AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME "
                "must be set in the environment."
            )
        _blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        _container_client = _blob_service_client.get_container_client(container)
        logger.info("Azure Blob container client initialised for: %s", container)
    return _container_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def blob_prefix_exists(prefix: str) -> bool:
    """
    Return True if there is at least one blob whose name starts with *prefix*.

    Used to check whether a claim folder (e.g. ``CLAIM_ID_123456/``) exists.
    """
    client = _get_container_client()
    blobs = client.list_blobs(name_starts_with=prefix)
    for _ in blobs:
        return True
    return False


def list_blobs_in_prefix(prefix: str) -> List[str]:
    """
    Return a list of blob names (full path) whose names start with *prefix*.

    Example: ``list_blobs_in_prefix("CLAIM_ID_123456/")`` returns
    ``["CLAIM_ID_123456/bill.pdf", "CLAIM_ID_123456/blood_report.pdf"]``.
    """
    client = _get_container_client()
    return [b.name for b in client.list_blobs(name_starts_with=prefix)]


def list_claim_ids() -> List[str]:
    """
    Return all unique top-level claim folder names (e.g. ``CLAIM_ID_XXXXXX``)
    present in the container.
    """
    client = _get_container_client()
    seen = set()
    for blob in client.list_blobs():
        # blob.name looks like "CLAIM_ID_123456/filename.pdf"
        parts = blob.name.split("/", 1)
        if parts and parts[0].startswith("CLAIM_ID_"):
            seen.add(parts[0])
    return sorted(seen)


def upload_blob(blob_name: str, data: bytes, overwrite: bool = True) -> None:
    """
    Upload *data* as a blob named *blob_name* in the configured container.

    Args:
        blob_name: Full blob path, e.g. ``"CLAIM_ID_123456/bill.pdf"``.
        data:      Raw bytes to upload.
        overwrite: If True (default), overwrite an existing blob.
    """
    client = _get_container_client()
    blob_client = client.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=overwrite)
    logger.info("Uploaded blob: %s (%d bytes)", blob_name, len(data))


def download_blob(blob_name: str) -> bytes:
    """
    Download and return the raw bytes of the blob named *blob_name*.

    Args:
        blob_name: Full blob path, e.g. ``"CLAIM_ID_123456/bill.pdf"``.

    Returns:
        Raw bytes content of the blob.

    Raises:
        azure.core.exceptions.ResourceNotFoundError: if the blob does not exist.
    """
    client = _get_container_client()
    blob_client = client.get_blob_client(blob_name)
    data: bytes = blob_client.download_blob().readall()
    logger.info("Downloaded blob: %s (%d bytes)", blob_name, len(data))
    return data
