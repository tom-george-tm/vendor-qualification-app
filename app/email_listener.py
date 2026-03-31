"""
Background Email Listener Service

Continuously polls an IMAP mailbox for new (unread) emails with PDF attachments.
When a matching email arrives, it:
  1. Extracts all PDF attachments
  2. Creates a new claim (generates CLAIM_ID, uploads PDFs to Azure Blob)
  3. Creates a DB record in the Claims collection
  4. Triggers the AI background analysis (same as upload_claim_documents)
  5. Marks the email as read so it isn't processed again

The listener runs as an asyncio background task tied to the FastAPI lifespan.
"""

import asyncio
import datetime
import email as email_lib
import imaplib
import logging
import random
from email.header import decode_header
from typing import List, Optional, Tuple

from app.azure_blob import blob_prefix_exists, upload_blob
from app.config import settings
from app.database import Claims
from app.workflows.claims_workflow import claim_workflow

# ──────────────────────────────────────────────────────────────
# Live status tracker (read by the /email-listener/status endpoint)
# ──────────────────────────────────────────────────────────────

_status: dict = {
    "running": False,
    "started_at": None,
    "last_poll_at": None,
    "last_poll_found": 0,       # emails with PDFs found in last poll
    "total_polls": 0,
    "total_emails_processed": 0,
    "total_claims_created": 0,
    "last_error": None,
    "last_claim_id": None,
    "last_claim_at": None,
    "recent_emails": [],        # last 20 processed emails with per-step status
}

# Max number of recent email records to keep in memory
_MAX_RECENT = 20


def _record_email_event(event: dict) -> None:
    """Append an email processing event to recent_emails, capped at _MAX_RECENT."""
    _status["recent_emails"].insert(0, event)
    if len(_status["recent_emails"]) > _MAX_RECENT:
        _status["recent_emails"] = _status["recent_emails"][:_MAX_RECENT]


def get_listener_status() -> dict:
    """Return a copy of the current listener status (safe to call from any coroutine)."""
    return dict(_status)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _generate_claim_id() -> str:
    """Generate a unique claim folder name like CLAIM_ID_765476."""
    unique_number = random.randint(100000, 999999)
    return f"CLAIM_ID_{unique_number}"


def _decode_subject(msg) -> str:
    """Safely decode the email subject header."""
    raw = msg.get("Subject", "(no subject)")
    parts = decode_header(raw)
    decoded_parts = []
    for fragment, charset in parts:
        if isinstance(fragment, bytes):
            decoded_parts.append(fragment.decode(charset or "utf-8", errors="replace"))
        else:
            decoded_parts.append(fragment)
    return " ".join(decoded_parts)


def _extract_pdf_attachments(msg) -> List[Tuple[str, bytes]]:
    """
    Walk through a parsed email message and return a list of (filename, raw_bytes)
    tuples for every PDF attachment found.
    """
    attachments: List[Tuple[str, bytes]] = []
    for part in msg.walk():
        content_disposition = str(part.get("Content-Disposition", ""))
        if "attachment" not in content_disposition:
            continue
        filename = part.get_filename()
        if not filename:
            continue
        # Decode filename if encoded
        decoded_parts = decode_header(filename)
        decoded_name = ""
        for fragment, charset in decoded_parts:
            if isinstance(fragment, bytes):
                decoded_name += fragment.decode(charset or "utf-8", errors="replace")
            else:
                decoded_name += fragment
        filename = decoded_name

        if not filename.lower().endswith(".pdf"):
            continue
        payload = part.get_payload(decode=True)
        if payload:
            attachments.append((filename, payload))
    return attachments


def _extract_sender(msg) -> str:
    """Extract the sender's email address from the email message."""
    from_header = msg.get("From", "unknown")
    # Handle "Name <email>" format
    if "<" in from_header and ">" in from_header:
        return from_header.split("<")[1].split(">")[0]
    return from_header


# ──────────────────────────────────────────────────────────────
# IMAP fetch (runs in a thread to avoid blocking the event loop)
# ──────────────────────────────────────────────────────────────

def _fetch_unread_emails() -> List[Tuple[str, str, List[Tuple[str, bytes]], str]]:
    """
    Connect to IMAP, fetch all UNSEEN emails that have PDF attachments.

    Returns a list of tuples:
        (email_uid, subject, [(filename, pdf_bytes), ...], sender_email)
    """
    host = settings.IMAP_HOST
    port = settings.IMAP_PORT
    username = settings.IMAP_USERNAME
    password = settings.IMAP_PASSWORD

    if not username or not password:
        print("IMAP credentials not configured — email listener skipping poll.")
        return []

    results: List[Tuple[str, str, List[Tuple[str, bytes]], str]] = []
    mail: Optional[imaplib.IMAP4_SSL] = None

    try:
        mail = imaplib.IMAP4_SSL(host, port)
        mail.login(username, password)
        mail.select("INBOX")

        # Search for unseen (unread) emails
        status, data = mail.uid("search", None, "UNSEEN")
        if status != "OK" or not data or not data[0]:
            return []

        uids = data[0].split()
        print("Found %d unread email(s) to inspect.", len(uids))

        for uid_bytes in uids:
            uid = uid_bytes.decode()
            status, msg_data = mail.uid("fetch", uid, "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue

            raw_email = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw_email)
            subject = _decode_subject(msg)
            sender = _extract_sender(msg)
            pdfs = _extract_pdf_attachments(msg)

            if pdfs:
                results.append((uid, subject, pdfs, sender))
                print(
                    "Email UID %s from <%s> has %d PDF attachment(s): %s",
                    uid, sender, len(pdfs),
                    [name for name, _ in pdfs],
                )
            else:
                print("Email UID %s from <%s> has no PDF attachments — skipping.", uid, sender)

    except imaplib.IMAP4.error as exc:
        print("IMAP error while fetching emails: %s", exc)
    except Exception as exc:
        print("Unexpected error while fetching emails: %s", exc)
    finally:
        if mail:
            try:
                mail.logout()
            except Exception:
                pass

    return results


def _mark_email_as_read(uid: str) -> None:
    """Connect to IMAP and mark a single email as SEEN (read)."""
    host = settings.IMAP_HOST
    port = settings.IMAP_PORT
    username = settings.IMAP_USERNAME
    password = settings.IMAP_PASSWORD

    if not username or not password:
        return

    mail: Optional[imaplib.IMAP4_SSL] = None
    try:
        mail = imaplib.IMAP4_SSL(host, port)
        mail.login(username, password)
        mail.select("INBOX")
        mail.uid("store", uid, "+FLAGS", "(\\Seen)")
        print("Marked email UID %s as read.", uid)
    except Exception as exc:
        print("Failed to mark email UID %s as read: %s", uid, exc)
    finally:
        if mail:
            try:
                mail.logout()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────
# Core processing logic (mirrors upload_claim_documents)
# ──────────────────────────────────────────────────────────────

async def _process_email(
    uid: str,
    subject: str,
    attachments: List[Tuple[str, bytes]],
    sender: str,
) -> Optional[str]:
    """
    Process one email: upload PDFs → create claim → trigger analysis.
    Returns the generated claim_id on success, None on failure.
    """
    # Per-email event record — registered into recent_emails IMMEDIATELY so the
    # status endpoint shows it even while processing is still in progress.
    event: dict = {
        "uid": uid,
        "received_at": datetime.datetime.utcnow().isoformat(),
        "sender": sender,
        "subject": subject,
        "attachments": [name for name, _ in attachments],
        "claim_id": None,
        "status": "fetched",           # fetched → processing → complete / failed
        "db_record_created": False,
        "blobs_uploaded": [],
        "blob_upload_status": "pending",
        "marked_as_read": False,
        "analysis_triggered": False,
        "error": None,
        "completed_at": None,
    }
    # Insert by reference — all in-place updates below will be visible immediately
    _record_email_event(event)

    # Generate a unique claim ID (retry if prefix already exists in blob storage)
    claim_id = _generate_claim_id()
    while blob_prefix_exists(f"{claim_id}/"):
        claim_id = _generate_claim_id()
    event["claim_id"] = claim_id
    event["status"] = "processing"

    print(
        "Processing email UID %s — creating claim %s with %d PDF(s)",
        uid, claim_id, len(attachments),
    )

    # ── Step 1: Create claim record in DB ──────────────────────
    try:
        await Claims.update_one(
            {"claim_id": claim_id},
            {
                "$set": {
                    "claim_id": claim_id,
                    "created_at": datetime.datetime.utcnow(),
                    "source": "email",
                    "source_email": sender,
                    "source_subject": subject,
                    "applicant_name": "Pending Analysis",
                    "policy_number": "Pending Analysis",
                    "applicant_age": None,
                    "patient_gender": None,
                    "medical_case": "Pending Analysis",
                    "diagnosis": "Pending Analysis",
                    "procedure": "Pending Analysis",
                    "hospital_name": "Pending Analysis",
                    "hospital_location": "Pending Analysis",
                    "claimed_amount": None,
                    "readiness_score": 0,
                    "risk_level": "N/A",
                    "submission_status": "Not Analyzed",
                    "uploaded_documents": [],
                    "is_analyzed": False,
                }
            },
            upsert=True,
        )
        event["db_record_created"] = True
    except Exception as exc:
        event["status"] = "failed"
        event["error"] = f"DB error: {exc}"
        event["completed_at"] = datetime.datetime.utcnow().isoformat()
        return None

    # ── Step 2: Upload each PDF to Azure Blob Storage ──────────
    try:
        for filename, pdf_bytes in attachments:
            blob_name = f"{claim_id}/{filename}"
            upload_blob(blob_name, pdf_bytes)
            event["blobs_uploaded"].append(blob_name)
            print("Uploaded blob from email: %s", blob_name)
        event["blob_upload_status"] = "success"
    except Exception as exc:
        event["blob_upload_status"] = f"failed: {exc}"
        event["status"] = "failed"
        event["error"] = f"Blob upload error: {exc}"
        event["completed_at"] = datetime.datetime.utcnow().isoformat()
        print("Failed to upload PDFs for claim %s from email UID %s: %s", claim_id, uid, exc)
        return None

    # ── Step 3: Trigger AI analysis ────────────────────────────
    try:
        await _trigger_background_analysis(claim_id)
        event["analysis_triggered"] = True
    except Exception as exc:
        event["error"] = f"Analysis error: {exc}"
        print("Background analysis failed for email claim %s: %s", claim_id, exc)

    event["status"] = "complete"
    event["completed_at"] = datetime.datetime.utcnow().isoformat()
    print(
        "✅ Email UID %s processed → claim %s (%d files uploaded)",
        uid, claim_id, len(event["blobs_uploaded"]),
    )
    return claim_id


async def _trigger_background_analysis(claim_id: str):
    """Trigger the LangGraph workflow for a claim (identical to claims.py)."""
    print("Triggering background analysis for email claim %s", claim_id)
    initial_state = {
        "session_id": f"bg-{claim_id}",
        "claim_id": claim_id,
        "claim_folder": f"{claim_id}/",
        "raw_files": [],
        "classified_files": [],
        "missing_docs": [],
        "document_analyses": [],
        "aggregated_result": {},
        "bill_amount": None,
        "settlement_amount": None,
        "deduction_percentage": 0,
        "is_ready": False,
        "response_message": "",
        "is_analyzed": False,
        "error": None,
    }
    await claim_workflow.ainvoke(initial_state)
    print("Background analysis completed for email claim %s", claim_id)


# ──────────────────────────────────────────────────────────────
# Main polling loop (runs forever as a background asyncio task)
# ──────────────────────────────────────────────────────────────

async def email_listener_loop():
    """
    Infinite loop that polls the IMAP inbox every EMAIL_POLL_INTERVAL_SECONDS.
    Should be started as an asyncio task during FastAPI lifespan startup.
    """
    interval = settings.EMAIL_POLL_INTERVAL_SECONDS
    _status["running"] = True
    _status["started_at"] = datetime.datetime.utcnow().isoformat()
    print(
        "📬 Email listener started — polling %s every %ds",
        settings.IMAP_USERNAME, interval,
    )

    while True:
        try:
            # Run the blocking IMAP fetch in a thread
            emails = await asyncio.to_thread(_fetch_unread_emails)

            _status["last_poll_at"] = datetime.datetime.utcnow().isoformat()
            _status["total_polls"] += 1
            _status["last_poll_found"] = len(emails)

            for uid, subject, attachments, sender in emails:
                try:
                    _status["total_emails_processed"] += 1
                    claim_id = await _process_email(uid, subject, attachments, sender)
                    if claim_id:
                        _status["total_claims_created"] += 1
                        _status["last_claim_id"] = claim_id
                        _status["last_claim_at"] = datetime.datetime.utcnow().isoformat()
                        # Only mark as read once we've successfully processed
                        await asyncio.to_thread(_mark_email_as_read, uid)
                        # Update the event record's marked_as_read flag
                        for ev in _status["recent_emails"]:
                            if ev["uid"] == uid:
                                ev["marked_as_read"] = True
                                break
                except Exception as exc:
                    _status["last_error"] = str(exc)
                    print(
                        "Failed to process email UID %s: %s", uid, exc
                    )

        except asyncio.CancelledError:
            _status["running"] = False
            print("Email listener received cancellation — shutting down.")
            break
        except Exception as exc:
            _status["last_error"] = str(exc)
            print("Email listener poll error: %s", exc)

        await asyncio.sleep(interval)

    print("📬 Email listener stopped.")