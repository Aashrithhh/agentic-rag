"""Document Ingestion Pipeline — chunk, embed, and store in pgvector.

Supported formats:
    .txt   — Plain text files
    .pdf   — PDF documents (with OCR fallback for scanned pages)
    .docx  — Microsoft Word documents
    .png / .jpg / .jpeg / .tiff / .bmp — Images (via OCR)
    .eml   — Standard email files (with image OCR & audio transcription for attachments)
    .msg   — Outlook message files (with image OCR & audio transcription for attachments)
    .pst   — Outlook data files (requires Outlook on Windows)

Usage:
    python -m app.ingest --dir ./data/raw_corpus
    python -m app.ingest --dir ./data/mixed_docs --doc-type "corporate-comms"
"""

from __future__ import annotations

import argparse
import base64
import email
import io
import logging
import os
import platform
import tempfile
from email import policy
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from app.config import settings
from app.db import init_db, upsert_chunks

logger = logging.getLogger(__name__)

# File extensions handled by each loader
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
_IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp",
                "image/tiff", "image/webp"}
_AUDIO_MIMES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/ogg",
                "audio/flac", "audio/mp4", "audio/m4a", "audio/webm", "audio/x-m4a"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
_ALL_SUPPORTED = {".txt", ".pdf", ".docx", *_IMAGE_EXTS, ".eml", ".msg", ".pst"}


# ─────────────────────────────────────────────────────────────────────
#  OCR helper (shared by image loader + scanned-PDF fallback)
# ─────────────────────────────────────────────────────────────────────
def _ocr_image(image) -> str:
    """Run Tesseract OCR on a PIL Image. Returns extracted text or ''."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
    except Exception as exc:
        logger.warning("OCR failed: %s (is Tesseract installed?)", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────
#  Attachment processing — GPT-4o Vision OCR + Whisper transcription
# ─────────────────────────────────────────────────────────────────────
def _ocr_image_gpt4o(image_bytes: bytes, filename: str = "image") -> str:
    """Extract text from an image using Azure GPT-4o vision.

    Sends the image as a base64 data-URI to GPT-4o and asks it to
    extract all visible text/content.  Falls back to Tesseract if
    the API call fails.
    """
    try:
        from openai import AzureOpenAI

        # Detect MIME type from extension or default to png
        ext = Path(filename).suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff",
                    ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/png")

        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"

        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        resp = client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": (
                    "You are an OCR assistant. Extract ALL text visible in the "
                    "image. Preserve formatting, tables, and structure as much as "
                    "possible. If the image contains no readable text, reply with "
                    "exactly: [NO TEXT FOUND]"
                )},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Extract all text from this image ({filename}):"},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                ]},
            ],
            max_tokens=4096,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        if text == "[NO TEXT FOUND]":
            logger.info("GPT-4o vision: no text found in %s", filename)
            return ""
        logger.info("GPT-4o vision OCR extracted %d chars from %s", len(text), filename)
        return text
    except Exception as exc:
        logger.warning("GPT-4o vision OCR failed for %s: %s — trying Tesseract fallback", filename, exc)
        # Tesseract fallback
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            return _ocr_image(img)
        except Exception:
            return ""


def _transcribe_audio(audio_bytes: bytes, filename: str = "audio.mp3") -> str:
    """Transcribe audio using Azure OpenAI Whisper, with fallbacks.

    Fallback chain:
      1. Azure OpenAI Whisper deployment
      2. GPT-4o audio input (base64-encoded audio in a chat message)
      3. Local ``whisper`` package (openai-whisper)
    """
    ext = Path(filename).suffix.lower() or ".mp3"

    # ── Attempt 1: Azure OpenAI Whisper ────────────────────────────
    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        # The API expects a file-like object with a name attribute
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        transcript = client.audio.transcriptions.create(
            model=settings.azure_whisper_deployment,
            file=audio_file,
            response_format="text",
        )
        text = transcript.strip() if isinstance(transcript, str) else transcript.text.strip()
        logger.info("Azure Whisper transcribed %d chars from %s", len(text), filename)
        return text
    except Exception as exc:
        logger.warning("Azure Whisper failed for %s: %s — trying GPT-4o audio", filename, exc)

    # ── Attempt 2: GPT-4o with audio input ─────────────────────────
    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        b64_audio = base64.b64encode(audio_bytes).decode("ascii")
        mime_map = {".mp3": "audio/mp3", ".wav": "audio/wav", ".m4a": "audio/mp4",
                    ".ogg": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm"}
        audio_mime = mime_map.get(ext, "audio/mp3")

        resp = client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": (
                    "You are a transcription assistant. Transcribe all spoken "
                    "words in the audio exactly as spoken. If the audio contains "
                    "no speech (only music, tones, or silence), reply with "
                    "exactly: [NO SPEECH DETECTED]"
                )},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Transcribe this audio file ({filename}):"},
                    {"type": "input_audio", "input_audio": {
                        "data": b64_audio,
                        "format": ext.lstrip(".") if ext.lstrip(".") in ("mp3", "wav") else "mp3",
                    }},
                ]},
            ],
            max_tokens=4096,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        if text == "[NO SPEECH DETECTED]":
            logger.info("GPT-4o audio: no speech in %s", filename)
            return ""
        logger.info("GPT-4o audio transcribed %d chars from %s", len(text), filename)
        return text
    except Exception as exc:
        logger.warning("GPT-4o audio failed for %s: %s — trying local whisper", filename, exc)

    # ── Attempt 3: Local whisper package ───────────────────────────
    try:
        import whisper

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            text = result.get("text", "").strip()
            logger.info("Local Whisper transcribed %d chars from %s", len(text), filename)
            return text
        finally:
            os.unlink(tmp_path)
    except ImportError:
        logger.warning("Local whisper package not installed. Run: pip install openai-whisper")
    except Exception as exc:
        logger.warning("Local whisper failed for %s: %s", filename, exc)

    return ""


def _process_attachment(content_type: str, payload: bytes, filename: str) -> str:
    """Process a single email attachment — returns extracted text or ''.

    Dispatches to GPT-4o vision for images and Whisper for audio files.
    """
    if not settings.process_attachments:
        return ""

    ct = content_type.lower()
    ext = Path(filename).suffix.lower() if filename else ""

    # Image attachments → OCR
    if ct in _IMAGE_MIMES or ext in _IMAGE_EXTS:
        logger.info("Processing image attachment: %s (%s)", filename, ct)
        text = _ocr_image_gpt4o(payload, filename)
        if text:
            return f"\n[ATTACHMENT: {filename} — Image OCR]\n{text}\n[/ATTACHMENT]"
        return ""

    # Audio attachments → Whisper
    if ct in _AUDIO_MIMES or ext in _AUDIO_EXTS:
        logger.info("Processing audio attachment: %s (%s)", filename, ct)
        text = _transcribe_audio(payload, filename)
        if text:
            return f"\n[ATTACHMENT: {filename} — Audio Transcript]\n{text}\n[/ATTACHMENT]"
        return ""

    # PDF attachments → extract text
    if ct == "application/pdf" or ext == ".pdf":
        logger.info("Processing PDF attachment: %s", filename)
        try:
            reader = PdfReader(io.BytesIO(payload))
            pages_text = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    pages_text.append(t)
            if pages_text:
                combined = "\n".join(pages_text)
                return f"\n[ATTACHMENT: {filename} — PDF Text]\n{combined}\n[/ATTACHMENT]"
        except Exception as exc:
            logger.warning("PDF attachment extraction failed for %s: %s", filename, exc)
        return ""

    logger.debug("Skipping unsupported attachment type: %s (%s)", filename, ct)
    return ""


# ─────────────────────────────────────────────────────────────────────
#  1. Plain text loader
# ─────────────────────────────────────────────────────────────────────
def load_texts(doc_dir: str | Path) -> list[Document]:
    """Load .txt files from a directory into Documents."""
    doc_dir = Path(doc_dir)
    docs: list[Document] = []
    for txt_path in sorted(doc_dir.glob("*.txt")):
        content = txt_path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            continue
        docs.append(Document(
            page_content=content,
            metadata={"source": txt_path.name, "page": 1, "file_type": "txt"},
        ))
    logger.info("Loaded %d .txt files from %s", len(docs), doc_dir)
    return docs


# ─────────────────────────────────────────────────────────────────────
#  2. PDF loader (with OCR fallback for scanned pages)
# ─────────────────────────────────────────────────────────────────────
def load_pdfs(doc_dir: str | Path) -> list[Document]:
    """Load .pdf files; falls back to OCR for pages with no extractable text."""
    doc_dir = Path(doc_dir)
    docs: list[Document] = []
    for pdf_path in sorted(doc_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # OCR fallback for scanned pages
            if not text.strip():
                try:
                    from PIL import Image
                    import io
                    for img_obj in page.images:
                        pil_img = Image.open(io.BytesIO(img_obj.data))
                        ocr_text = _ocr_image(pil_img)
                        if ocr_text:
                            text += "\n" + ocr_text
                except Exception as exc:
                    logger.debug("PDF OCR fallback skipped for %s p.%d: %s", pdf_path.name, page_num, exc)
            if not text.strip():
                continue
            docs.append(Document(
                page_content=text,
                metadata={"source": pdf_path.name, "page": page_num,
                           "total_pages": len(reader.pages), "file_type": "pdf"},
            ))
    logger.info("Loaded %d PDF pages from %s", len(docs), doc_dir)
    return docs


# ─────────────────────────────────────────────────────────────────────
#  3. Word (.docx) loader
# ─────────────────────────────────────────────────────────────────────
def load_docx(doc_dir: str | Path) -> list[Document]:
    """Load .docx files — extracts paragraphs and tables."""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        logger.warning("python-docx not installed — skipping .docx files. Run: pip install python-docx")
        return []

    doc_dir = Path(doc_dir)
    docs: list[Document] = []
    for docx_path in sorted(doc_dir.glob("*.docx")):
        try:
            word_doc = DocxDocument(str(docx_path))
            paragraphs: list[str] = []

            # Extract paragraphs
            for para in word_doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)

            # Extract tables
            for table in word_doc.tables:
                table_rows: list[str] = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    table_rows.append(" | ".join(cells))
                if table_rows:
                    paragraphs.append("\n[TABLE]\n" + "\n".join(table_rows) + "\n[/TABLE]")

            content = "\n".join(paragraphs)
            if not content.strip():
                continue

            docs.append(Document(
                page_content=content,
                metadata={"source": docx_path.name, "page": 1, "file_type": "docx"},
            ))
        except Exception as exc:
            logger.error("Failed to load %s: %s", docx_path.name, exc)

    logger.info("Loaded %d .docx files from %s", len(docs), doc_dir)
    return docs


# ─────────────────────────────────────────────────────────────────────
#  4. Image loader (OCR via Tesseract)
# ─────────────────────────────────────────────────────────────────────
def load_images(doc_dir: str | Path) -> list[Document]:
    """Load image files and extract text via Tesseract OCR."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed — skipping images. Run: pip install Pillow")
        return []

    doc_dir = Path(doc_dir)
    docs: list[Document] = []
    image_files = []
    for ext in _IMAGE_EXTS:
        image_files.extend(doc_dir.glob(f"*{ext}"))

    for img_path in sorted(image_files):
        try:
            img = Image.open(str(img_path))
            text = _ocr_image(img)
            if not text:
                logger.warning("No text extracted from image %s", img_path.name)
                continue
            docs.append(Document(
                page_content=text,
                metadata={"source": img_path.name, "page": 1, "file_type": "image",
                           "image_size": f"{img.width}x{img.height}"},
            ))
        except Exception as exc:
            logger.error("Failed to process image %s: %s", img_path.name, exc)

    logger.info("Loaded %d images (OCR) from %s", len(docs), doc_dir)
    return docs


# ─────────────────────────────────────────────────────────────────────
#  5. Email (.eml) loader
# ─────────────────────────────────────────────────────────────────────
def _extract_email_text(msg: email.message.EmailMessage) -> tuple[str, list[str]]:
    """Extract the plain-text body and process attachments from an email.

    Returns:
        (full_text, attachment_names) where full_text includes any text
        extracted from image/audio/PDF attachments.
    """
    body_parts: list[str] = []
    attachment_names: list[str] = []

    # Headers
    headers = []
    for h in ("From", "To", "Cc", "Date", "Subject"):
        val = msg.get(h)
        if val:
            headers.append(f"{h}: {val}")
    if headers:
        body_parts.append("\n".join(headers))

    # Body + attachments
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            fn = part.get_filename()

            if fn:
                # This is an attachment
                attachment_names.append(fn)
                payload = part.get_payload(decode=True)
                if payload:
                    att_text = _process_attachment(ct, payload, fn)
                    if att_text:
                        body_parts.append(att_text)
                continue

            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode("utf-8", errors="replace"))
            elif ct == "text/html":
                # Basic HTML stripping as fallback
                payload = part.get_payload(decode=True)
                if payload:
                    import re
                    html = payload.decode("utf-8", errors="replace")
                    text = re.sub(r"<[^>]+>", " ", html)
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        body_parts.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body_parts.append(payload.decode("utf-8", errors="replace"))

    return "\n\n".join(body_parts), attachment_names


def load_eml(doc_dir: str | Path) -> list[Document]:
    """Load .eml email files with full attachment processing."""
    doc_dir = Path(doc_dir)
    docs: list[Document] = []
    for eml_path in sorted(doc_dir.glob("*.eml")):
        try:
            raw = eml_path.read_bytes()
            msg = email.message_from_bytes(raw, policy=policy.default)
            text, attachments = _extract_email_text(msg)
            if not text.strip():
                continue

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": eml_path.name, "page": 1, "file_type": "eml",
                    "email_from": msg.get("From", ""),
                    "email_to": msg.get("To", ""),
                    "email_subject": msg.get("Subject", ""),
                    "email_date": msg.get("Date", ""),
                    "attachments": ", ".join(attachments) if attachments else "",
                },
            ))
        except Exception as exc:
            logger.error("Failed to load %s: %s", eml_path.name, exc)

    logger.info("Loaded %d .eml files from %s", len(docs), doc_dir)
    return docs


# ─────────────────────────────────────────────────────────────────────
#  6. Outlook message (.msg) loader
# ─────────────────────────────────────────────────────────────────────
def load_msg(doc_dir: str | Path) -> list[Document]:
    """Load .msg Outlook message files with attachment processing."""
    try:
        import extract_msg
    except ImportError:
        logger.warning("extract-msg not installed — skipping .msg files. Run: pip install extract-msg")
        return []

    doc_dir = Path(doc_dir)
    docs: list[Document] = []
    for msg_path in sorted(doc_dir.glob("*.msg")):
        try:
            msg = extract_msg.Message(str(msg_path))
            parts: list[str] = []

            # Headers
            headers = []
            if msg.sender:    headers.append(f"From: {msg.sender}")
            if msg.to:        headers.append(f"To: {msg.to}")
            if msg.cc:        headers.append(f"Cc: {msg.cc}")
            if msg.date:      headers.append(f"Date: {msg.date}")
            if msg.subject:   headers.append(f"Subject: {msg.subject}")
            if headers:
                parts.append("\n".join(headers))

            # Body
            body = msg.body or ""
            if body.strip():
                parts.append(body)

            # Process attachments (image OCR, audio transcription, PDF)
            att_names: list[str] = []
            for att in (msg.attachments or []):
                fn = att.longFilename or att.shortFilename or "unnamed"
                att_names.append(fn)
                if settings.process_attachments:
                    try:
                        att_data = att.data
                        if att_data:
                            # Guess content type from extension
                            ext = Path(fn).suffix.lower()
                            mime_map = {
                                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                                ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff",
                                ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/m4a",
                                ".ogg": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm",
                                ".pdf": "application/pdf",
                            }
                            ct = mime_map.get(ext, "application/octet-stream")
                            att_text = _process_attachment(ct, att_data, fn)
                            if att_text:
                                parts.append(att_text)
                    except Exception as exc:
                        logger.warning("Failed to process .msg attachment %s: %s", fn, exc)

            content = "\n\n".join(parts)
            if not content.strip():
                msg.close()
                continue

            docs.append(Document(
                page_content=content,
                metadata={
                    "source": msg_path.name, "page": 1, "file_type": "msg",
                    "email_from": msg.sender or "",
                    "email_to": msg.to or "",
                    "email_subject": msg.subject or "",
                    "email_date": str(msg.date or ""),
                    "attachments": ", ".join(att_names) if att_names else "",
                },
            ))
            msg.close()
        except Exception as exc:
            logger.error("Failed to load %s: %s", msg_path.name, exc)

    logger.info("Loaded %d .msg files from %s", len(docs), doc_dir)
    return docs


# ─────────────────────────────────────────────────────────────────────
#  7. Outlook PST loader (Windows only — uses Outlook COM)
# ─────────────────────────────────────────────────────────────────────
def load_pst(doc_dir: str | Path) -> list[Document]:
    """Load .pst Outlook data files via Outlook COM automation (Windows only).

    Requires Microsoft Outlook installed on the machine.
    """
    if platform.system() != "Windows":
        logger.warning("PST loading requires Windows + Outlook. Skipping .pst files.")
        return []

    doc_dir = Path(doc_dir)
    pst_files = sorted(doc_dir.glob("*.pst"))
    if not pst_files:
        return []

    try:
        import win32com.client
    except ImportError:
        logger.warning("pywin32 not installed — skipping .pst files. Run: pip install pywin32")
        return []

    docs: list[Document] = []
    try:
        outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    except Exception as exc:
        logger.error("Cannot start Outlook COM — is Outlook installed? %s", exc)
        return []

    for pst_path in pst_files:
        pst_abs = str(pst_path.resolve())
        try:
            # Add the PST as a store
            outlook.AddStore(pst_abs)
            # Find the store we just added (it appears as the last folder)
            store = None
            for folder in outlook.Folders:
                # Match by checking all folders; the added PST is typically last
                store = folder
            if store is None:
                logger.warning("Could not find added PST store for %s", pst_path.name)
                continue

            # Recursively extract emails from all sub-folders
            _extract_pst_folder(store, pst_path.name, docs)

            # Remove the PST store
            outlook.RemoveStore(store)
        except Exception as exc:
            logger.error("Failed to process PST %s: %s", pst_path.name, exc)

    logger.info("Loaded %d emails from .pst files in %s", len(docs), doc_dir)
    return docs


def _extract_pst_folder(folder, pst_name: str, docs: list[Document], max_emails: int = 5000) -> None:
    """Recursively extract emails from an Outlook folder tree."""
    try:
        items = folder.Items
        for i in range(1, min(items.Count + 1, max_emails + 1)):
            try:
                item = items.Item(i)
                if item.Class == 43:  # olMail
                    parts: list[str] = []
                    parts.append(f"From: {getattr(item, 'SenderName', '')}")
                    parts.append(f"To: {getattr(item, 'To', '')}")
                    if getattr(item, 'CC', ''):
                        parts.append(f"Cc: {item.CC}")
                    parts.append(f"Date: {getattr(item, 'ReceivedTime', '')}")
                    parts.append(f"Subject: {getattr(item, 'Subject', '')}")
                    body = getattr(item, "Body", "") or ""
                    if body.strip():
                        parts.append(f"\n{body}")

                    # Process attachments (image OCR, audio transcription)
                    att_names = []
                    for j in range(1, item.Attachments.Count + 1):
                        att = item.Attachments.Item(j)
                        fn = att.FileName
                        att_names.append(fn)

                        if settings.process_attachments:
                            ext = Path(fn).suffix.lower()
                            if ext in _IMAGE_EXTS or ext in _AUDIO_EXTS or ext == ".pdf":
                                try:
                                    # Save attachment to temp file, read bytes
                                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                                        att.SaveAsFile(tmp.name)
                                        tmp_path = tmp.name
                                    att_data = Path(tmp_path).read_bytes()
                                    os.unlink(tmp_path)

                                    mime_map = {
                                        ".png": "image/png", ".jpg": "image/jpeg",
                                        ".jpeg": "image/jpeg", ".gif": "image/gif",
                                        ".bmp": "image/bmp", ".tiff": "image/tiff",
                                        ".mp3": "audio/mpeg", ".wav": "audio/wav",
                                        ".m4a": "audio/m4a", ".ogg": "audio/ogg",
                                        ".pdf": "application/pdf",
                                    }
                                    ct = mime_map.get(ext, "application/octet-stream")
                                    att_text = _process_attachment(ct, att_data, fn)
                                    if att_text:
                                        parts.append(att_text)
                                except Exception as exc:
                                    logger.warning("PST attachment processing failed for %s: %s", fn, exc)

                    content = "\n".join(parts)
                    if content.strip():
                        docs.append(Document(
                            page_content=content,
                            metadata={
                                "source": pst_name,
                                "page": i,
                                "file_type": "pst",
                                "email_from": getattr(item, "SenderName", ""),
                                "email_to": getattr(item, "To", ""),
                                "email_subject": getattr(item, "Subject", ""),
                                "email_date": str(getattr(item, "ReceivedTime", "")),
                                "folder": folder.Name,
                                "attachments": ", ".join(att_names) if att_names else "",
                            },
                        ))
            except Exception:
                continue
    except Exception:
        pass

    # Recurse into sub-folders
    try:
        for j in range(1, folder.Folders.Count + 1):
            _extract_pst_folder(folder.Folders.Item(j), pst_name, docs, max_emails)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
#  Master loader — dispatches to all format-specific loaders
# ─────────────────────────────────────────────────────────────────────
def load_documents(doc_dir: str | Path) -> list[Document]:
    """Load all supported file types from a directory.

    Supported: .txt, .pdf, .docx, .png/.jpg/.tiff/.bmp (OCR),
               .eml, .msg, .pst
    """
    doc_dir = Path(doc_dir)
    docs: list[Document] = []

    # Count files by type for the summary
    file_counts: dict[str, int] = {}

    # Text
    if any(doc_dir.glob("*.txt")):
        loaded = load_texts(doc_dir)
        file_counts["txt"] = len(loaded)
        docs.extend(loaded)

    # PDF
    if any(doc_dir.glob("*.pdf")):
        loaded = load_pdfs(doc_dir)
        file_counts["pdf"] = len(loaded)
        docs.extend(loaded)

    # Word
    if any(doc_dir.glob("*.docx")):
        loaded = load_docx(doc_dir)
        file_counts["docx"] = len(loaded)
        docs.extend(loaded)

    # Images (OCR)
    has_images = any(any(doc_dir.glob(f"*{ext}")) for ext in _IMAGE_EXTS)
    if has_images:
        loaded = load_images(doc_dir)
        file_counts["images"] = len(loaded)
        docs.extend(loaded)

    # Email (.eml)
    if any(doc_dir.glob("*.eml")):
        loaded = load_eml(doc_dir)
        file_counts["eml"] = len(loaded)
        docs.extend(loaded)

    # Outlook (.msg)
    if any(doc_dir.glob("*.msg")):
        loaded = load_msg(doc_dir)
        file_counts["msg"] = len(loaded)
        docs.extend(loaded)

    # Outlook PST
    if any(doc_dir.glob("*.pst")):
        loaded = load_pst(doc_dir)
        file_counts["pst"] = len(loaded)
        docs.extend(loaded)

    summary = " | ".join(f"{ext}: {cnt}" for ext, cnt in file_counts.items() if cnt)
    logger.info("Total documents loaded: %d  (%s)", len(docs), summary or "none")
    return docs


def chunk_documents(docs: list[Document], chunk_size: int = settings.chunk_size, chunk_overlap: int = settings.chunk_overlap) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])
    chunks: list[Document] = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        for idx, text in enumerate(splits):
            chunks.append(Document(page_content=text, metadata={**doc.metadata, "chunk_index": idx}))
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def _get_embedding_model():
    """Return the Cohere embedding model (matching retriever)."""
    try:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(
            model=settings.cohere_embed_model,
            cohere_api_key=settings.cohere_api_key,
        )
    except Exception:
        # Fallback to HuggingFace (WARNING: 384-dim, won't match Cohere 1024-dim index!)
        logger.warning("Cohere embeddings unavailable, falling back to HuggingFace (dim mismatch risk!)")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def embed_chunks(chunks: list[Document], batch_size: int = 48) -> list[dict]:
    """Embed chunks in batches with rate-limit retry."""
    import time

    model = _get_embedding_model()
    texts = [c.page_content for c in chunks]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for attempt in range(5):
            try:
                embs = model.embed_documents(batch)
                all_embeddings.extend(embs)
                logger.info("Embedded batch %d/%d (%d chunks)", batch_num, total_batches, len(batch))
                break
            except Exception as exc:
                if "429" in str(exc) or "rate" in str(exc).lower():
                    wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                    logger.warning("Rate limited on batch %d, retrying in %ds...", batch_num, wait)
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed to embed batch {batch_num} after 5 retries")

        # Small delay between batches to stay under rate limit
        if i + batch_size < len(texts):
            time.sleep(2)

    rows = []
    for chunk, emb in zip(chunks, all_embeddings):
        meta = chunk.metadata
        rows.append({
            "content": chunk.page_content, "embedding": emb,
            "source": meta.get("source", "unknown"), "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"), "doc_type": meta.get("doc_type"),
            "entity_name": meta.get("entity_name"), "effective_date": meta.get("effective_date"),
            "metadata_extra": {k: v for k, v in meta.items() if k not in {"source", "page", "chunk_index", "doc_type", "entity_name", "effective_date"}},
        })
    return rows


def ingest(doc_dir: str | Path, *, doc_type: str | None = None, entity_name: str | None = None) -> int:
    engine = init_db()
    raw_docs = load_documents(doc_dir)
    if doc_type or entity_name:
        for d in raw_docs:
            if doc_type: d.metadata["doc_type"] = doc_type
            if entity_name: d.metadata["entity_name"] = entity_name
    chunks = chunk_documents(raw_docs)
    rows = embed_chunks(chunks)
    return upsert_chunks(engine, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into pgvector")
    parser.add_argument("--dir", dest="doc_dir", help="Directory with .txt and/or .pdf files")
    parser.add_argument("--pdf-dir", dest="pdf_dir", help="(Legacy) Directory with PDFs")
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--entity-name", default=None)
    args = parser.parse_args()
    doc_dir = args.doc_dir or args.pdf_dir
    if not doc_dir:
        parser.error("Provide --dir or --pdf-dir")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    count = ingest(doc_dir, doc_type=args.doc_type, entity_name=args.entity_name)
    print(f"\n✓ Ingested {count} chunks into pgvector.")


if __name__ == "__main__":
    main()
