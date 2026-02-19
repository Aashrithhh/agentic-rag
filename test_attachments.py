"""Test attachment processing in the email loader."""
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from app.ingest import load_eml

docs = load_eml("data/test_attachments")
print(f"\nLoaded {len(docs)} documents")
for doc in docs:
    src = doc.metadata.get("source", "?")
    att = doc.metadata.get("attachments", "")
    print(f"\n--- Source: {src} ---")
    print(f"Attachments: {att}")
    print(f"Content length: {len(doc.page_content)} chars")
    print(doc.page_content[:2000])
    if "[ATTACHMENT" in doc.page_content:
        print("\n*** ATTACHMENT TEXT EXTRACTED SUCCESSFULLY ***")
    else:
        print("\n*** NO ATTACHMENT TEXT FOUND ***")
