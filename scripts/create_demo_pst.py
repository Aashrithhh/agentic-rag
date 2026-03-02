"""Create a demo .pst file with realistic compliance investigation emails.

Scenario: TechCorp Q4 2024 Internal Financial Irregularities Investigation
- Unauthorized vendor payments discovered
- Whistleblower tip, legal hold, board notification, forensic audit trail

Requirements: Microsoft Outlook must be installed.
Usage:
    python scripts/create_demo_pst.py
    python scripts/create_demo_pst.py --output data/demo_pst/techcorp_investigation.pst
    python scripts/create_demo_pst.py --upload   # also upload to Azure blob
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Email content: TechCorp compliance investigation scenario ────────────────

EMAILS = [
    {
        "folder": "Inbox",
        "subject": "CONFIDENTIAL: Anonymous Tip — Vendor Payment Irregularities",
        "sender_name": "Ethics Hotline",
        "sender_email": "ethics-hotline@techcorp-anonymous.com",
        "to": "james.martinez@techcorp.com",
        "cc": "",
        "date": datetime(2024, 10, 3, 9, 15, 0),
        "body": """Dear Compliance Officer,

This message is submitted anonymously through the TechCorp Ethics Hotline.

I have observed a pattern of suspicious vendor payments that I believe warrants immediate investigation:

1. VENDOR: "Meridian Consulting LLC" — received $380,000 in Q3 2024 for "strategic advisory services." I cannot locate any approved contract or SOW for this engagement. The vendor address on file is a residential property in Austin, TX.

2. VENDOR: "Pacific Bridge Solutions" — received $215,000 in August 2024 for "IT infrastructure assessment." Our IT department has no record of receiving any deliverable or final report from this vendor.

3. Both vendors were added to the approved vendor list on the same day (June 14, 2024) and were approved by a single finance manager without the standard dual-approval required for vendors exceeding $100,000 threshold.

I am concerned that these payments may represent fraudulent activity, and I urge you to investigate before further disbursements are made. Q4 budget cycle is upcoming and I believe additional payments may be planned.

Please protect my identity. I am prepared to provide additional information if needed through this hotline.

— Anonymous Employee""",
    },
    {
        "folder": "Sent Items",
        "subject": "RE: CONFIDENTIAL: Anonymous Tip — Vendor Payment Irregularities",
        "sender_name": "James Martinez",
        "sender_email": "james.martinez@techcorp.com",
        "to": "robert.chen@techcorp.com",
        "cc": "sarah.kim@techcorp.com",
        "date": datetime(2024, 10, 3, 11, 42, 0),
        "body": """Robert,

I need to bring an urgent matter to your attention. This morning our Ethics Hotline received a tip alleging material financial irregularities involving two vendors: Meridian Consulting LLC and Pacific Bridge Solutions.

Combined exposure: approximately $595,000 in Q3 payments with possible Q4 follow-on.

I have looped in Sarah Kim (Legal) as we need to determine the appropriate response protocol. I recommend we:

1. Immediately freeze any pending payments to these two vendors
2. Pull all payment records, contracts, and approvals associated with both entities
3. Engage outside forensic accountants to review independently of our Finance team
4. Issue a legal hold notice covering all communications related to these vendors

I can have a preliminary findings memo to you by end of week.

James Martinez
Chief Compliance Officer | TechCorp
Direct: +1 (512) 555-0142""",
    },
    {
        "folder": "Inbox",
        "subject": "RE: CONFIDENTIAL: Anonymous Tip — Vendor Payment Irregularities",
        "sender_name": "Robert Chen",
        "sender_email": "robert.chen@techcorp.com",
        "to": "james.martinez@techcorp.com",
        "cc": "sarah.kim@techcorp.com; amanda.foster@techcorp.com",
        "date": datetime(2024, 10, 3, 14, 5, 0),
        "body": """James, Sarah,

Thank you for the immediate escalation. This is extremely concerning.

I am authorizing you to proceed with a full internal investigation. Please treat this as Priority 1.

Amanda Foster (CFO) is now copied and will cooperate fully with your requests for financial records. She is not a subject of the investigation.

Actions I am authorizing immediately:
- Payment freeze on both vendors effective immediately
- Forensic accountant engagement: please use Thornton & Associates (previously vetted, independent)
- Legal hold issuance — Sarah, please draft this today
- Preserve all email, financial, and system logs related to these vendors

I want weekly status updates and I expect this to be kept strictly confidential. Board notification will be required if the investigation confirms wrongdoing — we will discuss timing with Sarah.

Robert Chen
CEO | TechCorp""",
    },
    {
        "folder": "Sent Items",
        "subject": "LEGAL HOLD NOTICE — Investigation Matter TCI-2024-001",
        "sender_name": "Sarah Kim",
        "sender_email": "sarah.kim@techcorp.com",
        "to": "all-finance@techcorp.com; all-it@techcorp.com; james.martinez@techcorp.com",
        "cc": "robert.chen@techcorp.com",
        "date": datetime(2024, 10, 4, 8, 30, 0),
        "body": """LEGAL HOLD NOTICE — PLEASE READ AND RETAIN

Matter Reference: TCI-2024-001
Issued by: TechCorp Legal Department
Date: October 4, 2024

This Legal Hold Notice applies to all TechCorp employees, contractors, and agents who may possess documents or information related to the following:

SUBJECT MATTER:
- All communications, contracts, invoices, purchase orders, and payment records relating to vendors Meridian Consulting LLC and Pacific Bridge Solutions
- All vendor onboarding documentation for the above entities
- All finance system approvals, audit logs, and authorization records from January 1, 2024 to present
- All emails, chat messages, or other communications referencing the above vendors

YOUR OBLIGATIONS:
Effective immediately, you are required to:
1. PRESERVE all documents, records, and data (including electronic records) related to the above subject matter
2. SUSPEND any routine deletion, overwriting, or destruction of such records
3. NOT discuss the contents of this notice with anyone outside TechCorp unless directed by Legal

Failure to comply with this Legal Hold may result in disciplinary action and could constitute spoliation of evidence.

If you have questions, contact Sarah Kim, General Counsel, at sarah.kim@techcorp.com.

Sarah Kim
General Counsel | TechCorp""",
    },
    {
        "folder": "Sent Items",
        "subject": "Vendor Records Request — TCI-2024-001",
        "sender_name": "James Martinez",
        "sender_email": "james.martinez@techcorp.com",
        "to": "amanda.foster@techcorp.com",
        "cc": "sarah.kim@techcorp.com",
        "date": datetime(2024, 10, 4, 10, 15, 0),
        "body": """Amanda,

Per our investigation (TCI-2024-001), I need the following records from Finance as soon as possible:

1. All payment runs to Meridian Consulting LLC and Pacific Bridge Solutions from Jan 1, 2024 to present — include invoice numbers, amounts, GL codes, approver names
2. Original vendor onboarding files for both entities (W-9, certificate of incorporation, insurance certs, bank details)
3. The approval workflow records showing who authorized these vendors for the approved vendor list on June 14, 2024
4. All approved purchase orders and contracts associated with payments to these vendors
5. Any communications between Finance team members and these vendors

Please do not alert the approving finance manager until I advise — we need to gather records before any interviews.

Thornton & Associates will begin forensic review on Monday. They will contact you directly for system access.

James Martinez
Chief Compliance Officer""",
    },
    {
        "folder": "Inbox",
        "subject": "RE: Vendor Records Request — TCI-2024-001",
        "sender_name": "Amanda Foster",
        "sender_email": "amanda.foster@techcorp.com",
        "to": "james.martinez@techcorp.com",
        "cc": "sarah.kim@techcorp.com",
        "date": datetime(2024, 10, 4, 16, 50, 0),
        "body": """James,

I have pulled all available records. Here is what I found — and it is deeply troubling.

MERIDIAN CONSULTING LLC:
- Total paid Q2-Q3 2024: $380,000 (4 invoices, $95,000 each)
- No executed MSA or SOW on file — only a single-page "engagement letter" signed by one Finance Director
- Bank ACH routing leads to a First National account in Delaware
- Vendor was approved June 14 by Finance Director Mark Holloway without the standard dual-sign-off
- No deliverables received or documented in our project management system

PACIFIC BRIDGE SOLUTIONS:
- Total paid Q3 2024: $215,000 (2 invoices)
- No contract on file whatsoever
- Same approver: Mark Holloway
- Vendor onboarding file missing certificate of incorporation — only a handwritten note saying "verified by MH"
- IT has confirmed NO engagement with this vendor and NO deliverables received

RED FLAGS SUMMARY:
- Same sole approver for both non-standard vendors
- No deliverables documented for $595,000 in payments
- Onboarding files incomplete/missing for both
- Both vendors added same day by same approver

I am gravely concerned. I have pulled Mark Holloway's access to the payment system pending your investigation guidance.

Amanda Foster
CFO | TechCorp""",
    },
    {
        "folder": "Inbox",
        "subject": "IT Security Alert — Unusual System Access Pattern Detected",
        "sender_name": "Tom Bradley",
        "sender_email": "tom.bradley@techcorp.com",
        "to": "james.martinez@techcorp.com",
        "cc": "sarah.kim@techcorp.com; amanda.foster@techcorp.com",
        "date": datetime(2024, 10, 7, 9, 0, 0),
        "body": """James,

Per your request, IT Security has reviewed system logs for activity related to the TCI-2024-001 investigation. Findings below.

ANOMALOUS ACCESS — MARK HOLLOWAY (mholloway@techcorp.com):

1. VENDOR PORTAL ACCESS:
   - 47 vendor portal logins from unusual IP addresses outside corporate VPN (June–September 2024)
   - Vendor profile edits for Meridian Consulting LLC: bank account changed 3 times between June and August 2024
   - Same IP address used to access Pacific Bridge Solutions vendor profile

2. EMAIL ACTIVITY:
   - 23 emails sent to personal Gmail account (mholloway.personal@gmail.com) from corporate email — large attachments including what appear to be vendor payment summaries and ACH routing instructions
   - Emails sent at unusual hours (2:00–4:00 AM local time) on 8 separate occasions

3. FINANCE SYSTEM:
   - Approval audit trail shows Holloway's credentials used to approve his own vendor additions (standard system should require a second approver — this was bypassed using an administrative override code normally reserved for emergency payments)

4. VPN & DEVICE:
   - 3 logins from foreign IP addresses (Netherlands, Singapore) using Holloway's credentials in September 2024

I recommend immediate suspension of all Holloway system credentials and preservation of all logs.

Tom Bradley
Director of IT Security | TechCorp""",
    },
    {
        "folder": "Sent Items",
        "subject": "Investigation Update — TCI-2024-001 — Week 1",
        "sender_name": "James Martinez",
        "sender_email": "james.martinez@techcorp.com",
        "to": "robert.chen@techcorp.com",
        "cc": "sarah.kim@techcorp.com; amanda.foster@techcorp.com",
        "date": datetime(2024, 10, 10, 15, 30, 0),
        "body": """Robert,

Week 1 Investigation Update — Matter TCI-2024-001

EXECUTIVE SUMMARY:
Evidence gathered to date strongly suggests a scheme of fraudulent vendor payments orchestrated by Finance Director Mark Holloway. Total suspected fraudulent disbursements: $595,000. The scheme appears to involve shell companies receiving payment for services never rendered.

KEY FINDINGS TO DATE:
1. Both vendors (Meridian Consulting LLC and Pacific Bridge Solutions) were established by Holloway without valid contracts or deliverables
2. Holloway bypassed dual-approval controls using an emergency override code
3. IT Security has confirmed Holloway forwarded payment-related documents to a personal Gmail account
4. Bank records subpoenaed by outside counsel show ACH payments routed to accounts with no apparent business activity
5. Holloway's corporate credentials were used from foreign IP addresses on multiple occasions

ACTIONS TAKEN:
- Holloway placed on administrative leave (legal hold in place on all his devices/accounts)
- Law enforcement referral being prepared by Sarah Kim's team
- Thornton & Associates forensic review ongoing — full report expected Oct 25
- Q3 financials under review; Amanda Foster preparing restatement analysis

RECOMMENDED NEXT STEPS:
1. Board Audit Committee notification (Sarah is preparing disclosure materials)
2. Cyber insurance claim notification (policy covers employee fraud)
3. Bank hold/recovery attempt on remaining wire funds

I will provide another update next Friday.

James Martinez
Chief Compliance Officer""",
    },
    {
        "folder": "Sent Items",
        "subject": "Board Audit Committee — Urgent Disclosure: Matter TCI-2024-001",
        "sender_name": "Robert Chen",
        "sender_email": "robert.chen@techcorp.com",
        "to": "board-audit-committee@techcorp.com",
        "cc": "sarah.kim@techcorp.com; james.martinez@techcorp.com",
        "date": datetime(2024, 10, 11, 10, 0, 0),
        "body": """Members of the Audit Committee,

I am writing to inform you of a material internal matter requiring your immediate awareness and oversight per our governance obligations.

MATTER: TCI-2024-001 — Internal Financial Irregularities Investigation

SUMMARY:
On October 3, 2024, TechCorp's Ethics Hotline received an anonymous tip alleging that a Finance Director had caused the company to make approximately $595,000 in payments to two vendors — Meridian Consulting LLC and Pacific Bridge Solutions — for services that were never rendered. The Compliance and Legal teams have been conducting a full investigation since October 4.

PRELIMINARY FINDINGS:
Internal and forensic review has uncovered substantial evidence of a deliberate fraud scheme including: fabricated vendor relationships, bypassed financial controls, unauthorized approval overrides, and suspicious fund routing. The employee in question has been placed on administrative leave.

LAW ENFORCEMENT:
We expect to file a formal complaint with the FBI Financial Crimes unit by end of next week. Outside counsel (Perkins & Cowell LLP) is coordinating.

FINANCIAL IMPACT:
Preliminary estimate of direct loss: $595,000. We do not believe Q3 reported financials require restatement at this time, however this is under active review.

I will be available for a call at your earliest convenience. Please treat this communication as strictly confidential attorney-client privileged communication.

Robert Chen
CEO | TechCorp""",
    },
    {
        "folder": "Inbox",
        "subject": "Thornton & Associates — Forensic Review Final Report: TCI-2024-001",
        "sender_name": "David Thornton",
        "sender_email": "d.thornton@thorntonassoc.com",
        "to": "james.martinez@techcorp.com",
        "cc": "sarah.kim@techcorp.com; amanda.foster@techcorp.com",
        "date": datetime(2024, 10, 25, 16, 0, 0),
        "body": """Dear Mr. Martinez,

Please find below the executive summary of our forensic review completed in connection with Matter TCI-2024-001. A full written report (87 pages) will be delivered via secure file transfer.

FORENSIC REVIEW EXECUTIVE SUMMARY
Thornton & Associates | October 25, 2024

ENGAGEMENT SCOPE:
Review of all financial transactions, vendor records, system access logs, and email communications related to Meridian Consulting LLC and Pacific Bridge Solutions for the period January 1, 2023 – October 10, 2024.

TOTAL FRAUDULENT DISBURSEMENTS IDENTIFIED:
- Meridian Consulting LLC: $380,000 (Q2–Q3 2024)
- Pacific Bridge Solutions: $215,000 (Q3 2024)
- Total: $595,000

SCHEME MECHANICS:
The fraud was perpetrated by a single employee (Finance Director) who: (1) established or controlled shell vendor entities; (2) exploited an emergency override function in TechCorp's AP system to bypass dual-approval controls; (3) created fabricated engagement letters to support payment requests; and (4) routed funds to personal bank accounts via a nominee LLC structure.

BENEFICIAL OWNERSHIP FINDINGS:
Meridian Consulting LLC is registered in Delaware. Public records and bank subpoena results indicate the sole beneficial owner is an immediate family member of the Finance Director.
Pacific Bridge Solutions: shell entity with no operating activity. Sole signatory is associated with the Finance Director through prior shared addresses.

INTERNAL CONTROL FAILURES IDENTIFIED:
1. Emergency override function had no compensating monitoring control
2. Vendor onboarding required only one-level approval for expedited adds (policy gap)
3. No periodic vendor payment audits were conducted
4. Anomalous access patterns (after-hours, foreign IPs) were not flagged by SIEM

RECOMMENDATIONS:
1. Immediate remediation of emergency override controls
2. Mandatory dual-sign-off for all new vendor additions regardless of amount
3. Quarterly vendor payment audits by Internal Audit
4. Enhanced SIEM alerting for anomalous financial system access
5. Anti-fraud training for all Finance staff

We are available to present these findings to the Board at your convenience.

David Thornton, CFE, CPA
Managing Partner | Thornton & Associates""",
    },
    {
        "folder": "Sent Items",
        "subject": "FINAL: Investigation Summary & Remediation Plan — TCI-2024-001",
        "sender_name": "James Martinez",
        "sender_email": "james.martinez@techcorp.com",
        "to": "robert.chen@techcorp.com; board-audit-committee@techcorp.com",
        "cc": "sarah.kim@techcorp.com; amanda.foster@techcorp.com",
        "date": datetime(2024, 10, 28, 11, 0, 0),
        "body": """Robert and Audit Committee Members,

I am pleased to present the final summary and remediation plan for Matter TCI-2024-001.

MATTER CLOSED — KEY OUTCOMES:

1. CONFIRMED FRAUD: $595,000 in unauthorized payments to shell vendors controlled by Finance Director Mark Holloway confirmed by forensic review and law enforcement.

2. LAW ENFORCEMENT: FBI Financial Crimes complaint filed October 18, 2024. DOJ has opened a formal investigation. Civil recovery suit filed by Perkins & Cowell in Delaware federal court.

3. TERMINATION: Mark Holloway terminated for cause effective October 7, 2024.

4. FUND RECOVERY: $215,000 recovered via bank hold (Pacific Bridge Solutions account frozen). $380,000 (Meridian) recovery in progress via civil litigation.

5. INSURANCE CLAIM: Cyber/crime insurance claim submitted; carrier has acknowledged coverage. Expected recovery: up to $500,000.

REMEDIATION ACTIONS (COMPLETED):
- Emergency override function disabled pending redesign with mandatory dual-authorization
- All vendor additions now require dual approval + CFO sign-off for amounts over $50,000
- SIEM rules updated to alert on after-hours financial system access and foreign IP logins
- All Finance staff completed mandatory anti-fraud training (October 25)
- Internal Audit quarterly vendor payment audit program launched

CONTROL ENVIRONMENT RATING:
Prior to incident: DEFICIENT in vendor management and AP override controls
Post-remediation: SATISFACTORY — enhancements validated by Thornton & Associates

I recommend the Board formally accept the remediation plan and close this matter pending final fund recovery and criminal proceedings.

Respectfully,
James Martinez
Chief Compliance Officer | TechCorp
TCI-2024-001 — INVESTIGATION CLOSED""",
    },
]


def create_pst(output_path: Path) -> bool:
    """Create a .pst file with the compliance demo emails using Outlook COM."""
    import win32com.client
    import pywintypes

    # MAPI property tags for sender identity (read-only via normal props, settable via PropertyAccessor)
    PR_SENDER_NAME         = "http://schemas.microsoft.com/mapi/proptag/0x0C1A001F"
    PR_SENDER_EMAIL        = "http://schemas.microsoft.com/mapi/proptag/0x0C1F001F"
    PR_SENT_REPR_NAME      = "http://schemas.microsoft.com/mapi/proptag/0x0042001F"
    PR_SENT_REPR_EMAIL     = "http://schemas.microsoft.com/mapi/proptag/0x0065001F"

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file so we start fresh
    if output_path.exists():
        output_path.unlink()
        print(f"  Removed existing file: {output_path}")

    print(f"\nOpening Outlook...")
    outlook = win32com.client.Dispatch("Outlook.Application")
    ns = outlook.GetNamespace("MAPI")

    print(f"Creating PST store at: {output_path}")
    # olStoreUnicode = 3 — Unicode PST (modern format)
    ns.AddStoreEx(str(output_path), 3)

    # Find the newly added store by file path
    store_folder = None
    for folder in ns.Folders:
        try:
            # Store.FilePath is available on store objects
            fp = folder.FilePath if hasattr(folder, "FilePath") else ""
            if str(output_path).lower() in fp.lower():
                store_folder = folder
                break
        except Exception:
            continue

    if store_folder is None:
        # Fallback: take the last folder in namespace (newly added PST appears last)
        last = None
        for folder in ns.Folders:
            last = folder
        store_folder = last
        print(f"  Using store folder (fallback): {store_folder.Name}")
    else:
        print(f"  Found store folder: {store_folder.Name}")

    # Create subfolders
    subfolders: dict[str, object] = {}
    for fname in ("Inbox", "Sent Items"):
        try:
            sf = store_folder.Folders.Add(fname)
        except Exception:
            try:
                sf = store_folder.Folders[fname]
            except Exception:
                sf = store_folder
        subfolders[fname] = sf

    # MAPI PT_SYSTIME property tags (set AFTER Save() to bypass read-only restriction)
    PR_CLIENT_SUBMIT_TIME    = "http://schemas.microsoft.com/mapi/proptag/0x00390040"
    PR_MESSAGE_DELIVERY_TIME = "http://schemas.microsoft.com/mapi/proptag/0x0E060040"

    # Add emails
    print(f"\nAdding {len(EMAILS)} emails...")
    for i, email_data in enumerate(EMAILS, 1):
        try:
            mail = outlook.CreateItem(0)  # 0 = olMailItem
            mail.Subject = email_data["subject"]
            mail.To = email_data["to"]
            if email_data.get("cc"):
                mail.CC = email_data["cc"]

            # Prepend sender header to body (reliable fallback for sender identity)
            header = (
                f"From: {email_data['sender_name']} <{email_data['sender_email']}>\n"
                f"To: {email_data['to']}\n"
            )
            if email_data.get("cc"):
                header += f"CC: {email_data['cc']}\n"
            header += f"Date: {email_data['date'].strftime('%B %d, %Y %I:%M %p')}\n"
            header += f"Subject: {email_data['subject']}\n\n"
            mail.Body = header + email_data["body"]

            # Save first — MAPI properties become settable after Save()
            mail.Save()

            # Now set sender + time properties via PropertyAccessor
            dt = email_data["date"]
            try:
                pa = mail.PropertyAccessor
                pa.SetProperty(PR_SENDER_NAME,          email_data["sender_name"])
                pa.SetProperty(PR_SENDER_EMAIL,         email_data["sender_email"])
                pa.SetProperty(PR_SENT_REPR_NAME,       email_data["sender_name"])
                pa.SetProperty(PR_SENT_REPR_EMAIL,      email_data["sender_email"])
                pa.SetProperty(PR_CLIENT_SUBMIT_TIME,    pywintypes.Time(dt))
                pa.SetProperty(PR_MESSAGE_DELIVERY_TIME, pywintypes.Time(dt))
                mail.Save()  # Re-save to persist MAPI property changes
            except Exception as pe:
                print(f"     (MAPI props partial: {pe})")

            target_folder_name = email_data.get("folder", "Inbox")
            target_folder = subfolders.get(target_folder_name, subfolders["Inbox"])
            mail.Move(target_folder)
            print(f"  [{i:02d}/{len(EMAILS)}] Added: {email_data['subject'][:60]}")
            time.sleep(0.2)
        except Exception as exc:
            print(f"  [{i:02d}/{len(EMAILS)}] FAILED: {email_data['subject'][:50]} — {exc}")

    # Detach PST from Outlook profile so file is released
    try:
        ns.RemoveStore(store_folder)
        print(f"\nDetached PST from Outlook profile.")
    except Exception as exc:
        print(f"\nNote: Could not detach PST: {exc}")

    # Release COM objects so Outlook releases the file lock
    del store_folder, ns, outlook
    import gc
    gc.collect()
    time.sleep(2)  # Give Outlook time to release file handle

    size_kb = output_path.stat().st_size / 1024
    print(f"PST created: {output_path}  ({size_kb:.0f} KB)")
    return True


def upload_to_blob(pst_path: Path, blob_prefix: str = "demo-pst") -> None:
    """Upload the .pst file to Azure Blob Storage.

    Copies the file to a temp location first to avoid Outlook's exclusive lock.
    """
    import shutil, tempfile
    from dotenv import load_dotenv
    load_dotenv()

    from app.blob_storage import get_blob_client, is_blob_mode
    if not is_blob_mode():
        print("Azure Blob Storage not configured — skipping upload.")
        return

    # Copy to temp to bypass any remaining Outlook file lock
    with tempfile.NamedTemporaryFile(suffix=".pst", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    shutil.copy2(pst_path, tmp_path)

    client = get_blob_client()
    blob_name = f"{blob_prefix}/{pst_path.name}"
    size_kb = tmp_path.stat().st_size / 1024
    print(f"\nUploading {pst_path.name} ({size_kb:.0f} KB) -> {blob_name} ...")
    try:
        with open(tmp_path, "rb") as f:
            client._ensure_client()
            client._container_client.upload_blob(blob_name, f, overwrite=True)
        print(f"Upload complete: {blob_name}")
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    import subprocess
    import logging
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Create a demo PST with compliance emails")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/demo_pst/techcorp_investigation_2024.pst"),
        help="Output path for the .pst file",
    )
    parser.add_argument(
        "--upload", "-u",
        action="store_true",
        help="Upload to Azure Blob Storage after creation",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip PST creation — only upload an existing file",
    )
    parser.add_argument(
        "--blob-prefix",
        default="demo-pst",
        help="Blob folder prefix (default: demo-pst)",
    )
    args = parser.parse_args()

    if args.upload_only:
        # Called from subprocess after COM objects are released
        pst_path = args.output.resolve()
        if not pst_path.exists():
            print(f"ERROR: File not found: {pst_path}")
            sys.exit(1)
        upload_to_blob(pst_path, args.blob_prefix)
        print("\nUpload done.")
        return

    print("=" * 60)
    print("TechCorp Demo PST Creator — Compliance Investigation")
    print("=" * 60)
    print(f"Scenario : TCI-2024-001 Financial Fraud Investigation")
    print(f"Emails   : {len(EMAILS)} (Inbox + Sent Items)")
    print(f"Output   : {args.output}")

    ok = create_pst(args.output)

    if ok and args.upload:
        # Run upload in a SEPARATE subprocess so this process (which held
        # Outlook COM objects) has fully exited and released all file handles.
        pst_path = args.output.resolve()
        print(f"\nLaunching upload subprocess (releases Outlook file lock)...")
        result = subprocess.run(
            [sys.executable, __file__,
             "--upload-only",
             "--output", str(pst_path),
             "--blob-prefix", args.blob_prefix],
            capture_output=False,   # let output pass through
        )
        if result.returncode != 0:
            print(f"Upload subprocess failed (exit code {result.returncode})")

    print("\nDone.")


if __name__ == "__main__":
    main()
