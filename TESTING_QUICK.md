# Testing & Verification — Quick Reference

## ✅ Quick Start (No Setup Required)

### 1. Run Quick Verification
```powershell
python tests/verify.py
```
**What it checks:**
- Dependencies installed
- Configuration valid
- Graph compiles
- Demo runs end-to-end

**Expected output:** `4/4 checks passed`

---

### 2. Run Interactive Demo
```powershell
python demo.py
```
**What it demonstrates:**
- Retriever → 2 mock documents
- Grader → returns "no" (irrelevant)
- Rewriter → improves query (self-correction loop)
- Retriever → retrieves again
- Grader → returns "yes"
- Validator → verifies 1 claim
- Generator → produces cited answer

**Expected:** Loop count = 1, Status = pass

---

## 🔧 Advanced Testing (Requires Setup)

### 3. Database Tests (Requires PostgreSQL + pgvector)

**Setup:**
```powershell
# Start PostgreSQL with pgvector
docker run -d --name audit-rag-db `
  -e POSTGRES_USER=testuser `
  -e POSTGRES_PASSWORD=testpass `
  -e POSTGRES_DB=audit_rag `
  -p 5432:5432 `
  pgvector/pgvector:pg17

# Update .env
DATABASE_URL=postgresql+psycopg://testuser:testpass@localhost:5432/audit_rag
```

**Run tests:**
```powershell
python tests/test_db.py
```

**What it tests:**
- Connection to Postgres
- pgvector extension
- Table creation (HNSW + GIN indexes)
- Insert sample data
- Hybrid search query

---

### 4. Create & Ingest Sample Documents

**Create PDFs:**
```powershell
python tests/create_sample_docs.py
```
Creates 3 realistic financial documents:
- `Acme_10K_2024.pdf` — Material weaknesses disclosure
- `Acme_Audit_Report_2024.pdf` — Independent auditor report
- `Revenue_Recognition_Policy.pdf` — Compliance policies

**Ingest into database:**
```powershell
python -m app.ingest --pdf-dir tests/sample_docs --doc-type "10-K" --entity-name "Acme Corp"
```

---

### 5. Run Real Queries (Requires Anthropic API Key)

**Update .env:**
```env
ANTHROPIC_API_KEY=sk-ant-api03-your-real-key-here
```

**Simple query:**
```powershell
python main.py "Did Acme Corp disclose material weaknesses in FY2024?"
```

**With metadata filter:**
```powershell
python main.py "What are the revenue recognition policies?" --doc-type "10-K" --entity "Acme Corp"
```

**Verbose mode (see all nodes):**
```powershell
python main.py "List all audit findings" --verbose
```

**JSON output:**
```powershell
python main.py "Were material weaknesses disclosed?" --json > output.json
```

---

## 🧪 Component-Level Tests

### Test Individual Nodes
```powershell
python tests/test_nodes.py
```
Tests each graph node in isolation:
- Retriever
- Grader (requires API key)
- Rewriter (requires API key) 
- Validator
- Generator (requires API key)

---

## 📊 Expected Results

### ✅ Successful Test Output

**Demo:**
```
[RETRIEVER] Retrieved 2 documents
[GRADER] Score: no
[REWRITER] Rewritten: <improved query>
[RETRIEVER] Retrieved 2 documents
[GRADER] Score: yes
[VALIDATOR] [OK] Verified 1/1 claims
[GENERATOR] Generated answer with citations
[OK] Demo completed successfully!
Loop count: 1
```

**Verification:**
```
✓ All dependencies installed
✓ Config loaded
✓ Graph compiled successfully
✓ Demo passed
SUMMARY: 4/4 checks passed
```

**Real Query (with DB + API):**
```
============================================================
ANSWER
============================================================
**Material Weaknesses Disclosure**

Acme Corp disclosed material weaknesses in internal controls...
[Source: Acme_10K_2024.pdf, p.42].

Specifically, the internal audit identified deficiencies...
[Source: Acme_Audit_Report_2024.pdf, p.15].

🔄 Query was rewritten 1 time(s).
```

---

## 🔍 Troubleshooting

### ❌ Common Issues

**"Connection refused" (database)**
```powershell
# Check if Docker container is running
docker ps

# Check logs
docker logs audit-rag-db
```

**"Anthropic API key not found"**
- Edit `.env` and add real key
- Demo works without API key

**"No module named 'app'"**
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1
```

**Pydantic V1 warning**
- This is a known Python 3.14 compatibility warning
- System works correctly despite warning

---

## 📈 Performance Benchmarks

**Expected timing (on typical hardware):**
- Retrieval: < 200ms (10k documents)
- Grading: 1-3s per grader call
- Rewriting: 1-2s per rewrite
- Validation: 0.5-2s (parallel API calls)
- Generation: 2-5s
- **End-to-end with 1 loop:** 8-15s

---

## 🚀 What to Test Next

### Before Production:
1. ✅ Run `python tests/verify.py` → All pass
2. ✅ Set up database → `python tests/test_db.py`
3. ✅ Ingest real documents → `python -m app.ingest ...`
4. ✅ Run 5-10 test queries → `python main.py "..." --verbose`
5. ✅ Verify citations are accurate
6. ✅ Test self-correction loop fires correctly
7. ✅ Replace mock embedding model with production model
8. ✅ Set up validation API or mock endpoints
9. ✅ Add authentication & rate limiting
10. ✅ Configure monitoring (LangSmith)

---

## 📚 Full Documentation

- **Architecture:** See `README.md`
- **Detailed testing:** See `TESTING.md`
- **Database setup:** See `TESTING.md` → Database Setup
- **API integration:** See `TESTING.md` → External Validation API

---

## ⚡ TL;DR — Fastest Way to Verify

```powershell
# 1. Quick system check (30 seconds)
python tests/verify.py

# 2. See full demo (20 seconds)
python demo.py

# 3. If both pass → System is working! ✓
```
