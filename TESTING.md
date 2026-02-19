# Testing & Verification Guide

## Quick Tests (No Database Required)

### 1. Demo Mode (Already Verified ✅)
```powershell
python demo.py
```
This runs a complete mock flow showing all 5 nodes in action.

---

## Full System Testing (With Database)

### 2. Database Setup

#### Install PostgreSQL + pgvector
```powershell
# Using Docker (recommended):
docker run -d `
  --name audit-rag-db `
  -e POSTGRES_USER=testuser `
  -e POSTGRES_PASSWORD=testpass `
  -e POSTGRES_DB=audit_rag `
  -p 5432:5432 `
  pgvector/pgvector:pg17

# Or install PostgreSQL locally and add pgvector extension
```

#### Update .env
```env
DATABASE_URL=postgresql+psycopg://testuser:testpass@localhost:5432/audit_rag
ANTHROPIC_API_KEY=<your-real-key-here>
```

### 3. Test Database Connection
```powershell
python tests/test_db.py
```

### 4. Ingest Sample Documents
```powershell
# Create test PDFs first
python tests/create_sample_docs.py

# Ingest them
python -m app.ingest --pdf-dir ./tests/sample_docs --doc-type "10-K" --entity-name "Acme Corp"
```

### 5. Run End-to-End Query
```powershell
# Simple query
python main.py "What are the revenue recognition policies?"

# With metadata filter
python main.py "List audit findings" --doc-type "10-K" --entity "Acme Corp"

# Verbose mode (see all nodes)
python main.py "Did the company disclose any material weaknesses?" --verbose
```

---

## Component Tests

### Test Individual Nodes
```powershell
pytest tests/test_nodes.py -v
```

### Test Hybrid Search
```powershell
pytest tests/test_search.py -v
```

### Test Self-Correction Loop
```powershell
pytest tests/test_loop.py -v
```

---

## Manual Verification Checklist

### ✅ Graph Flow
- [ ] Retriever returns documents
- [ ] Grader produces yes/no score
- [ ] Rewriter fires when grader says "no"
- [ ] Validator calls external API (or mocks it)
- [ ] Generator produces cited answer

### ✅ Hybrid Search
- [ ] HNSW semantic search works
- [ ] GIN keyword search works
- [ ] RRF fusion combines both correctly
- [ ] Metadata filters apply

### ✅ Self-Correction
- [ ] Loop fires on irrelevant docs
- [ ] Query gets rewritten with better terms
- [ ] Max loop limit prevents infinite loops
- [ ] System falls through to generator after max loops

### ✅ Structured Outputs
- [ ] Grader returns valid GradeResult
- [ ] Validator extracts ExtractedClaim objects
- [ ] No JSON parsing errors

### ✅ Citations
- [ ] Final answer includes [Source: file.pdf, p.XX]
- [ ] All claims are properly attributed

---

## Performance Testing

### Benchmark Retrieval Speed
```powershell
python tests/benchmark.py
```

### Expected Performance
- **Retrieval**: < 200ms for 10k documents
- **Grading**: < 2s per query
- **End-to-End**: < 10s for single-loop query

---

## Troubleshooting

### Common Issues

**1. "No module named 'app'"**
```powershell
# Make sure venv is activated
.venv\Scripts\Activate.ps1
```

**2. "Connection refused" (database)**
```powershell
# Check if Postgres is running
docker ps
# Or check local Postgres service
```

**3. "Anthropic API key not found"**
```powershell
# Update .env with real key
notepad .env
```

**4. Pydantic V1 warning**
This is a known compatibility warning with Python 3.14 — system still works correctly.

---

## Next Steps

### Production Readiness
1. **Replace mock embedding model** in `app/nodes/retriever.py` and `app/ingest.py`
2. **Set up real validation API** or mock `/api/validate-all/` endpoints
3. **Add authentication** to main.py for production deployment
4. **Set up monitoring** (LangSmith tracing, error tracking)
5. **Add rate limiting** on LLM calls
6. **Create CI/CD pipeline** with automated tests
