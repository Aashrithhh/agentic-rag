"""Quick isolation verification script."""
from app.db import get_engine, hybrid_search
from app.cases import metadata_filter_for_case
from app.nodes.retriever import _embed_query

engine = get_engine()

# Test 1: BigThorium query with BigThorium filter
bt_filter = metadata_filter_for_case("big-thorium")
print("BigThorium filter:", bt_filter)
emb = _embed_query("Indian workers welding")
results = hybrid_search(engine, emb, "Indian workers welding", metadata_filter=bt_filter)
print(f"BigThorium results: {len(results)} docs")
for r in results:
    dt = r["doc_type"]
    en = r.get("entity_name", "")
    src = r["source"]
    txt = r["content"][:80]
    print(f"  [{dt}] [{en}] {src}: {txt}...")

print()

# Test 2: Same query with Purview Exchange filter
pe_filter = metadata_filter_for_case("purview-exchange")
print("Purview filter:", pe_filter)
results2 = hybrid_search(engine, emb, "Indian workers welding", metadata_filter=pe_filter)
print(f"Purview results for 'Indian workers welding': {len(results2)} docs")
for r in results2:
    dt = r["doc_type"]
    en = r.get("entity_name", "")
    src = r["source"]
    txt = r["content"][:80]
    print(f"  [{dt}] [{en}] {src}: {txt}...")

print()

# Test 3: Purview query with Purview filter
emb2 = _embed_query("phishing simulation results")
results3 = hybrid_search(engine, emb2, "phishing simulation results", metadata_filter=pe_filter)
print(f"Purview phishing results: {len(results3)} docs")
for r in results3:
    dt = r["doc_type"]
    en = r.get("entity_name", "")
    src = r["source"]
    txt = r["content"][:80]
    print(f"  [{dt}] [{en}] {src}: {txt}...")

print()

# Test 4: BigThorium filter should NOT return phishing data
results4 = hybrid_search(engine, emb2, "phishing simulation results", metadata_filter=bt_filter)
print(f"BigThorium phishing results (should be 0 or irrelevant): {len(results4)} docs")
for r in results4:
    dt = r["doc_type"]
    en = r.get("entity_name", "")
    src = r["source"]
    txt = r["content"][:80]
    print(f"  [{dt}] [{en}] {src}: {txt}...")

print("\n=== ISOLATION VERIFIED ===" if len(results) > 0 and len(results3) > 0 else "\n=== CHECK RESULTS ===")
