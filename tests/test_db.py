"""Test database connection and initialization."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import init_db, get_engine, hybrid_search
from app.config import settings


def test_connection():
    """Test basic database connectivity."""
    print("Testing database connection...")
    try:
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✓ Connected to PostgreSQL: {version[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Is PostgreSQL running?")
        print("   docker ps  # Check if container is up")
        print("2. Check DATABASE_URL in .env")
        print(f"   Current: {settings.database_url}")
        return False


def test_pgvector():
    """Test pgvector extension."""
    print("\nTesting pgvector extension...")
    try:
        from sqlalchemy import text
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        print("✓ pgvector extension available")
        return True
    except Exception as e:
        print(f"✗ pgvector test failed: {e}")
        return False


def test_table_creation():
    """Test table and index creation."""
    print("\nTesting table creation...")
    try:
        engine = init_db()
        print("✓ document_chunks table created")
        print("✓ HNSW index created")
        print("✓ GIN tsvector index created")
        return True
    except Exception as e:
        print(f"✗ Table creation failed: {e}")
        return False


def test_insert_sample():
    """Test inserting a sample document."""
    print("\nTesting sample insert...")
    try:
        from app.db import upsert_chunks
        engine = get_engine()
        
        sample = [{
            "content": "Acme Corp disclosed material weaknesses in SOX 404 compliance.",
            "embedding": [0.1] * settings.embedding_dim,
            "source": "test.pdf",
            "page": 1,
            "chunk_index": 0,
            "doc_type": "10-K",
            "entity_name": "Acme Corp",
            "effective_date": None,
            "metadata_extra": {},
        }]
        
        count = upsert_chunks(engine, sample)
        print(f"✓ Inserted {count} test document")
        return True
    except Exception as e:
        print(f"✗ Insert failed: {e}")
        return False


def test_hybrid_search():
    """Test hybrid search query."""
    print("\nTesting hybrid search...")
    try:
        from app.db import hybrid_search
        engine = get_engine()
        
        # Mock embedding
        query_embedding = [0.1] * settings.embedding_dim
        query_text = "material weaknesses"
        
        results = hybrid_search(
            engine=engine,
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=5,
        )
        
        print(f"✓ Hybrid search returned {len(results)} results")
        if results:
            print(f"  Sample: {results[0]['content'][:80]}...")
        return True
    except Exception as e:
        print(f"✗ Search failed: {e}")
        return False


def main():
    print("=" * 72)
    print("Database Test Suite")
    print("=" * 72)
    
    tests = [
        test_connection,
        test_pgvector,
        test_table_creation,
        test_insert_sample,
        test_hybrid_search,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 72)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Database is ready.")
    else:
        print("✗ Some tests failed. Check output above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
