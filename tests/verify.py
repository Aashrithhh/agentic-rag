"""Quick verification script — runs all checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Verify all required packages are installed."""
    print("Checking dependencies...")
    required = [
        "langgraph",
        "langchain",
        "langchain_openai",
        "langchain_cohere",
        "langchain_community",
        "pgvector",
        "sqlalchemy",
        "pydantic",
        "httpx",
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"✗ Missing packages: {', '.join(missing)}")
        print("  Install with: pip install " + " ".join(missing))
        return False
    
    print("✓ All dependencies installed")
    return True


def check_config():
    """Verify configuration is valid."""
    print("\nChecking configuration...")
    try:
        from app.config import settings
        print(f"✓ Config loaded")
        print(f"  Model: {settings.azure_openai_deployment}")
        print(f"  Embedding model: {settings.cohere_embed_model}")
        print(f"  Embedding dim: {settings.embedding_dim}")
        print(f"  Top-K: {settings.top_k}")
        print(f"  Max loops: {settings.max_rewrite_loops}")
        
        if not settings.azure_openai_api_key:
            print("⚠ Azure OpenAI API key not set (some features won't work)")
        if not settings.cohere_api_key:
            print("⚠ Cohere API key not set (embeddings won't work)")
        
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False


def check_graph():
    """Verify graph can be compiled."""
    print("\nChecking graph compilation...")
    try:
        from app.graph import compile_graph
        graph = compile_graph()
        print("✓ Graph compiled successfully")
        return True
    except Exception as e:
        print(f"✗ Graph compilation failed: {e}")
        return False


def run_demo():
    """Run the demo to verify end-to-end flow."""
    print("\nRunning demo (this takes ~10 seconds)...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "demo.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Check for success markers in output
        if result.returncode == 0 and "[OK] Demo completed successfully!" in result.stdout:
            print("✓ Demo passed")
            return True
        else:
            print(f"✗ Demo failed (exit code: {result.returncode})")
            if result.stderr:
                print("stderr:", result.stderr[:200])
            return False
    except Exception as e:
        print(f"✗ Demo error: {e}")
        return False


def main():
    print("=" * 72)
    print("QUICK VERIFICATION")
    print("=" * 72)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("Graph", check_graph),
        ("Demo", run_demo),
    ]
    
    results = []
    for name, check in checks:
        results.append(check())
    
    print("\n" + "=" * 72)
    passed = sum(results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ System is ready!")
        print("\nNext steps:")
        print("1. Set up database: See TESTING.md")
        print("2. Add real API key to .env")
        print("3. Ingest documents: python -m app.ingest --pdf-dir ./data")
        print("4. Run queries: python main.py 'your question here' --verbose")
    else:
        print("\n✗ Some checks failed. See output above.")
    
    print("=" * 72)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
