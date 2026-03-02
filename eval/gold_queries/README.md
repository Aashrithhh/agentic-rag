# Gold Evaluation Query Sets

This directory contains curated ground-truth query sets for continuous
evaluation of the Agentic RAG pipeline.

## Structure

Each case has its own JSONL file (or subdirectory).  Every line is a JSON
object with:

| Field                     | Type       | Description                                 |
|---------------------------|------------|---------------------------------------------|
| `query`                   | string     | The natural-language query                   |
| `expected_answer_contains`| string[]   | Keywords that MUST appear in the answer      |
| `expected_sources`        | string[]   | Source doc names the answer should cite       |
| `category`                | string     | Intent class: fact_lookup, summary, timeline, comparison, exploratory |
| `difficulty`              | string     | easy / medium / hard                         |

## Usage

```bash
# Run nightly eval against the gold set
python scripts/nightly_eval.py --case big_thorium --eval-file eval/gold_queries/general_corpus.jsonl

# Run all gold sets
python scripts/nightly_eval.py --all-cases
```

## Adding queries

1. Add a new line to the appropriate JSONL file.
2. Ensure `expected_answer_contains` uses keywords, not exact phrases.
3. Tag the correct `category` for intent routing evaluation.
4. Mark difficulty based on reasoning steps required.
