# iChunk Optimizer — Technical Deep Dive

## Table of Contents
1. Chunking Strategies (When and Why)
2. Embedding Generation (Models and Batching)
3. Retrieval and Reranking
4. Metadata Design and Filtering
5. Quality Controls and Evaluation

---

## 1) Chunking Strategies (When and Why)

Choosing the right strategy is decisive for retrieval quality (40–60% swing).

- Fixed-size (e.g., 400 chars, 50 overlap):
  - Pros: Simple, uniform
  - Cons: Splits sentences; context loss
  - Use: homogenous text, quick baseline

- Recursive (by separators: \n\n, \n, space):
  - Pros: Respects boundaries; more coherent chunks
  - Cons: Variable lengths
  - Use: articles, docs, reports

- Semantic grouping (cluster similar rows):
  - Pros: Topic-coherent chunks; better retrieval grounding
  - Cons: Additional compute (clustering)
  - Use: content-heavy text fields, knowledge bases

- Document-based (parent-child chunks):
  - Pros: Preserve per-record context; easy metadata mapping
  - Cons: May be large; consider sub-chunking
  - Use: tabular exports, records

- Campaign-specific (record/company/source-based):
  - Pros: Tailored for contact/campaign datasets
  - Cons: Domain-specific logic
  - Use: media/lead data for sales/marketing

Parameters:
- Chunk size/overlap tuned by content density
- Separator priority to maximize semantic continuity

---

## 2) Embedding Generation (Models and Batching)

Models:
- Local: sentence-transformers (e.g., MiniLM, BGE) — cost-effective, private
- API: OpenAI embeddings — higher quality, cost trade-offs

Batching:
- Batch size tuned to GPU/CPU memory for throughput
- Parallel workers (e.g., 6) increase tokens/sec

Preprocessing:
- Normalize whitespace, strip HTML, lowercasing (optional)
- Language detection for multilingual models, if needed

Quality trade-offs:
- Larger models → better semantic fidelity, higher cost/latency
- Domain-specific embeddings for verticals (e.g., biomed)

---

## 3) Retrieval and Reranking

ANN search:
- FAISS: flat (exact), IVF (inverted lists), HNSW (graph-based)
- ChromaDB: persistent collections with metadata filters

Similarity metrics:
- Cosine (directional), L2, dot-product; use same metric as model pretraining

Pipeline:
```
Query → Embed → ANN search (Top-N) → (Optional) Rerank → Assemble context → Return
```

Reranking (optional):
- Cross-encoder models re-score query–chunk pairs for improved precision
- Use when Top-N contains semi-relevant artifacts

Context assembly:
- Token-budget aware assembly of top chunks
- De-duplication and ordering by relevance

---

## 4) Metadata Design and Filtering

Metadata schema (examples):
- `source_file`, `row_range`, `strategy`, `timestamp`
- Domain fields (e.g., `company`, `industry`, `campaign_id`)

Use-cases:
- Filter queries: e.g., industry = healthcare, date range
- Audit trail: trace chunk to origin for verification
- Aggregations: reporting by metadata slices

Best practices:
- Keep metadata compact but discriminative
- Index frequently filtered fields

---

## 5) Quality Controls and Evaluation

Pre-ingest checks:
- Schema validation; column presence and types
- NaN/null handling policies per column

Chunk quality checks:
- Length distribution; extreme chunk detection
- Overlap correctness; boundary coherence heuristics

Embedding health:
- Outlier detection in vector space (norms, density)
- Language/encoding checks

Retrieval evaluation:
- Construct gold queries with expected sources; measure Recall@K and MRR
- Human-in-the-loop spot checks; feedback loop to adjust chunking/filters

Operational SLOs:
- Ingest throughput (rows/sec), latency (p95)
- Search latency (p50/p95), index size growth controls

Continuous improvement:
- A/B test strategies (fixed vs recursive vs semantic)
- Periodic re-embedding when models improve
