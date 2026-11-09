# Assignment 10: Improving RAG with Reranking, Chunking, and Conversational Memory

Dataset: `ccdv/arxiv-summarization`, document subset, first 100 train documents. Vector DB: ChromaDB. Embeddings: all MiniLM L6 v2. Reranker: ms marco MiniLM cross encoder.

| Comparison | P@5 | R@5 | MRR |
|---|---|---|---|
| Without reranking | 0.875 | 0.265 | 0.938 |
| With reranking | 0.900 | 0.304 | 1.000 |
| Fixed chunks | 0.900 | 0.254 | 1.000 |
| Sentence chunks | 0.875 | 0.316 | 0.938 |
| Follow up without memory | 0.050 | 0.042 | 0.083 |
| Follow up with memory | 0.350 | 0.274 | 1.000 |

The full discussion is in the last cell of the notebook.

## Files

* `notebooks/assignment10_rag.ipynb`, the notebook with outputs

## Rerunning

Uncomment the first cell to install packages, then run all cells (about 5 minutes on a laptop).
