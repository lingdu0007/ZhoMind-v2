"""Evaluation Retriever components for the Retrieval Evidence Baseline.

Sparse BM25 is the true lexical retrieval baseline and is exposed only through
the Evaluation Retriever's ``retrieval-evidence evaluate`` command (ADR-0001).
It never changes production Migration Retrieval, and the existing Lexical
Heuristic is never relabeled as BM25.
"""
