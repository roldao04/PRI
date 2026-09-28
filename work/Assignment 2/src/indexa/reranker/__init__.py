"""Neural reranking module for Assignment 2.

This module provides neural reranking capabilities using cross-encoder models
to improve the ranking quality of BM25 search results.
"""

from indexa.reranker.neural_reranker import NeuralReranker

__all__ = ["NeuralReranker"]
