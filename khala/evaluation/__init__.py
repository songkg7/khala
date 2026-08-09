"""Typed, side-effect-aware recall evaluation for Khala."""

from khala.evaluation.models import EvalCase, Judgment
from khala.evaluation.runner import evaluate_queries

__all__ = ["EvalCase", "Judgment", "evaluate_queries"]
