from typing import Dict
from .evaluator import Evaluator


class StringMatchEvaluator(Evaluator):
    """Evaluator that uses exact string matching to score responses."""

    CRITERIA: Dict[str, str] = {
        "exact_match": """
Score 0: The answer does not match the ground truth.
Score 1: The answer exactly matches the ground truth.
"""
    }

    def __init__(self, ground_truth: str, case_sensitive: bool = False, strip_whitespace: bool = True):
        """
        Initialize the string match evaluator.

        Args:
            ground_truth: The expected correct answer
            case_sensitive: Whether to perform case-sensitive matching (default: False)
            strip_whitespace: Whether to strip leading/trailing whitespace before comparison (default: True)
        """
        self.ground_truth = ground_truth
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def evaluate_response(self, response: str) -> int:
        """
        Evaluate a response using exact string matching.

        Args:
            response: The response to evaluate

        Returns:
            1 if the response matches the ground truth, 0 otherwise
        """
        # Prepare strings for comparison
        ground_truth = self.ground_truth
        response_to_check = response

        # Strip whitespace if requested
        if self.strip_whitespace:
            ground_truth = ground_truth.strip()
            response_to_check = response_to_check.strip()

        # Convert to lowercase if case-insensitive
        if not self.case_sensitive:
            ground_truth = ground_truth.lower()
            response_to_check = response_to_check.lower()

        # Perform exact match
        if response_to_check == ground_truth:
            return 1
        else:
            return 0