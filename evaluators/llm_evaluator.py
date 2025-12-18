from openai import OpenAI
from typing import Dict
from .evaluator import Evaluator


class LLMEvaluator(Evaluator):
    """Evaluator that uses LLM to score responses against ground truth."""

    CRITERIA: Dict[str, str] = {
        "accuracy": """
Score 0: The answer is completely wrong or unrelated.
Score 3: The answer has minor relevance but contains major inaccuracies.
Score 5: The answer is partially correct but missing key information.
Score 7: The answer is mostly correct with minor omissions.
Score 10: The answer is completely accurate and matches the ground truth.
"""
    }

    def __init__(self, api_key: str, base_url: str, ground_truth: str, question: str):
        """Initialize the LLM evaluator."""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.ground_truth = ground_truth
        self.question = question

    def evaluate_response(self, response: str) -> int:
        """Evaluate a response using LLM."""
        evaluation_prompt = f"""You are an expert evaluator. Your task is to score the answer based on how well it matches the ground truth.

Question: {self.question}
Ground Truth Answer: {self.ground_truth}
Answer: {response}

Scoring Criteria:
{self.CRITERIA['accuracy']}

Please evaluate the answer and respond with ONLY a single number from 0 to 10. Do not include any explanation or other text."""

        completion = self.client.chat.completions.create(
            model="ecnu-max",
            messages=[
                {"role": "system",
                 "content": "You are an expert evaluator. Respond only with a number from 0 to 10."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        score_text = completion.choices[0].message.content.strip()
        score = int(score_text)
        if score < 0 or score > 10:
            raise ValueError(f"Score out of range: {score}")
        return score
