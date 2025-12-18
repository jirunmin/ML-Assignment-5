from abc import ABC, abstractmethod
from typing import Dict


class Evaluator(ABC):
    CRITERIA: Dict[str, str] = {}

    @abstractmethod
    def evaluate_response(self, response: str) -> int:
        ...
