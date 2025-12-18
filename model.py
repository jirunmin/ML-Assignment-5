from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class ModelProvider(ABC):
    """
    Abstract base class for agent implementations.

    Implement this class to create your own agent system for the needle-in-haystack tests.
    """

    def __init__(self, api_key: str, base_url: str):
        """
        Initialize the model provider.

        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the LLM service
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "custom-agent"

    @abstractmethod
    async def evaluate_model(self, prompt: Dict) -> str:
        """
        Evaluate the model with the given prompt.

        This is where you implement your agent system logic.

        Args:
            prompt: Dictionary containing all necessary information

        Returns:
            The model's response
        """
        ...

    @abstractmethod
    def generate_prompt(self, **kwargs) -> Dict:
        """
        Generate the prompt structure for the model.

        Args:
            **kwargs: Flexible parameters depending on the test scenario

        Returns:
            Dictionary containing all prompt information
        """
        ...

    @abstractmethod
    def encode_text_to_tokens(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        ...

    @abstractmethod
    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs
            context_length: Optional number of tokens to decode

        Returns:
            Decoded text
        """
        ...
