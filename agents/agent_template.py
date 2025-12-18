from typing import List, Dict, Optional
from openai import AsyncOpenAI
import tiktoken
import random

from model import ModelProvider


class ExampleAgent(ModelProvider):
    """
    Example implementation of a multi-document retrieval agent.

    This baseline implementation demonstrates the interface but uses a naive strategy:
    - Randomly selects 1 text file
    - Randomly extracts 10000 tokens
    
    For better performance, consider implementing:
    - Retrieval from all relevant files
    - RAG (Retrieval-Augmented Generation) with vector search
    - Intelligent file selection based on relevance
    - Query-aware context extraction
    """

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.model_name = "ecnu-max"
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens_per_request = 10000

    async def evaluate_model(self, prompt: Dict) -> str:
        """
        Handle multi-document retrieval task.

        Baseline strategy:
        1. Randomly select 1 file from all available files
        2. Randomly extract 10000 tokens from that file
        3. Send to LLM for answering
        
        This is a naive approach for demonstration purposes.

        Args:
            prompt: Dictionary containing context_data and question

        Returns:
            Model response
        """
        context_data = prompt['context_data']
        question = prompt['question']

        # Use baseline random selection strategy
        selected_content = self._random_select_strategy(context_data)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer the question based on the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{selected_content}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=300
        )

        return response.choices[0].message.content

    def _random_select_strategy(self, context_data: Dict) -> str:
        """
        Baseline strategy: Randomly select 1 file and extract 10000 tokens.

        This is intentionally naive to demonstrate the interface.
        Implement smarter retrieval strategies for better performance.

        Args:
            context_data: Dictionary containing all file information

        Returns:
            Extracted text content
        """
        files = context_data['files']

        # Randomly select one file
        selected_file = random.choice(files)
        print(f"[Baseline] Randomly selected file: {selected_file['filename']}")

        content = selected_file['modified_content']
        tokens = self.encode_text_to_tokens(content)

        # If file is smaller than max tokens, return entire content
        if len(tokens) <= self.max_tokens_per_request:
            return content

        # Randomly extract a chunk
        max_start = len(tokens) - self.max_tokens_per_request
        start_pos = random.randint(0, max_start)
        end_pos = start_pos + self.max_tokens_per_request

        print(f"[Baseline] Randomly extracted tokens {start_pos}-{end_pos} from {len(tokens)} total")

        selected_tokens = tokens[start_pos:end_pos]
        return self.decode_tokens(selected_tokens)

    def generate_prompt(self, **kwargs) -> Dict:
        """
        Generate prompt structure for the model.

        Args:
            **kwargs: Flexible parameters (context_data, question, etc.)

        Returns:
            Dictionary containing all prompt information
        """
        return {
            'context_data': kwargs.get('context_data'),
            'question': kwargs.get('question')
        }

    def encode_text_to_tokens(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        """Decode token IDs to text."""
        if context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)
