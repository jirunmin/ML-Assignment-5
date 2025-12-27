from typing import List, Dict, Optional
from openai import AsyncOpenAI
import tiktoken
import re

from model import ModelProvider


class KeywordRetrievalAgent(ModelProvider):
    """
    A simple and safe baseline agent using keyword-based retrieval.

    Strategy:
    1. Extract keywords from the question
    2. Scan all documents for relevant paragraphs
    3. Select top-k relevant chunks
    4. Ask the LLM to answer using only these chunks
    """

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.model_name = "ecnu-max"
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens_per_request = 10000

    # ============================================================
    # 核心函数
    # ============================================================
    async def evaluate_model(self, prompt: Dict) -> str:
        context_data = prompt["context_data"]
        question = prompt["question"]

        # 1. 基于关键词的检索
        selected_content = self._keyword_retrieval_strategy(
            context_data=context_data,
            question=question
        )

        # 2. 构造 LLM 输入
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant.\n"
                    "Answer with ONLY the final answer.\n"
                    "Do NOT explain.\n"
                    "Do NOT add extra text.\n"
                    "If the answer is a day of the week, output only that word.\n"
                    "If the answer is not found, output: I do not know."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{selected_content}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer:"
                )
            }
        ]

        # 3. 调用模型
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=300,
        )

        return response.choices[0].message.content

    # ============================================================
    # 关键词检索逻辑（核心）
    # ============================================================
    def _keyword_retrieval_strategy(
        self,
        context_data: Dict,
        question: str,
        top_k: int = 6,
        window_chars: int = 400
    ) -> str:
        """
        Extract relevant text chunks using keyword overlap.
        """

        # --- Step 1: extract keywords from question ---
        words = re.findall(r"[A-Za-z]{3,}", question.lower())
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "whose",
            "why", "how", "about", "there", "their", "this", "that",
            "with", "from", "have", "will", "would", "could", "should"
        }
        keywords = [w for w in words if w not in stop_words][:8]

        # --- Step 2: scan all documents ---
        scored_chunks = []

        for file in context_data["files"]:
            content = file.get("modified_content", "")
            if not content:
                continue

            paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

            for p in paragraphs:
                score = sum(1 for kw in keywords if kw in p.lower())
                if score > 0:
                    scored_chunks.append((score, p))

        # --- Step 3: rank by relevance ---
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # --- Step 4: select top-k chunks ---
        selected_chunks = [p[:window_chars] for _, p in scored_chunks[:top_k]]

        return "\n\n".join(selected_chunks)

    # ============================================================
    # Required interface functions
    # ============================================================
    def generate_prompt(self, **kwargs) -> Dict:
        return {
            "context_data": kwargs.get("context_data"),
            "question": kwargs.get("question"),
        }

    def encode_text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        if context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)
