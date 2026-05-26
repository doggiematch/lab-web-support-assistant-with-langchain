import os
from typing import Sequence

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def get_hf_token() -> str:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Falta HUGGINGFACEHUB_API_TOKEN en .env. Crea un token Read en "
            "Hugging Face y añádelo al archivo .env."
        )
    return token


def get_embedding_model() -> str:
    return os.getenv("HF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_chat_model() -> str:
    return os.getenv("HF_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def _pool_embedding(values) -> list[float]:
    array = np.asarray(values, dtype=np.float32).squeeze()
    if array.ndim == 0:
        return [float(array)]
    if array.ndim > 1:
        array = array.mean(axis=0)

    norm = np.linalg.norm(array)
    if norm:
        array = array / norm
    return array.astype(float).tolist()


class HuggingFaceInferenceEmbeddings(Embeddings):
    def __init__(self, model: str | None = None):
        self.model = model or get_embedding_model()
        self.client = InferenceClient(token=get_hf_token())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        result = self.client.feature_extraction(
            text,
            model=self.model,
            normalize=True,
            truncate=True,
        )
        return _pool_embedding(result)


def chat_completion(messages: Sequence[dict[str, str]], max_tokens: int = 500) -> str:
    client = InferenceClient(token=get_hf_token())
    response = client.chat_completion(
        messages=list(messages),
        model=get_chat_model(),
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
