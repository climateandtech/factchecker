"""Ollama embedding that uses the batch API (/api/embed with input list) for multiple texts."""

from typing import List

from llama_index.embeddings.ollama import OllamaEmbedding


class BatchedOllamaEmbedding(OllamaEmbedding):
    """
    Ollama embedding that uses the batch API so multiple texts are sent in one request
    (or chunked by embed_batch_size) instead of one request per text.
    """

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings via Ollama batch API (embed with input list)."""
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            chunk = texts[i : i + self.embed_batch_size]
            result = self._client.embed(
                model=self.model_name,
                input=chunk,
                options=self.ollama_additional_kwargs,
            )
            # embed() returns dict with "embeddings" key (list of vectors)
            embeddings = result.get("embeddings", [])
            out.extend(embeddings)
        return out

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings asynchronously via Ollama batch API."""
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            chunk = texts[i : i + self.embed_batch_size]
            result = await self._async_client.embed(
                model=self.model_name,
                input=chunk,
                options=self.ollama_additional_kwargs,
            )
            embeddings = result.get("embeddings", [])
            out.extend(embeddings)
        return out
