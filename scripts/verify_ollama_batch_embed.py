#!/usr/bin/env python3
"""
Verify Ollama batch embed API: POST /api/embed with input=[text1, text2, ...].
Run with: python scripts/verify_ollama_batch_embed.py
Requires: Ollama running (e.g. ollama serve), an embedding model (e.g. ollama pull nomic-embed-text).
"""
import json
import os
import urllib.error
import urllib.request

BASE = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL") or os.getenv("OLLAMA_MODEL", "nomic-embed-text")


def main() -> None:
    texts = ["first", "second text", "third"]
    body = json.dumps({"model": MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"Ollama not reachable at {BASE}: {e}")
        raise SystemExit(1)
    except urllib.error.HTTPError as e:
        print(f"Ollama error: {e.code} {e.reason}")
        if e.fp:
            print(e.fp.read().decode())
        raise SystemExit(1)

    embeddings = data.get("embeddings", [])
    if len(embeddings) != len(texts):
        print(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
        raise SystemExit(1)
    dim = len(embeddings[0])
    for i, emb in enumerate(embeddings):
        if len(emb) != dim:
            print(f"Embedding {i} length {len(emb)} != {dim}")
            raise SystemExit(1)
    print(f"OK: batch embed returned {len(embeddings)} vectors of dimension {dim}")


if __name__ == "__main__":
    main()
