"""Wikipedia passage retrieval for KC-1 (mockable API layer)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _api_search(query: str, k: int) -> list[str]:
    """Live search — override in tests via patch."""
    import urllib.parse
    import urllib.request

    url = (
        "https://en.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": k,
            }
        )
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    snippets = [hit.get("snippet", "") for hit in data.get("query", {}).get("search", [])]
    return [s for s in snippets if s][:k]


def retrieve_passages(
    question: str,
    answer: str,
    k: int = 3,
    cache_dir: Path | None = None,
) -> list[str]:
    query = f"{question} {answer}"
    cache_key = hashlib.sha256(query.encode()).hexdigest()
    cache_file = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
    snippets = _api_search(query, k)
    if cache_file is not None:
        cache_file.write_text(json.dumps(snippets))
    return snippets
