"""
Alias generation providers.

Use HeuristicAliasProvider for deterministic, offline synonyms.
Optionally add GPTAliasProvider for LLM-based augmentation (not enabled by default).
"""
from __future__ import annotations

from typing import List, Protocol, Dict, Any
import hashlib

from text_normalization import TextNormalizer


class AliasProvider(Protocol):
    def generate(self, item: Dict[str, Any], n: int = 20) -> List[str]:
        ...


class HeuristicAliasProvider:
    def __init__(self, normalizer: TextNormalizer | None = None) -> None:
        self.normalizer = normalizer or TextNormalizer()

    def generate(self, item: Dict[str, Any], n: int = 20) -> List[str]:
        name = item.get("name") or item.get("item") or ""
        name_norm = self.normalizer.normalize_text(name)
        base = [name_norm]
        # Expand lightweight deterministic aliases
        base += self.normalizer.expand_aliases(name_norm)
        # Include model codes from metadata if present
        model = (item.get("model") or "").strip()
        if model:
            base.append(self.normalizer.normalize_text(model))
        # De-dup and cap
        uniq: List[str] = []
        seen = set()
        for t in base:
            if t and t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq[:max(1, n)]


def md5_key(*parts: str) -> str:
    m = hashlib.md5()
    for p in parts:
        m.update((p or "").encode("utf-8"))
    return m.hexdigest()

