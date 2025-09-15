"""
Simple filesystem caches for aliases and embeddings.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import json
import hashlib
import numpy as np


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class AliasCache:
    def __init__(self, base_dir: str | Path = "models/alias_cache") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        sub = key[:2]
        p = self.base / sub
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path_for_key(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, key: str, data: Dict[str, Any]) -> None:
        p = self._path_for_key(key)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class EmbeddingCache:
    def __init__(self, base_dir: str | Path = "models/emb_cache") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, model_name: str, key: str) -> Path:
        safe_model = model_name.replace("/", "_")
        sub = key[:2]
        p = self.base / safe_model / sub
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{key}.npy"

    def get(self, model_name: str, key: str) -> Optional[np.ndarray]:
        p = self._path_for_key(model_name, key)
        if not p.exists():
            return None
        try:
            return np.load(p)
        except Exception:
            return None

    def put(self, model_name: str, key: str, array: np.ndarray) -> None:
        p = self._path_for_key(model_name, key)
        np.save(p, array)

