"""
Text normalization and canonicalization utilities for Strustore.

Goals:
- Normalize Unicode/width, punctuation, and spacing
- Map JP/EN gaming terms and abbreviations to canonical forms
- Provide context-aware rules to avoid false positives (e.g., Spanish "con")
- Offer helper methods for indexing and query preprocessing
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple


class TextNormalizer:
    def __init__(self) -> None:
        # Core lexicon maps common variants to canonical forms (lowercase)
        # Keep brief and evidence-backed; expand as corpus grows.
        self.lexicon: Dict[str, str] = {
            # Platforms and brands
            "nds": "nintendo ds",
            "nintendods": "nintendo ds",
            "ds": "ds",  # keep generic; context rules refine later
            "dsl": "nintendo ds lite",
            "dslite": "nintendo ds lite",
            "ds lite": "nintendo ds lite",
            "3ds": "nintendo 3ds",
            "2ds": "nintendo 2ds",
            "gb": "game boy",
            "gameboy": "game boy",
            "gba": "game boy advance",
            "gbc": "game boy color",
            "gc": "gamecube",
            "n64": "nintendo 64",
            "snes": "snes",
            "nes": "nes",
            "ps": "playstation",
            "psx": "playstation",
            "ps1": "playstation",
            "ps2": "playstation 2",
            "ps3": "playstation 3",
            "ps4": "playstation 4",
            "ps5": "playstation 5",
            "psp": "psp",
            "vita": "playstation vita",
            "x360": "xbox 360",
            "xbone": "xbox one",
            # Accessories and parts
            "joycon": "joy-con",
            "joy con": "joy-con",
            "ジョイコン": "joy-con",
            "dual shock": "dualshock",
            "dual shock 2": "dualshock 2",
            "dual shock 3": "dualshock 3",
            "dual shock 4": "dualshock 4",
            # JP -> EN mapping for common terms
            "本体": "console",
            "コントローラー": "controller",
            "動作確認済み": "tested working",
            "新品": "new",
            "中古": "used",
            "美品": "mint condition",
            # Colors
            "ホワイト": "white",
            "ブラック": "black",
            "ブルー": "blue",
            "レッド": "red",
            "ピンク": "pink",
            "シルバー": "silver",
            # Conditions/short forms
            "cib": "complete in box",
            "junk": "for parts",
        }

        # Hyphen/space/punctuation normalizations (applied before lexicon)
        self._dash_rx = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]+")  # hyphen-like chars
        self._spaces_rx = re.compile(r"\s+")
        self._punct_rx = re.compile(r"[\u3000\u00A0]")  # full-width/nbsp -> space

        # Model code patterns: CUH-ZCT1, SCPH-10010, AGS-001, etc.
        self._model_code_rx = re.compile(r"\b([A-Z]{2,4}-?\d{3,4}[A-Z]{0,2})\b", re.IGNORECASE)

        # Context guard words to avoid mapping false positives (e.g., Spanish "con")
        self._stop_terms = {"con": True}  # never map globally without hardware context

        # Hardware context words (for safe mapping of ambiguous short tokens)
        self._hardware_context = {
            "console", "controller", "gamepad", "nintendo", "playstation", "sony", "xbox", "sega",
            "ds", "3ds", "2ds", "gb", "gba", "game boy", "gameboy", "gamecube", "wii", "n64", "snes",
            "ps1", "ps2", "ps3", "ps4", "ps5", "psp", "vita", "joy-con", "dualshock", "scph",
        }

    # ---------- Core primitives ----------
    def normalize_unicode(self, text: str) -> str:
        if not text:
            return ""
        t = unicodedata.normalize("NFKC", text)
        t = self._punct_rx.sub(" ", t)
        t = self._dash_rx.sub("-", t)
        t = t.replace("_", " ")
        t = self._spaces_rx.sub(" ", t).strip()
        return t

    def lowercase(self, text: str) -> str:
        return text.lower() if text else ""

    def _normalize_token_basic(self, token: str) -> str:
        return self.lowercase(self.normalize_unicode(token))

    def _apply_lexicon(self, token: str, context: str = "") -> str:
        # Avoid mapping ambiguous stop terms unless hardware context is present
        if token in self._stop_terms:
            if not any(hw in context for hw in self._hardware_context):
                return token
        return self.lexicon.get(token, token)

    # ---------- Public API ----------
    def normalize_text(self, text: str) -> str:
        """Full normalization for free text: Unicode→spaces/hyphens→lower→lexicon where safe."""
        t = self._normalize_token_basic(text)
        # Apply lexicon word-by-word to keep structure
        words = t.split()
        context = " ".join(words)
        mapped = [self._apply_lexicon(w, context) for w in words]
        return self._spaces_rx.sub(" ", " ".join(mapped)).strip()

    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Normalize a list of tokens with context-aware mapping."""
        basic = [self._normalize_token_basic(t) for t in tokens if isinstance(t, str) and t.strip()]
        context = " ".join(basic)
        out = [self._apply_lexicon(t, context) for t in basic]
        # Normalize model codes to uppercase canonical dashed form when present
        norm_codes = []
        for t in out:
            m = self._model_code_rx.findall(t)
            if m:
                for code in m:
                    c = code.upper()
                    if "-" not in c and any(c.startswith(prefix) for prefix in ("CUH", "SCPH", "AGS", "AGB", "CGB", "NTR", "USG", "TWL", "CTR", "HEG", "HDH")):
                        # Insert a dash before numeric part if missing
                        c = re.sub(r"([A-Z]{2,4})(\d)", r"\1-\2", c)
                    norm_codes.append(c)
                    t = t.replace(code, c)
            norm_codes.append(t)
        return norm_codes

    def normalize_for_index(self, text: str) -> str:
        """Aggressive but safe normalization for index passages.
        Keeps semantic hints, standardizes variants, and normalizes model codes.
        """
        t = self.normalize_text(text)
        # Ensure hyphenated hardware terms become consistent
        t = t.replace("joy con", "joy-con").replace("dual shock", "dualshock")
        # Collapse multiple spaces
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def expand_aliases(self, canonical_name: str) -> List[str]:
        """Generate lightweight synonyms for a canonical name using deterministic rules.
        Use GPT provider for richer augmentation when enabled elsewhere.
        """
        name = self.normalize_text(canonical_name)
        aliases: List[str] = []
        # Joy-Con variants
        if "joy-con" in name:
            aliases += ["joycon", "joy con", "ジョイコン"]
        # PlayStation shorthand
        if "playstation" in name:
            if " 2" in name:
                aliases += ["ps2", "ps 2"]
            elif " 3" in name:
                aliases += ["ps3", "ps 3"]
            elif " 4" in name:
                aliases += ["ps4", "ps 4"]
            elif " 5" in name:
                aliases += ["ps5", "ps 5"]
            else:
                aliases += ["ps", "psx", "ps1"]
        # Nintendo DS family
        if "nintendo ds lite" in name:
            aliases += ["dsl", "ds lite", "dslite"]
        elif "nintendo ds" in name:
            aliases += ["nds", "ds", "nintendods", "ニンテンドーds", "ニンテンドーds"]
        # Game Boy family
        if "game boy advance" in name:
            aliases += ["gba"]
        if "game boy color" in name:
            aliases += ["gbc"]
        if "game boy" in name:
            aliases += ["gb", "gameboy"]
        # Xbox
        if "xbox 360" in name:
            aliases += ["x360"]
        if "xbox one" in name:
            aliases += ["xbone"]
        # Return normalized unique list
        uniq = []
        seen = set()
        for a in aliases:
            na = self.normalize_text(a)
            if na and na not in seen:
                seen.add(na)
                uniq.append(na)
        return uniq

