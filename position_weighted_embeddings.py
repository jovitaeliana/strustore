"""
Position-weighted token classifier utilities used by the vector database pipeline.

This module is intentionally lightweight and dependency-free. It provides
basic, deterministic heuristics for hardware-related token analysis so that
existing scripts can run without behavioral surprises.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set
import re


class PositionWeightedTokenClassifier:
    """
    Simple heuristics to assess hardware-related tokens and predict high-level
    attributes. This is not a ML model; it is deterministic text logic intended
    to enrich contextual text and metadata.
    """

    # Core hardware-related tokens (lowercase) commonly found in this repo
    hardware_terms: Set[str] = {
        # device types
        "console", "controller", "handheld", "gamepad", "system", "bundle",
        # nintendo ecosystem
        "nintendo", "任天堂", "switch", "joycon", "joy-con", "ds", "dsi", "dslite",
        "3ds", "2ds", "gba", "gb", "gameboy", "game boy", "gamecube", "gc",
        "wii", "n64", "snes", "super famicom", "famicom", "shvc", "hvc",
        # playstation ecosystem
        "playstation", "ps", "ps1", "ps2", "ps3", "ps4", "ps5", "psp", "vita",
        "dualshock", "dualsense", "sixaxis", "scph",
        # xbox ecosystem
        "xbox", "xbox360", "xbox one", "series x", "series s",
        # accessories
        "memory", "card", "memory card", "charger", "dock", "cable", "adapter",
        # common condition/color terms used in pairs (context only)
        "used", "中古", "new", "新品", "tested", "動作確認済み", "ホワイト", "ブラック",
    }

    # Brand to indicative terms (all lowercase substrings)
    brand_terms: Dict[str, List[str]] = {
        "Nintendo": [
            "nintendo", "任天堂", "switch", "joycon", "joy-con",
            "ds", "dslite", "dsi", "3ds", "2ds",
            "gba", "gameboy", "game boy", "gamecube", "gc",
            "wii", "n64", "snes", "super famicom", "famicom", "shvc", "hvc",
            # common model codes
            "ntr-001", "usg-001", "twl-001", "ctr-001", "hds-001", "hds",
            "heg-001", "hdh-001", "ags-001", "agb-001", "cgb-001", "dmg-01",
        ],
        "Sony": [
            "sony", "playstation", "ps", "ps1", "ps2", "ps3", "ps4", "ps5",
            "dualsense", "dualshock", "sixaxis", "psp", "vita", "scph",
        ],
        "Microsoft": [
            "microsoft", "xbox", "xbox360", "xbox 360", "xbox one",
            "series x", "series s",
        ],
        "Sega": [
            "sega", "saturn", "megadrive", "genesis", "dreamcast",
        ],
    }

    # Regexes for common model code patterns seen in repo/positive pairs
    _model_code_patterns: List[re.Pattern] = [
        re.compile(r"\b[A-Z]{2,4}-?\d{3,4}\b", re.IGNORECASE),  # NTR-001, USG-001, AGB001
        re.compile(r"\b\d{3,4}[A-Z]{0,2}\b", re.IGNORECASE),     # 1537, 1914, CUH-ZCT1 variants
    ]

    def classify_hardware_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Classify token list with simple heuristics.

        Returns a dict with:
          - hardware_relevance_score: float in [0,1]
          - predicted_hardware: {brand, device_type, model_codes}
          - token_weights: optional per-token weights for transparency
        """
        if not tokens:
            return {
                "hardware_relevance_score": 0.0,
                "predicted_hardware": {"brand": None, "device_type": None, "model_codes": []},
                "token_weights": [],
            }

        tokens_norm = [t.strip().lower() for t in tokens if isinstance(t, str) and t.strip()]

        # Brand scoring
        brand_scores: Dict[str, float] = {b: 0.0 for b in self.brand_terms}
        for t in tokens_norm:
            for brand, terms in self.brand_terms.items():
                if any(term in t for term in terms):
                    brand_scores[brand] += 1.0

        best_brand = max(brand_scores, key=brand_scores.get)
        brand = best_brand if brand_scores[best_brand] > 0 else None

        # Device type detection (very lightweight)
        device_type = None
        joined = " ".join(tokens_norm)
        if any(x in joined for x in ["controller", "gamepad", "dualshock", "dualsense", "joycon", "joy-con", "sixaxis"]):
            device_type = "controller"
        elif any(x in joined for x in ["handheld", "ds", "3ds", "2ds", "gb", "gameboy", "psp", "vita"]):
            device_type = "handheld"
        elif "console" in joined or any(x in joined for x in ["ps1", "ps2", "ps3", "ps4", "ps5", "xbox", "switch", "wii", "n64", "snes", "gamecube"]):
            device_type = "console"

        # Model codes
        model_codes: List[str] = []
        for t in tokens_norm:
            for pat in self._model_code_patterns:
                model_codes.extend(pat.findall(t))
        model_codes = sorted(set(mc.upper() for mc in model_codes))

        # Hardware relevance: proportion of tokens that intersect with hardware_terms or brand terms
        hardware_hits = 0
        for t in tokens_norm:
            if t in self.hardware_terms:
                hardware_hits += 1
                continue
            # Also count if any brand indicator appears in token
            if any(any(term in t for term in terms) for terms in self.brand_terms.values()):
                hardware_hits += 1

        base_score = hardware_hits / max(1, len(tokens_norm))
        # Light boosts when we have brand/device_type/model_codes evidence
        boost = 0.0
        if brand:
            boost += 0.1
        if device_type:
            boost += 0.1
        if model_codes:
            boost += 0.1
        hardware_relevance_score = max(0.0, min(1.0, base_score + boost))

        # Optional simple position weights (exponential decay on index)
        token_weights = []
        for idx, t in enumerate(tokens_norm):
            # Higher weight for earlier tokens
            position_weight = pow(2.71828, -0.1 * idx)
            is_hw = (t in self.hardware_terms) or any(
                any(term in t for term in terms) for terms in self.brand_terms.values()
            )
            token_weights.append({
                "token": t,
                "position_index": idx,
                "position_weight": position_weight,
                "is_hardware": is_hw,
            })

        return {
            "hardware_relevance_score": float(hardware_relevance_score),
            "predicted_hardware": {
                "brand": brand,
                "device_type": device_type,
                "model_codes": model_codes,
            },
            "token_weights": token_weights,
        }

