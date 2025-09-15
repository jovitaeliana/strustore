# Text Normalization & Canonicalization

This document describes the production normalization pipeline added to Strustore and how it integrates with indexing and query processing, while keeping the database classification sourced strictly from `items.json`.

## Goals
- Normalize noisy real‑world tokens to canonical gaming terms (JP/EN). Example: `nds → nintendo ds`, `joycon/joy-con → joy-con`, `本体 → console`, `コントローラー → controller`.
- Apply the same normalization to both index passages and queries for symmetry.
- Preserve database truth: item IDs and categories always come from `items.json`. `itemtypes.json` is display‑only.
- Maintain strict “No Match” when confidence is low.

## Modules
- `text_normalization/normalizer.py`
  - `TextNormalizer.normalize_text(text)`: Unicode/width normalization, hyphen/space unification, lexicon mapping, model‑code standardization.
  - `TextNormalizer.normalize_tokens(tokens)`: Context‑aware token mapping with guards (e.g., do not map Spanish “con”).
  - `TextNormalizer.normalize_for_index(text)`: Aggressive but safe normalization for corpus passages.
  - `TextNormalizer.expand_aliases(name)`: Deterministic alias variants (e.g., DS, GBA, PlayStation versions).
- `aliases/generator.py`
  - `HeuristicAliasProvider`: Offline, deterministic alias generation using the normalizer.
  - Interface for future GPT alias provider (optional, cached).
- `aliases/cache.py`
  - `AliasCache` and `EmbeddingCache` for MD5‑keyed persistence.
- `position_weighted_embeddings.py`
  - `PositionWeightedTokenClassifier`: Deterministic hardware token heuristics used by the vector DB builder.

## Integration
- Indexing (`strustore-vector-classification/src/create_vector_database.py`)
  - Adds CLI/env and optional normalization of index passages before embedding.
  - E5 prefixes preserved: `passage:` for corpus; `query:` for queries.
  - CLI: `python src/create_vector_database.py --model intfloat/multilingual-e5-base --items ./items.json --out models/vector_database [--no-normalize] [--use-aliases]`
- GDINO Enhancement (`strustore-vector-classification/src/enhance_gdino_results.py`)
  - Normalizes GDINO tokens before search when enabled.
  - Classification uses items.json only; if no vector DB match, returns “No Match”.
  - Display: `gdino_improved_readable` is always items.json `reference_name`, or “No Match”. Itemtypes are never used for display.
  - QA fields: `official_name`, `official_similarity`, `official_consistency` from itemtypes are saved in JSON and logged for analytics only.
  - CLI: `python src/enhance_gdino_results.py --vector-db models/vector_database --gdino-dir ../gdinoOutput/final [--no-normalize] [--no-enhanced-classification] [--official-threshold 0.60] [--no-consistency-gate]`

## “No Match” Contract
When confidence is insufficient, the enhancer writes a strict no‑match record per detection:
- `id: "No Match"`
- `reference_name: "No Match"`
- `category/model/brand: ""`
- `similarity_score: 0.0`
This prevents accidental assignment to `items.json` IDs. Display will show exactly “No Match”.

## Model Choice
- Embeddings: `intfloat/multilingual-e5-base` (robust JP/EN retrieval with `query:/passage:` conventions).
- Aliases: optional GPT augmentation is planned via `AliasProvider` (use with MD5 caching). Normalizer de‑dups and maps all generated aliases to canonical forms before indexing.

## Examples
- Normalize tokens: `TextNormalizer().normalize_tokens(["ジョイコン", "PS2", "本体"]) → ["joy-con", "playstation 2", "console"]`
- Canonical passage: `normalize_for_index("Nintendo DS Lite console | 本体 | USG001") → "nintendo ds lite console | console | USG-001"`

## Notes
- Display comes strictly from items.json (`reference_name`). Itemtypes are used only for QA fields in JSON/logs and never affect display or ID.
- Normalization is enabled by default for both index and queries; disable with `--no-normalize` to compare behavior.
