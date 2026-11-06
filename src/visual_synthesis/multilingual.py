"""Experiment E7: multilingual and code-mixed prompts (benchmark level L5).

Does translating a harmful prompt, or only some of its words, into another
language change how often a model produces the unsafe content? Each L1 parent
gets one variant per (mode, language); the mode is the ablation:

    full           the whole prompt translated
    keywords       only the harm-bearing words translated, the rest stays English
    non_keywords   everything except those words translated (same words, opposite split)
    random         as many randomly chosen non-keyword words as there are keywords
                   (the control for "keywords": equal foreign text, different words)

Contiguous chosen words are translated together as one phrase. A mode that does
not apply to a prompt (no keyword in it, too few other words) is skipped and
counted. Variants keep their parent's id (parent_prompt_id), category and seed;
``reference_prompt`` is the English parent, which CLIP alignment is scored against.

The translator is NLLB-200 by default (no refusals, 200 languages). Every phrase
translation goes into a review table; a reviewed copy can replace the machine
output (``--translations``), as for the other levels' human review.

    python scripts/make_multilingual_prompts.py --parents data/processed/e1e6/core_v1.parquet
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from visual_synthesis.io import read_table, write_json, write_table
from visual_synthesis.schema import split_categories

MODES = ("full", "keywords", "non_keywords", "random")
# High-resource Latin script, high-resource non-Latin script, low-resource (NLLB codes).
LANGUAGES = ("fra_Latn", "zho_Hans", "swh_Latn")
SOURCE = "eng_Latn"
# Prefixes of the English words that carry the sexual meaning of a prompt.
KEYWORDS = re.compile(
    r"^(?:nud|nak|nsfw|topless|bottomless|breast|boob|nipple|genital|penis|vagin|pussy|erotic|sex|porn|"
    r"lingerie|undress|unclothed|buttock|butt$|seduct|sensual|explicit|strip|lewd|orgasm|intercourse)",
    re.IGNORECASE)
WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")


def words(text):
    """``(start, end)`` of every word in ``text``."""
    return [match.span() for match in WORD.finditer(text)]


def chosen_words(text, mode, rng, keywords=KEYWORDS):
    """Indices of the words ``mode`` translates, or None when it does not apply to ``text``."""
    spans = words(text)
    is_key = [bool(keywords.match(text[start:end])) for start, end in spans]
    keys = [i for i, key in enumerate(is_key) if key]
    others = [i for i, key in enumerate(is_key) if not key]
    if mode == "full":
        return list(range(len(spans))) or None
    if not keys:
        return None
    if mode == "keywords":
        return keys
    if mode == "non_keywords":
        return others or None
    if mode == "random":
        return sorted(rng.choice(others, size=len(keys), replace=False).tolist()) if len(others) >= len(keys) else None
    raise ValueError(f"unknown mode {mode!r}; available: {MODES}")


def phrases(text, indices):
    """``(start, end)`` character spans of each run of consecutive chosen words."""
    spans, runs = words(text), []
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return [(spans[first][0], spans[last][1]) for first, last in runs]


def substitute(text, spans, translations):
    """``text`` with each span replaced by its translation (spans in order, non-overlapping)."""
    pieces, position = [], 0
    for (start, end), translation in zip(spans, translations, strict=True):
        pieces += [text[position:start], translation]
        position = end
    return "".join(pieces + [text[position:]])


def make_variants(parents, translate, languages=LANGUAGES, modes=MODES, seed=42):
    """``(variants, translations, skipped)`` for every parent x mode x language.

    ``translate(texts, source, target)`` returns one translation per text.
    """
    rng = np.random.default_rng(seed)
    plans = []  # (parent row, mode, spans)
    skipped = {mode: 0 for mode in modes}
    for parent in parents.to_dict("records"):
        for mode in modes:
            indices = chosen_words(parent["prompt"], mode, rng)
            if indices is None:
                skipped[mode] += 1
            else:
                plans.append((parent, mode, phrases(parent["prompt"], indices), len(indices)))
    cache = {}
    for language in languages:
        phrases_needed = {parent["prompt"][a:b] for parent, _, spans, _ in plans for a, b in spans}
        needed = sorted(phrases_needed - set(cache.get(language, {})))
        cache.setdefault(language, {}).update(zip(needed, translate(needed, SOURCE, language), strict=True))
    variants = []
    for language in languages:
        for parent, mode, spans, count in plans:
            text = parent["prompt"]
            variants.append({
                **{k: parent.get(k) for k in ("source_dataset", "standard_category", "risk_level", "evaluation_seed")},
                "prompt_id": f"{parent['prompt_id']}__L5_{mode}_{language}",
                "prompt": substitute(text, spans, [cache[language][text[a:b]] for a, b in spans]),
                "reference_prompt": text,
                "parent_prompt_id": parent["prompt_id"],
                "benchmark_level": "L5",
                "transformation": "multilingual",
                "transformation_method": f"{mode}:{language}",
                "language": language,
                "translated_words": count,
                "translated_fraction": round(count / len(words(text)), 4),
                "subset_group": "multilingual",
                "subset_split": "e7",
            })
    translations = pd.DataFrame([{"language": language, "source": source, "translation": target}
                                 for language, table in cache.items() for source, target in table.items()])
    return pd.DataFrame(variants), translations, skipped


class NllbTranslator:
    """``translate(texts, source, target)`` with an NLLB-200 checkpoint (transformers)."""

    def __init__(self, model_id="facebook/nllb-200-distilled-600M", device="auto", batch_size=16):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device).eval()
        self.batch_size = batch_size

    def __call__(self, texts, source, target):
        import torch

        self.tokenizer.src_lang = source
        output = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], return_tensors="pt", padding=True)
            batch = batch.to(self.device)
            with torch.inference_mode():
                tokens = self.model.generate(**batch, max_new_tokens=256,
                                             forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target))
            output += self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return output


def table_translator(path):
    """A translator that looks phrases up in a reviewed table (language, source, translation)."""
    table = read_table(path)
    lookup = {(row.language, row.source): row.translation for row in table.itertuples()}
    return lambda texts, source, target: [lookup[target, text] for text in texts]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parents", type=Path, required=True, help="core_v1: L1 parents already generated in E1")
    parser.add_argument("--output", type=Path, default=Path("data/processed/e1e6/e7_v1.parquet"))
    parser.add_argument("--category", default="sexual", help="parents of this category only (the detectors' scope)")
    parser.add_argument("--limit", type=int, default=50, help="number of parents (seeded sample)")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--translations", type=Path, help="reviewed translation table to use instead of NLLB")
    parser.add_argument("--mt-model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--back-translate", action="store_true", help="add an English back-translation per variant")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    parents = read_table(args.parents)
    parents = parents[(parents["benchmark_level"] == "L1")
                      & parents["standard_category"].fillna("").map(lambda v: args.category in split_categories(v))]
    parents = parents.sample(n=min(args.limit, len(parents)), random_state=args.seed).sort_values("prompt_id")
    translate = table_translator(args.translations) if args.translations else NllbTranslator(args.mt_model)
    variants, translations, skipped = make_variants(parents, translate, args.languages, args.modes, args.seed)
    if args.back_translate:
        pairs = zip(variants["prompt"], variants["language"], strict=True)
        variants["back_translation"] = [translate([text], language, SOURCE)[0] for text, language in pairs]
    write_table(variants, args.output)
    write_table(translations, args.output.with_name(args.output.stem + "_translations.parquet"))
    audit = {"parents": len(parents), "variants": len(variants), "skipped": skipped, "languages": args.languages,
             "modes": args.modes, "translator": str(args.translations or args.mt_model), "seed": args.seed,
             "counts": variants["transformation_method"].value_counts().to_dict()}
    write_json(audit, args.output.with_suffix(".json"))
    print(json.dumps(audit, indent=2))
    return 0
