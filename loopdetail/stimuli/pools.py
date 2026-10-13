"""Per-tokenizer candidate pools: every answer string (with its leading space, as it
appears after the prompt) must be ONE token for the tokenizer of the model under test."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..tokenization import TextTokenizer
from .vocab import BOX_LABELS, CATEGORY_ITEMS, CATEGORY_PREFERENCE, NAMES


@dataclass
class Pool:
    tokenizer: str
    categories: dict[str, list[str]]  # category name -> items (equal sizes)
    names: list[str]
    labels: list[str]
    dropped: dict[str, list[str]]  # words removed because they are multi-token

    def item_category(self) -> dict[str, str]:
        return {it: c for c, items in self.categories.items() for it in items}

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Pool":
        return cls(**d)


def is_single_token(tok: TextTokenizer, word: str) -> bool:
    return len(tok.encode(" " + word, add_special=False)) == 1


def resolve_pool(tok: TextTokenizer, n_categories: int = 8, min_items: int = 10,
                 min_names: int = 32) -> Pool:
    dropped: dict[str, list[str]] = {"items": [], "categories": [], "names": []}
    survivors: dict[str, list[str]] = {}
    for cat in CATEGORY_PREFERENCE:
        if not is_single_token(tok, cat):
            dropped["categories"].append(cat)
            continue
        items = [w for w in CATEGORY_ITEMS[cat] if is_single_token(tok, w)]
        dropped["items"] += [w for w in CATEGORY_ITEMS[cat] if w not in items]
        if len(items) >= min_items:
            survivors[cat] = items
        else:
            dropped["categories"].append(cat)
    if len(survivors) < max(4, min(n_categories, 6)):
        raise RuntimeError(
            f"only {len(survivors)} categories survive single-token filtering for {tok.name}; "
            "extend loopdetail/stimuli/vocab.py"
        )
    chosen = list(survivors)[:n_categories]
    size = min(len(survivors[c]) for c in chosen)
    categories = {c: survivors[c][:size] for c in chosen}
    single_names = [n for n in NAMES if is_single_token(tok, n)]
    dropped["names"] = [n for n in NAMES if n not in single_names]
    names = single_names + [n for n in NAMES if n not in single_names]
    if len(names) < min_names:
        raise RuntimeError(f"need >= {min_names} names, have {len(names)}")
    return Pool(tokenizer=tok.name, categories=categories, names=names, labels=list(BOX_LABELS),
                dropped=dropped)
