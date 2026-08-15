"""UNG node adapters for neardupe — pure, JSON-in/JSON-out wrappers.

Each function wraps a documented neardupe API with JSON-serializable inputs
and outputs only (dict/list/str/int/float/bool/None); shingle sets become
sorted lists and signature/pair tuples become lists. Single-output nodes
return the output value directly. No I/O, no network, no filesystem access.
"""

from __future__ import annotations

from .dedupe import (
    clusters as _clusters,
    find_duplicates as _find_duplicates,
    unique as _unique,
)
from .lsh import candidate_pairs as _candidate_pairs
from .minhash import jaccard as _jaccard, shingles as _shingles, signature as _signature


def find_duplicate_pairs(texts: list, threshold: float = 0.8, k: int = 5,
                         n: int = 128, bands: int = 32) -> list:
    """Return near-duplicate [i, j, similarity] triples sorted by descending sim."""
    return [[i, j, sim] for i, j, sim in
            _find_duplicates(texts, threshold=threshold, k=k, n=n, bands=bands)]


def cluster_texts(texts: list, threshold: float = 0.8, k: int = 5,
                  n: int = 128, bands: int = 32) -> list:
    """Group text indices into near-duplicate clusters (singletons included)."""
    return _clusters(texts, threshold=threshold, k=k, n=n, bands=bands)


def unique_texts(texts: list, threshold: float = 0.8, k: int = 5,
                 n: int = 128, bands: int = 32) -> list:
    """Return the indices to keep: the first member of each duplicate cluster."""
    return _unique(texts, threshold=threshold, k=k, n=n, bands=bands)


def shingle_text(text: str, k: int = 5) -> list:
    """Return the sorted k-word shingles of the text (char k-grams for tiny texts)."""
    return sorted(_shingles(text, k=k))


def minhash_signature(text: str, k: int = 5, n: int = 128) -> list:
    """Return the deterministic n-position MinHash signature of the text."""
    return list(_signature(_shingles(text, k=k), n=n))


def pairwise_similarity(text_a: str, text_b: str, k: int = 5, n: int = 128) -> float:
    """Estimate Jaccard similarity of two texts via their MinHash signatures."""
    sig_a = _signature(_shingles(text_a, k=k), n=n)
    sig_b = _signature(_shingles(text_b, k=k), n=n)
    return _jaccard(sig_a, sig_b)


def lsh_candidate_pairs(signatures: list, bands: int = 32) -> list:
    """Return sorted [i, j] pairs sharing at least one LSH band bucket."""
    sigs = [tuple(s) for s in signatures]
    return [list(p) for p in sorted(_candidate_pairs(sigs, bands=bands))]


_TAGS = ["license.mit", "runtime.python", "dependency-free"]

_TUNING = [
    {"name": "threshold", "value_type": "number", "default": 0.8, "required": False},
    {"name": "k", "value_type": "integer", "default": 5, "required": False},
    {"name": "n", "value_type": "integer", "default": 128, "required": False},
    {"name": "bands", "value_type": "integer", "default": 32, "required": False},
]

_TEXTS_IN = [{"name": "texts", "type_id": "amarel.types.text-list",
              "description": "The corpus texts, index-addressed."}]


def _node(fn, action, caps, summary, inputs, outputs, parameters):
    return {
        "fn": fn,
        "id": "amarel.neardupe." + action,
        "capabilities": caps,
        "summary": summary,
        "inputs": inputs,
        "outputs": outputs,
        "parameters": parameters,
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    }


NODES = [
    _node(find_duplicate_pairs, "find-duplicates", ["records.find-duplicates"],
          "Find near-duplicate text pairs with MinHash + LSH banding, "
          "verified against a Jaccard threshold.",
          _TEXTS_IN,
          [{"name": "pairs", "type_id": "amarel.types.json-value",
            "description": "[i, j, similarity] triples, best first."}],
          _TUNING),
    _node(cluster_texts, "cluster", ["records.cluster-duplicates"],
          "Union-find near-duplicate pairs into clusters of text indices "
          "(singletons included).",
          _TEXTS_IN,
          [{"name": "clusters", "type_id": "amarel.types.json-value",
            "description": "Sorted index groups ordered by smallest member."}],
          _TUNING),
    _node(unique_texts, "unique", ["records.deduplicate"],
          "Return the indices to keep after near-duplicate removal: the "
          "first member of each cluster.",
          _TEXTS_IN,
          [{"name": "kept", "type_id": "amarel.types.json-value",
            "description": "Ascending kept indices."}],
          _TUNING),
    _node(shingle_text, "shingles", ["text.shingle"],
          "Split a text into sorted overlapping k-word shingles (character "
          "k-grams for tiny texts).",
          [{"name": "text", "type_id": "amarel.types.text",
            "description": "The text to shingle."}],
          [{"name": "shingles", "type_id": "amarel.types.text-list",
            "description": "The sorted shingle strings."}],
          [{"name": "k", "value_type": "integer", "default": 5, "required": False}]),
    _node(minhash_signature, "minhash", ["text.fingerprint"],
          "Compute the deterministic blake2b MinHash signature of a text "
          "(identical across runs, processes, and machines).",
          [{"name": "text", "type_id": "amarel.types.text",
            "description": "The text to fingerprint."}],
          [{"name": "signature", "type_id": "amarel.types.signature",
            "description": "The n-position integer signature."}],
          [{"name": "k", "value_type": "integer", "default": 5, "required": False},
           {"name": "n", "value_type": "integer", "default": 128, "required": False}]),
    _node(pairwise_similarity, "pairwise-similarity", ["text.estimate-similarity"],
          "Estimate the Jaccard similarity of two texts from their MinHash "
          "signatures.",
          [{"name": "text_a", "type_id": "amarel.types.text",
            "description": "The first text."},
           {"name": "text_b", "type_id": "amarel.types.text",
            "description": "The second text."}],
          [{"name": "similarity", "type_id": "amarel.types.number",
            "description": "Estimated Jaccard similarity in [0, 1]."}],
          [{"name": "k", "value_type": "integer", "default": 5, "required": False},
           {"name": "n", "value_type": "integer", "default": 128, "required": False}]),
    _node(lsh_candidate_pairs, "candidate-pairs", ["records.propose-candidates"],
          "Bucket MinHash signatures with LSH banding and return the [i, j] "
          "pairs sharing any bucket — no O(n^2) scan.",
          [{"name": "signatures", "type_id": "amarel.types.signatures",
            "description": "One signature list per item, equal lengths."}],
          [{"name": "pairs", "type_id": "amarel.types.json-value",
            "description": "Sorted [i, j] candidate pairs (i < j)."}],
          [{"name": "bands", "value_type": "integer", "default": 32,
            "required": False}]),
]
