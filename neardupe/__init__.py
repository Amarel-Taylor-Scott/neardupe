"""neardupe — find near-duplicate texts with MinHash + LSH (stdlib only).

Beyond exact dedup: shingle each text, MinHash it into a fixed-length signature,
use LSH banding to surface candidate pairs without an O(n²) scan, then verify
each candidate with the Jaccard estimate. Dataset hygiene for training corpora
and RAG stores.
"""

from .minhash import shingles, signature, jaccard
from .lsh import band, candidate_pairs
from .dedupe import find_duplicates, clusters, unique, signatures

__all__ = [
    "shingles", "signature", "jaccard",
    "band", "candidate_pairs",
    "find_duplicates", "clusters", "unique", "signatures",
]
__version__ = "0.1.0"
