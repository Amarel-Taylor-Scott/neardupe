"""Near-duplicate detection: MinHash signatures + LSH candidates + verify.

``text → shingles → signature → LSH candidate pairs → Jaccard verification``.
:func:`find_duplicates` returns scored pairs, :func:`clusters` groups them with
union-find, and :func:`unique` keeps the first member of each cluster — the
indices to keep when scrubbing a corpus.
"""

from __future__ import annotations

from .lsh import candidate_pairs
from .minhash import jaccard, shingles, signature


def signatures(texts, *, k=5, n=128):
    """Return the MinHash signature for each text (aligned to input order)."""
    return [signature(shingles(t, k=k), n=n) for t in texts]


def find_duplicates(texts, *, threshold=0.8, k=5, n=128, bands=32):
    """Return near-duplicate ``(i, j, sim)`` triples sorted by descending sim.

    Only LSH candidate pairs are scored, so cost tracks the number of bucket
    collisions rather than n². A pair is reported when its MinHash Jaccard
    estimate is ``>= threshold``.
    """
    sigs = signatures(texts, k=k, n=n)
    out = []
    for i, j in candidate_pairs(sigs, bands=bands):
        sim = jaccard(sigs[i], sigs[j])
        if sim >= threshold:
            out.append((i, j, sim))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out


def clusters(texts, *, threshold=0.8, k=5, n=128, bands=32):
    """Group indices into near-duplicate clusters via union-find over dup pairs.

    Every index appears in exactly one group (singletons included). Each group
    is sorted ascending; groups are ordered by their smallest index.
    """
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, j, _ in find_duplicates(
        texts, threshold=threshold, k=k, n=n, bands=bands
    ):
        union(i, j)

    groups = {}
    for x in range(len(texts)):
        groups.setdefault(find(x), []).append(x)
    return [sorted(g) for g in sorted(groups.values(), key=min)]


def unique(texts, *, threshold=0.8, k=5, n=128, bands=32):
    """Return the kept indices: the first member of each near-duplicate cluster."""
    groups = clusters(texts, threshold=threshold, k=k, n=n, bands=bands)
    return sorted(min(g) for g in groups)
