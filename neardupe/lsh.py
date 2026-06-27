"""LSH banding over MinHash signatures — find candidates without O(n²) scans.

Each signature is split into ``bands`` contiguous bands; items are bucketed by
each band's hash. Two signatures that agree on *any* whole band share a bucket,
so near-duplicates surface as candidate pairs while dissimilar items almost
never collide — letting :func:`candidate_pairs` skip the full cross product.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict


def _band_hash(chunk):
    data = ",".join(map(str, chunk)).encode("utf-8")
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def band(signatures, bands=32):
    """Bucket item indices by band: ``{(band_no, band_hash): [idx, ...]}``.

    Every signature is cut into ``bands`` contiguous bands of equal length, so
    ``bands`` must divide the signature length. Items sharing a band hash land
    in the same bucket.
    """
    buckets = defaultdict(list)
    if not signatures:
        return dict(buckets)
    length = len(signatures[0])
    if bands <= 0 or length % bands:
        raise ValueError(
            "bands (%d) must divide signature length (%d)" % (bands, length)
        )
    rows = length // bands
    for idx, sig in enumerate(signatures):
        if len(sig) != length:
            raise ValueError("all signatures must share one length")
        for b in range(bands):
            chunk = sig[b * rows:(b + 1) * rows]
            buckets[(b, _band_hash(chunk))].append(idx)
    return dict(buckets)


def candidate_pairs(signatures, bands=32):
    """Return ``{(i, j), ...}`` with ``i < j`` for items sharing any bucket."""
    pairs = set()
    for idxs in band(signatures, bands).values():
        if len(idxs) < 2:
            continue
        # idxs is ascending (enumerate order), so (idxs[a], idxs[b]) has i < j
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs.add((idxs[a], idxs[b]))
    return pairs
