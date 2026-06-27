"""MinHash signatures over text shingles — deterministic, stdlib only.

A document becomes a set of overlapping word k-grams (*shingles*); MinHash turns
that set into a fixed-length integer *signature* whose matching positions
estimate the Jaccard similarity of the underlying sets. Each of the ``n``
permutations is a ``hashlib.blake2b`` keyed by a per-permutation salt, so a
signature is identical across runs, processes, and machines — unlike the salted
builtin ``hash()``, which is not.
"""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"\w+")
_MAX = (1 << 64) - 1          # sentinel: larger than any 64-bit blake2b digest


def shingles(text, k=5):
    """Return the set of ``k``-word shingles (lowercased) for ``text``.

    Long texts → overlapping windows of ``k`` consecutive words. Texts with at
    least two but fewer than ``k`` words fall back to the bare word set, and
    single-word / punctuation-only texts fall back to character ``k``-grams, so
    every non-empty text still yields at least one shingle (hence a signature).
    """
    words = _WORD.findall(text.lower())
    if len(words) >= k:
        return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}
    if len(words) >= 2:
        return set(words)
    s = "".join(text.lower().split())
    if not s:
        return set()
    if len(s) <= k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def signature(shingles, n=128):
    """Return the MinHash signature of a shingle set: a tuple of ``n`` minima.

    Position ``i`` holds the smallest ``blake2b`` digest of any shingle under
    permutation ``i`` (selected by salting the hash with ``i``). An empty set
    yields an all-sentinel signature, so two empty texts compare as identical.
    """
    sig = [_MAX] * n
    if not shingles:
        return tuple(sig)
    encoded = [s.encode("utf-8") for s in shingles]
    for i in range(n):
        salt = i.to_bytes(16, "little")          # blake2b salt is max 16 bytes
        low = _MAX
        for b in encoded:
            v = int.from_bytes(
                hashlib.blake2b(b, digest_size=8, salt=salt).digest(), "little"
            )
            if v < low:
                low = v
        sig[i] = low
    return tuple(sig)


def jaccard(sig_a, sig_b):
    """Estimate Jaccard similarity as the fraction of equal MinHash positions."""
    if len(sig_a) != len(sig_b):
        raise ValueError(
            "signatures differ in length: %d vs %d" % (len(sig_a), len(sig_b))
        )
    if not sig_a:
        return 1.0
    equal = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return equal / len(sig_a)
