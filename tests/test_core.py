"""Offline tests for neardupe — deterministic MinHash + LSH, no network."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neardupe import (  # noqa: E402
    shingles, signature, jaccard, candidate_pairs,
    find_duplicates, clusters, unique,
)

# A long base sentence; near-duplicates differ by a single content word so they
# still share most of their 5-word shingles (MinHash detects them reliably).
BASE = ("the quick brown fox jumps over the lazy dog while the curious cat "
        "watches silently from the old wooden fence near the river bank under "
        "the bright morning sun on a calm summer day in july")
NEAR = BASE.replace("lazy", "sleepy")            # one word changed
NEAR2 = BASE.replace("morning", "evening")       # a different word changed
OTHER = ("economic policy debates dominated the parliamentary session as "
         "legislators argued about taxation healthcare and infrastructure "
         "spending throughout the long autumn afternoon")


def test_near_dup_found_above_low_threshold():
    dups = find_duplicates([BASE, NEAR], threshold=0.6)
    assert (0, 1) in [(i, j) for i, j, _ in dups]


def test_near_dup_not_found_above_high_threshold():
    assert find_duplicates([BASE, NEAR], threshold=0.95) == []


def test_exact_duplicate_found():
    dups = find_duplicates([BASE, OTHER, BASE], threshold=0.8)
    scored = {(i, j): sim for i, j, sim in dups}
    assert (0, 2) in scored
    assert scored[(0, 2)] == 1.0


def test_transitive_cluster_and_singleton():
    cl = clusters([BASE, NEAR, NEAR2, OTHER], threshold=0.6)
    assert [0, 1, 2] in cl       # transitively similar (NEAR↔NEAR2 only via BASE)
    assert [3] in cl             # unrelated text is its own cluster
    assert len(cl) == 2


def test_unique_keeps_one_per_cluster_and_first_index():
    texts = [BASE, NEAR, NEAR2, OTHER]
    cl = clusters(texts, threshold=0.6)
    keep = unique(texts, threshold=0.6)
    assert keep == [0, 3]                  # the first index of each cluster
    assert len(keep) == len(cl)


def test_distinct_texts_have_no_duplicates():
    texts = [
        OTHER,
        "the astronauts repaired the orbiting telescope during a long spacewalk "
        "high above the blue curve of the pacific ocean shortly after dawn",
        "she planted tomatoes basil and peppers in the small garden behind her "
        "grandmother's stone cottage early that rainy spring weekend",
    ]
    assert find_duplicates(texts, threshold=0.6) == []


def test_jaccard_of_identical_signatures_is_one():
    sig = signature(shingles(BASE))
    assert jaccard(sig, sig) == 1.0


def test_signature_is_deterministic():
    a = signature(shingles(BASE))
    b = signature(shingles(BASE))
    assert isinstance(a, tuple)
    assert a == b


def test_candidate_pairs_surface_near_dup():
    sigs = [signature(shingles(t)) for t in (BASE, NEAR, OTHER)]
    assert (0, 1) in candidate_pairs(sigs)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("\n%d passed" % len(fns))
