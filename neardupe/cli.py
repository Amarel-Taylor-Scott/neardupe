"""``neardupe`` CLI — find near-duplicate texts in a dataset from the shell.

    neardupe corpus.jsonl                       # pairs:  i  j  sim
    neardupe corpus.jsonl --mode clusters       # one cluster (indices) per line
    neardupe corpus.txt --mode unique           # kept indices, count on stderr
    cat corpus.jsonl | neardupe - --field body  # read stdin, JSON field "body"
"""

from __future__ import annotations

import argparse
import json
import sys

from .dedupe import clusters, find_duplicates, unique


def _read_texts(source, field):
    raw = sys.stdin.read() if source == "-" else open(source, encoding="utf-8").read()
    texts = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        obj = None
        if s[0] in "{[":
            try:
                obj = json.loads(s)
            except Exception:
                obj = None
        if isinstance(obj, dict):
            texts.append(str(obj.get(field, "")))
        else:
            texts.append(s)
    return texts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="neardupe",
        description="Find near-duplicate texts (MinHash + LSH, stdlib only).",
    )
    p.add_argument("file", help="input file, or '-' for stdin")
    p.add_argument("--field", default="text",
                   help="JSON field to read when a line is a JSON object (default: text)")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Jaccard similarity cutoff (default: 0.8)")
    p.add_argument("--k", type=int, default=5,
                   help="shingle size in words (default: 5)")
    p.add_argument("--n", type=int, default=128,
                   help="number of MinHash permutations (default: 128)")
    p.add_argument("--bands", type=int, default=32,
                   help="LSH bands; must divide --n (default: 32)")
    p.add_argument("--mode", choices=("pairs", "clusters", "unique"),
                   default="pairs", help="output mode (default: pairs)")
    a = p.parse_args(argv)

    kw = dict(threshold=a.threshold, k=a.k, n=a.n, bands=a.bands)

    try:
        texts = _read_texts(a.file, a.field)
        if a.mode == "pairs":
            for i, j, sim in find_duplicates(texts, **kw):
                print("%d\t%d\t%.4f" % (i, j, sim))
        elif a.mode == "clusters":
            for group in clusters(texts, **kw):
                print(" ".join(str(i) for i in group))
        else:  # unique
            kept = unique(texts, **kw)
            for i in kept:
                print(i)
            sys.stderr.write("neardupe: kept %d of %d\n" % (len(kept), len(texts)))
    except Exception as e:  # noqa: BLE001 — CLI boundary
        sys.stderr.write("neardupe: %s\n" % e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
