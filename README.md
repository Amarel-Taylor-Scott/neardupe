# neardupe

> Find **near-duplicate** texts in a dataset — not just exact matches — with
> MinHash + LSH banding, **stdlib only.** Dataset hygiene for training corpora
> and RAG stores: surface the fuzzy dupes, cluster them, and keep one per
> cluster. No embeddings, no model call, no dependencies.

```python
from neardupe import find_duplicates, clusters, unique

texts = [
    "machine learning models trained on duplicated web text tend to memorize passages and waste compute during the long training run",
    "machine learning models trained on duplicated web text tend to memorize passages and waste compute during the lengthy training run",
    "a quick reminder to buy milk eggs bread and coffee on the way home tonight",
]

find_duplicates(texts, threshold=0.6)   # [(0, 1, 0.63)]  — texts 0 & 1 ~63% similar
clusters(texts, threshold=0.6)          # [[0, 1], [2]]
unique(texts, threshold=0.6)            # [0, 2]  — keep one per cluster, drop the rest
```

## The problem

Exact dedup (a `set()`, a hash column) misses the duplicates that actually bloat
a corpus: the same paragraph with one word changed, a reformatted copy, a
near-identical FAQ answer. And comparing every pair to catch them is O(n²) —
hopeless past a few thousand documents.

`neardupe` estimates pairwise similarity with **MinHash** and uses **LSH
banding** to only ever compare pairs that are *likely* similar — so it scales to
large corpora while still catching the fuzzy duplicates exact dedup leaves
behind.

## How it works

1. **Shingle** each text into overlapping word k-grams — `shingles`.
2. **MinHash** each shingle set into a fixed-length integer signature using `n`
   deterministic `blake2b` permutations — `signature`. The fraction of matching
   positions estimates the true Jaccard similarity (`jaccard`).
3. **Band** the signatures into `bands` groups and bucket items by each band's
   hash — `band` / `candidate_pairs`. Anything sharing a bucket is a *candidate*;
   this replaces the O(n²) all-pairs scan.
4. **Verify** each candidate against your `threshold` (`find_duplicates`),
   **cluster** the survivors with union-find (`clusters`), and **keep the first**
   of each (`unique`).

Hashing is `hashlib.blake2b` with a per-permutation salt, so signatures are
**identical across runs, processes, and machines** — never the salted builtin
`hash()`.

## API

```python
from neardupe import (
    shingles, signature, jaccard,       # minhash
    band, candidate_pairs,              # lsh
    find_duplicates, clusters, unique,  # dedupe
)

shingles(text, k=5)                     # set of word k-grams (char k-grams for tiny texts)
signature(shingles, n=128)             # tuple[int] MinHash signature (deterministic)
jaccard(sig_a, sig_b)                  # fraction of equal positions ≈ Jaccard similarity

candidate_pairs(signatures, bands=32)  # {(i, j), ...} sharing at least one LSH bucket

find_duplicates(texts, threshold=0.8)  # [(i, j, sim), ...] sorted by descending sim
clusters(texts, threshold=0.8)         # [[i, ...], ...] union-find groups (incl. singletons)
unique(texts, threshold=0.8)           # [i, ...] first index of each cluster
```

`threshold`, `k`, `n`, and `bands` flow through every dedupe function. The
default `threshold` is a strict `0.8`; lower it to catch looser matches.

## CLI

```bash
neardupe corpus.jsonl                         # pairs:  i <TAB> j <TAB> sim
neardupe corpus.jsonl --mode clusters         # one cluster (indices) per line
neardupe corpus.jsonl --mode unique           # kept indices; "kept N of M" on stderr
neardupe corpus.txt --field text --threshold 0.7
cat corpus.jsonl | neardupe - --field body    # read stdin, pull JSON field "body"
```

Each input line is read as a JSON object (use `--field`, default `text`) or, if
it is not JSON, as a raw text. Pass `-` to read stdin. `--k`, `--n`, and
`--bands` are also exposed.

## Layout

```
neardupe/
  minhash.py   shingles, MinHash signatures, Jaccard estimate
  lsh.py       LSH banding → candidate pairs (no O(n²) scan)
  dedupe.py    find_duplicates / clusters / unique (union-find)
  cli.py       the `neardupe` command
```

MIT. Stdlib only — no dependencies, no network, no API keys.
