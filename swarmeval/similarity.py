"""Text similarity backends for the redundancy analysis (Phase 2).

Redundancy needs "are these two texts saying the same thing?".  Two backends:

* ``LexicalSimilarity`` (default, standard library only): Jaccard similarity
  of word 3-shingles.  It catches copies and light rewording, not
  paraphrase.  Candidate pairs come from an inverted shingle index, so texts
  that share nothing are never compared.
* ``EmbeddingSimilarity``: cosine similarity of sentence embeddings, which
  catches paraphrase.  It needs an ``embed_fn`` or the optional
  ``sentence-transformers`` package (``pip install 'swarmeval[embeddings]'``),
  checked for when the backend is built and imported only when first used.  Every pair is a candidate, so the number of
  pairs is capped (``max_pairs``) and ``truncated`` says when the cap hit.

Anything with ``name``, ``threshold``, ``candidates(texts)`` and
``scores(texts, pairs)`` can be passed instead, plus ``truncated`` (True when
the max_pairs cap dropped candidate pairs; read as False when absent);
``resolve_similarity`` turns a name, an import path (``"pkg.module:attr"``)
or an instance into one.

Classes:
- Similarity: A similarity backend. ``candidates`` yields index pairs ``(i, j)`` with ``i < j`` worth scoring;
  ``scores`` returns one score in [0, 1] per pair (higher = more similar); pairs scoring ``>= threshold`` count as
  similar.
- LexicalSimilarity: Jaccard similarity of lowercase word ``shingle``-grams. A pair where either text has fewer than
  ``shingle`` words is compared on word sets instead (a 2-word text has no 3-shingles to share). Candidates are
  capped at ``max_pairs`` (``truncated`` is then True and later pairs are not compared, so redundancy can be under-
  counted).
- EmbeddingSimilarity: Cosine similarity of embeddings. ``embed_fn(list_of_texts) -> list_of_vectors``; without one,
  ``sentence-transformers`` must be installed (checked here, ImportError otherwise) and is imported on first use,
  when ``model`` is loaded. Every pair is a candidate, capped at ``max_pairs`` (``truncated`` is then True and later
  pairs are not compared, so redundancy can be under-counted).
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

Pair = tuple[int, int]
_WORD = re.compile(r"\w+")


@runtime_checkable
class Similarity(Protocol):
    name: str
    threshold: float
    truncated: bool

    def candidates(self, texts: Sequence[str]) -> Iterable[Pair]:
        """Index pairs ``(i, j)`` with ``i < j`` worth scoring."""

    def scores(self, texts: Sequence[str], pairs: Sequence[Pair]) -> list[float]:
        """One score in [0, 1] per pair, higher meaning more similar."""


def similar_pairs(sim: Any, texts: Sequence[str]) -> list[tuple[int, int, float]]:
    """All candidate pairs scoring at or above the backend's threshold."""
    pairs = list(sim.candidates(texts))
    if not pairs:
        return []
    return [(i, j, s) for (i, j), s in zip(pairs, sim.scores(texts, pairs), strict=True) if s >= sim.threshold]


def words(text: str) -> list[str]:
    """Lowercase words of ``text``."""
    return _WORD.findall(text.lower())


def jaccard(a: set[Any], b: set[Any]) -> float:
    """Jaccard similarity of two sets, 1.0 when both are empty."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class LexicalSimilarity:
    name = "lexical"

    def __init__(self, threshold: float = 0.6, shingle: int = 3, max_pairs: int = 200000) -> None:
        if shingle < 1:
            raise ValueError("shingle must be >= 1")
        self.threshold = threshold
        self.shingle = shingle
        self.max_pairs = max_pairs
        self.truncated = False

    def _features(self, texts: Sequence[str]) -> tuple[list[set[str]], list[frozenset[tuple[str, ...]]]]:
        """Word sets and shingle sets of each text."""
        uni, shingles = [], []
        k = self.shingle
        for t in texts:
            w = words(t)
            uni.append(set(w))
            shingles.append(frozenset(tuple(w[i:i + k]) for i in range(len(w) - k + 1)) if len(w) >= k
                            else frozenset())
        return uni, shingles

    def similarity(self, a: str, b: str) -> float:
        """Similarity of two texts."""
        return self.scores([a, b], [(0, 1)])[0]

    def candidates(self, texts: Sequence[str]) -> list[Pair]:
        """Pairs sharing at least one shingle (or, involving a short text, one
        word), found through an inverted index rather than all pairs."""
        uni, shingles = self._features(texts)
        by_shingle: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i, sh in enumerate(shingles):
            for s in sh:
                by_shingle[s].append(i)

        def generate() -> Iterator[Pair]:
            for posting in by_shingle.values():
                for x in range(len(posting)):
                    for y in range(x + 1, len(posting)):
                        yield (posting[x], posting[y])
            short = [i for i, sh in enumerate(shingles) if not sh and uni[i]]
            if short:
                by_word: dict[str, list[int]] = defaultdict(list)
                for i, ws in enumerate(uni):
                    for w in ws:
                        by_word[w].append(i)
                for i in short:
                    for w in uni[i]:
                        for j in by_word[w]:
                            if j != i:
                                yield (min(i, j), max(i, j))

        self.truncated = False
        out: set[Pair] = set()
        for pair in generate():
            if len(out) >= self.max_pairs and pair not in out:
                self.truncated = True
                break
            out.add(pair)
        return sorted(out)

    def scores(self, texts: Sequence[str], pairs: Sequence[Pair]) -> list[float]:
        """One similarity per pair: shingle Jaccard, or word-set Jaccard for short texts."""
        uni, shingles = self._features(texts)
        out = []
        for i, j in pairs:
            if shingles[i] and shingles[j]:
                out.append(jaccard(set(shingles[i]), set(shingles[j])))
            else:
                out.append(jaccard(uni[i], uni[j]))
        return out

    def __repr__(self) -> str:
        return f"LexicalSimilarity(threshold={self.threshold}, shingle={self.shingle}, max_pairs={self.max_pairs})"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors, 0.0 when either is zero."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


_NEEDS_ST = "EmbeddingSimilarity needs sentence-transformers: pip install 'swarmeval[embeddings]', or pass embed_fn="


class EmbeddingSimilarity:
    name = "embeddings"

    def __init__(self, embed_fn: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
                 model: str = "all-MiniLM-L6-v2", threshold: float = 0.85, max_pairs: int = 5000) -> None:
        if embed_fn is None and importlib.util.find_spec("sentence_transformers") is None:
            raise ImportError(_NEEDS_ST)
        self.embed_fn = embed_fn
        self.model = model
        self.threshold = threshold
        self.max_pairs = max_pairs
        self.truncated = False

    def _embedder(self) -> Callable[[list[str]], Sequence[Sequence[float]]]:
        """The embedding function, loading sentence-transformers on first use."""
        if self.embed_fn is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(_NEEDS_ST) from exc
            st = SentenceTransformer(self.model)
            self.embed_fn = lambda texts: [[float(x) for x in v] for v in st.encode(list(texts))]
        return self.embed_fn

    def candidates(self, texts: Sequence[str]) -> list[Pair]:
        """All index pairs, capped at ``max_pairs``."""
        out: list[Pair] = []
        n = len(texts)
        self.truncated = False
        for i in range(n):
            for j in range(i + 1, n):
                if len(out) >= self.max_pairs:
                    self.truncated = True
                    return out
                out.append((i, j))
        return out

    def scores(self, texts: Sequence[str], pairs: Sequence[Pair]) -> list[float]:
        """Cosine similarity per pair, embedding each distinct text once."""
        if not pairs:
            return []
        needed = sorted({i for p in pairs for i in p})
        uniq: dict[str, int] = {}
        for i in needed:
            uniq.setdefault(texts[i], len(uniq))
        vecs = list(self._embedder()(list(uniq)))
        vec_of = {i: vecs[uniq[texts[i]]] for i in needed}
        return [cosine(vec_of[i], vec_of[j]) for i, j in pairs]

    def __repr__(self) -> str:
        return (f"EmbeddingSimilarity(model={self.model!r}, threshold={self.threshold}, "
                f"max_pairs={self.max_pairs}, custom_embed_fn={self.embed_fn is not None})")


SIMILARITIES: dict[str, Callable[[], Any]] = {"lexical": LexicalSimilarity, "embeddings": EmbeddingSimilarity}


def _is_similarity(obj: Any) -> bool:
    """True if ``obj`` is an instance with the similarity backend interface."""
    return (not inspect.isclass(obj) and callable(getattr(obj, "candidates", None))
            and callable(getattr(obj, "scores", None)) and hasattr(obj, "threshold"))


def resolve_similarity(spec: Any = None) -> Any:
    """A similarity backend from ``None`` (lexical), a registry name, a ``"pkg.module:attr"`` path to an
    instance or argument-free class/factory, or an instance."""
    if spec is None:
        return LexicalSimilarity()
    if _is_similarity(spec):
        return spec
    if isinstance(spec, str):
        if spec in SIMILARITIES:
            return SIMILARITIES[spec]()
        if ":" in spec:
            mod_name, _, attr = spec.partition(":")
            try:
                obj = getattr(importlib.import_module(mod_name), attr)
            except (ImportError, AttributeError) as exc:
                raise ValueError(f"similarity {spec!r}: cannot import: {exc}") from exc
            if _is_similarity(obj):
                return obj
            if callable(obj):
                try:
                    made = obj()
                except TypeError as exc:
                    raise ValueError(f"similarity {spec!r}: could not be built without arguments: {exc}") from exc
                if _is_similarity(made):
                    return made
            raise ValueError(f"similarity {spec!r} is not a similarity (needs threshold, candidates(), scores())")
        raise ValueError(f"unknown similarity {spec!r}; use one of {sorted(SIMILARITIES)} or 'pkg.module:attr'")
    raise ValueError(f"cannot use {spec!r} as a similarity; pass a name, 'pkg.module:attr' or an instance "
                     f"with threshold, candidates() and scores()")
