"""Similarity backends and their resolution."""
import builtins
import importlib.util
import sys

import pytest

from swarmeval import EmbeddingSimilarity, LexicalSimilarity, resolve_similarity
from swarmeval.similarity import cosine, similar_pairs


def test_lexical_boundaries():
    lex = LexicalSimilarity()
    assert lex.similarity("The quick brown fox jumps", "the  QUICK brown fox jumps") == 1.0
    assert lex.similarity("alpha beta gamma delta", "one two three four") == 0.0
    assert lex.similarity("a b c d", "a b c e") == pytest.approx(1 / 3)
    assert lex.similarity("hello world", "world hello") == 1.0
    assert lex.similarity("hello", "hello there friend") == pytest.approx(1 / 3)
    assert lex.similarity("", "") == 1.0


def test_lexical_candidates_use_index():
    lex = LexicalSimilarity()
    texts = [
        "Aurora was first released in 2015",
        "Aurora was first released in 2016",
        "completely unrelated sentence about cooking pasta",
        "hi",
        "hi there",
    ]
    cands = lex.candidates(texts)
    assert (0, 1) in cands and (3, 4) in cands
    assert all(2 not in p for p in cands)
    assert (0, 2) not in cands and (0, 3) not in cands
    hits = similar_pairs(lex, texts)
    assert [(i, j) for i, j, _ in hits] == [(0, 1)]


def fake_embed(texts):
    table = {"cat": [1.0, 0.0], "kitten": [0.9, 0.1], "car": [0.0, 1.0]}
    return [table[t] for t in texts]


def test_embedding_with_fake_embed_fn():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return fake_embed(texts)

    emb = EmbeddingSimilarity(embed_fn=embed)
    texts = ["cat", "kitten", "car", "cat"]
    pairs = emb.candidates(texts)
    assert len(pairs) == 6 and emb.truncated is False
    scores = emb.scores(texts, pairs)
    assert scores[pairs.index((0, 1))] == pytest.approx(cosine([1, 0], [0.9, 0.1]))
    assert scores[pairs.index((0, 2))] == 0.0
    assert scores[pairs.index((0, 3))] == pytest.approx(1.0)
    assert len(calls) == 1 and sorted(calls[0]) == ["car", "cat", "kitten"]
    hits = {(i, j) for i, j, _ in similar_pairs(emb, texts)}
    assert hits == {(0, 1), (0, 3), (1, 3)}


def test_embedding_pair_cap():
    emb = EmbeddingSimilarity(embed_fn=fake_embed, max_pairs=2)
    assert emb.candidates(["cat", "kitten", "car"]) == [(0, 1), (0, 2)]
    assert emb.truncated is True
    emb.candidates(["cat", "car"])
    assert emb.truncated is False


def _fake_find_spec(monkeypatch, found):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **kw: (
        (object() if found else None) if name == "sentence_transformers" else real(name, *a, **kw)))


def test_embedding_missing_extra(monkeypatch):
    _fake_find_spec(monkeypatch, found=True)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    emb = EmbeddingSimilarity()
    with pytest.raises(ImportError, match=r"pip install 'swarmeval\[embeddings\]', or pass embed_fn="):
        emb.scores(["a", "b"], [(0, 1)])


def test_embedding_missing_extra_fails_at_construction(monkeypatch):
    _fake_find_spec(monkeypatch, found=False)
    with pytest.raises(ImportError, match=r"needs sentence-transformers: pip install 'swarmeval\[embeddings\]'"):
        EmbeddingSimilarity()
    with pytest.raises(ImportError, match="needs sentence-transformers"):
        resolve_similarity("embeddings")
    assert EmbeddingSimilarity(embed_fn=lambda texts: [[1.0]] * len(texts)).embed_fn is not None


def test_missing_embeddings_extra_stops_evaluate_before_any_run(monkeypatch, tmp_path):
    from swarmeval import evaluate
    from swarmeval.cli import main
    from swarmeval.examples.research_swarm import build_benchmark
    _fake_find_spec(monkeypatch, found=False)
    ran = []

    def swarm(task, ctx):
        ran.append(task.task_id)
        return "x"

    with pytest.raises(ImportError, match="needs sentence-transformers"):
        evaluate(swarm, build_benchmark(2), similarity="embeddings")
    assert ran == []
    with pytest.raises(SystemExit, match="^swarmeval: --similarity: .*needs sentence-transformers"):
        main(["demo", "--trials", "1", "--similarity", "embeddings", "--out", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_resolve_similarity(monkeypatch):
    assert isinstance(resolve_similarity(None), LexicalSimilarity)
    assert isinstance(resolve_similarity("lexical"), LexicalSimilarity)
    real_import = builtins.__import__

    def guarded(name, *a, **kw):
        if name.startswith("sentence_transformers"):
            raise AssertionError("imported too early")
        return real_import(name, *a, **kw)

    _fake_find_spec(monkeypatch, found=True)
    monkeypatch.setattr(builtins, "__import__", guarded)
    emb = resolve_similarity("embeddings")
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert isinstance(emb, EmbeddingSimilarity) and emb.embed_fn is None

    inst = LexicalSimilarity(threshold=0.3)
    assert resolve_similarity(inst) is inst
    assert isinstance(resolve_similarity("swarmeval.similarity:LexicalSimilarity"), LexicalSimilarity)
    assert isinstance(resolve_similarity("test_similarity:CUSTOM"), LexicalSimilarity)
    for bad in ("nope", "swarmeval.similarity:missing", "no_such_module_xyz:thing", "swarmeval.similarity:cosine",
                42):
        with pytest.raises(ValueError):
            resolve_similarity(bad)


CUSTOM = LexicalSimilarity(threshold=0.5)


def test_lexical_candidates_capped_when_one_phrase_is_everywhere():
    sim = LexicalSimilarity(max_pairs=500)
    pairs = sim.candidates([f"no reliable information found item {i}" for i in range(200)])
    assert len(pairs) <= 500 and sim.truncated
