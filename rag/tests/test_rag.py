import numpy as np

from rag.documents.chunker import chunk_text
from rag.embeddings.encoder import HashingTfidfEncoder


def test_chunker_basic():
    text = ("This is a sentence. " * 200).strip()
    chunks = chunk_text(text, target_chars=400, overlap_chars=80)
    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)


def test_encoder_dim_and_norm():
    enc = HashingTfidfEncoder(dim=256)
    enc.fit(["hello world", "the quick brown fox", "merhaba dünya"])
    v = enc.encode_one("hello world")
    assert v.shape == (256,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_encoder_finds_similar():
    enc = HashingTfidfEncoder(dim=512)
    docs = [
        "Paris is the capital of France.",
        "Bananas are yellow fruits grown in tropical regions.",
        "Python is a programming language used in data science.",
    ]
    enc.fit(docs)
    vecs = enc.encode(docs)
    q = enc.encode_one("What is the capital of France?")
    sims = [float(np.dot(q, v)) for v in vecs]
    # Top result should be the Paris doc
    assert sims.index(max(sims)) == 0
