'''
Tests for the run modes: full (embed + cluster), embed (stop after writing the
embeddings), and cluster (cluster a saved embeddings file).  Small CPU pipeline.
'''
import json

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from nemi.workflow import DEFAULT_EMBEDDINGS_PATH, NEMI


def _blobs(n=60):
    X, _ = make_blobs(n_samples=n, n_features=5, centers=3, random_state=0)
    return X


def _cpu_params():
    return {"device": "cpu",
            "embedding_dict": {"n_components": 3, "n_neighbors": 15, "min_dist": 0.0},
            "clustering_dict": {"method": "agglomerative", "linkage": "ward",
                                "n_clusters": 3, "n_neighbors": 15}}


def test_embed_mode_writes_embeddings_and_stops(tmp_path):
    emb = tmp_path / "emb.npz"
    X = _blobs()
    nemi = NEMI(params=_cpu_params())
    nemi.run(X, n=3, mode="embed", embeddings=str(emb))

    d = np.load(emb)
    assert d["embeddings"].shape == (3, X.shape[0], 3)
    assert json.loads(d["params"].item())["device"] == "cpu"
    assert not hasattr(nemi, "clusters")              # clustering intentionally skipped


def test_full_mode_also_writes_embeddings(tmp_path):
    emb, out = tmp_path / "emb.npz", tmp_path / "out.npz"
    X = _blobs()
    NEMI(params=_cpu_params()).run(X, n=3, output=str(out), embeddings=str(emb))

    assert np.load(emb)["embeddings"].shape == (3, X.shape[0], 3)
    assert np.load(out)["clusters"].shape == (X.shape[0],)


def test_cluster_mode_reuses_saved_embeddings(tmp_path):
    emb, out = tmp_path / "emb.npz", tmp_path / "out.npz"
    X = _blobs()
    NEMI(params=_cpu_params()).run(X, n=3, mode="embed", embeddings=str(emb))

    nemi = NEMI(params=_cpu_params())
    nemi.run(mode="cluster", embeddings=str(emb), output=str(out))

    # the pack holds exactly the saved embeddings, not a fresh fit
    saved = np.load(emb)["embeddings"]
    assert np.array_equal(np.stack([m.embedding for m in nemi.nemi_pack]), saved)
    assert nemi.clusters.shape == (X.shape[0],)
    assert np.load(out)["member_clusters"].shape == (3, X.shape[0])


def test_cluster_mode_single_member_round_trip(tmp_path):
    emb, out = tmp_path / "emb.npz", tmp_path / "out.npz"
    X = _blobs()
    NEMI(params=_cpu_params()).run(X, n=1, mode="embed", embeddings=str(emb))
    assert np.load(emb)["embeddings"].shape == (1, X.shape[0], 3)

    nemi = NEMI(params=_cpu_params())
    nemi.run(mode="cluster", embeddings=str(emb), output=str(out))
    assert set(np.load(out).files) == {"embedding", "clusters"}
    assert nemi.clusters.shape == (X.shape[0],)


def test_cluster_mode_missing_embeddings_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no saved embeddings"):
        NEMI(params=_cpu_params()).run(mode="cluster",
                                       embeddings=str(tmp_path / "nope.npz"))


def test_default_embeddings_path_is_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X = _blobs()
    NEMI(params=_cpu_params()).run(X, n=2, mode="embed")
    assert np.load(DEFAULT_EMBEDDINGS_PATH)["embeddings"].shape[0] == 2

    NEMI(params=_cpu_params()).run(X, n=3, mode="embed")
    assert np.load(DEFAULT_EMBEDDINGS_PATH)["embeddings"].shape[0] == 3


def test_embed_mode_requires_X():
    with pytest.raises(ValueError, match="requires X"):
        NEMI(params=_cpu_params()).run(mode="embed")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        NEMI(params=_cpu_params()).run(_blobs(), mode="bogus")
