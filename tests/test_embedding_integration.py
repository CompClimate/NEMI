'''
Integration test for device-dispatched embedding (nemi.workflow.fit_embedding).

Generates a UMAP embedding on CPU (umap-learn) and GPU (cuML) from the same
toy dataset.  UMAP is stochastic and the CPU/GPU implementations differ, so the
embeddings are NOT expected to match — we assert each is a valid, same-shaped,
finite embedding.  The GPU checks auto-skip when cuML is unavailable, so this
meaningfully runs only on a RAPIDS/GPU machine (the server).
'''
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from nemi.workflow import SingleNemi

N_SAMPLES = 200
N_COMPONENTS = 3
EMBED = {"n_components": N_COMPONENTS, "n_neighbors": 15, "min_dist": 0.0}
TSNE_EMBED = {"method": "tsne", "n_components": 2, "perplexity": 30.0,
              "early_exaggeration": 12.0, "learning_rate": 200.0, "max_iter": 250}


def _toy():
    # deterministic data (only UMAP is stochastic)
    X, _ = make_blobs(n_samples=N_SAMPLES, n_features=5, centers=4, random_state=0)
    return X


def _embed(device, X, embed=None):
    nm = SingleNemi(params={"device": device,
                            "embedding_dict": embed if embed else EMBED})
    nm.fit_embedding(X)
    return nm.embedding


def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def test_cpu_embedding_valid():
    emb = _to_numpy(_embed("cpu", _toy()))
    assert emb.shape == (N_SAMPLES, N_COMPONENTS)
    assert np.all(np.isfinite(emb))


def test_gpu_embedding_valid():
    pytest.importorskip("cuml")
    emb = _to_numpy(_embed("gpu", _toy()))
    assert emb.shape == (N_SAMPLES, N_COMPONENTS)
    assert np.all(np.isfinite(emb))


def test_cpu_and_gpu_same_shape_but_differ():
    pytest.importorskip("cuml")
    X = _toy()
    cpu = _to_numpy(_embed("cpu", X))
    gpu = _to_numpy(_embed("gpu", X))
    assert cpu.shape == gpu.shape == (N_SAMPLES, N_COMPONENTS)
    assert np.all(np.isfinite(cpu)) and np.all(np.isfinite(gpu))
    # stochastic + different implementations -> embeddings are not identical
    assert not np.allclose(cpu, gpu)


def test_cpu_tsne_embedding_valid():
    emb = _to_numpy(_embed("cpu", _toy(), TSNE_EMBED))
    assert emb.shape == (N_SAMPLES, 2)
    assert np.all(np.isfinite(emb))


def test_gpu_tsne_embedding_valid():
    pytest.importorskip("cuml")
    emb = _to_numpy(_embed("gpu", _toy(), TSNE_EMBED))
    assert emb.shape == (N_SAMPLES, 2)
    assert np.all(np.isfinite(emb))


def test_gpu_tsne_rejects_n_components_above_two():
    pytest.importorskip("cuml")
    with pytest.raises(ValueError, match="n_components=2 only"):
        _embed("gpu", _toy(), {**TSNE_EMBED, "n_components": 3})


def test_unknown_embedding_method_raises():
    with pytest.raises(ValueError, match="unknown embedding method"):
        _embed("cpu", _toy(), {"method": "bogus", "n_components": 2})
