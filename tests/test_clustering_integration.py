'''
Integration tests for the GPU clustering dispatch (nemi.workflow.__cluster_gpu).

Each test auto-skips when cuML is unavailable, so it meaningfully runs only on a
RAPIDS/GPU machine (the server).  GPU clusters the embedding for every method
(single-linkage agglomerative); the CPU/GPU agglomerative comparison uses a
fixed n_clusters so both must return exactly k labels.
'''
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from nemi.workflow import SingleNemi

N = 90


def _embedding():
    X, _ = make_blobs(n_samples=N, n_features=3, centers=3, random_state=0)
    return X


def _labels(device, method, **cluster_kwargs):
    nm = SingleNemi(params={"device": device,
                            "clustering_dict": {"method": method, **cluster_kwargs}})
    emb = _embedding()
    nm.X = emb
    nm.embedding = emb
    return nm.predict_clusters()


@pytest.mark.parametrize("method,kw", [
    ("agglomerative", {"linkage": "single", "n_clusters": 3, "n_neighbors": 10}),
    ("dbscan", {"eps": 1.5, "min_samples": 5}),
    ("hdbscan", {"min_cluster_size": 5, "min_samples": 5}),
])
def test_gpu_clustering_valid(method, kw):
    pytest.importorskip("cuml")
    labels = _labels("gpu", method, **kw)
    assert labels.shape == (N,)
    assert np.asarray(labels).dtype.kind in "iu"


def test_cpu_gpu_agglomerative_both_return_k():
    pytest.importorskip("cuml")
    kw = {"linkage": "single", "n_clusters": 3, "n_neighbors": 10}
    cpu = _labels("cpu", "agglomerative", **kw)
    gpu = _labels("gpu", "agglomerative", **kw)
    assert cpu.shape == gpu.shape == (N,)
    assert len(np.unique(cpu)) == 3 and len(np.unique(gpu)) == 3   # fixed k


def test_gpu_full_pipeline_runs():
    pytest.importorskip("cuml")
    X, _ = make_blobs(n_samples=200, n_features=5, centers=4, random_state=0)
    nm = SingleNemi(params={
        "device": "gpu",
        "embedding_dict": {"n_components": 3, "n_neighbors": 15, "min_dist": 0.0},
        "clustering_dict": {"method": "agglomerative", "linkage": "single",
                            "n_clusters": 5, "n_neighbors": 15},
    })
    nm.run(X)                          # UMAP (cuML) + clustering (cuML) end-to-end
    assert nm.clusters.shape == (200,)
