'''
Integration tests for the ensemble path: NEMI.run(X, n>1) -> assess_overlap.

Runs the real embedding + clustering + co-location end to end.  CPU tests run
anywhere; the GPU test auto-skips without cuML (runs on the server) and verifies
the cuML embedding/clustering -> host-NumPy -> assess_overlap handoff.
'''
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from nemi.workflow import NEMI

N = 150


def _blobs():
    X, _ = make_blobs(n_samples=N, n_features=5, centers=4, random_state=0)
    return X


def _params(device):
    return {
        "device": device,
        "embedding_dict": {"n_components": 3, "n_neighbors": 15, "min_dist": 0.0},
        "clustering_dict": {"method": "agglomerative",
                            "linkage": "single" if device == "gpu" else "ward",
                            "n_clusters": 4, "n_neighbors": 15},
    }


def test_cpu_ensemble_produces_consensus(embeddings_path):
    nemi = NEMI(params=_params("cpu"))
    nemi.run(_blobs(), n=3, embeddings=embeddings_path)
    assert nemi.base_id == 0
    assert nemi.clusters.shape == (N,)


def test_assess_overlap_false_keeps_pack_only(embeddings_path):
    nemi = NEMI(params=_params("cpu"))
    nemi.run(_blobs(), n=3, assess_overlap=False, embeddings=embeddings_path)
    assert len(nemi.nemi_pack) == 3
    assert not hasattr(nemi, "clusters")             # consensus intentionally skipped


def test_gpu_ensemble_produces_consensus(embeddings_path):
    pytest.importorskip("cuml")
    nemi = NEMI(params=_params("gpu"))
    nemi.run(_blobs(), n=3, embeddings=embeddings_path)
    assert nemi.clusters.shape == (N,)
