'''
Unit tests for the CPU clustering dispatch (nemi.workflow.predict_clusters /
__cluster_cpu).  A precomputed toy embedding is injected so no UMAP or network
is involved.  Covers each clustering method, the fixed-k guarantee, unknown
method handling, and the printed params readout.
'''
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from nemi.workflow import SingleNemi

N = 90


def _embedding():
    X, _ = make_blobs(n_samples=N, n_features=3, centers=3, random_state=0)
    return X


def _run_cpu(method, **cluster_kwargs):
    nm = SingleNemi(params={"device": "cpu",
                            "clustering_dict": {"method": method, **cluster_kwargs}})
    emb = _embedding()
    nm.X = emb
    nm.embedding = emb
    return nm.predict_clusters()


def test_cpu_agglomerative_ward_gives_k_clusters():
    labels = _run_cpu("agglomerative", linkage="ward", n_clusters=3, n_neighbors=10)
    assert labels.shape == (N,)
    assert len(np.unique(labels)) == 3          # fixed n_clusters


def test_cpu_agglomerative_single_gives_k_clusters():
    labels = _run_cpu("agglomerative", linkage="single", n_clusters=4, n_neighbors=10)
    assert len(np.unique(labels)) == 4


def test_cpu_dbscan_runs_with_integer_labels():
    labels = _run_cpu("dbscan", eps=1.5, min_samples=5)
    assert labels.shape == (N,)
    assert labels.dtype.kind in "iu"            # integer labels (incl. -1 noise)


def test_cpu_hdbscan_runs():
    labels = _run_cpu("hdbscan", min_cluster_size=5, min_samples=5)
    assert labels.shape == (N,)
    assert labels.dtype.kind in "iu"


def test_unknown_method_raises():
    nm = SingleNemi(params={"device": "cpu", "clustering_dict": {"method": "bogus"}})
    nm.X = nm.embedding = _embedding()
    with pytest.raises(ValueError, match="unknown clustering method"):
        nm.predict_clusters()


def test_predict_clusters_prints_params(capsys):
    _run_cpu("agglomerative", linkage="ward", n_clusters=3, n_neighbors=10)
    out = capsys.readouterr().out
    assert "Clustering | device=cpu" in out
    assert "'method': 'agglomerative'" in out
    assert "Clusters found:" in out
