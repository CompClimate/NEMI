'''
Tests for saving NEMI outputs: SingleNemi.save_outputs / run(output=...) and
NEMI.run consensus saving.  save_outputs is tested directly (no pipeline); the
run-level saves use a small CPU pipeline.
'''
import numpy as np
from sklearn.datasets import make_blobs

from nemi.workflow import NEMI, SingleNemi


def _blobs(n=60):
    X, _ = make_blobs(n_samples=n, n_features=5, centers=3, random_state=0)
    return X


def _cpu_params():
    return {"device": "cpu",
            "embedding_dict": {"n_components": 3, "n_neighbors": 15, "min_dist": 0.0},
            "clustering_dict": {"method": "agglomerative", "linkage": "ward",
                                "n_clusters": 3, "n_neighbors": 15}}


def test_save_outputs_writes_embedding_and_clusters(tmp_path):
    nm = SingleNemi()
    nm.embedding = np.arange(12.0).reshape(6, 2)
    nm.clusters = np.array([0, 0, 1, 1, 2, 2])
    out = tmp_path / "out.npz"
    nm.save_outputs(str(out))

    d = np.load(out)
    assert set(d.files) == {"embedding", "clusters"}
    assert np.array_equal(d["embedding"], nm.embedding)
    assert np.array_equal(d["clusters"], nm.clusters)


def test_singlenemi_run_saves_outputs(tmp_path):
    out = tmp_path / "single.npz"
    X = _blobs()
    SingleNemi(params=_cpu_params()).run(X, output=str(out))

    d = np.load(out)
    assert d["embedding"].shape == (X.shape[0], 3)
    assert d["clusters"].shape == (X.shape[0],)


def test_ensemble_run_saves_consensus(tmp_path):
    out = tmp_path / "ens.npz"
    X = _blobs()
    NEMI(params=_cpu_params()).run(X, n=3, output=str(out))

    d = np.load(out)
    assert set(d.files) == {"embedding", "clusters"}
    assert d["clusters"].shape == (X.shape[0],)


def test_ensemble_no_assess_writes_nothing(tmp_path, capsys):
    out = tmp_path / "skip.npz"
    NEMI(params=_cpu_params()).run(_blobs(), n=3, assess_overlap=False, output=str(out))

    assert not out.exists()
    assert "output not saved" in capsys.readouterr().out
