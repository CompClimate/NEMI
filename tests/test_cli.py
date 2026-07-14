'''
Unit tests for the NEMI CLI argument management (nemi.cli).

Offline only: parsing, config merge/precedence, validation (incl. the
GPU + ward hard error), and main() up to NEMI instantiation.  No GPU and no
real clustering run.
'''
import numpy as np
import pytest
import yaml
from sklearn.datasets import make_blobs

from nemi import cli
from nemi.cli import NemiConfig


# --- NemiConfig.to_params --------------------------------------------------

def test_to_params_agglomerative_default():
    params = NemiConfig(input="x.npy", output="o.npy").to_params()
    assert params["device"] == "cpu"
    assert params["embedding_dict"] == {"min_dist": 0.0, "n_components": 3, "n_neighbors": 20}
    assert params["clustering_dict"] == {
        "method": "agglomerative", "linkage": "ward", "n_clusters": 30, "n_neighbors": 40}


def test_to_params_dbscan_only_has_its_keys():
    params = NemiConfig(input="x", output="o", clustering="dbscan",
                        eps=0.2, min_samples=7).to_params()
    assert params["clustering_dict"] == {"method": "dbscan", "eps": 0.2, "min_samples": 7}


def test_to_params_hdbscan_only_has_its_keys():
    params = NemiConfig(input="x", output="o", clustering="hdbscan",
                        min_cluster_size=25, min_samples=5).to_params()
    assert params["clustering_dict"] == {
        "method": "hdbscan", "min_cluster_size": 25, "min_samples": 5}


# --- validate --------------------------------------------------------------

def test_gpu_agglomerative_ward_is_hard_error():
    cfg = NemiConfig(input="x", output="o", device="gpu",
                     clustering="agglomerative", linkage="ward")
    with pytest.raises(ValueError, match="single linkage only"):
        cfg.validate()


def test_gpu_agglomerative_single_ok():
    NemiConfig(input="x", output="o", device="gpu",
               clustering="agglomerative", linkage="single").validate()


def test_gpu_dbscan_ok():
    NemiConfig(input="x", output="o", device="gpu", clustering="dbscan").validate()


def test_missing_input_raises():
    with pytest.raises(ValueError, match="no input"):
        NemiConfig(output="o").validate()


def test_missing_output_raises():
    with pytest.raises(ValueError, match="no output"):
        NemiConfig(input="x").validate()


def test_bad_n_members_raises():
    with pytest.raises(ValueError, match="n_members"):
        NemiConfig(input="x", output="o", n_members=0).validate()


def test_irrelevant_param_warns():
    cfg = NemiConfig(input="x", output="o", clustering="dbscan")
    with pytest.warns(UserWarning, match="linkage"):
        cfg.validate(provided={"linkage", "eps"})


# --- parser + config merge -------------------------------------------------

def _cfg(argv):
    args = cli.build_parser().parse_args(argv)
    return cli._merge_config(args)


def test_defaults_from_minimal_args():
    cfg, provided = _cfg(["data.npy", "-o", "out.npy"])
    assert cfg.input == "data.npy" and cfg.output == "out.npy"
    assert cfg.device == "cpu" and cfg.clustering == "agglomerative"
    assert cfg.linkage == "ward" and cfg.n_clusters == 30
    assert provided == {"input", "output"}


def test_cli_overrides():
    cfg, _ = _cfg(["d.npy", "-o", "o.npy", "--device", "gpu",
                   "--clustering", "hdbscan", "--min-cluster-size", "12", "-n", "5"])
    assert cfg.device == "gpu" and cfg.clustering == "hdbscan"
    assert cfg.min_cluster_size == 12 and cfg.n_members == 5


def test_config_file_values(tmp_path):
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(yaml.safe_dump(
        {"input": "d.npy", "output": "o.npy", "clustering": "dbscan", "eps": 0.9}))
    cfg, _ = _cfg(["--config", str(cfgfile)])
    assert cfg.clustering == "dbscan" and cfg.eps == 0.9


def test_cli_overrides_config(tmp_path):
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(yaml.safe_dump({"input": "d.npy", "output": "o.npy", "device": "cpu"}))
    cfg, _ = _cfg(["--config", str(cfgfile), "--device", "gpu", "--clustering", "dbscan"])
    assert cfg.device == "gpu"          # CLI wins over config


def test_assess_overlap_defaults_true():
    cfg, _ = _cfg(["d.npy", "-o", "o.npy"])
    assert cfg.assess_overlap is True


def test_no_assess_overlap_flag():
    cfg, _ = _cfg(["d.npy", "-o", "o.npy", "--no-assess-overlap"])
    assert cfg.assess_overlap is False


def test_unknown_config_key_raises(tmp_path):
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(yaml.safe_dump({"input": "d", "output": "o", "bogus": 1}))
    with pytest.raises(ValueError, match="unknown config/arg keys"):
        _cfg(["--config", str(cfgfile)])


# --- main(): up to NEMI instantiation -------------------------------------

def test_main_cpu_runs_and_saves(tmp_path):
    data = tmp_path / "d.npy"
    np.save(data, make_blobs(n_samples=60, n_features=4, centers=3, random_state=0)[0])
    out = tmp_path / "o.npz"
    nemi = cli.main([str(data), "-o", str(out),
                     "--embed-n-neighbors", "10", "--n-clusters", "3",
                     "--cluster-n-neighbors", "10"])
    assert nemi.clusters.shape == (60,)
    assert out.exists()


def test_main_gpu_ward_raises_before_gpu_check(tmp_path):
    data = tmp_path / "d.npy"
    np.save(data, np.random.random((20, 4)))
    with pytest.raises(ValueError, match="single linkage only"):
        cli.main([str(data), "-o", str(tmp_path / "o.npy"), "--device", "gpu"])


def test_main_gpu_without_cuml_exits(tmp_path, monkeypatch):
    data = tmp_path / "d.npy"
    np.save(data, np.random.random((20, 4)))

    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name in ("cuml", "cupy"):
            raise ImportError("no gpu here")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit):
        # dbscan avoids the ward hard-error so we reach the gpu import guard
        cli.main([str(data), "-o", str(tmp_path / "o.npy"),
                  "--device", "gpu", "--clustering", "dbscan"])
