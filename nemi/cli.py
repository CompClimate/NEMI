"""
Command-line interface for NEMI.

Parses CLI arguments (and an optional YAML config), validates them into a
``NemiConfig``, loads the input data, and instantiates the ``NEMI`` workflow.
Backwards compatible: the default device is CPU and the default clustering is
agglomerative/ward, matching ``nemi.workflow.default_params``.

Precedence (lowest to highest): dataclass defaults < --config file < explicit
CLI arguments.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import yaml

from nemi.workflow import NEMI

DEVICES = ("cpu", "gpu")
CLUSTERINGS = ("agglomerative", "dbscan", "hdbscan")
LINKAGES = ("ward", "single")


@dataclass
class NemiConfig:
    # I/O
    input: str | None = None
    output: str | None = None
    # backend & ensemble
    device: str = "cpu"
    n_members: int = 1
    seed: int | None = None
    scale: bool = True
    assess_overlap: bool = True
    # embedding (UMAP)
    n_components: int = 3
    embed_n_neighbors: int = 20
    min_dist: float = 0.0
    # clustering
    clustering: str = "agglomerative"
    n_clusters: int = 30
    linkage: str = "ward"
    cluster_n_neighbors: int = 40
    eps: float = 0.5
    min_samples: int = 5
    min_cluster_size: int = 50

    # Params that actually apply to each clustering method (drives the
    # irrelevant-argument warnings).
    _METHOD_PARAMS = {
        "agglomerative": {"n_clusters", "linkage", "cluster_n_neighbors"},
        "dbscan": {"eps", "min_samples"},
        "hdbscan": {"min_cluster_size", "min_samples"},
    }

    def validate(self, provided: set[str] | None = None) -> None:
        if self.input is None:
            raise ValueError("no input given (positional 'input' or config 'input')")
        if self.output is None:
            raise ValueError("no output given (--output or config 'output')")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got '{self.device}'")
        if self.clustering not in CLUSTERINGS:
            raise ValueError(f"clustering must be one of {CLUSTERINGS}, got '{self.clustering}'")
        if self.linkage not in LINKAGES:
            raise ValueError(f"linkage must be one of {LINKAGES}, got '{self.linkage}'")
        if self.n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {self.n_members}")

        # GPU agglomerative is single-linkage only (cuML has no ward). Hard error.
        if (self.device == "gpu" and self.clustering == "agglomerative"
                and self.linkage == "ward"):
            raise ValueError(
                "GPU agglomerative clustering supports single linkage only "
                "(cuML has no ward). Use --linkage single or --device cpu."
            )

        # Warn about explicitly-set params that don't apply to the chosen method.
        if provided:
            relevant = self._METHOD_PARAMS[self.clustering]
            all_params = set().union(*self._METHOD_PARAMS.values())
            for name in sorted((provided & all_params) - relevant):
                warnings.warn(f"--{name.replace('_', '-')} is ignored for "
                              f"--clustering {self.clustering}")

    def to_params(self) -> dict:
        """Build the params dict consumed by ``nemi.workflow.NEMI``."""
        clustering = {"method": self.clustering}
        if self.clustering == "agglomerative":
            clustering.update(linkage=self.linkage, n_clusters=self.n_clusters,
                              n_neighbors=self.cluster_n_neighbors)
        elif self.clustering == "dbscan":
            clustering.update(eps=self.eps, min_samples=self.min_samples)
        elif self.clustering == "hdbscan":
            clustering.update(min_cluster_size=self.min_cluster_size,
                              min_samples=self.min_samples)
        return dict(
            device=self.device,
            embedding_dict=dict(min_dist=self.min_dist,
                                n_components=self.n_components,
                                n_neighbors=self.embed_n_neighbors),
            clustering_dict=clustering,
        )


def build_parser() -> argparse.ArgumentParser:
    # SUPPRESS: unset args stay out of the namespace, so a --config value is
    # only overridden by an *explicitly passed* CLI arg.
    p = argparse.ArgumentParser(
        prog="nemi", description="Native Emergent Manifold Interrogation")
    p.add_argument("input", nargs="?", default=argparse.SUPPRESS,
                   help="input data .npy of shape (n_samples, n_features)")
    p.add_argument("-o", "--output", default=argparse.SUPPRESS,
                   help="path to write cluster labels (.npy)")
    p.add_argument("--config", default=None,
                   help="YAML config; explicit CLI args override its values")

    g = p.add_argument_group("backend & ensemble")
    g.add_argument("--device", choices=DEVICES, default=argparse.SUPPRESS)
    g.add_argument("-n", "--n-members", type=int, dest="n_members",
                   default=argparse.SUPPRESS)
    g.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    g.add_argument("--scale", dest="scale", action="store_true",
                   default=argparse.SUPPRESS)
    g.add_argument("--no-scale", dest="scale", action="store_false",
                   default=argparse.SUPPRESS)
    g.add_argument("--assess-overlap", dest="assess_overlap", action="store_true",
                   default=argparse.SUPPRESS)
    g.add_argument("--no-assess-overlap", dest="assess_overlap", action="store_false",
                   default=argparse.SUPPRESS)

    g = p.add_argument_group("embedding (UMAP)")
    g.add_argument("--n-components", type=int, dest="n_components",
                   default=argparse.SUPPRESS)
    g.add_argument("--embed-n-neighbors", type=int, dest="embed_n_neighbors",
                   default=argparse.SUPPRESS)
    g.add_argument("--min-dist", type=float, dest="min_dist",
                   default=argparse.SUPPRESS)

    g = p.add_argument_group("clustering")
    g.add_argument("--clustering", choices=CLUSTERINGS, default=argparse.SUPPRESS)
    g.add_argument("--n-clusters", type=int, dest="n_clusters",
                   default=argparse.SUPPRESS)
    g.add_argument("--linkage", choices=LINKAGES, default=argparse.SUPPRESS)
    g.add_argument("--cluster-n-neighbors", type=int, dest="cluster_n_neighbors",
                   default=argparse.SUPPRESS)
    g.add_argument("--eps", type=float, default=argparse.SUPPRESS)
    g.add_argument("--min-samples", type=int, dest="min_samples",
                   default=argparse.SUPPRESS)
    g.add_argument("--min-cluster-size", type=int, dest="min_cluster_size",
                   default=argparse.SUPPRESS)
    return p


def _merge_config(args: argparse.Namespace):
    """defaults < config file < explicit CLI.  Returns (NemiConfig, provided)."""
    cli = {k: v for k, v in vars(args).items() if k != "config"}
    file_cfg = {}
    if getattr(args, "config", None):
        with open(args.config) as fh:
            file_cfg = yaml.safe_load(fh) or {}
    merged = {**file_cfg, **cli}          # explicit CLI wins over config
    valid = set(NemiConfig.__dataclass_fields__)
    unknown = set(merged) - valid
    if unknown:
        raise ValueError(f"unknown config/arg keys: {sorted(unknown)}")
    return NemiConfig(**merged), set(merged)


def _require_gpu() -> None:
    try:
        import cuml   # noqa: F401
        import cupy   # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "device=gpu requires RAPIDS cuml + cupy (install the 'gpu' extra or "
            f"use a RAPIDS conda env); import failed: {exc}"
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg, provided = _merge_config(args)
    cfg.validate(provided)

    if cfg.device == "gpu":
        _require_gpu()

    X = np.load(cfg.input)
    nemi = NEMI(params=cfg.to_params())

    print(f"Running NEMI | device={cfg.device} clustering={cfg.clustering} "
          f"n_members={cfg.n_members} assess_overlap={cfg.assess_overlap} "
          f"X={X.shape} -> {cfg.output}")
    # TODO: cfg.scale is captured but not wired — run() does not apply
    # StandardScaler yet (see SingleNemi.scale_data).
    # TODO: cfg.seed is captured but not wired — no reproducible-ensemble seeding.
    nemi.run(X, n=cfg.n_members, assess_overlap=cfg.assess_overlap, output=cfg.output)
    return nemi


if __name__ == "__main__":
    main()
