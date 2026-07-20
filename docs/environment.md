# Environment

The GPU pathway needs RAPIDS **cuML** + **cupy** and PyTorch, which are not
pip-installable from the default index. This combination of packages is kind 
of a nightmare so it is recommended you use the tools provided in this repo if 
you want to use the GPU pathway.

`build_cuml_env_nrp.sh` builds a working
conda env. This was built and tested for NRP JupyterHub. On nrp this script will
build the conda env you need and register a Jupyter kernel. It could probably be used on other
systems but may need some updating. 

The CPU pathway needs only the base `pip install nemi-learn` deps.

## GPU env
Built by `build_cuml_env_nrp.sh`. Everything (cache + temp)
stays on `/home` to survive ephemeral eviction. It creates a conda env from the
`rapidsai`/`conda-forge`/`nvidia` channels, then pip-installs the pip-only project
deps (`dbof`, NEMI editable, `umap-learn`, …).

## Pinned versions
| Package | Pin | Why |
|---|---|---|
| **cuml** | `=25.06` (keep `<=25.08`) | GPU agglomerative passes `n_neighbors`; cuML renamed it to `c` after 25.08 (see below) |
| numpy | `<2.5` | numba (via cuML) ceiling |
| scikit-learn | `=1.5.*` (`<1.6`) | cuML 25.06 sklearn-compat shim; 1.6 renamed `check_array`'s finite kwarg |
| umap-learn | `<0.5.7` | 0.5.7+ calls sklearn 1.6's `ensure_all_finite` |

The pip `gpu` extra (`pip install nemi-learn[gpu]`) carries the satellite caps
(`cupy-cuda12x`, `scikit-learn<1.6`, `umap-learn<0.5.7`, `numpy<2.5`); cuML itself
comes from conda.

You should not need to use the toml file after installing via the bash script. 

## The cuML agglomerative `n_neighbors` → `c` change
`SingleNemi.__cluster_gpu` calls cuML `AgglomerativeClustering` with `n_neighbors`.
The keyword is version-specific:

| cuML | KNN-graph param |
|---|---|
| ≤ 25.08 | `n_neighbors` (default 10, range 2–1023) |
| ≥ 25.10 | `c` (default 15); `n_neighbors = log(n_samples) + c` — `n_neighbors` removed |

The constructor is keyword-only with no `**kwargs`, so on cuML ≥ 25.10 the current
call raises `TypeError: unexpected keyword argument 'n_neighbors'`. Keep cuML
pinned to `<=25.08`, or update `__cluster_gpu` to pass `c` before bumping. See
[GPU and CPU pathway](gpu_and_cpu_pathway.md).
