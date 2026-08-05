===================
GPU and CPU Pathway
===================

``--device`` (or ``params['device']``) selects the backend for both the
embedding and the clustering step. ``cpu`` is the default and is backwards
compatible with the original workflow.

Backends
========

.. list-table::
   :header-rows: 1

   * - Step
     - CPU (``cpu``)
     - GPU (``gpu``)
   * - Embedding
     - ``umap-learn`` UMAP
     - cuML ``UMAP`` (cupy)
   * - Clustering
     - scikit-learn
     - cuML ``cluster`` (cupy)

Selection happens in ``SingleNemi.__embedding_algo`` and ``__cluster_cpu`` /
``__cluster_gpu`` (``nemi/workflow.py``).

Key behavioural difference
==========================

* **CPU agglomerative** builds a kNN connectivity graph and computes **ward**
  distances on the embedding.
* **GPU** clusters the embedding for *every* method. cuML agglomerative supports
  **single linkage only** (no ward) and builds its own kNN connectivity, so both
  linkage and distances are on the embedding.

Because of this, ``--device gpu --clustering agglomerative --linkage ward`` is a
**hard error** — use ``--linkage single`` or ``--device cpu``. ``dbscan``,
``hdbscan`` and ``kmeans`` map directly to their cuML equivalents.

Requirements
============

See :doc:`environment <environment>`.
