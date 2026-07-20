============
Running NEMI
============

Runs the NEMI workflow (UMAP embedding → clustering → optional ensemble
consensus) from the command line on a saved feature array.

Entry point: CLI ``nemi`` (``nemi.cli:main``). Orchestration lives in
``nemi.workflow.NEMI``.

CLI usage
=========

.. code-block:: bash

    nemi INPUT.npy -o OUTPUT.npz [options]
    nemi INPUT.npy -o OUTPUT.npz --config run.yaml   # YAML config

A YAML config can supply any option; explicit CLI args override it. Precedence
(low → high): dataclass defaults < ``--config`` file < CLI args.

Input options
=============

* ``input`` (positional) — ``.npy`` array of shape ``(n_samples, n_features)``.
* ``-o/--output`` — path to write results (``.npz``).

Backend & ensemble:

* ``--device`` — ``cpu`` (default) or ``gpu``. See
  :doc:`GPU and CPU pathway <gpu_and_cpu_pathway>`.
* ``-n/--n-members`` — ensemble members (default 1).
* ``--assess-overlap`` / ``--no-assess-overlap`` — run the cross-member
  co-location majority vote for a consensus labelling (default on; ignored when
  ``n == 1``).
* ``--scale`` / ``--no-scale``, ``--seed`` — captured but not yet wired (see
  TODOs in ``cli.py``).

Embedding (UMAP): ``--n-components``, ``--embed-n-neighbors``, ``--min-dist``.

Clustering: ``--clustering`` — ``agglomerative`` (default), ``dbscan``, or
``hdbscan``. Each method reads only its own params; setting others warns:

* agglomerative — ``--n-clusters``, ``--linkage`` (``ward``/``single``),
  ``--cluster-n-neighbors``
* dbscan — ``--eps``, ``--min-samples``
* hdbscan — ``--min-cluster-size``, ``--min-samples``

Output
======

Results are written to the ``--output`` ``.npz``:

* **Single member** (``n == 1``) — ``embedding`` ``(N, d)``, ``clusters`` ``(N,)``.
* **Ensemble** (``n > 1``) — per-member ``embeddings`` ``(n, N, d)`` and
  ``member_clusters`` ``(n, N)``. With ``--assess-overlap``, also consensus
  ``clusters`` ``(N,)``, base ``embedding`` ``(N, d)``, and per-sample
  ``entropy`` ``(N,)``.
