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
    nemi INPUT.npy --mode embed -e EMB.npz           # embed only
    nemi --mode cluster -e EMB.npz -o OUTPUT.npz     # cluster saved embeddings

A YAML config can supply any option; explicit CLI args override it. Precedence
(low → high): dataclass defaults < ``--config`` file < CLI args.

Input options
=============

* ``input`` (positional) — ``.npy`` array of shape ``(n_samples, n_features)``.
  Not required with ``--mode cluster``.
* ``-o/--output`` — path to write results (``.npz``).
  Not required with ``--mode embed``.
* ``-e/--embeddings`` — ensemble embeddings ``.npz``. Written by ``--mode
  full``/``embed``, read by ``--mode cluster``. Defaults to
  ``nemi_embeddings.npz`` in the working directory, overwritten on every run.

Modes
=====

``--mode`` selects how much of the pipeline runs, so an expensive UMAP fit can
be reused across clustering experiments:

* ``full`` (default) — embed, then cluster. Writes both files.
* ``embed`` — stop once the embeddings are written. No ``--output``.
* ``cluster`` — skip the embedding and cluster the saved ensemble. No ``input``.

The embeddings file holds ``embeddings`` ``(n, N, d)`` — one entry per ensemble
member, including when ``n == 1`` — plus ``params``, a JSON record of the
device and embedding settings that produced it.

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

Clustering: ``--clustering`` — ``agglomerative`` (default), ``dbscan``,
``hdbscan``, or ``kmeans``. Each method reads only its own params; setting
others warns:

* agglomerative — ``--n-clusters``, ``--linkage`` (``ward``/``single``),
  ``--cluster-n-neighbors``
* dbscan — ``--eps``, ``--min-samples``
* hdbscan — ``--min-cluster-size``, ``--min-samples``
* kmeans — ``--n-clusters``

Output
======

Results are written to the ``--output`` ``.npz``:

* **Single member** (``n == 1``) — ``embedding`` ``(N, d)``, ``clusters`` ``(N,)``.
* **Ensemble** (``n > 1``) — per-member ``embeddings`` ``(n, N, d)`` and
  ``member_clusters`` ``(n, N)``. With ``--assess-overlap``, also consensus
  ``clusters`` ``(N,)``, base ``embedding`` ``(N, d)``, and per-sample
  ``entropy`` ``(N,)``.

Testing
======
To test run the following command. (note this assumes you are in a gpu capable environment)

.. code-block:: bash

    python -m pytest tests/ -v -o addopts= -p no:cacheprovider