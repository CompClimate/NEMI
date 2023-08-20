==========
Quickstart
==========

Given an array ``X`` with dimensions (``n_samples``, ``n_features``), 
these Python commands will run the NEMI workflow and bring up a plot::

    from nemi import NEMI
    nemi = NEMI()
    nemi.run(X)
    nemi.plot('clusters')