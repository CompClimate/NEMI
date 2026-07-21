'''
Unit tests for NEMI.entropy_map — per-sample normalized Shannon entropy of the
aligned ensemble vote distribution.  overlap_votes is set by hand (as
assess_overlap would produce it) so the entropy is fully deterministic.
'''
import numpy as np

from nemi.workflow import NEMI


def _nemi(overlap_votes):
    nemi = NEMI()
    nemi.overlap_votes = np.asarray(overlap_votes, dtype=float)  # (K, N)
    return nemi


def test_entropy_zero_when_members_agree():
    H = _nemi([[4.], [0.]]).entropy_map()      # all votes -> cluster 0
    assert H.shape == (1,)
    assert np.allclose(H, 0.0)


def test_entropy_one_when_evenly_split():
    H = _nemi([[2.], [2.]]).entropy_map()      # even 2-way split -> max entropy
    assert np.allclose(H, 1.0)


def test_entropy_between_for_partial_agreement():
    H = _nemi([[3.], [1.]]).entropy_map()
    assert 0.0 < H[0] < 1.0


def test_entropy_zero_for_no_votes():
    H = _nemi([[0.], [0.]]).entropy_map()      # noise-everywhere sample
    assert np.allclose(H, 0.0)


def test_entropy_shape_and_attribute():
    nemi = _nemi([[4, 2, 3, 0],
                  [0, 2, 1, 0]])
    H = nemi.entropy_map()
    assert H.shape == (4,)
    assert nemi.entropy is H
