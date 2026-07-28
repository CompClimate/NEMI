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
    nemi = _nemi([[4.], [0.]]); nemi.n_members = 4   # all 4 members -> cluster 0
    H = nemi.entropy_map()
    assert H.shape == (1,)
    assert np.allclose(H, 0.0)


def test_entropy_one_when_evenly_split():
    nemi = _nemi([[2.], [2.]]); nemi.n_members = 4   # even 2-way split, no noise -> max
    H = nemi.entropy_map()
    assert np.allclose(H, 1.0)


def test_entropy_between_for_partial_agreement():
    nemi = _nemi([[3.], [1.]]); nemi.n_members = 4   # uneven split, no noise
    H = nemi.entropy_map()
    assert 0.0 < H[0] < 1.0


def test_entropy_noise_raises_uncertainty():
    # identical cluster votes; the noisy case has an extra member that abstained
    clustered = _nemi([[2.], [1.], [0.]]); clustered.n_members = 3
    noisy = _nemi([[2.], [1.], [0.]]); noisy.n_members = 4   # 1 unassigned vote
    assert noisy.entropy_map()[0] > clustered.entropy_map()[0]


def test_entropy_zero_for_all_noise():
    nemi = _nemi([[0.], [0.]]); nemi.n_members = 4   # every member called it noise
    H = nemi.entropy_map()                            # -> confidently unassigned
    assert np.allclose(H, 0.0)


def test_entropy_shape_and_attribute():
    nemi = _nemi([[4, 2, 3, 0],
                  [0, 2, 1, 0]])
    nemi.n_members = 4
    H = nemi.entropy_map()
    assert H.shape == (4,)
    assert nemi.entropy is H
