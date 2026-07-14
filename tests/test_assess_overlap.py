'''
Unit tests for NEMI.assess_overlap (the ensemble co-location + majority vote).

The nemi_pack is built by hand with fixed per-member label vectors, so the
consensus is fully deterministic — no UMAP, clustering, or randomness involved.
Covers reproduction of identical members, cross-member label alignment,
majority-vote tie-breaking, and the attributes it sets.
'''
import numpy as np

from nemi.workflow import NEMI, SingleNemi


def _nemi(member_labels):
    nemi = NEMI()
    pack = []
    for labels in member_labels:
        m = SingleNemi()
        m.clusters = np.asarray(labels)
        m.embedding = np.zeros((len(labels), 2))   # base member's is exposed as nemi.embedding
        pack.append(m)
    nemi.nemi_pack = pack
    return nemi


def test_identical_members_reproduce_labels():
    base = [0, 0, 0, 1, 1, 1]
    nemi = _nemi([base, base, base])
    nemi.assess_overlap(base_id=0)
    assert list(nemi.clusters) == base


def test_permuted_labels_are_aligned():
    # member 2 uses swapped label *numbers* for the same two groups;
    # co-location must match them back to the base labelling.
    nemi = _nemi([[0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0, 0]])
    nemi.assess_overlap(base_id=0)
    assert list(nemi.clusters) == [0, 0, 0, 1, 1, 1]


def test_majority_vote_resolves_disagreement():
    # sample index 2 is cluster 0 in members 0 & 1 but cluster 1 in member 2
    nemi = _nemi([[0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1]])
    nemi.assess_overlap(base_id=0)
    assert nemi.clusters[2] == 0                     # 2 of 3 members win
    assert list(nemi.clusters) == [0, 0, 0, 1, 1, 1]


def test_sets_expected_attributes():
    nemi = _nemi([[0, 0, 1, 1], [0, 0, 1, 1]])
    nemi.assess_overlap(base_id=0)
    assert nemi.base_id == 0
    assert nemi.clusters.shape == (4,)
    assert nemi.embedding is nemi.nemi_pack[0].embedding


def test_base_id_selects_reference_member():
    nemi = _nemi([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    nemi.assess_overlap(base_id=1)
    assert nemi.base_id == 1
    assert nemi.embedding is nemi.nemi_pack[1].embedding
