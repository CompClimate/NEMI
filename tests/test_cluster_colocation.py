'''
Unit tests for nemi.cluster_colocation.

assess_overlap_fast is exercised directly on hand-built label vectors, so every
consensus is deterministic -- no UMAP, clustering, or randomness involved.
Covers base-member selection, cross-member label alignment, majority voting,
noise handling, and the shapes/values of everything returned.

The final test pins assess_overlap_fast to assess_overlap_legacy (the original
mask-based implementation) over 100 randomised ensembles.
'''
import numpy as np
import pytest

from nemi.cluster_colocation import (assess_overlap_fast, assess_overlap_legacy,
                                     default_base_id)


def _labels(*rows):
    """Member label arrays; None marks a noise sample (NaN, as sort_clusters emits)."""
    return [np.array([np.nan if v is None else float(v) for v in row]) for row in rows]


# ---------------------------------------------------------------- base member

def test_default_base_id_picks_member_with_most_clusters():
    members = _labels([0, 0, 1, 1],
                      [0, 1, 2, 3],
                      [0, 0, 0, 1])
    assert default_base_id(members) == 1


def test_default_base_id_ignores_noise_when_counting():
    members = _labels([0, 1, None, None],
                      [0, 0, 0, None])
    assert default_base_id(members) == 0


def test_base_id_defaults_to_member_with_most_clusters():
    # member 1 has 3 clusters, so the consensus can express all 3
    members = _labels([0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 2, 2],
                      [0, 0, 1, 1, 1, 1])
    clusters, votes, _, _ = assess_overlap_fast(members)
    assert votes.shape[0] == 3


def test_explicit_base_id_is_honoured():
    # base 0 has only 2 clusters, so the consensus is capped at 2
    members = _labels([0, 0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 2, 2],
                      [0, 0, 1, 1, 1, 1])
    _, votes, _, _ = assess_overlap_fast(members, base_id=0)
    assert votes.shape[0] == 2


# ------------------------------------------------------------------ alignment

def test_identical_members_reproduce_the_base_labelling():
    base = [0, 0, 0, 1, 1, 1]
    members = _labels(base, base, base)
    clusters, _, _, _ = assess_overlap_fast(members, base_id=0)
    assert list(clusters) == base


def test_permuted_label_numbers_are_aligned():
    # member 2 uses swapped label *numbers* for the same two groups
    members = _labels([0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 0])
    clusters, _, _, _ = assess_overlap_fast(members, base_id=0)
    assert list(clusters) == [0, 0, 0, 1, 1, 1]


def test_fully_permuted_three_cluster_member_is_aligned():
    base = [0, 0, 1, 1, 2, 2]
    rotated = [2, 2, 0, 0, 1, 1]           # same groups, labels rotated
    members = _labels(base, base, rotated)
    clusters, votes, _, _ = assess_overlap_fast(members, base_id=0)
    assert list(clusters) == base
    # every member agrees once aligned, so each sample gets all 3 votes
    assert votes.sum(axis=0).tolist() == [3.0] * 6


def test_matching_is_one_to_one():
    # member 1 merges the base's two clusters into one; only one of them can
    # claim it, so the other base cluster must match a different compare cluster
    members = _labels([0, 0, 1, 1],
                      [0, 0, 0, 1],
                      [0, 0, 1, 1])
    _, votes, _, _ = assess_overlap_fast(members, base_id=0)
    assert votes.sum(axis=0).max() <= 3.0


# --------------------------------------------------------------------- voting

def test_majority_wins_over_dissenting_member():
    # members 0 and 1 agree; member 2 splits sample 2 off on its own
    members = _labels([0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1])
    clusters, votes, _, _ = assess_overlap_fast(members, base_id=0)
    assert list(clusters) == [0, 0, 0, 1, 1, 1]
    assert votes[0, 2] == 2.0 and votes[1, 2] == 1.0


def test_vote_counts_never_exceed_member_count():
    members = _labels([0, 0, 1, 1, 2, 2],
                      [0, 1, 1, 2, 2, 0],
                      [2, 2, 0, 0, 1, 1],
                      [0, 0, 1, 1, 2, 2])
    _, votes, n_members, _ = assess_overlap_fast(members)
    assert n_members == 4
    assert votes.sum(axis=0).max() <= n_members


# ---------------------------------------------------------------------- noise

def test_sample_all_members_call_noise_becomes_minus_one():
    members = _labels([0, 0, 1, None],
                      [0, 0, 1, None],
                      [0, 0, 1, None])
    clusters, _, _, unassigned = assess_overlap_fast(members, base_id=0)
    assert clusters[3] == -1
    assert unassigned[3] == 1.0


def test_minority_noise_does_not_win():
    # only 1 of 3 members calls sample 3 noise, so it still gets a cluster
    members = _labels([0, 0, 1, 1],
                      [0, 0, 1, 1],
                      [0, 0, 1, None])
    clusters, _, _, unassigned = assess_overlap_fast(members, base_id=0)
    assert clusters[3] == 1
    assert unassigned[3] == pytest.approx(1 / 3)


def test_unassigned_frac_is_graded():
    # sample 3: noise in 2 of 4 members
    members = _labels([0, 0, 1, 1],
                      [0, 0, 1, 1],
                      [0, 0, 1, None],
                      [0, 0, 1, None])
    _, _, n_members, unassigned = assess_overlap_fast(members, base_id=0)
    assert n_members == 4
    assert unassigned.tolist() == [0.0, 0.0, 0.0, 0.5]


def test_base_member_noise_does_not_vote():
    # the base itself calls sample 3 noise, so it casts no vote there
    members = _labels([0, 0, 1, None],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1])
    _, votes, _, unassigned = assess_overlap_fast(members, base_id=0)
    assert votes.sum(axis=0)[3] == 2.0
    assert unassigned[3] == pytest.approx(1 / 3)


def test_ties_between_cluster_and_noise_go_to_the_cluster():
    # sample 2: one member votes cluster 0, one abstains -> cluster wins the tie
    members = _labels([0, 0, 0],
                      [0, 0, None])
    clusters, _, _, _ = assess_overlap_fast(members, base_id=0)
    assert clusters[2] == 0


# -------------------------------------------------------------- shapes / args

def test_returns_expected_shapes_and_types():
    members = _labels([0, 0, 1, 1, 2, 2],
                      [0, 0, 1, 1, 2, 2],
                      [1, 1, 2, 2, 0, 0])
    clusters, votes, n_members, unassigned = assess_overlap_fast(members)
    assert clusters.shape == (6,)
    assert votes.shape == (3, 6)
    assert unassigned.shape == (6,)
    assert n_members == 3
    assert np.issubdtype(clusters.dtype, np.integer)


def test_max_clusters_limits_the_consensus_labels():
    members = _labels([0, 0, 1, 1, 2, 2],
                      [0, 0, 1, 1, 2, 2],
                      [0, 0, 1, 1, 2, 2])
    _, votes, _, _ = assess_overlap_fast(members, base_id=0, max_clusters=2)
    assert votes.shape[0] == 2
    # the dropped cluster's samples are left to the unassigned bin
    clusters, _, _, _ = assess_overlap_fast(members, base_id=0, max_clusters=2)
    assert list(clusters[4:]) == [-1, -1]


def test_single_member_returns_its_own_labelling():
    members = _labels([0, 0, 1, 1, 2])
    clusters, votes, n_members, unassigned = assess_overlap_fast(members)
    assert list(clusters) == [0, 0, 1, 1, 2]
    assert n_members == 1
    assert unassigned.tolist() == [0.0] * 5


def test_members_with_different_cluster_counts():
    # 8, 9 and 10 clusters across members, as HDBSCAN produces
    rng = np.random.default_rng(3)
    members = [rng.integers(0, k, 200).astype(float) for k in (8, 9, 10)]
    clusters, votes, n_members, unassigned = assess_overlap_fast(members)
    assert votes.shape == (10, 200)          # base is the 10-cluster member
    assert clusters.max() <= 9
    assert n_members == 3


# ------------------------------------------------------- fast vs legacy pin

def _random_ensemble(rng):
    n_samples = int(rng.integers(20, 400))
    members = []
    for _ in range(int(rng.integers(2, 8))):
        n_clusters = int(rng.integers(2, 10))
        labels = rng.integers(0, n_clusters, n_samples).astype(float)
        labels[rng.random(n_samples) < rng.uniform(0.0, 0.4)] = np.nan   # noise
        members.append(labels)
    return members


def test_fast_matches_legacy_on_random_ensembles():
    """The fast path must reproduce the original implementation exactly."""
    rng = np.random.default_rng(0)
    checked = 0
    while checked < 100:
        members = _random_ensemble(rng)
        if any(np.all(np.isnan(m)) for m in members):
            continue
        kwargs = {}
        if checked % 2:                                   # exercise both paths
            kwargs['base_id'] = int(rng.integers(0, len(members)))
        fast = assess_overlap_fast(members, **kwargs)
        legacy = assess_overlap_legacy(members, **kwargs)
        for name, got, expected in zip(
                ('clusters', 'overlap_votes', 'n_members', 'unassigned_frac'),
                fast, legacy):
            assert np.array_equal(got, expected, equal_nan=True), (
                f'{name} differs on ensemble {checked}')
        checked += 1
