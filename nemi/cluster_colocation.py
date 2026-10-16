""" Cross-member cluster co-location for a NEMI ensemble.

``assess_overlap_fast`` is the implementation the pipeline uses.
``assess_overlap_legacy`` is the original mask-based version, kept as the
reference that the fast path is tested against.  Both take the per-member label
arrays, match clusters with :func:`greedy_selection`, and return the same four
values.
"""
import numpy as np
from tqdm import tqdm

__all__ = ['assess_overlap_fast', 'assess_overlap_legacy', 'default_base_id']


def _num_clusters(labels):
    """Number of clusters in a label vector, ignoring NaN noise."""
    finite = labels[~np.isnan(labels)]
    return int(finite.max()) + 1 if finite.size else 0


def default_base_id(member_clusters):
    """ Member to use as the base: the one with the most clusters.

    Its cluster count then covers every other member, which variable-k methods
    like HDBSCAN require.
    """
    return int(np.argmax([_num_clusters(labels) for labels in member_clusters]))


def _bin_index(labels, k):
    """This just marks the noise label as k which is useful for creating our contingency table"""
    return np.where((labels >= 0) & (labels < k), labels, k).astype(np.int64)


def intersection_union_inclusion_exclusion(T, max_clusters, num_clusters) :
    """Calculates the intersection and union between two sets of clusters."""
    intersection = T[:max_clusters, :num_clusters] # Base clusters n comp clusters

    #|A| + |B|
    A = T[:max_clusters, :].sum(1)[:, None] # row count as [max clusters, 1] shape array
    B = T[:, :num_clusters].sum(0)[None, :] # column count as [1, num_clusters] shape array
    union = A + B - intersection

    return intersection, union


def contingency_table_per_ensemble(base_idx, comp_member_comp_idx, max_clusters, num_clusters):
    """Calculates a contigency table between base clusters and an ensemble member's clusters."""
    bins = base_idx + comp_member_comp_idx # This gives a unique bin for each combination of cluster overlap in the data.
    bin_cnt = np.bincount(bins, minlength=(num_clusters + 1)*(max_clusters + 1))
    T = bin_cnt.reshape(max_clusters + 1, num_clusters + 1) # this is including noise bins overlaps which we do not consider in intersection

    intersection, union = intersection_union_inclusion_exclusion(T, max_clusters, num_clusters)

    with np.errstate(invalid='ignore'): # both empty pairs are 0/0 -> NaN, as before
        contingency_table = ((intersection / union) * 100).T

    return contingency_table


def greedy_selection(contingency_table, max_clusters, num_clusters):
    """ Match each base cluster to at most one compare cluster, greedily by IoU.

    Base clusters are visited largest first (labels are sorted by size) and each
    takes the unused compare cluster it overlaps most.  Only a positive IoU is a
    match: a base cluster that shares no points with any unused compare cluster
    stays unmatched, so a compare cluster lying entirely in base noise casts no
    vote.  Equal IoUs go to the lowest compare label, so the matching does not
    depend on sort internals and is the same on every machine.

    Args:
        contingency_table (num_clusters, max_clusters): IoU (%) of each compare
            cluster (row) with each base cluster (column).  NaN (both clusters
            empty) counts as no overlap.
        max_clusters (int): number of base clusters to match.
        num_clusters (int): number of compare cluster labels (table rows).

    Returns:
        match (num_clusters + 1,): base cluster for each compare cluster, -1
            where unmatched.  The last entry is the noise bin and is always -1.
    """
    available = np.ones(contingency_table.shape[0], dtype=bool)
    match = np.full(num_clusters + 1, -1, dtype=np.int64)

    for c1 in range(max_clusters):
        iou = contingency_table[:, c1]
        candidates = np.where(available & (iou > 0), iou, 0.0)
        best = np.argmax(candidates)  # first maximum, so ties go to the lowest label
        if candidates[best] > 0:
            match[best] = c1
            available[best] = False

    return match


def assess_overlap_fast(member_clusters, base_id=None, max_clusters=None):
    """ Consensus clusters from an ensemble, by co-location matching + vote.

    Algorithm :
        1. Use bin count to find intersection O(N)
        2. Compute Union with inclusion-exclusion O(K^2)
        3. Greedy one-to-one matching on positive IoU (:func:`greedy_selection`)
        4. Majority vote across the aligned members

    Args:
        member_clusters (list): per-member label arrays of shape (N,), sorted by
            descending cluster size, with NaN for noise.
        base_id (int, optional): member used as the reference labelling.
            Defaults to :func:`default_base_id`.
        max_clusters (int, optional): number of base clusters to match.
            Defaults to all of them.

    Returns:
        clusters (N,): consensus labels, -1 where the ensemble mostly abstained.
        overlap_votes (K, N): per-sample vote count for each consensus cluster.
        n_members (int): number of ensemble members that voted.
        unassigned_frac (N,): fraction of members that cast no matched vote.
    """
    if base_id is None:
        base_id = default_base_id(member_clusters)

    compare_ids = [i for i in range(len(member_clusters))]
    compare_ids.pop(base_id)

    base_labels = member_clusters[base_id]
    num_clusters = _num_clusters(base_labels)

    if max_clusters is None:
        max_clusters = num_clusters

    base_bin = _bin_index(base_labels, max_clusters)
    base_idx = base_bin * (num_clusters + 1)

    sample_idx = np.arange(base_labels.shape[0])
    aggOverlaps = np.zeros((max_clusters, base_labels.shape[0]))

    for compare_id in tqdm(compare_ids, desc='Assessing overlap'):
        compare_idx = _bin_index(member_clusters[compare_id], num_clusters)
        contingency_table = contingency_table_per_ensemble(base_idx, compare_idx, max_clusters, num_clusters)

        match = greedy_selection (contingency_table, max_clusters, num_clusters)

        # this line permutes the compare labels to the greedily selected matching base labels
        # See how this works :
            # >>> labels
            # array([0, 1, 2, 1, 1, 2])
            # >>> match
            # array([1, 0, 2])
            # >>> match[labels]
            # array([1, 0, 2, 0, 0, 2])
        aligned = match[compare_idx]

        voted = aligned >= 0 # masks noise and unmatched compare clusters
        aggOverlaps[aligned[voted], sample_idx[voted]] += 1 #increment row cluster, column data points

     # fill in the base entry
    based = base_bin < max_clusters # remove noise
    aggOverlaps[base_bin[based], sample_idx[based]] += 1

    # majority vote, with an "unassigned" bin competing so samples the
    # ensemble mostly left as noise get -1 rather than a spurious cluster 0
    n_members = len(member_clusters)
    unassigned = n_members - aggOverlaps.sum(axis=0)        # (N,) members that abstained (noise)
    augmented = np.vstack([aggOverlaps, unassigned])        # (K+1, N)
    voteOverlaps = np.argmax(augmented, axis=0)
    voteOverlaps[voteOverlaps == aggOverlaps.shape[0]] = -1  # noise bin won -> -1

    return voteOverlaps, aggOverlaps, n_members, unassigned / n_members


def assess_overlap_legacy(member_clusters, base_id=None, max_clusters=None):
    """ Assess the overlap between the clusters.

    The original mask-based implementation, kept as the reference that
    :func:`assess_overlap_fast` is tested against.  Same arguments and returns.
    Both share :func:`greedy_selection`, so the pin covers the overlap counts
    and the vote, and the matching rule is tested on its own.
    """
    if base_id is None:
        base_id = default_base_id(member_clusters)

    # list of ensemble members we are comparing to the base
    compare_ids = [i for i in range(len(member_clusters))]
    compare_ids.pop(base_id)

    # identify clusters from the base ensemble member
    base_labels = member_clusters[base_id]

    # (NaN-safe: HDBSCAN/DBSCAN emit -1 noise -> NaN)
    num_clusters = _num_clusters(base_labels)

    if max_clusters is None:
        max_clusters = num_clusters

    sortedOverlap=np.zeros((len(compare_ids)+1, max_clusters, base_labels.shape[0]))*np.nan

    summaryStats=np.zeros((num_clusters, max_clusters))

    # compile sorted cluster data
    dataVector=[labels for id, labels in enumerate(member_clusters) if id != base_id]

    # loop over ensemble members, not including the base member
    for compare_cnt, compare_id in enumerate(compare_ids):
        # grab clusters of ensemble member
        compare_labels= dataVector[compare_cnt]

        # go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area)
        for c1 in range(max_clusters):
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            # mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1==base_labels)] = 1

            # go through each cluster
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int)

                # mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2==compare_labels)] = 1

                # Sum of flags where the two datasets of that cluster are both present
                num_overlap=np.sum(data1_M*data2_M)

                #Sum of where they overlap
                num_total=np.sum(data1_M | data2_M)

                summaryStats[c2, c1]=(num_overlap / num_total)*100 # Add percentage of coverage

            #Filled in 'summaryStatistics' matrix results of percentage overlaps

        # same matching rule as the fast path; each matched compare cluster's
        # samples then vote for its base cluster
        match = greedy_selection(summaryStats, max_clusters, num_clusters)
        for c2 in np.flatnonzero(match >= 0):
            sortedOverlap[compare_id, match[c2], compare_labels == c2] = 1

    # fill in the base entry in the sorted overlap
    for c1 in range(max_clusters):
        sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

    # majority vote, with an "unassigned" bin competing so samples the
    # ensemble mostly left as noise get -1 rather than a spurious cluster 0
    aggOverlaps = np.nansum(sortedOverlap, axis=0)          # (K, N)
    n_members = len(member_clusters)
    unassigned = n_members - aggOverlaps.sum(axis=0)        # (N,) members that abstained (noise)
    augmented = np.vstack([aggOverlaps, unassigned])        # (K+1, N)
    voteOverlaps = np.argmax(augmented, axis=0)
    voteOverlaps[voteOverlaps == aggOverlaps.shape[0]] = -1  # noise bin won -> -1

    return voteOverlaps, aggOverlaps, n_members, unassigned / n_members
