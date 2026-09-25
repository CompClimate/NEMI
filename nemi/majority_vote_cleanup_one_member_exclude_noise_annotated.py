#!/usr/bin/env python3
"""

Notes: 
- This code smooths the results after DBSCAN. (In my case, after two rounds of DBSCAN.) 
- DBSCAN noise labels are assigned a -1. 
- Majority vote smoothing: for each grid cell look at 8 surrounding grid cells (for r=1)
    - 3 by 3 squares for r=1, 5 by 5 for r=2 
    - Change center grid cell based on majority of (8) surrounding labels, not including noise labels 
- Repeat for each grid cell for each month (in my case 12 months) 
- Noise can be relabeled using non-noise neighbors 

-----------------------------------------------------------------------
DATA SHAPES 

  1. "Filtered 1-D stack" : A flat one dimensional array that has one entry 
      per grid cell that resulted from the previous step of removing high latitude steaks 
      in embedding. 

  2. "Full valid-ocean 1-D stack": A flat one dimensional array with one entry for 
      each grid cell that has non-nan values in the PFT input data. (So this 
      is the shape of the data before filtering out the high latitude points 
      in the embedding.) Used to plot the grid for the 12 months, including the 
      high latitude regions that were filtered. 

  3. "2-D month grid": A regular (lat,lon) array for a single month for the smoother 
      to see who the neighbors are. 
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr


# Root folder 
BASE = Path("/quobyte/maikesgrp/makayla/12_month_embedding")

# DBSCAN second round output 
RUN = BASE / "noise_recluster_dbscan" / "ensemble50_eps0.075_ms40"

# kept indicies records indicies for filtered cells before filtering
FILTERED_ROOT = BASE / "embeddings" / "without_high_latitude_streaks"
KEPT_INDICES_FILE = FILTERED_ROOT / "ensemble10" / "kept_indices.npy"

# Input PFT files to find valid non-nan cells 
PFT_FILES = [
    Path("/group/maikesgrp/makayla/Cocco_monthly_clim.nc"),
    Path("/group/maikesgrp/makayla/Diatom_monthly_clim.nc"),
    Path("/group/maikesgrp/makayla/Diazo_monthly_clim.nc"),
    Path("/group/maikesgrp/makayla/MixDino_monthly_clim_skipna.nc"),
    Path("/group/maikesgrp/makayla/PicoEuk_monthly_clim.nc"),
    Path("/group/maikesgrp/makayla/PicoPro_monthly_clim.nc"),
]

#Marks grid cells that were filtered out 
FILL_VALUE = -999999

# DBSCAN noise 
NOISE_LABEL = -1


def parse_args():
    """
    Define command-line options 
    """
    p = argparse.ArgumentParser()

    # Which ensemble member to smooth
    p.add_argument("--ensemble", type=int, default=1)

    p.add_argument(
        "--final-file",
        type=Path,
        default=None,
        help=(
            "Override for the input final_combined_labels.npy path. "
            "Defaults to ensembleXX/final_combined/final_combined_labels.npy."
        ),
    )

    p.add_argument(
        "--radius",
        type=int,
        default=1,
        help=(
            "Radius of the voting window. radius=1 means the "
            "8 surrounding cells (a 3x3 window); radius=2 means a 5x5 "
            "window, etc."
        ),
    )
    
    p.add_argument(
        "--include-center",
        action="store_true",
        help=(
            "Include the cell's own current label as one of its own "
            "votes. Default: only neighbors vote. Note: if the cell's "
            "current label is noise, this self-vote is still excluded, "
            "same as any other noise-labeled voter."
        ),
    )
    p.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help=(
            "Minimum number of valid non noise neighbor votes required "
            "to relabel a cell."
        ),
    )

    p.add_argument(
        "--require-strict-majority",
        action="store_true",
        help=(
            "Require the winning label to hold strictly more than half "
            "of the valid NON-NOISE neighbor votes."
        ),
    )
    # Without this flag, the most common label among neighbors wins even
    # if it is, say, only 3 votes out of 8 (a plurality). With this flag,
    # that same 3-out-of-8 result would NOT be enough to relabel the
    # cell, because 3 is not strictly more than half of 8.
    
    p.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of majority-vote passes to apply per month.",
    )
    # Note smoothing can be repeated for multiple iterations 
    # but I only tried one iteration for each radius (r=1, r=2) 

    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    
    return p.parse_args()


def first_data_array(path: Path) -> xr.DataArray:
    """
    Opens a NetCDF climatology file to get the gridded PFT values 
    dimensions: (month, lat, lon) 
    """
    ds = xr.open_dataset(path)

    if not ds.data_vars:
        raise ValueError(f"No data variables in {path}")

    for name in ds.data_vars:
        da = ds[name]
        if da.ndim >= 3:
            return da

    return ds[next(iter(ds.data_vars))]


def normalize_month_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """
    Accounts for NetCDF files that have dimensions in different orders
    or have different dimension names so build_valid_masks() is consistent. 
    """

    dims = list(da.dims)

    month_dim = next((d for d in dims if d.lower() in {"month", "time"}), None)
    lat_dim = next((d for d in dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in dims if "lon" in d.lower()), None)

    if month_dim is None or lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not identify month/lat/lon dimensions in {da.dims}")

    da = da.transpose(month_dim, lat_dim, lon_dim)

    if da.shape[0] != 12:
        raise ValueError(f"Expected 12 monthly slices, got shape {da.shape}")

    return da


def build_valid_masks():
    """
    For each of the 12 months finds which grid cells are non-nan in PFT input.
    Needs to match order of the filtered one dimensional label. 

    Returns
    -------
    valid : bool array, shape (12, nlat, nlon)
        True where a cell has real data in every PFT file, that month.
    lat, lon : 1-D coordinate arrays
        The latitude and longitude values for the grid, taken from the
        first PFT file (all files share the same grid).
    """
    arrays = [normalize_month_lat_lon(first_data_array(path)) for path in PFT_FILES]

    shape = arrays[0].shape
    for da in arrays[1:]:
        if da.shape != shape:
            raise ValueError(f"PFT grid mismatch: expected {shape}, got {da.shape}")

    # Start by assuming every cell is valid, then AND in each PFT's own
    # finite-data mask. A cell only survives if ALL SIX files agree it
    # has real data there.
    valid = np.ones(shape, dtype=bool)
    for da in arrays:
        valid &= np.isfinite(np.asarray(da.values))

    lat = arrays[0][arrays[0].dims[1]].values
    lon = arrays[0][arrays[0].dims[2]].values

    return valid, lat, lon


def neighbor_offsets(radius: int, include_center: bool):
    """
    Build the list of (row_offset, col_offset) pairs that define "who
    counts as a neighbor" for a given radius.

    For radius=1 this produces 8 neighbors: one row up,
    one row down, or the same row, combined with one column left, one
    column right, or the same column, minus the (0, 0) pair (the cell
    itself) unless include_center is True. For radius=2 it produces a
    5x5 block of offsets (24 neighbors, or 25 with the center), and so
    on for larger radii.
    """
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0 and not include_center:
                continue
            offsets.append((dr, dc))
    return offsets


def gather_neighbor_stack(grid: np.ndarray, vote_mask: np.ndarray, offsets):
    """
    For a single month's 2-D grid, build a stack of "what does each of
    my neighbors look like" arrays, one layer per offset in `offsets`.
    Recall in the r=1 case we are looking at a 5x5 block of offsets. 

    Returns two arrays of shape (K, nlat, nlon), where
    K is the number of neighbor offsets:

        stack_val[k, r, c]   = the label found at offset k relative to
                                cell (r, c) (whatever value happens to
                                be there, even if it is not a valid
                                voter)
        stack_valid[k, r, c] = whether that neighbor is actually allowed
                                to vote (i.e. it exists, is inside the
                                grid/pole edges, and passed `vote_mask`,
                                which already has noise-labeled cells excluded)

    Longitude ("columns") is periodic: the ocean wraps around the globe,
    so the cell just east of the last column is the first column again.
     `np.roll` shifts an array around a circular axis.

    Latitude ("rows") is NOT periodic: the poles are not connected to
    each other. (no such neighbor exists for
    cells right at the top or bottom edge of the grid).
    """
    nlat, nlon = grid.shape
    K = len(offsets)

    stack_val = np.zeros((K, nlat, nlon), dtype=grid.dtype)
    stack_valid = np.zeros((K, nlat, nlon), dtype=bool)

    for k, (dr, dc) in enumerate(offsets):
        # Shift the whole grid horizontally by -dc columns. Longitude
        # wraps, np.roll: a column that rolls off one edge reappears on
        # the other
        col_val = np.roll(grid, shift=-dc, axis=1)
        col_valid = np.roll(vote_mask, shift=-dc, axis=1)

        if dr == 0:
            # Same row, just the longitude shift from above.
            stack_val[k] = col_val
            stack_valid[k] = col_valid
        elif dr > 0:
            # Looking "down" (toward higher row indices) by dr rows.
            # Rows near the bottom edge have no such neighbor, so they
            # are simply left as invalid (their initial np.zeros/False
            # values from stack_val/stack_valid above).
            stack_val[k, : nlat - dr, :] = col_val[dr:, :]
            stack_valid[k, : nlat - dr, :] = col_valid[dr:, :]
            # Remaining bottom rows stay invalid (no such neighbor).
        else:
            # Looking "up" (toward lower row indices) by |dr| rows.
            # Same idea, mirrored: rows near the top edge have no such
            # neighbor.
            d = -dr
            stack_val[k, d:, :] = col_val[: nlat - d, :]
            stack_valid[k, d:, :] = col_valid[: nlat - d, :]
            # Remaining top rows stay invalid (no such neighbor).

    return stack_val, stack_valid


def majority_vote_pass(
    labels2d: np.ndarray,
    valid_mask: np.ndarray,
    offsets,
    min_votes: int,
    require_strict_majority: bool,
):
    """
    Run ONE neighborhood majority vote pass over a month's
    2D grid: every prefiltered cell (non-steak and non-nan) cell
    looks at its neighbors and turns into whichever label its neighbors
    agree on most (exluding noise votes). 

    Returns
    -------
    new_labels : the grid after this one pass of relabeling
    changed_mask : bool grid, True wherever a cell's label actually
        changed during this pass
    """
    # "exclude noise from voting"
    # a cell can vote only if it is inside the retained domain AND its
    # current label is not noise.
    vote_mask = valid_mask & (labels2d != NOISE_LABEL)

    # For every neighbor offset, gather what label each neighbor has
    # and whether that neighbor is allowed to vote at all.
    stack_val, stack_valid = gather_neighbor_stack(labels2d, vote_mask, offsets)
    K = stack_val.shape[0]

    # Count the votes -------------------------------------------------
    # for each neighbor position k, count how many OTHER
    # eligible neighbors share that exact same label. Doing this for
    # every k gives us, for every cell, a per-neighbor vote count that
    # we can then take the max of to find the winning label. This is a
    # fully vectorized way of doing "count votes per candidate label"
    # without knowing ahead of time how many distinct labels exist.
    counts = np.zeros((K,) + labels2d.shape, dtype=np.int16)
    for i in range(K):
        # eq[k, r, c] is True if neighbor k (at cell r,c's neighborhood)
        # has the same label as neighbor i, AND neighbor k is a valid
        # (non-noise, in-domain, on-grid) voter.
        eq = (stack_val == stack_val[i]) & stack_valid
        counts[i] = eq.sum(axis=0)

    # Neighbors that are not valid voters should never be picked as the
    # "winning" candidate, so force their vote counts down to -1
    # (lower than any real count, which is always >= 0).
    counts_masked = np.where(stack_valid, counts, -1)

    # For each cell, the highest vote count found among all candidate
    # neighbor labels.
    maxcount = counts_masked.max(axis=0)

    # Which neighbor slots achieved that highest count (there can be
    # more than one, if two different labels are tied for most common).
    is_max = (counts_masked == maxcount[None, :, :]) & stack_valid

    # Pick the first tied slot as a possible "winning label," 
    # so we have a single  label to compare the others against
    # in the tie check  below. If there's no tie, this the
    # winning label.
    first_idx = np.argmax(is_max, axis=0)
    first_label = np.take_along_axis(stack_val, first_idx[None, :, :], axis=0)[0]

    # A tie happends if some other slot also achieved the max count
    # but has a different label than the one identified as a possible label.
  
    mismatch = is_max & (stack_val != first_label[None, :, :])
    has_tie = mismatch.any(axis=0)

    has_valid_neighbor = stack_valid.any(axis=0)
    total_valid = stack_valid.sum(axis=0)

    # A cell is eligible to be relabeled if:
    #   - it has at least one valid (non-noise) neighbor at all, and
    #   - there is no tie for the most popular label, and
    #   - the winning label's vote count meets --min-votes.
    eligible = has_valid_neighbor & ~has_tie & (maxcount >= min_votes)

    if require_strict_majority:
        # "Strictly more than half": maxcount*2 > total_valid 
        # i.e. maxcount > total_valid / 2 without any rounding issues
        eligible &= (maxcount * 2) > total_valid

    # Important distinction: `eligible` up to this point was computed
    # from `vote_mask` (which excludes noise), but the cell being
    # examined does not itself need to be non-noise to receive a new
    # label. A noise cell is allowed to be relabeled by its
    # non-noise neighbors. So the final
    # eligibility check uses the full retained domain, `valid_mask`,
    # not the noise-excluding `vote_mask`.
    eligible &= valid_mask

    new_labels = np.where(eligible, first_label, labels2d)
    changed = eligible & (new_labels != labels2d)

    # --- Sanity checks -----------------------------------------------
    # Make sure to not included high lat streaks that were filtered out. 
    
    if np.any(new_labels[~valid_mask] != labels2d[~valid_mask]):
        raise RuntimeError("A cell outside the retained cleanup domain changed.")
    if np.any(changed & ~valid_mask):
        raise RuntimeError("changed_mask contains cells outside the retained domain.")

    return new_labels, changed


def smooth_month(
    labels2d: np.ndarray,
    valid_mask: np.ndarray,
    radius: int,
    include_center: bool,
    min_votes: int,
    require_strict_majority: bool,
    iterations: int,
):
    """
    Run majority_vote_pass() over one month's grid. (Can run for more 
    than one iteration, but I did not.) 

    Because `vote_mask` inside majority_vote_pass() is recomputed from
    whatever the CURRENT labels are on every pass (not the original
    labels), a cell that was noise and got filled in during pass 1
    becomes eligible to cast a vote of its own starting in pass 2.

    Returns
    -------
    current : the grid after all passes
    total_changed : Checks whether final result is different from start 
    per_iteration_changed : list of how many cells changed on each
        individual pass
    """
    offsets = neighbor_offsets(radius, include_center)

    current = labels2d.copy()
    ever_changed = np.zeros(labels2d.shape, dtype=bool)
    per_iteration_changed = []

    for _ in range(iterations):
        current, changed = majority_vote_pass(
            labels2d=current,
            valid_mask=valid_mask,
            offsets=offsets,
            min_votes=min_votes,
            require_strict_majority=require_strict_majority,
        )
        ever_changed |= changed
        per_iteration_changed.append(int(changed.sum()))

        if not np.any(changed):
            # If nothing changes in 1 pass stop process 
            break
    # Compare final labels to original 
    total_changed = valid_mask & (current != labels2d)

    return current, total_changed, per_iteration_changed


def main():
    args = parse_args()
    tag = f"{args.ensemble:02d}"  # e.g. 1 -> "01", 12 -> "12"

    # Where to read this ensemble member's DBSCAN
    # cluster labels from 
    
    final_file = (
        args.final_file
        if args.final_file is not None
        else RUN / f"ensemble{tag}" / "final_combined" / "final_combined_labels.npy"
    )

    if not final_file.exists():
        raise FileNotFoundError(final_file)
    if not KEPT_INDICES_FILE.exists():
        raise FileNotFoundError(KEPT_INDICES_FILE)

    # Everything this run produces goes into a folder specific
    # to this ensemble member and this cleanup variant
    
    out_root = (
        args.output_root
        if args.output_root is not None
        else RUN / f"ensemble{tag}" / "final_combined_majority_vote_smoothed_exclude_noise"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"MAJORITY VOTE SMOOTHING (NOISE EXCLUDED FROM VOTE) ENSEMBLE {args.ensemble}")
    print("=" * 72)
    print("radius:", args.radius, " window:", f"{2*args.radius+1}x{2*args.radius+1}")
    print("include_center:", args.include_center)
    print("min_votes:", args.min_votes, "(non-noise votes only)")
    print("require_strict_majority:", args.require_strict_majority)
    print("iterations:", args.iterations)
    print("noise_votes: EXCLUDED (cells currently labeled noise never cast a vote)")
    print("output:", out_root)
    print()

    # "filtered 1-D stack" 
    # one cluster label per filtered ocean cell
    final_labels = np.load(final_file)
    print("Filtered points:", f"{len(final_labels):,}")

    # Rebuilds from the raw climatology files which grid cells
    # count as valid ocean for each of the 12 months 
    print("Building 12 monthly valid-ocean masks...")
    valid_monthly, lat, lon = build_valid_masks()
    n_full_valid = int(valid_monthly.sum())
    print("Full valid stack:", f"{n_full_valid:,}")

    # kept_indices: for every entry in the filtered 1-D array
    # which position that value occupies in the full
    # valid-ocean 1-D stack. 
    
    kept_indices = np.load(KEPT_INDICES_FILE).astype(np.int64)

    if kept_indices.size != final_labels.size:
        raise ValueError(
            f"kept_indices length {kept_indices.size:,} != "
            f"filtered labels {final_labels.size:,}"
        )
    if kept_indices.min() < 0 or kept_indices.max() >= n_full_valid:
        raise ValueError("kept_indices falls outside reconstructed full valid-ocean stack.")

    # Build a full valid-sized array, fill it with the
    # "not part of the retained domain" flag (yellow high-lat steak points) 
    # then drop filtered labels into their correct positions using kept_indices.
    # Cells that were valid ocean but got filtered out earlier ( the
    # high-latitude streak cells) are left as FILL_VALUE and marked
    # False in full_valid_kept below, so the smoothing step will never
    # change them. 
    full_valid_labels = np.full(n_full_valid, FILL_VALUE, dtype=final_labels.dtype)
    full_valid_labels[kept_indices] = final_labels

    full_valid_kept = np.zeros(n_full_valid, dtype=bool)
    full_valid_kept[kept_indices] = True

    smoothed_full_valid = full_valid_labels.copy()
    changed_full_valid = np.zeros(n_full_valid, dtype=bool)

    month_records = []
    stack_offset = 0  # Goes through the full-valid 1-D stack, 12 months at a time

    for month in range(12):
        valid2d = valid_monthly[month]
        nvalid = int(valid2d.sum())

        # Slice out just this month's chunk of the full-valid 1-D stack.
        # Because build_valid_masks() and the earlier pipeline step that
        # produced kept_indices both use the same "month 0 cells, then
        # month 1 cells, ..." ordering, this slice lines up correctly.
        month_values = full_valid_labels[stack_offset : stack_offset + nvalid]
        month_kept = full_valid_kept[stack_offset : stack_offset + nvalid]

        # Scatter this month's 1-D values back onto a proper 2-D
        # (lat, lon) grid, using valid2d ( "is this cell
        # real ocean" mask) to know where each 1-D value goes.
        grid = np.full(valid2d.shape, FILL_VALUE, dtype=final_labels.dtype)
        grid[valid2d] = month_values

        kept_grid = np.zeros(valid2d.shape, dtype=bool)
        kept_grid[valid2d] = month_kept

        # cleanup step: smooth this one month's 2-D
        # grid using neighborhood majority voting excluding noise.
        smoothed_grid, changed_grid, per_iter = smooth_month(
            labels2d=grid,
            valid_mask=kept_grid,
            radius=args.radius,
            include_center=args.include_center,
            min_votes=args.min_votes,
            require_strict_majority=args.require_strict_majority,
            iterations=args.iterations,
        )

        if np.any(changed_grid & ~kept_grid):
            raise RuntimeError(f"Month {month + 1:02d}: change escaped retained domain.")

        # Fold this month's smoothed 2-D grid back down into the
        # full-valid 1-D stack, at the same offset it came from.
        smoothed_full_valid[stack_offset : stack_offset + nvalid] = smoothed_grid[valid2d]
        changed_full_valid[stack_offset : stack_offset + nvalid] = changed_grid[valid2d]

        month_records.append(
            {
                "month": month + 1,
                "n_changed": int(changed_grid.sum()),
                "per_iteration_changed": per_iter,
            }
        )

        print(
            f"Month {month + 1:02d}: "
            f"changed={int(changed_grid.sum()):,}, "
            f"iterations_run={len(per_iter)}, "
            f"per_iteration={per_iter}"
        )

        stack_offset += nvalid

    if stack_offset != n_full_valid:
        raise RuntimeError(f"Monthly reconstruction consumed {stack_offset:,}, expected {n_full_valid:,}.")

    # Now go the other direction: pull the smoothed
    # full-valid stack back down to just the filtered cells, using
    # kept_indices again, so the output lines up with the
    # original input array (final_labels) 
    smoothed_filtered = smoothed_full_valid[kept_indices]
    changed_filtered = changed_full_valid[kept_indices]

    # Sanity check: None of the high lat steaks should make it into filtered output 
    fill_leak = smoothed_filtered == FILL_VALUE
    if np.any(fill_leak):
        raise RuntimeError(
            f"Internal fill value leaked into {int(fill_leak.sum()):,} retained filtered pixels."
        )

    # Sanity check: Check that the "changed" bookkeeping
    #should agree exactly with just directly comparing the final labels to the original ones.
    actual_change = smoothed_filtered != final_labels
    if not np.array_equal(changed_filtered, actual_change):
        n_unflagged = int(np.sum(actual_change & ~changed_filtered))
        n_false_flag = int(np.sum(changed_filtered & ~actual_change))
        raise RuntimeError(
            "Final changed mask disagrees with actual label changes: "
            f"unflagged={n_unflagged:,}, flagged-but-same={n_false_flag:,}."
        )

    # Split the changed cells into two categories : cells
    # that started as noise and got filled in with a real cluster label
    # and cells that already had a real cluster label but got switched to a different
    # one by the vote 
    original_noise = final_labels == NOISE_LABEL
    from_noise = changed_filtered & original_noise
    from_real = changed_filtered & ~original_noise

    np.save(out_root / "final_combined_labels_majority_smoothed.npy", smoothed_filtered)
    np.save(out_root / "changed_mask.npy", changed_filtered)

    # summary of the run: settings used 
    summary = {
        "ensemble": args.ensemble,
        "method": "neighborhood_majority_vote_exclude_noise_votes",
        "final_file": str(final_file),
        "kept_indices_file": str(KEPT_INDICES_FILE),
        "radius": args.radius,
        "window": f"{2 * args.radius + 1}x{2 * args.radius + 1}",
        "include_center": args.include_center,
        "exclude_noise_from_votes": True,
        "min_votes": args.min_votes,
        "require_strict_majority": args.require_strict_majority,
        "iterations_requested": args.iterations,
        "cleanup_domain": "retained_filtered_cells_only",
        "longitude": "periodic",
        "latitude": "not_periodic",
        "protected_clusters": "none - every valid cell is eligible",
        "tie_rule": "keep original label on a tie among top-voted non-noise labels",
        "n_filtered_points": int(final_labels.size),
        "n_changed": int(changed_filtered.sum()),
        "percent_changed": float(100.0 * changed_filtered.mean()),
        "n_noise_relabeled": int(from_noise.sum()),
        "n_real_cluster_pixels_relabeled": int(from_real.sum()),
        "output_root": str(out_root),
    }

    with open(out_root / "smoothing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_root / "monthly_records.json", "w") as f:
        json.dump(month_records, f, indent=2)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print("Changed points:", f"{summary['n_changed']:,}")
    print("Percent changed:", f"{summary['percent_changed']:.6f}%")
    print("Noise pixels relabeled:", f"{summary['n_noise_relabeled']:,}")
    print("Real-cluster pixels relabeled:", f"{summary['n_real_cluster_pixels_relabeled']:,}")
    print("Output:", out_root)


if __name__ == "__main__":
    main()
