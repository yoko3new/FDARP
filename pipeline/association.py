"""
Associate detected AR components with NOAA SRS regions within one frame.

A region is linked to every component within `threshold_deg` (great-circle
distance to the nearest component pixel). Linked components and regions are
grouped, and each group is classified:

    1:1  one component, one region
    1:N  one component, several regions
    N:1  several components, one region
    M:N  several of each
    1:0  component with no region
    0:1  region with no component
"""

from collections import defaultdict

import cv2
import numpy as np

from solar_utils import (advance_longitudes, great_circle_deg, latlon_to_pixel,
                         pixel_to_latlon)

CLASSES = ("1:1", "1:N", "N:1", "M:N", "1:0", "0:1")


def nearest_on_surface(mask, lat0, lon0, r_px, cx, cy, b0, subsample=4):
    """
    Nearest mask pixel to a heliographic position.
    Returns (x_px, y_px, dist_deg); dist is inf for an empty mask.
    """
    ys, xs = np.nonzero(mask)
    if subsample > 1:
        xs, ys = xs[::subsample], ys[::subsample]
    if len(xs) == 0:
        return None, None, np.inf
    lat, lon, on_disk = pixel_to_latlon(xs, ys, r_px, cx, cy, b0)
    if not on_disk.any():
        return None, None, np.inf
    xs, ys, lat, lon = xs[on_disk], ys[on_disk], lat[on_disk], lon[on_disk]
    d = great_circle_deg(lat0, lon0, lat, lon)
    k = int(np.argmin(d))
    return int(xs[k]), int(ys[k]), float(d[k])


def component_distances(labels, ncomp, lat0, lon0, r_px, cx, cy, b0, subsample=4):
    """Great-circle distance (deg) from a position to every component: {label: deg}."""
    out = {}
    for idx in range(1, ncomp):
        _, _, d = nearest_on_surface(labels == idx, lat0, lon0,
                                     r_px, cx, cy, b0, subsample)
        if np.isfinite(d):
            out[idx] = d
    return out


def build_groups(links, comp_ids, region_ids):
    """Connected groups of the bipartite (component, region) link graph."""
    adj_c, adj_r = defaultdict(set), defaultdict(set)
    for c, r in links:
        adj_c[c].add(r)
        adj_r[r].add(c)

    seen_c, seen_r, groups = set(), set(), []
    for c0 in comp_ids:
        if c0 in seen_c or c0 not in adj_c:
            continue
        gc, gr = set(), set()
        stack = [("c", c0)]
        while stack:
            kind, node = stack.pop()
            if kind == "c" and node not in gc:
                gc.add(node)
                stack.extend(("r", r) for r in adj_c[node])
            elif kind == "r" and node not in gr:
                gr.add(node)
                stack.extend(("c", c) for c in adj_r[node])
        seen_c |= gc
        seen_r |= gr
        groups.append({"components": sorted(gc), "regions": sorted(gr)})

    groups += [{"components": [c], "regions": []} for c in comp_ids if c not in seen_c]
    groups += [{"components": [], "regions": [r]} for r in region_ids if r not in seen_r]
    return groups


def classify(group):
    nc, nr = len(group["components"]), len(group["regions"])
    if nc == 1 and nr == 1:
        return "1:1"
    if nc == 1 and nr == 0:
        return "1:0"
    if nc == 0 and nr == 1:
        return "0:1"
    if nc == 1:
        return "1:N"
    if nr == 1:
        return "N:1"
    return "M:N"


def associate_frame(frame, srs_regions, threshold_deg=3.0, subsample=4,
                    rotate=True, min_area=0):
    """
    Associate one frame (from solar_utils.load_frame) with SRS regions.

    SRS positions are valid at 00:00 UT; with rotate=True they are advanced to
    the frame time. Returns labels/stats of the components, the visible regions
    with their pixel positions, the groups and the class counts.
    """
    mask = frame["mask"].astype(np.uint8)
    cx, cy, r_px, b0 = frame["cx"], frame["cy"], frame["r_px"], frame["b0"]
    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    time = frame["time"]
    hours = int(time[:2]) + int(time[2:]) / 60.0
    regions = advance_longitudes(srs_regions, hours) if rotate else list(srs_regions)

    xs, ys, vis = latlon_to_pixel([r["lat"] for r in regions],
                                  [r["lon"] for r in regions], r_px, cx, cy, b0)
    visible = [dict(r, x=float(x), y=float(y))
               for r, x, y, v in zip(regions, xs, ys, vis) if v]

    comp_ids = [i for i in range(1, ncomp)
                if stats[i, cv2.CC_STAT_AREA] >= min_area]
    links, dists = [], {}
    for r in visible:
        dd = component_distances(labels, ncomp, r["lat"], r["lon"],
                                 r_px, cx, cy, b0, subsample)
        dists[r["number"]] = dd
        links += [(c, r["number"]) for c, d in dd.items()
                  if d <= threshold_deg and c in comp_ids]

    groups = build_groups(links, comp_ids, [r["number"] for r in visible])
    counts = defaultdict(int)
    for g in groups:
        counts[classify(g)] += 1

    return {"labels": labels, "stats": stats, "comp_ids": comp_ids,
            "regions": visible, "distances": dists, "groups": groups,
            "counts": dict(counts)}
