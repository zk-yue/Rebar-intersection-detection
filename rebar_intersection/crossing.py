"""Crossing / intersection point computation between two line groups."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from sympy import symbols, solve


def compute_crossings(
    index_class: list[np.ndarray],
    directions: list[np.ndarray],
    points_on_line: list[np.ndarray],
    marker_radius: float = 0.0125,
) -> tuple[np.ndarray, list[o3d.geometry.TriangleMesh]]:
    """
    Compute pairwise XY intersections of two orientation groups.

    Z is taken as the average of the two lines' Z at the intersection
    parameters (lines may not be perfectly coplanar).
    """
    if len(index_class) < 2:
        raise ValueError("index_class must contain two orientation groups.")

    t1, t2 = symbols("t_1 t_2")
    cross_points = []

    for m in index_class[0]:
        for n in index_class[1]:
            a0, b0 = directions[m], points_on_line[m]
            a1, b1 = directions[n], points_on_line[n]
            result = solve(
                [
                    a0[0] * t1 + b0[0] - (a1[0] * t2 + b1[0]),
                    a0[1] * t1 + b0[1] - (a1[1] * t2 + b1[1]),
                ],
                [t1, t2],
            )
            if not result:
                continue
            cx = float(a0[0] * result[t1] + b0[0])
            cy = float(a0[1] * result[t1] + b0[1])
            cz1 = float(a0[2] * result[t1] + b0[2])
            cz2 = float(a1[2] * result[t2] + b1[2])
            cross_points.append(np.array([cx, cy, (cz1 + cz2) / 2.0], dtype=float))

    cross_point_set = np.asarray(cross_points, dtype=float) if cross_points else np.zeros((0, 3))
    print(f"-> Intersection count: {cross_point_set.shape[0]}")

    markers: list[o3d.geometry.TriangleMesh] = []
    for point in cross_point_set:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius, resolution=20)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.translate(point)
        markers.append(sphere)

    return cross_point_set, markers
