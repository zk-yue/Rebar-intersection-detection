"""Top-layer plane selection via iterative RANSAC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class PlaneSelectConfig:
    distance_threshold: float = 0.03
    ransac_n: int = 3
    num_iterations: int = 500
    min_plane_points: int = 15000


def select_top_plane(
    point_cloud: o3d.geometry.PointCloud,
    config: PlaneSelectConfig | None = None,
) -> tuple[list[o3d.geometry.PointCloud], list, o3d.geometry.PointCloud]:
    """
    Iteratively segment planes with RANSAC and pick the top layer.

    The top plane is chosen as the plane with the smallest mean Z among
    planes that contain enough points (closest to the camera if Z points
    toward the sensor / depth increases with distance — here smaller mean Z).
    """
    config = config or PlaneSelectConfig()
    remaining = o3d.geometry.PointCloud(point_cloud)
    planes: list[o3d.geometry.PointCloud] = []
    plane_models: list = []

    while len(remaining.points) >= config.ransac_n:
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=config.distance_threshold,
            ransac_n=config.ransac_n,
            num_iterations=config.num_iterations,
        )
        if len(inliers) == 0:
            break
        inlier_cloud = remaining.select_by_index(inliers)
        planes.append(inlier_cloud)
        plane_models.append(plane_model)
        remaining = remaining.select_by_index(inliers, invert=True)
        # Stop if leftover is too small to form another plane
        if len(remaining.points) < config.min_plane_points // 2:
            break

    if not planes:
        raise RuntimeError("No planes were fitted from the point cloud.")

    average_z = []
    point_cnt = []
    for plane in planes:
        pts = np.asarray(plane.points)
        average_z.append(float(np.mean(pts[:, 2])))
        point_cnt.append(pts.shape[0])

    print(f"-> Fitted {len(planes)} plane(s)")
    print(f"-> Mean Z per plane: {average_z}")
    print(f"-> Point count per plane: {point_cnt}")

    filtered_z = list(average_z)
    for i, count in enumerate(point_cnt):
        if count < config.min_plane_points:
            filtered_z[i] = float("inf")
    print(f"-> Mean Z after filtering small planes: {filtered_z}")

    if all(z == float("inf") for z in filtered_z):
        raise RuntimeError(
            "No plane passed the min_plane_points filter. "
            "Try lowering min_plane_points."
        )

    min_index = int(np.argmin(filtered_z))
    selected_plane = planes[min_index]
    return planes, plane_models, selected_plane
