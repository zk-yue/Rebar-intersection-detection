"""Line fitting, clustering, and outlier filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
from sklearn.cluster import KMeans

from .utils import cosine_similarity


@dataclass
class LineFitConfig:
    min_points: int = 5000
    distance_threshold: float = 0.015
    max_iteration: int = 200


@dataclass
class OutlierFilterConfig:
    num_clusters: int = 2
    similarity_threshold: float = 0.997


def fit_lines(
    selected_plane: o3d.geometry.PointCloud,
    config: LineFitConfig | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[o3d.geometry.PointCloud]]:
    """Fit multiple lines on a planar point cloud with RANSAC."""
    config = config or LineFitConfig()
    remaining = o3d.geometry.PointCloud(selected_plane)
    segments: list[o3d.geometry.PointCloud] = []
    directions_normalized: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    points_on_line: list[np.ndarray] = []
    iters = 0

    while len(remaining.points) > config.min_points:
        points = np.asarray(remaining.points)
        line = pyrsc.Line()
        direction, point, inliers = line.fit(
            points, thresh=config.distance_threshold, maxIteration=config.max_iteration
        )
        if len(inliers) < config.min_points:
            break

        line_cloud = remaining.select_by_index(inliers)
        color = np.random.uniform(0, 1, 3)
        line_cloud.paint_uniform_color(color)
        remaining = remaining.select_by_index(inliers, invert=True)

        direction = np.asarray(direction, dtype=float)
        point = np.asarray(point, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        directions_normalized.append(direction / norm)
        directions.append(direction)
        points_on_line.append(point)
        segments.append(line_cloud)
        iters += 1
        print(f"---> Fitted {iters} line(s)")

    print(f"-> Total valid lines: {iters}")
    return directions_normalized, directions, points_on_line, segments


def filter_line_outliers(
    directions_normalized: list[np.ndarray],
    segments: list[o3d.geometry.PointCloud],
    config: OutlierFilterConfig | None = None,
) -> tuple[list[np.ndarray], list[o3d.geometry.PointCloud]]:
    """
    Cluster lines into two orientation groups and drop outliers.

    Returns:
        index_class: list of two index arrays (one per orientation group)
        segments: segments with outliers removed
    """
    config = config or OutlierFilterConfig()
    lines_vector = np.asarray(directions_normalized, dtype=float)
    if lines_vector.ndim != 2 or lines_vector.shape[0] < config.num_clusters:
        raise RuntimeError(
            f"Need at least {config.num_clusters} lines for clustering, "
            f"got {0 if lines_vector.ndim != 2 else lines_vector.shape[0]}."
        )

    n = lines_vector.shape[0]
    similarities = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            sim = abs(cosine_similarity(lines_vector[i], lines_vector[j]))
            similarities[i, j] = sim
            similarities[j, i] = sim

    kmeans = KMeans(n_clusters=config.num_clusters, n_init="auto", random_state=0)
    line_classes = kmeans.fit_predict(similarities)
    print(f"-> Line class labels: {line_classes}")

    index_mem = np.arange(n)
    index_class: list[np.ndarray] = []
    outliers_index: list[int] = []

    for cluster_id in range(config.num_clusters):
        cluster_indices = index_mem[line_classes == cluster_id]
        index_class.append(cluster_indices.copy())
        cluster_lines = lines_vector[line_classes == cluster_id].copy()
        if cluster_lines.shape[0] == 0:
            continue

        for i in range(1, cluster_lines.shape[0]):
            if cosine_similarity(cluster_lines[0], cluster_lines[i]) < 0:
                cluster_lines[i] = -cluster_lines[i]

        mean_vector = cluster_lines.mean(axis=0).reshape(1, 3)
        centroid_similarity = np.array(
            [abs(cosine_similarity(mean_vector.ravel(), cluster_lines[i])) for i in range(cluster_lines.shape[0])]
        )
        outlier_local = np.where(centroid_similarity < config.similarity_threshold)[0]
        outliers_index.extend(cluster_indices[outlier_local].tolist())

    print(f"-> Class 0 indices: {index_class[0]}")
    print(f"-> Class 1 indices: {index_class[1]}")
    print(f"-> Outlier count: {len(outliers_index)}")

    # Keep indices aligned with directions / points_on_line arrays.
    for outlier_idx in outliers_index:
        index_class[0] = np.delete(index_class[0], np.where(index_class[0] == outlier_idx))
        index_class[1] = np.delete(index_class[1], np.where(index_class[1] == outlier_idx))

    for index in sorted(outliers_index, reverse=True):
        del segments[index]

    print(f"-> Class 0 after outlier removal: {index_class[0]}")
    print(f"-> Class 1 after outlier removal: {index_class[1]}")
    return index_class, segments


def build_line_point_clouds(
    index_class: list[np.ndarray],
    directions: list[np.ndarray],
    points_on_line: list[np.ndarray],
    t_range: float = 0.8,
    num_samples: int = 100,
) -> list[o3d.geometry.PointCloud]:
    """Sample parametric lines into Open3D point clouds for visualization."""
    clouds: list[o3d.geometry.PointCloud] = []
    for class_indices in index_class:
        for idx in class_indices:
            p0 = points_on_line[idx]
            direction = directions[idx]
            t_values = np.linspace(-t_range, t_range, num=num_samples)
            line_points = [p0 + t * direction for t in t_values]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(line_points)
            color = np.random.uniform(0, 1, 3)
            cloud.paint_uniform_color(color)
            clouds.append(cloud)
    return clouds
