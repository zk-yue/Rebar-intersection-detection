"""End-to-end detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

from .crossing import compute_crossings
from .lines import (
    LineFitConfig,
    OutlierFilterConfig,
    build_line_point_clouds,
    filter_line_outliers,
    fit_lines,
)
from .plane import PlaneSelectConfig, select_top_plane


@dataclass
class DetectionConfig:
    plane: PlaneSelectConfig = field(default_factory=PlaneSelectConfig)
    line: LineFitConfig = field(default_factory=LineFitConfig)
    outlier: OutlierFilterConfig = field(default_factory=OutlierFilterConfig)
    marker_radius: float = 0.0125


@dataclass
class DetectionResult:
    intersections: np.ndarray
    directions: list[np.ndarray]
    points_on_line: list[np.ndarray]
    index_class: list[np.ndarray]
    top_plane: o3d.geometry.PointCloud
    line_segments: list[o3d.geometry.PointCloud]
    fitted_line_clouds: list[o3d.geometry.PointCloud]
    markers: list[o3d.geometry.TriangleMesh]
    all_planes: list[o3d.geometry.PointCloud]


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Failed to load point cloud or cloud is empty: {path}")
    return cloud


def detect_intersections(
    point_cloud: o3d.geometry.PointCloud | str | Path,
    config: DetectionConfig | None = None,
) -> DetectionResult:
    """
    Run the full rebar intersection detection pipeline.

    Steps:
      1. Select the top rebar plane
      2. Fit multiple lines with RANSAC
      3. Cluster lines into two orientations and drop outliers
      4. Compute pairwise crossings between the two groups
    """
    config = config or DetectionConfig()
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        point_cloud = load_point_cloud(point_cloud)

    print("------------- step 1: load point cloud ---------------")
    print(f"-> Points: {len(point_cloud.points)}")

    print("------- step 2: select top plane -----------")
    all_planes, _, top_plane = select_top_plane(point_cloud, config.plane)

    print("------------- step 3: fit lines ---------------")
    directions_norm, directions, points_on_line, segments = fit_lines(top_plane, config.line)

    print("-------- step 4: classify lines & remove outliers ---------")
    index_class, segments = filter_line_outliers(directions_norm, segments, config.outlier)

    print("------------- step 5: compute crossings ---------------")
    intersections, markers = compute_crossings(
        index_class, directions, points_on_line, marker_radius=config.marker_radius
    )
    fitted_line_clouds = build_line_point_clouds(index_class, directions, points_on_line)

    return DetectionResult(
        intersections=intersections,
        directions=directions,
        points_on_line=points_on_line,
        index_class=index_class,
        top_plane=top_plane,
        line_segments=segments,
        fitted_line_clouds=fitted_line_clouds,
        markers=markers,
        all_planes=all_planes,
    )


def visualize_result(result: DetectionResult, show_segments: bool = True) -> None:
    """Open Open3D windows for key intermediate and final results."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    o3d.visualization.draw_geometries([result.top_plane, frame], window_name="Top plane")
    if result.line_segments:
        o3d.visualization.draw_geometries(
            [*result.line_segments, frame], window_name="Line segments"
        )
    geometries = [*result.fitted_line_clouds, *result.markers, frame]
    if show_segments:
        geometries.extend(result.line_segments)
    o3d.visualization.draw_geometries(geometries, window_name="Intersections")
