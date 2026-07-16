#!/usr/bin/env python3
"""Generate matplotlib result figures for the README (headless)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from rebar_intersection.pipeline import detect_intersections, load_point_cloud

OUT = ROOT / "docs" / "images"
SAMPLE = ROOT / "data" / "sample" / "point_cloud_00000.pcd"


def _downsample(cloud, max_points: int = 80000) -> np.ndarray:
    pts = np.asarray(cloud.points)
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
    return pts[idx]


def _set_equal_3d(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2 * 1.05
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def save_fig(fig, name: str) -> None:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    raw = load_point_cloud(SAMPLE)
    result = detect_intersections(raw)

    # 1) Raw point cloud (downsample)
    raw_pts = _downsample(raw, 60000)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2],
        s=0.2, c=raw_pts[:, 2], cmap="viridis", alpha=0.5,
    )
    ax.view_init(elev=25, azim=-60)
    _set_equal_3d(ax, raw_pts)
    ax.set_title(f"Raw point cloud ({len(raw.points)} points)")
    save_fig(fig, "demo_01_raw.png")

    # 2) Top plane
    top_pts = _downsample(result.top_plane, 50000)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(top_pts[:, 0], top_pts[:, 1], top_pts[:, 2], s=0.3, c="#e85d04", alpha=0.55)
    ax.view_init(elev=30, azim=-55)
    _set_equal_3d(ax, top_pts)
    ax.set_title(f"Selected top plane ({len(result.top_plane.points)} points)")
    save_fig(fig, "demo_02_top_plane.png")

    # 3) Line segments
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    cmap = plt.get_cmap("tab20")
    for i, seg in enumerate(result.line_segments):
        pts = _downsample(seg, 8000)
        all_pts.append(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.6, c=[cmap(i % 20)], alpha=0.8)
    all_pts = np.vstack(all_pts) if all_pts else np.zeros((1, 3))
    ax.view_init(elev=35, azim=-50)
    _set_equal_3d(ax, all_pts)
    ax.set_title(f"RANSAC line segments ({len(result.line_segments)} lines)")
    save_fig(fig, "demo_03_lines.png")

    # 4) Intersections overlay
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, seg in enumerate(result.line_segments):
        pts = _downsample(seg, 6000)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.4, c=[cmap(i % 20)], alpha=0.55)
    for cloud in result.fitted_line_clouds:
        pts = np.asarray(cloud.points)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.0, alpha=0.9)
    cross = result.intersections
    if cross.size:
        ax.scatter(
            cross[:, 0], cross[:, 1], cross[:, 2],
            s=35, c="red", depthshade=False, label="intersections",
        )
    ax.view_init(elev=40, azim=-45)
    base = all_pts if all_pts.size else cross
    _set_equal_3d(ax, base)
    ax.set_title(f"Detected intersections (N={cross.shape[0]})")
    ax.legend(loc="upper right")
    save_fig(fig, "demo_04_intersections.png")

    # 5) Top-down XY view
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111)
    for i, seg in enumerate(result.line_segments):
        pts = _downsample(seg, 6000)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c=[cmap(i % 20)], alpha=0.5)
    for cloud in result.fitted_line_clouds:
        pts = np.asarray(cloud.points)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1.2, alpha=0.85)
    if cross.size:
        ax.scatter(
            cross[:, 0], cross[:, 1],
            s=40, c="red", zorder=5, label=f"intersections ({cross.shape[0]})",
        )
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Top-down view (XY)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    save_fig(fig, "demo_05_topdown.png")

    print("done")


if __name__ == "__main__":
    main()
