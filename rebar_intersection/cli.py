"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pipeline import DetectionConfig, detect_intersections, visualize_result
from .plane import PlaneSelectConfig
from .lines import LineFitConfig, OutlierFilterConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect rebar mesh intersection points from a 3D point cloud."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/sample/point_cloud_00000.pcd",
        help="Input point cloud (.pcd / .ply / ...).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/intersections.npy",
        help="Output path for intersection coordinates (.npy).",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="output/intersections.json",
        help="Optional JSON export of intersection coordinates.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show Open3D visualization windows (requires a display).",
    )
    parser.add_argument("--plane-dist", type=float, default=0.03, help="RANSAC plane distance threshold.")
    parser.add_argument("--min-plane-points", type=int, default=15000, help="Min points to keep a plane.")
    parser.add_argument("--line-dist", type=float, default=0.015, help="RANSAC line distance threshold.")
    parser.add_argument("--min-line-points", type=int, default=5000, help="Min inliers per line.")
    parser.add_argument("--outlier-threshold", type=float, default=0.997, help="Direction similarity threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectionConfig(
        plane=PlaneSelectConfig(
            distance_threshold=args.plane_dist,
            min_plane_points=args.min_plane_points,
        ),
        line=LineFitConfig(
            distance_threshold=args.line_dist,
            min_points=args.min_line_points,
        ),
        outlier=OutlierFilterConfig(similarity_threshold=args.outlier_threshold),
    )

    result = detect_intersections(args.input, config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, result.intersections)
    print(f"-> Saved intersections to {out_path}  shape={result.intersections.shape}")

    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "count": int(result.intersections.shape[0]),
            "points": result.intersections.tolist(),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"-> Saved JSON to {json_path}")

    if args.visualize:
        visualize_result(result)


if __name__ == "__main__":
    main()
