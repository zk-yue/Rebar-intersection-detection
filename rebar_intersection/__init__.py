"""Rebar mesh intersection detection from 3D point clouds."""

from .pipeline import detect_intersections, DetectionResult

__all__ = ["detect_intersections", "DetectionResult"]
__version__ = "1.0.0"
