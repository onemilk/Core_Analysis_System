"""Tests for MorphologyEngine"""
import numpy as np
import cv2
from engine.morphology_engine import MorphologyEngine
from engine.models import MaskRegion


def _make_square_region():
    """Create a simple square MaskRegion."""
    cnt = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)
    return MaskRegion(
        contour=cnt.tolist(),
        area_px=1600.0,
        centroid=(30.0, 30.0),
        bbox=(10, 10, 40, 40)
    )


class TestMorphologyEngine:
    def test_dilate_region(self):
        r = _make_square_region()
        result = MorphologyEngine.dilate_region(r, kernel_size=5, iterations=2)
        assert result.area_px >= r.area_px

    def test_erode_region(self):
        r = _make_square_region()
        result = MorphologyEngine.erode_region(r, kernel_size=5, iterations=1)
        assert result.area_px <= r.area_px

    def test_denoise_by_area(self):
        regions = [
            MaskRegion(contour=[], area_px=500, centroid=(0, 0), bbox=(0, 0, 10, 10)),
            MaskRegion(contour=[], area_px=5, centroid=(0, 0), bbox=(0, 0, 2, 2)),
            MaskRegion(contour=[], area_px=300, centroid=(0, 0), bbox=(0, 0, 8, 8)),
        ]
        filtered = MorphologyEngine.denoise_by_area(regions, min_area_px=100)
        assert len(filtered) == 2

    def test_fill_holes(self):
        """Hole filling should produce a filled contour."""
        r = _make_square_region()
        r.contour = [[10, 10], [50, 10], [50, 50], [30, 30], [10, 50]]  # with dent
        result = MorphologyEngine.fill_holes(r, max_hole_size=200)
        assert result is not None

    def test_erode_too_much_returns_none(self):
        r = _make_square_region()
        result = MorphologyEngine.erode_region(r, kernel_size=99, iterations=3)
        assert result is None
