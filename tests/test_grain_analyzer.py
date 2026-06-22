"""Tests for GrainAnalyzer"""
import numpy as np
import cv2
from engine.grain_analyzer import GrainAnalyzer
from engine.models import MaskRegion


def _make_circular_region():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 25, 255, -1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return MaskRegion(
        contour=cnts[0].squeeze(1).tolist(),
        area_px=cv2.contourArea(cnts[0]),
        centroid=(50, 50),
        bbox=(25, 25, 50, 50)
    )


class TestGrainAnalyzer:
    def test_analyze_single_grain(self):
        region = _make_circular_region()
        scale = 0.05
        image_area_px = 10000
        results, summary = GrainAnalyzer.analyze([region], scale, image_area_px)
        assert len(results) == 1
        r = results[0]
        assert r["area_mm2"] > 0
        assert r["equivalent_d_mm"] > 0
        assert r["feret_long_mm"] > 0
        assert 0 < r["circularity"] <= 1.0

    def test_feret_dimensions(self):
        region = _make_circular_region()
        scale = 0.05
        image_area_px = 10000
        results, _ = GrainAnalyzer.analyze([region], scale, image_area_px)
        r = results[0]
        ratio = r["feret_long_mm"] / max(r["feret_short_mm"], 0.001)
        assert ratio < 1.5

    def test_size_classification(self):
        region = _make_circular_region()
        scale = 1.0
        image_area_px = 10000
        results, _ = GrainAnalyzer.analyze([region], scale, image_area_px)
        assert results[0]["size_category"] == "砾"

    def test_empty_regions(self):
        results, summary = GrainAnalyzer.analyze([], 0.05, 10000)
        assert results == []
        assert summary["total_count"] == 0
        assert summary["avg_diameter_mm"] == 0.0

    def test_summary_statistics(self):
        region = _make_circular_region()
        scale = 0.05
        image_area_px = 10000
        _, summary = GrainAnalyzer.analyze([region], scale, image_area_px)
        assert summary["total_count"] == 1
        assert summary["avg_diameter_mm"] > 0
        assert "md_diameter_mm" in summary
        assert "std_dev_mm" in summary
        assert "size_distribution" in summary
