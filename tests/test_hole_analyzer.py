"""Tests for hole analyzer."""
import numpy as np, cv2
from engine.hole_analyzer import HoleAnalyzer

def _make_hole_image():
    img = np.ones((120, 160, 3), dtype=np.uint8) * 200
    cv2.circle(img, (40, 60), 18, (30, 30, 30), -1)
    cv2.circle(img, (120, 60), 22, (35, 35, 35), -1)
    return img

class TestHoleAnalyzer:
    def test_analyze_finds_holes(self):
        img = _make_hole_image()
        results, summary, images = HoleAnalyzer.analyze(img, threshold=90, min_area=10, max_area=5000)
        assert summary["hole_count"] >= 2
        assert summary["total_area"] > 0
        assert images["gray"] is not None
        assert images["binary"] is not None
        assert images["result"] is not None

    def test_return_format(self):
        img = _make_hole_image()
        results, summary, images = HoleAnalyzer.analyze(img)
        required = ["hole_count", "total_area", "avg_area", "avg_circularity", "avg_diameter_mm", "porosity_percent", "diameters"]
        for k in required:
            assert k in summary, f"Missing: {k}"

    def test_empty_image(self):
        results, summary, images = HoleAnalyzer.analyze(None)
        assert "error" in summary
