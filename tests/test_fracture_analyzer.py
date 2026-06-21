"""Tests for fracture analyzer."""
import numpy as np, cv2
from engine.fracture_analyzer import FractureAnalyzer

def _make_crack_image():
    img = np.ones((150, 200, 3), dtype=np.uint8) * 200
    cv2.line(img, (30, 60), (170, 70), (40, 40, 40), 2)
    cv2.line(img, (50, 30), (55, 120), (40, 40, 40), 2)
    return img

class TestFractureAnalyzer:
    def test_analyze_finds_cracks(self):
        img = _make_crack_image()
        results, summary, images = FractureAnalyzer.analyze(img, threshold=60, min_area=50)
        assert summary["crack_count"] > 0
        assert "avg_width" in summary
        assert images["result"] is not None

    def test_return_format(self):
        img = _make_crack_image()
        _, summary, _ = FractureAnalyzer.analyze(img)
        required = ["crack_count", "total_area", "avg_width", "max_length"]
        for k in required:
            assert k in summary, f"Missing: {k}"

    def test_empty_image(self):
        _, summary, _ = FractureAnalyzer.analyze(None)
        assert "error" in summary
