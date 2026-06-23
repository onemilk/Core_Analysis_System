"""Tests for fracture analyzer."""
import numpy as np, cv2
from engine.fracture_analyzer import FractureAnalyzer

def _make_crack_image():
    """创建含仿真裂缝的测试图像——粗锯齿线，易于边缘检测。"""
    img = np.ones((150, 200, 3), dtype=np.uint8) * 220
    # 单条锯齿裂缝，宽8px，深色，高长宽比确保被检测
    pts = np.array([[20,75],[40,68],[60,78],[80,70],[100,76],[120,67],[140,74],[160,70],[180,73]], dtype=np.int32)
    cv2.polylines(img, [pts], False, (10, 10, 10), 8)
    return img

class TestFractureAnalyzer:
    def test_analyze_finds_cracks(self):
        img = _make_crack_image()
        results, summary, images = FractureAnalyzer.analyze(img, threshold=30, min_area=100)
        assert summary["crack_count"] >= 1, f"Expected >=1 cracks, got {summary['crack_count']}"
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
