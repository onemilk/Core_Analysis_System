"""Tests for ImageProcessor"""
import numpy as np
import cv2
from core_analysis.engine.image_processor import ImageProcessor


def _make_test_image(w=100, h=80):
    """Create a simple BGR test image with a dark circle."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    cv2.circle(img, (50, 40), 15, (50, 50, 50), -1)
    return img


class TestImageProcessor:
    def test_auto_levels_produces_valid_image(self):
        img = _make_test_image()
        result = ImageProcessor.auto_levels(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_auto_levels_on_dark_image(self):
        """Auto levels on dark image should increase brightness."""
        dark = (_make_test_image() * 0.3).astype(np.uint8)
        result = ImageProcessor.auto_levels(dark)
        assert result.mean() > dark.mean()

    def test_to_grayscale(self):
        img = _make_test_image()
        gray = ImageProcessor.to_grayscale(img)
        assert len(gray.shape) == 2

    def test_gaussian_blur(self):
        img = _make_test_image()
        blurred = ImageProcessor.gaussian_blur(img, kernel_size=5)
        assert blurred.shape == img.shape

    def test_sharpen(self):
        img = _make_test_image()
        sharp = ImageProcessor.sharpen(img)
        assert sharp.shape == img.shape

    def test_canny_edges(self):
        img = _make_test_image()
        edges = ImageProcessor.canny_edges(img, 50, 150)
        assert len(edges.shape) == 2
        assert edges.dtype == np.uint8

    def test_adjust_brightness_contrast(self):
        img = _make_test_image()
        result = ImageProcessor.adjust_brightness_contrast(img, brightness=30, contrast=1.2)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
