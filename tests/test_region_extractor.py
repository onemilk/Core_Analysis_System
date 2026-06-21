"""Tests for RegionExtractor"""
import numpy as np
import cv2
from core_analysis.engine.region_extractor import RegionExtractor


def _make_image_with_holes():
    """Create an image with two distinct dark circular regions."""
    img = np.ones((120, 160, 3), dtype=np.uint8) * 220
    cv2.circle(img, (40, 60), 18, (30, 30, 30), -1)
    cv2.circle(img, (120, 60), 22, (35, 35, 35), -1)
    return img


class TestRegionExtractor:
    def test_extract_by_color_sample(self):
        img = _make_image_with_holes()
        sample_color = img[60, 40]
        regions = RegionExtractor.extract_by_color_sample(
            img, sample_color, match_tolerance=40
        )
        assert len(regions) >= 1
        for r in regions:
            assert r.area_px > 0
            assert len(r.contour) > 2
            assert r.bbox[2] > 0 and r.bbox[3] > 0

    def test_filter_by_area(self):
        img = _make_image_with_holes()
        sample_color = img[60, 40]
        regions = RegionExtractor.extract_by_color_sample(
            img, sample_color, match_tolerance=40
        )
        big_regions = RegionExtractor.filter_by_area(regions, min_area_px=100)
        assert len(big_regions) <= len(regions)

    def test_get_mask_from_regions(self):
        img = _make_image_with_holes()
        sample_color = img[60, 40]
        regions = RegionExtractor.extract_by_color_sample(
            img, sample_color, match_tolerance=40
        )
        mask = RegionExtractor.get_mask_from_regions(regions, img.shape[:2])
        assert mask.shape == img.shape[:2]
        assert mask.dtype == np.uint8
        assert mask.max() == 255
