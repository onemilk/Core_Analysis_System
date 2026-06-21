"""RegionExtractor — color-based region segmentation via LAB color space."""

import cv2
import numpy as np
from engine.models import MaskRegion


class RegionExtractor:
    @staticmethod
    def extract_by_color_sample(image: np.ndarray, sample_color: np.ndarray,
                                match_tolerance: int = 30,
                                continuous_only: bool = False) -> list:
        """Extract regions matching a sampled color using LAB color space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        sample_lab = cv2.cvtColor(np.uint8([[sample_color]]), cv2.COLOR_BGR2LAB)[0][0]

        # Cast to int16 to avoid uint8 underflow when subtracting tolerance
        s = sample_lab.astype(np.int16)
        lower = np.clip(s - match_tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(s + match_tolerance, 0, 255).astype(np.uint8)

        mask = cv2.inRange(lab, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 1:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = 0.0, 0.0
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append(MaskRegion(
                contour=cnt.squeeze(1).tolist() if len(cnt.shape) == 3 else [],
                area_px=area,
                centroid=(cx, cy),
                bbox=(x, y, w, h)
            ))
        return regions

    @staticmethod
    def filter_by_area(regions: list, min_area_px: float = 0,
                       max_area_px: float = float('inf')) -> list:
        return [r for r in regions if min_area_px <= r.area_px <= max_area_px]

    @staticmethod
    def extract_continuous_at_point(image: np.ndarray, sample_color: np.ndarray,
                                    px: int, py: int, match_tolerance: int = 30) -> list:
        """Extract only the connected region containing (px, py) matching sample color."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        sample_lab = cv2.cvtColor(np.uint8([[sample_color]]), cv2.COLOR_BGR2LAB)[0][0]
        s = sample_lab.astype(np.int16)
        lower = np.clip(s - match_tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(s + match_tolerance, 0, 255).astype(np.uint8)
        mask = cv2.inRange(lab, lower, upper)

        # Flood fill from click point to isolate connected component
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(mask, flood_mask, (px, py), 255,
                      flags=cv2.FLOODFILL_MASK_ONLY)
        filled = flood_mask[1:-1, 1:-1]

        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1:
                continue
            M = cv2.moments(cnt)
            cx = M["m10"] / M["m00"] if M["m00"] != 0 else 0.0
            cy = M["m01"] / M["m00"] if M["m00"] != 0 else 0.0
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append(MaskRegion(
                contour=cnt.squeeze(1).tolist() if len(cnt.shape) == 3 else [],
                area_px=area, centroid=(cx, cy), bbox=(x, y, w, h)))
        return regions

    @staticmethod
    def get_mask_from_regions(regions: list, image_shape: tuple) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        for r in regions:
            pts = np.array(r.contour, dtype=np.int32)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                cv2.drawContours(mask, [pts], -1, 255, -1)
        return mask
