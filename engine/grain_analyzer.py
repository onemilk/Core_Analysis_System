"""GrainAnalyzer — quantitative grain analysis from extracted regions."""

import math
import cv2
import numpy as np
from engine.models import MaskRegion, GrainResult


class GrainAnalyzer:
    @staticmethod
    def analyze(regions: list, scale_mm_per_px: float,
                image_area_px: float) -> tuple:
        results = []
        diameters = []

        for i, region in enumerate(regions):
            area_mm2 = region.area_px * (scale_mm_per_px ** 2)
            # Perimeter via arcLength
            contour_pts = np.array(region.contour, dtype=np.int32)
            if contour_pts.ndim == 2 and contour_pts.shape[0] > 2:
                perimeter_px = cv2.arcLength(contour_pts.reshape(-1, 1, 2), True)
            else:
                perimeter_px = 0.0
            perimeter_mm = perimeter_px * scale_mm_per_px
            d = 2.0 * math.sqrt(area_mm2 / math.pi)
            circularity = (4.0 * math.pi * area_mm2 / (perimeter_mm ** 2)) \
                          if perimeter_mm > 0 else 0.0

            # Feret diameters via minAreaRect
            pts = contour_pts.reshape(-1, 1, 2) if contour_pts.ndim == 2 else np.array([[0,0]])
            rect = cv2.minAreaRect(pts) if len(pts) >= 5 else None
            if rect:
                feret_long = max(rect[1]) * scale_mm_per_px
                feret_short = min(rect[1]) * scale_mm_per_px
            else:
                feret_long = feret_short = d

            size_cat = GrainAnalyzer._classify_size(d)

            result = GrainResult(
                region_index=i,
                area_mm2=round(area_mm2, 4),
                equivalent_d_mm=round(d, 4),
                perimeter_mm=round(perimeter_mm, 4),
                feret_long_mm=round(feret_long, 4),
                feret_short_mm=round(feret_short, 4),
                circularity=round(circularity, 4),
                size_category=size_cat
            )
            results.append(result)
            diameters.append(d)

        n = len(results)
        avg_d = round(sum(diameters) / n, 4) if n > 0 else 0.0
        md = round(float(np.percentile(diameters, 50)), 4) if diameters else 0.0
        std = round(float(np.std(diameters)), 4) if diameters else 0.0

        size_dist = {"砾": 0, "砂": 0, "粉砂": 0, "泥": 0}
        for r in results:
            if r.size_category in size_dist:
                size_dist[r.size_category] += 1

        summary = {
            "total_count": n,
            "avg_diameter_mm": avg_d,
            "md_diameter_mm": md,
            "std_dev_mm": std,
            "max_diameter_mm": round(max(diameters), 4) if diameters else 0.0,
            "min_diameter_mm": round(min(diameters), 4) if diameters else 0.0,
            "size_distribution": size_dist,
            "diameters": diameters,
        }
        return results, summary

    @staticmethod
    def _classify_size(diameter_mm: float) -> str:
        if diameter_mm > 2:
            return "砾"
        elif diameter_mm >= 0.0625:
            return "砂"
        elif diameter_mm >= 0.0039:
            return "粉砂"
        else:
            return "泥"