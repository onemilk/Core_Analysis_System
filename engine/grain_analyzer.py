"""GrainAnalyzer — watershed-based grain segmentation + quantitative analysis."""

import math, cv2, numpy as np
from skimage.feature import peak_local_max


class GrainAnalyzer:
    @staticmethod
    def analyze_direct(bgr, scale_mm_per_px=0.05, min_area_px=30, threshold_block=21):
        """Watershed-based grain segmentation. Works directly on BGR image.
        threshold_block: smaller = more sensitive to small grains (odd, 11-51)
        min_area_px: filter noise below this size"""
        if bgr is None:
            return [], {"error": "Image is None"}, {}

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleanup: aggressive open+close to denoise and fill grain interiors
        kernel3 = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=3)

        # Distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return [], {"total_count": 0, "error": "No grains detected"}, {"gray": gray, "binary": binary, "result": bgr}

        # Peak local max markers
        local_max = peak_local_max(dist, min_distance=15, exclude_border=False, labels=binary//255)
        markers = np.zeros(dist.shape, dtype=np.int32)
        for i, (y, x) in enumerate(local_max):
            markers[y, x] = i + 1

        # Watershed with line separation for touching grains
        labels = cv2.watershed(bgr, np.int32(markers))
        labels[binary == 0] = 0

        results, diameters = [], []
        result_img = bgr.copy()

        for label in range(1, markers.max() + 1):
            if label == -1: continue  # skip watershed boundary
            mask = np.uint8(labels == label) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(cnt)
            if area_px < min_area_px:
                continue

            rect = cv2.minAreaRect(cnt)
            feret_long_px = max(rect[1])
            feret_short_px = min(rect[1])

            # Shape filter: reject elongated cracks/gaps between grains
            aspect_ratio = feret_long_px / max(feret_short_px, 1.0)
            if aspect_ratio > 4.0 and area_px < min_area_px * 5:
                continue

            area_mm2 = float(area_px * (scale_mm_per_px ** 2))
            perimeter_px = cv2.arcLength(cnt, True)
            perimeter_mm = float(perimeter_px * scale_mm_per_px)
            d_mm = float(2.0 * math.sqrt(area_mm2 / math.pi))
            circularity = float((4.0 * math.pi * area_mm2 / (perimeter_mm ** 2)) if perimeter_mm > 0 else 0.0)

            feret_long = float(feret_long_px * scale_mm_per_px)
            feret_short = float(feret_short_px * scale_mm_per_px)
            size_cat = GrainAnalyzer._classify_size(d_mm)

            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
            # Annotate diameter value next to each grain
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.putText(result_img, f"{d_mm:.1f}", (cx-15, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            diameters.append(d_mm)
            results.append({"area_mm2": area_mm2, "d_mm": d_mm,
                           "feret_long": feret_long, "feret_short": feret_short,
                           "circularity": circularity, "size": size_cat})

        n = len(results)
        avg_d = float(round(sum(diameters) / n, 4)) if n > 0 else 0.0
        md = float(round(np.percentile(diameters, 50), 4)) if diameters else 0.0
        std = float(round(np.std(diameters), 4)) if diameters else 0.0

        size_dist = {"砾": 0, "砂": 0, "粉砂": 0, "泥": 0}
        for r in results:
            if r["size"] in size_dist:
                size_dist[r["size"]] += 1

        summary = {
            "total_count": int(n),
            "avg_diameter_mm": avg_d,
            "md_diameter_mm": md,
            "d10_mm": float(round(np.percentile(diameters, 10), 4)) if diameters else 0.0,
            "d50_mm": float(round(np.median(diameters), 4)) if diameters else 0.0,
            "d90_mm": float(round(np.percentile(diameters, 90), 4)) if diameters else 0.0,
            "std_dev_mm": std,
            "max_diameter_mm": float(round(max(diameters), 4)) if diameters else 0.0,
            "min_diameter_mm": float(round(min(diameters), 4)) if diameters else 0.0,
            "size_distribution": size_dist,
            "diameters": [float(d) for d in diameters],
        }
        return results, summary, {"result": result_img, "binary": binary, "gray": gray}

    @staticmethod
    def analyze(regions: list, scale_mm_per_px: float, image_area_px: float) -> tuple:
        """Legacy method — kept for backward compatibility."""
        results, diameters = [], []
        for i, region in enumerate(regions):
            area_mm2 = region.area_px * (scale_mm_per_px ** 2)
            d = float(2.0 * math.sqrt(area_mm2 / math.pi))
            size_cat = GrainAnalyzer._classify_size(d)
            results.append(GrainResult(region_index=i, area_mm2=round(area_mm2,4),
                equivalent_d_mm=round(d,4), size_category=size_cat))
            diameters.append(d)
        n = len(results)
        size_dist = {"砾":0,"砂":0,"粉砂":0,"泥":0}
        for r in results:
            if r.size_category in size_dist: size_dist[r.size_category] += 1
        summary = {"total_count": n, "avg_diameter_mm": round(sum(diameters)/n,4) if n else 0.0,
            "md_diameter_mm": round(float(np.percentile(diameters,50)),4) if diameters else 0.0,
            "std_dev_mm": round(float(np.std(diameters)),4) if diameters else 0.0,
            "size_distribution": size_dist, "diameters": diameters}
        return results, summary

    @staticmethod
    def _classify_size(diameter_mm: float) -> str:
        if diameter_mm > 2: return "砾"
        elif diameter_mm >= 0.0625: return "砂"
        elif diameter_mm >= 0.0039: return "粉砂"
        else: return "泥"