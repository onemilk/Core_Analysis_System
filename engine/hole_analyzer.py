"""Hole analyzer — grayscale threshold + contour extraction (senior project algorithm)."""
import cv2, math, numpy as np

class HoleAnalyzer:
    @staticmethod
    def analyze(image, threshold=100, min_area=10, max_area=10000, scale_mm_per_px=0.05):
        if image is None:
            return [], {"error": "Image is None"}, {}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_img = image.copy()
        results, diameters, areas, circularities = [], [], [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            d_mm = 2 * math.sqrt(area / math.pi) * scale_mm_per_px
            area_mm2 = area * (scale_mm_per_px ** 2)
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
            areas.append(area)
            diameters.append(d_mm)
            circularities.append(circularity)
            results.append({"area_px": float(area), "area_mm2": float(area_mm2), "diameter_mm": float(d_mm), "circularity": float(circularity)})

        n = len(results)
        total_area_mm2 = sum(r["area_mm2"] for r in results)
        image_area_px = gray.shape[0] * gray.shape[1]
        image_area_mm2 = image_area_px * (scale_mm_per_px ** 2)
        porosity = (total_area_mm2 / image_area_mm2 * 100) if image_area_mm2 > 0 else 0
        summary = {
            "hole_count": int(n), "total_area": float(round(total_area_mm2, 2)),
            "avg_area": float(round(total_area_mm2 / n, 2)) if n > 0 else 0.0,
            "avg_circularity": float(round(float(np.mean(circularities)), 4)) if circularities else 0.0,
            "avg_diameter_mm": float(round(float(np.mean(diameters)), 4)) if diameters else 0.0,
            "max_diameter_mm": float(round(float(max(diameters)), 4)) if diameters else 0.0,
            "min_diameter_mm": float(round(float(min(diameters)), 4)) if diameters else 0.0,
            "porosity_percent": float(round(porosity, 2)),
            "diameters": [float(d) for d in diameters],
            "size_distribution": {"大洞(>10mm)": int(sum(1 for d in diameters if d > 10)),
                                  "中洞(5-10mm)": int(sum(1 for d in diameters if 5 <= d <= 10)),
                                  "小洞(1-5mm)": int(sum(1 for d in diameters if 1 <= d < 5)),
                                  "针孔(<1mm)": int(sum(1 for d in diameters if d < 1))}
        }
        return results, summary, {"gray": gray, "binary": binary, "result": result_img}
