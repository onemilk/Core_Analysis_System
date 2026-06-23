"""Fracture analyzer — adaptive threshold + distance transform (senior project algorithm)."""
import cv2, numpy as np, math
from scipy import ndimage

class FractureAnalyzer:
    @staticmethod
    def analyze(image, threshold=80, min_area=100, max_area=float('inf'), scale_mm_per_px=0.05):
        if image is None:
            return [], {"error": "Image is None"}, {}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
        # 计算图像亮度，明亮图片用 Otsu（效果等同孔洞分析的阈值法）
        brightness = float(np.median(gray))

        if brightness > 140:
            # 明亮图片：Otsu 自动阈值（与孔洞分析相同的算法）
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # 暗色图片：原算法（自适应 + 全局阈值）
            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
            _, global_thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_or(adaptive, global_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        labeled, num_features = ndimage.label(binary)
        sizes = ndimage.sum(binary, labeled, range(num_features + 1))
        mask = sizes > int(min_area / 10)
        binary = (mask[labeled]).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_img = image.copy()
        results, widths, lengths = [], [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity >= 0.7:
                continue

            length = cv2.arcLength(cnt, True)
            crack_mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(crack_mask, [cnt], -1, 1, -1)
            dist = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)
            w = np.max(dist * 2) if np.any(crack_mask) else 0

            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
            results.append({"area_px": area, "length_px": length, "width_px": w, "solidity": solidity})
            widths.append(w)
            lengths.append(length)

        n = len(results)
        summary = {
            "crack_count": int(n),
            "total_area": float(sum(r["area_px"] for r in results)),
            "avg_width": float(round(float(np.mean(widths)), 2)) if widths else 0.0,
            "max_width": float(round(float(max(widths)), 2)) if widths else 0.0,
            "max_length": float(round(float(max(lengths)), 2)) if lengths else 0.0,
            "avg_length": float(round(float(np.mean(lengths)), 2)) if lengths else 0.0,
        }
        # Ensure all results use native Python types
        clean_results = []
        for r in results:
            clean_results.append({
                "area_px": float(r["area_px"]),
                "length_px": float(r["length_px"]),
                "width_px": float(r["width_px"]),
                "solidity": float(r["solidity"])
            })
        return clean_results, summary, {"gray": enhanced, "binary": binary, "result": result_img}
