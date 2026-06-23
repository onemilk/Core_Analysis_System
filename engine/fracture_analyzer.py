"""Fracture analyzer — adaptive threshold + distance transform (senior project algorithm)."""
import cv2, numpy as np, math
from scipy import ndimage

class FractureAnalyzer:
    @staticmethod
    def analyze(image, threshold=80, min_area=30, max_area=float('inf'), min_elongation=1.5, scale_mm_per_px=0.05):
        if image is None:
            return [], {"error": "Image is None"}, {}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.median(gray))  # 图像亮度中值

        # 统一的预处理管线：CLAHE 增强 → 双边滤波（保边缘）→ 自适应+全局阈值
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # 双边滤波保留裂缝边缘（裂缝很细，高斯模糊会将其抹入背景）
        blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
        _, global_thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_or(adaptive, global_thresh)

        if brightness > 140:
            # 明亮图片额外处理：黑帽变换提取亮背景下的暗色裂缝（标准CV技术）
            kernel_n = max(3, min(gray.shape[0], gray.shape[1]) // 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_n, kernel_n))
            blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
            _, bh_binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 合并黑帽结果与自适应阈值结果，确保裂缝完整提取
            binary = cv2.bitwise_or(binary, bh_binary)

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
            # 实体度过滤：裂缝为凹形/细长，实体度应低于阈值
            max_solidity = 0.95 if brightness > 140 else 0.85
            if solidity >= max_solidity:
                continue
            # 狭长度过滤：裂缝应为狭长形状，最小外接旋转矩形长宽比 >= min_elongation
            # 这是裂缝区别于孔洞/噪点的核心形态特征
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            if min(rw, rh) > 0:
                elongation = max(rw, rh) / min(rw, rh)
            else:
                elongation = 1.0
            if elongation < min_elongation:
                continue

            length = cv2.arcLength(cnt, True)
            crack_mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(crack_mask, [cnt], -1, 1, -1)
            dist = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)
            w = np.max(dist * 2) if np.any(crack_mask) else 0

            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
            results.append({"area_px": area, "length_px": length, "width_px": w, "solidity": solidity, "elongation": elongation})
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
                "solidity": float(r["solidity"]),
                "elongation": float(r["elongation"])
            })
        return clean_results, summary, {"gray": enhanced, "binary": binary, "result": result_img}
