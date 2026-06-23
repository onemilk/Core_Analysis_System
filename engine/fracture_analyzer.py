"""裂缝分析器 — 自适应阈值 + 距离变换测宽 + 亮度自适应 + 形状过滤。"""
import cv2, numpy as np, math
from scipy import ndimage

class FractureAnalyzer:
    @staticmethod
    def analyze(image, threshold=80, min_area=100, max_area=float('inf'), scale_mm_per_px=0.05):
        """分析裂缝。threshold 为基准值，实际阈值会根据图像亮度自动调整。"""
        if image is None:
            return [], {"error": "Image is None"}, {}

        # 1. 预处理：灰度化 + 增强对比度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 根据图像亮度自适应调整 CLAHE
        brightness = float(np.median(gray))
        clip_limit = 3.0 if brightness > 150 else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # 高斯滤波去噪（保留边缘比双边滤波更好）
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 2. 边缘检测 + 阈值：根据亮度自适应选择策略
        if brightness > 140:
            # 明亮图片：Canny 降低阈值 + Otsu 辅助捕捉细裂缝
            canny_low = max(15, int(30 * 100 / brightness))
            edges = cv2.Canny(blurred, canny_low, canny_low * 2)
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_or(edges, thresh)
            binary = cv2.bitwise_or(binary, otsu)  # 明亮图片用 Otsu 补充
        else:
            # 暗色图片：Canny 标准阈值 + 自适应阈值 + 全局阈值
            edges = cv2.Canny(blurred, 30, 100)
            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
            _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_or(edges, thresh)
            binary = cv2.bitwise_or(binary, adaptive)

        # 3. 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4. 区域生长去噪（去掉小斑点）
        labeled, num_features = ndimage.label(binary)
        sizes = ndimage.sum(binary, labeled, range(num_features + 1))
        mask = sizes > int(min_area / 10)
        binary = (mask[labeled]).astype(np.uint8) * 255

        # 5. 轮廓提取 + 形状过滤
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_img = image.copy()
        results, widths, lengths = [], [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            # 凸包分析：实体度 = 面积/凸包面积，裂缝通常 < 0.5
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            # 实体度过滤：裂缝实体度低，拒绝圆形/块状误检
            if solidity >= 0.65:
                continue

            # 长宽比过滤：裂缝是细长形状，拒绝块状区域
            rect = cv2.minAreaRect(cnt)
            feret_long = max(rect[1])
            feret_short = min(rect[1])
            aspect = feret_long / max(feret_short, 1.0)
            if aspect < 2.0:
                continue
                continue

            length = cv2.arcLength(cnt, True)
            # 距离变换计算裂缝宽度
            crack_mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(crack_mask, [cnt], -1, 1, -1)
            dist = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)
            w = float(np.max(dist * 2)) if np.any(crack_mask) else 0.0

            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
            results.append({"area_px": float(area), "length_px": float(length),
                           "width_px": w, "solidity": float(solidity)})
            widths.append(w)
            lengths.append(length)

        # 6. 统计汇总
        n = len(results)
        summary = {
            "crack_count": int(n),
            "total_area": float(sum(r["area_px"] for r in results)),
            "avg_width": float(round(float(np.mean(widths)), 2)) if widths else 0.0,
            "max_width": float(round(float(max(widths)), 2)) if widths else 0.0,
            "max_length": float(round(float(max(lengths)), 2)) if lengths else 0.0,
            "avg_length": float(round(float(np.mean(lengths)), 2)) if lengths else 0.0,
        }
        return results, summary, {"gray": enhanced, "binary": binary, "result": result_img}
