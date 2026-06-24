"""Fracture analyzer — 亮暗双路径分流 + 深度学习融合（仅亮图）。

亮暗分界阈值: brightness_median = 140
  暗图 (≤140): 原裂缝算法（CLAHE→Bilateral→自适应+全局阈值），不做DL融合
  亮图 (>140): 改进管线（+黑帽变换 + CrackAwareNet DL融合）
"""

import cv2, numpy as np, math
from scipy import ndimage
from engine.fracture_dl_model import FractureDLModel

# 亮暗分界阈值（灰度中值）
BRIGHTNESS_THRESHOLD = 140


class FractureAnalyzer:
    @staticmethod
    def analyze(image, threshold=80, min_area=30, max_area=float('inf'),
                min_elongation=None, use_dl=True, scale_mm_per_px=0.05):
        if image is None:
            return [], {"error": "Image is None"}, {}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.median(gray))  # 图像亮度中值
        # 图像纹理强度（灰度标准差）—— 纹理越强，自适应阈值需越保守
        texture_level = float(np.std(gray))
        dl_available = False

        # 自适应阈值 C 值 —— 根据图像类型和纹理强度自动调节
        # 亮图：背景亮、裂缝暗，但暗色噪点也多 → C 偏保守 (4~8)
        # 暗图：背景暗、裂缝更暗，暗色噪点少 → C 偏敏感 (2~4)
        if brightness > BRIGHTNESS_THRESHOLD:
            # 亮图：以黑帽+DL为主力，自适应阈值做辅助，C取保守值
            if texture_level < 15:
                adaptive_c = 2   # 极干净图，保留敏感度
            elif texture_level > 40:
                adaptive_c = 6   # 纹理重图，强力抑制噪点
            else:
                adaptive_c = 4   # 中等纹理
        else:
            # 暗图：自适应阈值为主力，C取敏感值捕获微弱裂缝
            adaptive_c = 3 if texture_level > 50 else 2

        # ================================================================
        # 预处理阶段 —— 亮暗图走完全不同的管线
        # ================================================================

        if brightness > BRIGHTNESS_THRESHOLD:
            # ==========================================
            # 亮图管线：DL 优先（DL 模型专训裂缝识别，天然抗纹理噪点）
            # DL 可用时直接用 DL 掩码替代 CV；不可用时回退到 CV
            # ==========================================

            enhanced = gray  # 默认（DL模式不需要CLAHE增强）
            if use_dl and FractureDLModel.is_available():
                # DL 直接输出裂缝掩码（大图自动分块推理）
                binary = FractureDLModel.predict(image, threshold=0.3)
                dl_available = True
            else:
                # CV 回退：CLAHE + Bilateral + 自适应+全局 + 黑帽
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
                adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 11, adaptive_c)
                _, global_thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
                binary = cv2.bitwise_or(adaptive, global_thresh)
                kernel_n = max(3, min(gray.shape[0], gray.shape[1]) // 100)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_n, kernel_n))
                blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
                bh_thresh = max(30.0, float(np.median(blackhat)) * 2.5)
                _, bh_binary = cv2.threshold(blackhat, bh_thresh, 255, cv2.THRESH_BINARY)
                binary = cv2.bitwise_or(binary, bh_binary)

        else:
            # ==========================================
            # 暗图管线：原裂缝算法，不做任何改动
            # CLAHE → Bilateral → 自适应阈值 + 全局阈值
            # 暗图 DL 模型不参与（避免引入噪声误检）
            # ==========================================

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)

            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, adaptive_c)
            _, global_thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_or(adaptive, global_thresh)

        # ================================================================
        # 后处理阶段 —— 亮暗图共享（形态学 → 标签过滤 → 轮廓 → 形状筛选）
        # ================================================================

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
            # 亮暗自适应阈值：亮图偏严格(1.5)过滤圆形噪点，暗图偏宽松(1.15)避免漏检短裂缝
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            if min(rw, rh) > 0:
                elongation = max(rw, rh) / min(rw, rh)
            else:
                elongation = 1.0
            _min_elong = min_elongation if min_elongation is not None else (1.5 if brightness > 140 else 1.15)
            if elongation < _min_elong:
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
            "dl_available": dl_available,
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
