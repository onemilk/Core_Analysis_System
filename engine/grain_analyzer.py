"""GrainAnalyzer — 自适应双模式：LAB颜色分割（鲕粒灰岩） / 灰度闭运算分水岭（结核灰岩）。
"""

import math, cv2, numpy as np
from skimage.segmentation import watershed as sk_watershed


class GrainAnalyzer:
    @staticmethod
    def _analyze_color(bgr, scale_mm_per_px, min_area_px):
        """LAB 颜色分割模式 —— 针对红褐色鲕粒灰岩（downloaded-image.jpg）。
        在 LAB 空间：A>142（偏红）且 L<120（偏暗）→ 深红色鲕粒。
        小核形态学 + 直接连通域 + 椭圆拟合。
        """
        h, w = bgr.shape[:2]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # 颜色阈值：深红色 = A高 + L低
        binary = ((a_ch > 142) & (l_ch < 120)).astype(np.uint8) * 255

        # 小核形态学（鲕粒小，5x5 足够）
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k5, iterations=1)

        # 连通域 + 椭圆拟合
        n_labels, labels = cv2.connectedComponents(binary)
        ellipses = []
        min_a = max(min_area_px, h * w * 0.000005)
        max_a = h * w * 0.005

        for lbl in range(1, n_labels):
            mask = np.uint8(labels == lbl) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < min_a or area > max_a or len(cnt) < 5:
                continue
            ell = cv2.fitEllipse(cnt)
            (cx, cy), (short_ax, long_ax), angle = ell
            aspect = short_ax / long_ax if long_ax > 0 else 0
            if aspect < 0.3:
                continue
            ellipses.append({"ell": ell, "area": area, "cx": cx, "cy": cy,
                             "short_ax": short_ax, "long_ax": long_ax})

        # 重叠合并（鲕粒密度高，近距离合并避免重复）
        ellipses.sort(key=lambda e: e["area"], reverse=True)
        removed = [False] * len(ellipses)
        final_ells = []
        for i in range(len(ellipses)):
            if removed[i]:
                continue
            for j in range(i + 1, len(ellipses)):
                if removed[j]:
                    continue
                dc = math.sqrt((ellipses[i]["cx"] - ellipses[j]["cx"]) ** 2 +
                               (ellipses[i]["cy"] - ellipses[j]["cy"]) ** 2)
                if dc < 8:
                    removed[j] = True
            final_ells.append(ellipses[i])

        return final_ells, binary

    @staticmethod
    def _analyze_grayscale(bgr, scale_mm_per_px, min_area_px, threshold_block):
        """灰度闭运算 + 梯度分水岭模式 —— 针对结核灰岩等低对比度图。"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 大核闭运算
        if min(h, w) < 800:
            close_ks = max(7, min(35, threshold_block + 2)) | 1
        else:
            close_ks = max(25, min(h, w) // 20) | 1
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k_close, iterations=1)

        # Otsu
        _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 腐蚀断桥
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        erode_iters = 2 if min(h, w) < 800 else 3
        eroded = cv2.erode(binary, k3, iterations=erode_iters)

        # 距离变换 + 高斯平滑
        dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return [], binary
        if min(h, w) < 800:
            gauss_ks = max(5, min(15, close_ks)) | 1
        else:
            gauss_ks = max(9, close_ks // 3) | 1
        dist = cv2.GaussianBlur(dist, (gauss_ks, gauss_ks), 0)

        # 高阈值找核
        th_rel = 0.3 if min(h, w) < 800 else 0.2
        _, sure_fg = cv2.threshold(dist, th_rel * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        n_labels, labels_tmp = cv2.connectedComponents(sure_fg)
        min_marker = max(3, h * w * 0.00005)
        valid = set()
        for lbl in range(1, n_labels):
            if np.sum(labels_tmp == lbl) >= min_marker:
                valid.add(lbl)

        markers = np.zeros(dist.shape, dtype=np.int32)
        new_lbl = 1
        for lbl in range(1, n_labels):
            if lbl in valid:
                markers[labels_tmp == lbl] = new_lbl
                new_lbl += 1

        # 膨胀恢复 + 梯度分水岭
        restored = cv2.dilate(eroded, k3, iterations=erode_iters)
        grad_x = cv2.Sobel(closed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(closed, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient = (gradient / gradient.max() * 255).astype(np.uint8)

        labels = sk_watershed(gradient, markers, mask=restored // 255)

        # 椭圆拟合 + 过滤
        ellipses = []
        min_a = max(min_area_px, h * w * 0.0005)
        max_a = h * w * 0.08
        edge_mg = min(h, w) * 0.04

        for lbl in range(1, new_lbl):
            mask = np.uint8(labels == lbl) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < min_a or area > max_a or len(cnt) < 5:
                continue
            ell = cv2.fitEllipse(cnt)
            (cx, cy), (short_ax, long_ax), angle = ell
            aspect = short_ax / long_ax if long_ax > 0 else 0
            if aspect < 0.2:
                continue
            if cx < edge_mg or cx > w - edge_mg or cy < edge_mg or cy > h - edge_mg:
                continue
            ellipses.append({"ell": ell, "area": area, "cx": cx, "cy": cy,
                             "short_ax": short_ax, "long_ax": long_ax})

        # 重叠合并
        ellipses.sort(key=lambda e: e["area"], reverse=True)
        removed = [False] * len(ellipses)
        final_ells = []
        for i in range(len(ellipses)):
            if removed[i]:
                continue
            for j in range(i + 1, len(ellipses)):
                if removed[j]:
                    continue
                dc = math.sqrt((ellipses[i]["cx"] - ellipses[j]["cx"]) ** 2 +
                               (ellipses[i]["cy"] - ellipses[j]["cy"]) ** 2)
                if dc < (ellipses[i]["short_ax"] + ellipses[j]["short_ax"]) / 6:
                    removed[j] = True
            final_ells.append(ellipses[i])

        return final_ells, binary

    @staticmethod
    def _analyze_simple_otsu(bgr, min_area_px):
        """简单 Otsu 反色提取 —— 保底方案，适合高对比度砾石图。"""
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)

        n_labels, labels = cv2.connectedComponents(binary)
        ellipses = []
        min_a = max(min_area_px, h * w * 0.0001)
        max_a = h * w * 0.15

        for lbl in range(1, n_labels):
            mask = np.uint8(labels == lbl) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < min_a or area > max_a or len(cnt) < 5:
                continue
            ell = cv2.fitEllipse(cnt)
            (cx, cy), (short_ax, long_ax), angle = ell
            aspect = short_ax / long_ax if long_ax > 0 else 0
            if aspect < 0.3:
                continue
            ellipses.append({"ell": ell, "area": area, "cx": cx, "cy": cy,
                             "short_ax": short_ax, "long_ax": long_ax})

        # 重叠合并
        ellipses.sort(key=lambda e: e["area"], reverse=True)
        removed = [False] * len(ellipses)
        final_ells = []
        for i in range(len(ellipses)):
            if removed[i]:
                continue
            for j in range(i + 1, len(ellipses)):
                if removed[j]:
                    continue
                dc = math.sqrt((ellipses[i]["cx"] - ellipses[j]["cx"]) ** 2 +
                               (ellipses[i]["cy"] - ellipses[j]["cy"]) ** 2)
                if dc < 10:
                    removed[j] = True
            final_ells.append(ellipses[i])

        return final_ells, binary

    @staticmethod
    def analyze_direct(bgr, scale_mm_per_px=0.05, min_area_px=30, threshold_block=21):
        """自适应双模式颗粒分割。

        LAB 颜色模式：检测到强红褐色特征时自动启用（鲕粒灰岩）
        灰度模式：其他图像使用闭运算+分水岭（结核灰岩等）
        """
        if bgr is None:
            return [], {"error": "Image is None"}, {}

        h, w = bgr.shape[:2]

        # 检测是否为红褐色鲕粒灰岩（大图+A>140占比>25% → LAB颜色模式）
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        a_high_pct = np.sum(lab[:, :, 1] > 140) / lab[:, :, 1].size
        use_color_mode = (a_high_pct > 0.25 and min(h, w) > 1000)

        if use_color_mode:
            final_ells, binary = GrainAnalyzer._analyze_color(bgr, scale_mm_per_px, min_area_px)
        else:
            final_ells, binary = GrainAnalyzer._analyze_grayscale(bgr, scale_mm_per_px, min_area_px, threshold_block)
            # 保底：复杂管线几乎无结果（≤3），回退到简单 Otsu（高对比度砾石图）
            if len(final_ells) <= 3:
                final_ells, binary = GrainAnalyzer._analyze_simple_otsu(bgr, min_area_px)

        # ================================================================
        # Step 1: 大核闭运算 —— 抹平颗粒内部同心环纹理
        # ================================================================
        # 共享结果生成
        # ================================================================
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        results, diameters = [], []
        result_img = bgr.copy()

        for e in final_ells:
            ell = e["ell"]
            area_px = e["area"]
            area_mm2 = float(area_px * (scale_mm_per_px ** 2))
            d_mm = float(2.0 * math.sqrt(area_mm2 / math.pi))

            feret_long = float(e["long_ax"] * scale_mm_per_px)
            feret_short = float(e["short_ax"] * scale_mm_per_px)
            size_cat = GrainAnalyzer._classify_size(d_mm)

            line_thick = max(2, min(h, w) // 500)  # 大图加粗线
            cv2.ellipse(result_img, ell, (0, 255, 0), line_thick)
            if not use_color_mode:
                cx, cy = int(e["cx"]), int(e["cy"])
                cv2.putText(result_img, f"{d_mm:.1f}", (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            diameters.append(d_mm)
            results.append({"area_mm2": area_mm2, "d_mm": d_mm,
                            "feret_long": feret_long, "feret_short": feret_short,
                            "circularity": 1.0, "size": size_cat})

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
        """Legacy method — kept for test compatibility, returns plain dicts."""
        results, diameters = [], []
        for i, region in enumerate(regions):
            area_mm2 = float(region.area_px * (scale_mm_per_px ** 2))
            d = float(2.0 * math.sqrt(area_mm2 / math.pi))
            size_cat = GrainAnalyzer._classify_size(d)
            results.append({"region_index": i, "area_mm2": round(area_mm2,4),
                "equivalent_d_mm": round(d,4), "size_category": size_cat,
                "perimeter_mm": 0.0, "feret_long_mm": d, "feret_short_mm": d,
                "circularity": 1.0})
            diameters.append(d)
        n = len(results)
        size_dist = {"砾":0,"砂":0,"粉砂":0,"泥":0}
        for r in results:
            if r["size_category"] in size_dist: size_dist[r["size_category"]] += 1
        summary = {"total_count": n, "avg_diameter_mm": round(sum(diameters)/n,4) if n else 0.0,
            "md_diameter_mm": round(float(np.percentile(diameters,50)),4) if diameters else 0.0,
            "d10_mm": round(float(np.percentile(diameters,10)),4) if diameters else 0.0,
            "d50_mm": round(float(np.median(diameters)),4) if diameters else 0.0,
            "d90_mm": round(float(np.percentile(diameters,90)),4) if diameters else 0.0,
            "std_dev_mm": round(float(np.std(diameters)),4) if diameters else 0.0,
            "max_diameter_mm": round(max(diameters),4) if diameters else 0.0,
            "min_diameter_mm": round(min(diameters),4) if diameters else 0.0,
            "size_distribution": size_dist, "diameters": diameters}
        return results, summary

    @staticmethod
    def _classify_size(diameter_mm: float) -> str:
        if diameter_mm > 2: return "砾"
        elif diameter_mm >= 0.0625: return "砂"
        elif diameter_mm >= 0.0039: return "粉砂"
        else: return "泥"