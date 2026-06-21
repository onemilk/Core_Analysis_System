# 岩心孔洞裂缝分析系统 v1.0 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 Flask 桌面化岩心分析系统：孔洞/裂缝/粒度分析 + 报告生成 + 知识库，融合学长算法和旧项目引擎。

**Architecture:** Flask Web 后端 + HTML5/Chart.js 前端，引擎层复用旧项目 57 tests，孔洞裂缝用学长灰度和距离变换算法。

**Tech Stack:** Python 3.12, Flask, OpenCV, NumPy, SciPy, matplotlib, Chart.js, SQLite, PyInstaller

---

### Task 1: Project Scaffold + Copy Engine

**Files:** 8 files copied, 2 created

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p engine templates static
```

- [ ] **Step 2: Copy engine modules from old project**

```bash
cp ../岩心分析教学系统/core_analysis/engine/image_processor.py engine/
cp ../岩心分析教学系统/core_analysis/engine/region_extractor.py engine/
cp ../岩心分析教学系统/core_analysis/engine/morphology_engine.py engine/
cp ../岩心分析教学系统/core_analysis/engine/report_generator.py engine/
cp ../岩心分析教学系统/core_analysis/data/models.py engine/
cp ../岩心分析教学系统/core_analysis/engine/grain_analyzer.py engine/
```

- [ ] **Step 3: Create engine/__init__.py**

```python
# Core Analysis Engine
```

- [ ] **Step 4: Copy knowledge base**

```bash
cp ../岩心分析教学系统/core_analysis/data/sedimentary_knowledge.json data/knowledge.json 2>/dev/null || cp ../岩心分析教学系统/core_analysis/data/sedimentary_knowledge.json knowledge.json
```

- [ ] **Step 5: Copy tests**

```bash
cp -r ../岩心分析教学系统/tests/ tests/ 2>/dev/null
```

- [ ] **Step 6: Verify engine imports**

```bash
D:/python/python312/python.exe -c "from engine.image_processor import ImageProcessor; from engine.region_extractor import RegionExtractor; from engine.morphology_engine import MorphologyEngine; from engine.grain_analyzer import GrainAnalyzer; print('Engine OK')"
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: scaffold project, copy engine modules and tests"
```

---

### Task 2: New Hole Analyzer (Fused Algorithm)

**Files:**
- Create: `engine/hole_analyzer.py`
- Create: `tests/test_hole_analyzer.py`

- [ ] **Step 1: Write failing test**

Write `tests/test_hole_analyzer.py`:
```python
"""Tests for hole analyzer."""
import numpy as np, cv2
from engine.hole_analyzer import HoleAnalyzer

def _make_hole_image():
    img = np.ones((120, 160, 3), dtype=np.uint8) * 200
    cv2.circle(img, (40, 60), 18, (30, 30, 30), -1)
    cv2.circle(img, (120, 60), 22, (35, 35, 35), -1)
    return img

class TestHoleAnalyzer:
    def test_analyze_finds_holes(self):
        img = _make_hole_image()
        results, summary, images = HoleAnalyzer.analyze(img, threshold=90, min_area=10, max_area=5000)
        assert summary["hole_count"] >= 2
        assert summary["total_area"] > 0
        assert images["gray"] is not None
        assert images["binary"] is not None
        assert images["result"] is not None

    def test_return_format(self):
        img = _make_hole_image()
        results, summary, images = HoleAnalyzer.analyze(img)
        required = ["hole_count", "total_area", "avg_area", "avg_circularity", "avg_diameter_mm", "porosity_percent", "diameters"]
        for k in required:
            assert k in summary, f"Missing: {k}"

    def test_empty_image(self):
        results, summary, images = HoleAnalyzer.analyze(None)
        assert "error" in summary
```

Run: `D:/python/python312/python.exe -m pytest tests/test_hole_analyzer.py -v`
Expected: FAIL

- [ ] **Step 2: Implement HoleAnalyzer**

Write `engine/hole_analyzer.py`:
```python
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
            results.append({"area_px": area, "area_mm2": area_mm2, "diameter_mm": d_mm, "circularity": circularity})

        n = len(results)
        total_area_mm2 = sum(r["area_mm2"] for r in results)
        image_area_px = gray.shape[0] * gray.shape[1]
        image_area_mm2 = image_area_px * (scale_mm_per_px ** 2)
        porosity = (total_area_mm2 / image_area_mm2 * 100) if image_area_mm2 > 0 else 0
        summary = {
            "hole_count": n, "total_area": total_area_mm2,
            "avg_area": total_area_mm2 / n if n > 0 else 0,
            "avg_circularity": float(np.mean(circularities)) if circularities else 0,
            "avg_diameter_mm": float(np.mean(diameters)) if diameters else 0,
            "max_diameter_mm": max(diameters) if diameters else 0,
            "min_diameter_mm": min(diameters) if diameters else 0,
            "porosity_percent": round(porosity, 2), "diameters": diameters,
            "size_distribution": {"大洞(>10mm)": sum(1 for d in diameters if d > 10),
                                  "中洞(5-10mm)": sum(1 for d in diameters if 5 <= d <= 10),
                                  "小洞(1-5mm)": sum(1 for d in diameters if 1 <= d < 5),
                                  "针孔(<1mm)": sum(1 for d in diameters if d < 1)}
        }
        return results, summary, {"gray": gray, "binary": binary, "result": result_img}
```

Run: `D:/python/python312/python.exe -m pytest tests/test_hole_analyzer.py -v`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add engine/hole_analyzer.py tests/test_hole_analyzer.py
git commit -m "feat: add fused hole analyzer (grayscale threshold + old project classification)"
```

---

### Task 3: New Fracture Analyzer (Fused Algorithm)

**Files:**
- Create: `engine/fracture_analyzer.py`
- Create: `tests/test_fracture_analyzer.py`

- [ ] **Step 1: Write failing test**

Write `tests/test_fracture_analyzer.py`:
```python
"""Tests for fracture analyzer."""
import numpy as np, cv2
from engine.fracture_analyzer import FractureAnalyzer

def _make_crack_image():
    img = np.ones((150, 200, 3), dtype=np.uint8) * 200
    cv2.line(img, (30, 60), (170, 70), (40, 40, 40), 2)
    cv2.line(img, (50, 30), (55, 120), (40, 40, 40), 2)
    return img

class TestFractureAnalyzer:
    def test_analyze_finds_cracks(self):
        img = _make_crack_image()
        results, summary, images = FractureAnalyzer.analyze(img, threshold=60, min_area=50)
        assert summary["crack_count"] > 0
        assert "avg_width" in summary
        assert images["result"] is not None

    def test_return_format(self):
        img = _make_crack_image()
        _, summary, _ = FractureAnalyzer.analyze(img)
        required = ["crack_count", "total_area", "avg_width", "max_length"]
        for k in required:
            assert k in summary, f"Missing: {k}"

    def test_empty_image(self):
        _, summary, _ = FractureAnalyzer.analyze(None)
        assert "error" in summary
```

Run: `D:/python/python312/python.exe -m pytest tests/test_fracture_analyzer.py -v`
Expected: FAIL

- [ ] **Step 2: Implement FractureAnalyzer**

Write `engine/fracture_analyzer.py`:
```python
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

        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
        _, global_thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_or(adaptive, global_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        labeled, _ = ndimage.label(binary)
        sizes = ndimage.sum(binary, labeled, range(nn + 1))
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
            results.append({"area_px": area, "length_px": length, "width_px": w,
                            "solidity": solidity})
            widths.append(w)
            lengths.append(length)

        n = len(results)
        summary = {
            "crack_count": n, "total_area": sum(r["area_px"] for r in results),
            "avg_width": float(np.mean(widths)) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "max_length": max(lengths) if lengths else 0,
            "avg_length": float(np.mean(lengths)) if lengths else 0,
        }
        return results, summary, {"gray": enhanced, "binary": binary, "result": result_img}
```

Run: `D:/python/python312/python.exe -m pytest tests/test_fracture_analyzer.py -v`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add engine/fracture_analyzer.py tests/test_fracture_analyzer.py
git commit -m "feat: add fused fracture analyzer (adaptive threshold + distance transform)"
```

---

### Task 4: Flask Web Skeleton

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write app.py**

Write `app.py`:
```python
"""Core Analysis System — Flask web application."""
import os, io, base64, json
import cv2, numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from engine.hole_analyzer import HoleAnalyzer
from engine.fracture_analyzer import FractureAnalyzer
from engine.grain_analyzer import GrainAnalyzer
from engine.report_generator import ReportGenerator
from engine.image_processor import ImageProcessor as IP
from engine.region_extractor import RegionExtractor
from engine.morphology_engine import MorphologyEngine

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_knowledge():
    path = os.path.join(os.path.dirname(__file__), "knowledge.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"categories": {}}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        img_b64 = data["image"]
        analysis_type = data["type"]
        params = data.get("params", {})

        img_bytes = base64.b64decode(img_b64.split(",")[-1] if "," in img_b64 else img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({"error": "Invalid image"}), 400

        if analysis_type == "hole":
            results, summary, images = HoleAnalyzer.analyze(bgr, **params)
        elif analysis_type == "fracture":
            results, summary, images = FractureAnalyzer.analyze(bgr, **params)
        elif analysis_type == "grain":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            regions = RegionExtractor.extract_by_color_sample(bgr, bgr[gray.shape[0]//2, gray.shape[1]//2], params.get("tolerance", 30))
            regions = MorphologyEngine.denoise_by_area(regions, params.get("denoise", 10))
            grain_results, summary = GrainAnalyzer.analyze(regions, params.get("scale", 0.05), gray.shape[0] * gray.shape[1])
            results = [{"area_mm2": r.area_mm2, "d_mm": r.equivalent_d_mm, "feret_long": r.feret_long_mm, "feret_short": r.feret_short_mm, "circularity": r.circularity, "size": r.size_category} for r in grain_results]
            images = {"result": bgr}
        else:
            return jsonify({"error": "Unknown type"}), 400

        # Encode images for response
        encoded = {}
        for key, img in images.items():
            if img is not None:
                _, buf = cv2.imencode(".png", img)
                encoded[key] = "data:image/png;base64," + base64.b64encode(buf).decode()

        return jsonify({"results": results, "summary": summary, "images": encoded})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/knowledge")
def knowledge():
    return jsonify(load_knowledge())


if __name__ == "__main__":
    app.run(port=5000, debug=True)
```

- [ ] **Step 2: Verify Flask starts**

```bash
D:/python/python312/python.exe -m flask --app app run --port 5000 &
sleep 2
curl http://localhost:5000/ | head -1
kill %1
```
Expected: HTML response.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Flask application with analyze and knowledge endpoints"
```

---

### Task 5: Frontend — Main Interface

**Files:**
- Create: `templates/index.html`
- Create: `static/app.js`
- Modify: `app.py` (add static route if needed)

- [ ] **Step 1: Write HTML template**

Write `templates/index.html`:
```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>岩心孔洞裂缝分析系统</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Microsoft YaHei",sans-serif; display: flex; flex-direction: column; height: 100vh; }
.toolbar { background: #2c3e50; padding: 10px 20px; display: flex; gap: 12px; align-items: center; }
.toolbar button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; background: #3498db; color: #fff; font-size: 14px; }
.toolbar button.active { background: #e74c3c; }
.main { display: flex; flex: 1; overflow: hidden; }
.viewer { flex: 1; padding: 12px; display: flex; flex-direction: column; align-items: center; overflow: auto; background: #1a1a1a; }
.viewer img { max-width: 100%; max-height: 70vh; object-fit: contain; }
.view-tabs { display: flex; gap: 4px; margin-bottom: 8px; }
.view-tabs button { padding: 4px 12px; border: 1px solid #555; background: #333; color: #aaa; cursor: pointer; border-radius: 4px 4px 0 0; font-size: 12px; }
.view-tabs button.current { background: #444; color: #fff; border-bottom: 2px solid #3498db; }
.panel { width: 300px; padding: 16px; overflow-y: auto; background: #f8f9fa; border-left: 1px solid #ddd; }
.panel label { display: block; margin: 8px 0 4px; font-size: 13px; color: #555; }
.panel input, .panel select { width: 100%; padding: 6px; font-size: 13px; margin-bottom: 8px; }
.panel button { width: 100%; padding: 10px; background: #2c3e50; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; margin: 8px 0; }
.summary { background: #fff; padding: 12px; border-radius: 6px; margin: 8px 0; font-size: 12px; }
.summary td { padding: 2px 8px; }
.chart-area { padding: 12px; background: #fff; border-top: 1px solid #ddd; max-height: 200px; }
</style>
</head>
<body>
<div class="toolbar">
  <button id="btnOpen">📁 打开图像</button>
  <input type="file" id="fileInput" accept="image/*" style="display:none">
  <button id="btnHole" class="active">🕳️ 孔洞分析</button>
  <button id="btnFracture">⚡ 裂缝分析</button>
  <button id="btnGrain">📏 粒度分析</button>
  <span style="flex:1"></span>
  <button id="btnKnowledge">📚 知识库</button>
  <button id="btnReport">📊 生成报告</button>
</div>
<div class="main">
  <div class="viewer">
    <div class="view-tabs">
      <button class="current" data-view="result">结果图</button>
      <button data-view="original">原图</button>
      <button data-view="gray">灰度图</button>
      <button data-view="binary">二值图</button>
    </div>
    <img id="mainImage" src="" alt="请打开图像">
  </div>
  <div class="panel">
    <div id="holeParams">
      <label>阈值</label><input type="range" id="threshold" min="50" max="200" value="100">
      <span id="thresholdVal">100</span>
      <label>最小面积(px)</label><input type="number" id="minArea" value="10">
      <label>最大面积(px)</label><input type="number" id="maxArea" value="10000">
      <label>标尺(mm/px)</label><input type="number" id="scale" value="0.05" step="0.01">
    </div>
    <button id="btnAnalyze">🔍 开始分析</button>
    <div class="summary" id="summary"><i>等待分析...</i></div>
    <button id="btnExportJSON">📥 导出JSON</button>
    <button id="btnExportCSV">📊 导出CSV</button>
  </div>
</div>
<div class="chart-area"><canvas id="chart"></canvas></div>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write JavaScript**

Write `static/app.js`:
```javascript
let currentImage = null, currentType = 'hole', resultData = null, imageData = null, chart = null;

document.getElementById('btnOpen').onclick = () => document.getElementById('fileInput').click();
document.getElementById('fileInput').onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = ev => { currentImage = ev.target.result; document.getElementById('mainImage').src = currentImage; };
  reader.readAsDataURL(f);
};

['btnHole','btnFracture','btnGrain'].forEach(id => {
  document.getElementById(id).onclick = function() {
    document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    currentType = this.id === 'btnHole' ? 'hole' : this.id === 'btnFracture' ? 'fracture' : 'grain';
  };
});

document.getElementById('threshold').oninput = function() { document.getElementById('thresholdVal').textContent = this.value; };

document.querySelectorAll('.view-tabs button').forEach(b => {
  b.onclick = function() {
    document.querySelectorAll('.view-tabs button').forEach(x => x.classList.remove('current'));
    this.classList.add('current');
    const view = this.dataset.view;
    if (imageData && imageData[view]) document.getElementById('mainImage').src = imageData[view];
    else if (view === 'original' && currentImage) document.getElementById('mainImage').src = currentImage;
  };
});

document.getElementById('btnAnalyze').onclick = async () => {
  if (!currentImage) return alert('请先打开图像');
  const params = {};
  if (currentType === 'hole') params.threshold = +document.getElementById('threshold').value;
  params.min_area = +document.getElementById('minArea').value;
  params.max_area = +document.getElementById('maxArea').value;
  params.scale_mm_per_px = +document.getElementById('scale').value;

  const res = await fetch('/api/analyze', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({image: currentImage, type: currentType, params})
  });
  const data = await res.json();
  resultData = data;
  imageData = data.images;
  if (data.error) { alert(data.error); return; }

  document.getElementById('mainImage').src = data.images.result || data.images.gray || currentImage;
  let s = '';
  for (const [k,v] of Object.entries(data.summary)) {
    if (k === 'diameters' || k === 'size_distribution') continue;
    s += `<tr><td><b>${k}</b></td><td>${typeof v === 'number' ? v.toFixed(2) : v}</td></tr>`;
  }
  document.getElementById('summary').innerHTML = `<table>${s}</table>`;

  if (data.summary.diameters && data.summary.diameters.length > 0) {
    const ctx = document.getElementById('chart').getContext('2d');
    if (chart) chart.destroy();
    const diameters = data.summary.diameters;
    chart = new Chart(ctx, {
      type: 'bar',
      data: { labels: diameters.map((_,i) => `#${i+1}`), datasets: [{ label: '直径(mm)', data: diameters }] },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }
};

document.getElementById('btnExportJSON').onclick = () => {
  if (!resultData) return;
  const blob = new Blob([JSON.stringify(resultData, null, 2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'analysis.json'; a.click();
};

document.getElementById('btnExportCSV').onclick = () => {
  if (!resultData || !resultData.summary) return;
  let csv = Object.entries(resultData.summary).map(([k,v]) => `${k},${v}`).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'analysis.csv'; a.click();
};
```

- [ ] **Step 3: Verify frontend loads**

```bash
D:/python/python312/python.exe -m flask --app app run --port 5000 &
sleep 2 && curl http://localhost:5000/ 2>/dev/null | grep -o "<title>.*</title>"
kill %1
```
Expected: `<title>岩心孔洞裂缝分析系统</title>`

- [ ] **Step 4: Commit**

```bash
git add templates/ static/ app.py
git commit -m "feat: add main interface with 4-view, params panel, and chart"
```

---

### Task 6: Report Generation + Knowledge Base Page

**Files:**
- Create: `templates/report.html` (copy from old project)
- Modify: `app.py` (+ report route, + knowledge page route)

- [ ] **Step 1: Add report and knowledge routes to app.py**

Add routes before `if __name__`:
```python
@app.route("/report")
def report_page():
    return render_template("report.html")

@app.route("/knowledge")
def knowledge_page():
    return render_template("knowledge.html")
```

- [ ] **Step 2: Copy and adapt report template**

```bash
cp ../岩心分析教学系统/core_analysis/templates/hole_report.html templates/report.html
```

Create `templates/knowledge.html`:
```html
<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>沉积知识库</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;padding:20px;max-width:800px;margin:0 auto}
input{padding:8px;width:100%;margin-bottom:12px;font-size:14px}
.entry{margin:12px 0;padding:12px;border-left:3px solid #3498db;background:#f8f9fa}
.entry h3{margin:0 0 4px}.entry p{margin:4px 0;font-size:13px;color:#555}
.related a{color:#3498db;margin-right:8px;font-size:12px;cursor:pointer}</style></head><body>
<h2>沉积知识库</h2><input type="text" id="search" placeholder="搜索..." oninput="filter()">
<div id="entries"></div>
<script>
fetch('/api/knowledge').then(r=>r.json()).then(data=>{
  window.kb=data.categories;
  renderAll();
});
function renderAll(filter=''){
  document.getElementById('entries').innerHTML='';
  let html='';
  for(const[cat,entries] of Object.entries(window.kb)){
    for(const[name,item] of Object.entries(entries)){
      if(filter && !name.includes(filter) && !(item.definition||'').includes(filter)) continue;
      html+=`<div class="entry"><h3>${name}</h3><p>${item.definition||''}</p>
        <div class="related">${(item.related||[]).map(r=>`<a href="javascript:searchRel('${r}')">${r}</a>`).join('')}</div></div>`;
    }
  }
  document.getElementById('entries').innerHTML=html||'<p>无匹配结果</p>';
}
function filter(){renderAll(document.getElementById('search').value);}
function searchRel(t){document.getElementById('search').value=t;filter();}
</script></body></html>
```

- [ ] **Step 3: Commit**

```bash
git add templates/ app.py
git commit -m "feat: add report and knowledge base pages"
```

---

### Task 7: Launch Script + E2E Test

**Files:**
- Create: `launch.py`
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write launch.py**

```python
"""Launch script — starts Flask and opens browser."""
import os, sys, webbrowser, subprocess, time, threading

def main():
    port = 5000
    threading.Thread(target=lambda: __import__('app').app.run(port=port, debug=False), daemon=True).start()
    time.sleep(1)
    webbrowser.open(f"http://localhost:{port}")
    print(f"系统已启动: http://localhost:{port}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests**

```bash
D:/python/python312/python.exe -m pytest tests/ -q
```

- [ ] **Step 3: Commit and push**

```bash
git add -A && git commit -m "feat: add launch script and E2E verification" && git push origin master 2>&1 || echo "No remote yet"
```

---

## Summary

| Task | Component | Files |
|---|---|---|
| 1 | Scaffold + Copy Engine | engine/, tests/, knowledge.json |
| 2 | Hole Analyzer | engine/hole_analyzer.py + test |
| 3 | Fracture Analyzer | engine/fracture_analyzer.py + test |
| 4 | Flask Skeleton | app.py |
| 5 | Frontend | templates/index.html, static/app.js |
| 6 | Report + Knowledge | templates/report.html, templates/knowledge.html |
| 7 | Launch + E2E | launch.py + all tests |
