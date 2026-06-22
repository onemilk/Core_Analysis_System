"""Core Analysis System — Flask web application."""
import os, io, base64, json
import cv2, numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from engine.hole_analyzer import HoleAnalyzer
from engine.fracture_analyzer import FractureAnalyzer
from engine.grain_analyzer import GrainAnalyzer
from engine.report_generator import ReportGenerator
from engine.region_extractor import RegionExtractor
from engine.morphology_engine import MorphologyEngine

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

from flask.json.provider import DefaultJSONProvider
import numpy as np

class NumpyProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

app.json = NumpyProvider(app)

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

        # Apply ROI polygon: white-out outside, draw green border
        roi_polygon = params.pop("roi_polygon", None)
        if roi_polygon:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in roi_polygon], dtype=np.int32)
            roi_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [pts], 255)
            bgr[roi_mask == 0] = (255, 255, 255)

        if analysis_type == "hole":
            results, summary, images = HoleAnalyzer.analyze(bgr, **params)
        elif analysis_type == "fracture":
            results, summary, images = FractureAnalyzer.analyze(bgr, **params)
        elif analysis_type == "grain":
            results, summary, images = GrainAnalyzer.analyze_direct(bgr,
                scale_mm_per_px=params.get("scale", 0.05),
                min_area_px=params.get("min_area", 50),
                threshold_block=params.get("block_size", 21))
        else:
            return jsonify({"error": "Unknown type"}), 400

        # Draw ROI border on result image
        if roi_polygon and "result" in images:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in roi_polygon], dtype=np.int32)
            cv2.polylines(images["result"], [pts], True, (255, 0, 0), 2)  # blue border for ROI

        encoded = {}
        for key, img in images.items():
            if img is not None:
                _, buf = cv2.imencode(".png", img)
                encoded[key] = "data:image/png;base64," + base64.b64encode(buf).decode()

        import json as _j
        def _clean(o):
            import numpy as _n
            if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [_clean(i) for i in o]
            if isinstance(o, (_n.integer,)): return int(o)
            if isinstance(o, (_n.floating,)): return float(o)
            if isinstance(o, _n.ndarray): return o.tolist()
            if hasattr(o, 'item') and callable(o.item): return o.item()
            return o
        return _j.dumps(_clean({"results": results, "summary": summary, "images": encoded})), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/knowledge")
def knowledge():
    return jsonify(load_knowledge())

@app.route("/api/report/generate", methods=["POST"])
def generate_report():
    """Generate professional HTML report using Jinja2 + matplotlib, save to report/."""
    import os, sys
    data = request.json
    analysis_type = data.get("type", "hole")
    summary = data.get("summary", {})
    results = data.get("results", [])
    info = {"image_id":"","well":"","depth":"","layer":"","lithology":"","scale":"","date":"","analyst":""}
    from datetime import datetime
    info["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    if analysis_type == "hole":
        # Map new keys to old report format
        s = {"total_count": summary.get("hole_count",0),
             "total_area_mm2": summary.get("total_area",0),
             "avg_area_mm2": summary.get("avg_area",0),
             "porosity_percent": summary.get("porosity_percent",0),
             "avg_equivalent_d_mm": summary.get("avg_diameter_mm",0),
             "max_equivalent_d_mm": summary.get("max_diameter_mm",0),
             "min_equivalent_d_mm": summary.get("min_diameter_mm",0),
             "size_distribution": summary.get("size_distribution",{}),
             "diameters": summary.get("diameters",[])}
        fill_stats = [{"status":"未充填","count":s["total_count"],"area":s["total_area_mm2"],"percent":100}]
        effect = {"valid":s["total_count"],"semi_valid":0,"invalid":0}
        html = ReportGenerator.generate_hole_report(s, fill_stats, effect, info)
    elif analysis_type == "fracture":
        s = {"total_count": summary.get("crack_count",0),
             "total_area_mm2": summary.get("total_area",0),
             "porosity_percent": 0,
             "total_length_mm": summary.get("avg_length",0) * summary.get("crack_count",1),
             "surface_density": 0, "linear_density": 0, "avg_spacing_mm": 0}
        fractures = [{"length_mm":r.get("length_px",0),"width_mm":r.get("width_px",0),"area_mm2":r.get("area_px",0),"fracture_type":"构造缝","fill_status":"张开缝","effectiveness":"有效"} for r in results]
        type_stats = [{"type":"构造缝","count":len(results),"total_length":sum(r.get("length_px",0) for r in results)}]
        html = ReportGenerator.generate_fracture_report(s, fractures, type_stats, info)
    else:
        s = {"total_count": summary.get("total_count",0),
             "avg_diameter_mm": summary.get("avg_diameter_mm",0),
             "md_diameter_mm": summary.get("d50_mm",0),
             "std_dev_mm": summary.get("std_dev_mm",0),
             "max_diameter_mm": summary.get("max_diameter_mm",0),
             "min_diameter_mm": summary.get("min_diameter_mm",0),
             "size_distribution": summary.get("size_distribution",{}),
             "diameters": summary.get("diameters",[]),
             "feret_data": [(r.get("feret_long",0), r.get("feret_short",0)) for r in results]}
        html = ReportGenerator.generate_grain_report(s, info)

    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(base_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    filename = f"岩心分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    filepath = os.path.join(report_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    # Open in browser
    import webbrowser
    webbrowser.open('file:///' + filepath.replace('\\', '/'))
    return jsonify({"status": "ok", "path": filepath, "filename": filename})

@app.route("/report")
def report_page():
    return render_template("report.html",
        info={"image_id":"","well":"","depth":"","layer":"","lithology":"","scale":"","date":"","analyst":""},
        summary={"total_count":0,"total_area_mm2":0,"avg_area_mm2":0,"porosity_percent":0,"avg_d_mm":0,"max_d_mm":0,"min_d_mm":0},
        fill_stats=[], effect={"valid":0,"semi_valid":0,"invalid":0},
        size_dist=[], charts={})

@app.route("/knowledge")
def knowledge_page():
    return render_template("knowledge.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
