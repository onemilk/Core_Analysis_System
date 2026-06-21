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

def _sanitize(obj):
    """Recursively convert numpy types to Python native types."""
    import numpy as np
    if isinstance(obj, dict): return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_sanitize(i) for i in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if hasattr(obj, 'item'): return obj.item()
    return str(obj) if type(obj).__module__ == 'numpy' else obj

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
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            center_color = bgr[gray.shape[0]//2, gray.shape[1]//2]
            regions = RegionExtractor.extract_by_color_sample(bgr, center_color, params.get("tolerance", 30))
            regions = MorphologyEngine.denoise_by_area(regions, params.get("denoise", 10))
            grain_results, summary = GrainAnalyzer.analyze(regions, params.get("scale", 0.05), gray.shape[0] * gray.shape[1])
            results = [{"area_mm2": r.area_mm2, "d_mm": r.equivalent_d_mm, "feret_long": r.feret_long_mm, "feret_short": r.feret_short_mm, "circularity": r.circularity, "size": r.size_category} for r in grain_results]
            images = {"result": bgr}
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

        import json as _json
        def _conv(o):
            if hasattr(o, 'item'): return o.item()
            if hasattr(o, 'tolist'): return o.tolist()
            return float(o)
        from flask import Response
        return Response(_json.dumps({"results": results, "summary": summary, "images": encoded}, default=_conv), mimetype='application/json')
    except Exception as e:
        import json as _json, traceback
        from flask import Response
        return Response(_json.dumps({"error": str(e), "trace": traceback.format_exc()}), status=500, mimetype='application/json')

@app.route("/api/knowledge")
def knowledge():
    return jsonify(load_knowledge())

@app.route("/report")
def report_page():
    return render_template("report.html")

@app.route("/knowledge")
def knowledge_page():
    return render_template("knowledge.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
