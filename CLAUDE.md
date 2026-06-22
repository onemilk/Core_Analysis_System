# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flask-based desktop analysis system for rock core images. Browser-based UI, Python 3.12 engine. Single-user teaching tool for hole/fracture/grain analysis.

## Common Commands

```bash
# Run all tests
D:/python/python312/python.exe -m pytest tests/ -q

# Run single test file
D:/python/python312/python.exe -m pytest tests/test_hole_analyzer.py -v

# Start development server (auto-kills old processes on port 5000)
D:/python/python312/python.exe launch.py

# Build standalone exe
D:/python/python312/python.exe -m PyInstaller --name="CoreAnalysisSystem" --onefile --windowed --add-data="templates;templates" --add-data="static;static" --add-data="knowledge.json;." --hidden-import=scipy.ndimage launch.py
```

## Architecture

```
app.py              Flask routes (/, /api/analyze, /api/knowledge, /report, /knowledge)
launch.py           Entry point — kills old port 5000 processes, starts Flask, opens browser
engine/             Analysis modules (no Flask dependency)
  hole_analyzer.py    Grayscale threshold + contour detection (senior project algorithm)
  fracture_analyzer.py Adaptive threshold + distance transform + solidity filtering
  grain_analyzer.py   Feret diameter + circularity + Udden-Wentworth classification
  image_processor.py  Filters, rotation, edge detection
  region_extractor.py LAB color segmentation
  morphology_engine.py Dilate/erode/denoise
  report_generator.py Jinja2 HTML reports + matplotlib charts
templates/          Jinja2 HTML templates (index.html, report.html, knowledge.html)
static/app.js       All frontend logic — file upload, API calls, Chart.js, ROI polygon, report generation
knowledge.json      34-entry sedimentary knowledge base
```

## JSON Serialization

Numpy types (float32, int64) are not JSON-serializable. Engine outputs were converted to native Python types (`float()`, `int()`) at source. The API route uses a recursive `_clean()` function + `json.dumps()` for safety. Do NOT use `jsonify()` directly on numpy-containing data.

## ROI Polygon

Frontend: HTML5 Canvas overlay on the image. User clicks to add vertices, double-click to close. Polygon coordinates (in natural image pixels) are sent as `params.roi_polygon` to the API. Backend: `cv2.fillPoly` creates white mask, bgr pixels outside polygon are set to white (255,255,255) before analysis. Blue border drawn on result image after analysis.

## Port Management

Multiple Flask instances accumulate on port 5000 because Windows taskkill via bash/Cygwin is unreliable. `launch.py` now auto-kills existing listeners on port 5000 before starting. If live API returns 500 but test_client returns 200, old processes are running stale code — kill all Python and restart.
