# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

基于 Flask 的岩心图像分析桌面化系统。浏览器界面，Python 3.12 引擎，单机教学工具，支持孔洞/裂缝/粒度分析。

## 常用命令

```bash
# 运行所有测试
D:/python/python312/python.exe -m pytest tests/ -q

# 运行单个测试文件
D:/python/python312/python.exe -m pytest tests/test_hole_analyzer.py -v

# 启动开发服务器（自动杀掉端口 5000 上的旧进程）
D:/python/python312/python.exe launch.py

# 打包成 exe
D:/python/python312/python.exe -m PyInstaller --name="CoreAnalysisSystem" --onefile --windowed --add-data="templates;templates" --add-data="static;static" --add-data="knowledge.json;." --hidden-import=scipy.ndimage launch.py
```

## 架构

```
app.py              主文件，Flask 路由（/, /api/analyze, /api/knowledge, /report, /knowledge）
launch.py           启动脚本——自动杀旧进程 → 启动 Flask → 打开浏览器
engine/             分析引擎（不依赖 Flask，旧项目直接搬来）
  hole_analyzer.py    灰度阈值 + 轮廓提取（学长算法）
  fracture_analyzer.py 自适应阈值 + 距离变换测宽 + 实体度过滤（学长算法）
  grain_analyzer.py   Feret 直径 + 圆度 + Udden-Wentworth 分类（旧项目引擎）
  image_processor.py  滤波器、旋转、边缘检测
  region_extractor.py  LAB 颜色空间分割
  morphology_engine.py 膨胀/腐蚀/去噪
templates/          Jinja2 模板（index.html, report.html, knowledge.html）
static/app.js       前端全部逻辑——文件上传、API 调用、Chart.js 图表、ROI 多边形框选、报告生成
knowledge.json      沉积学知识库（6 分类 34 条目）
```

## JSON 序列化注意事项

Numpy 类型（float32、int64）不能直接 JSON 序列化。引擎输出已在源头显式转为 Python 原生类型（`float()`、`int()`）。API 路由使用递归 `_clean()` 函数 + `json.dumps()` 兜底。不要直接用 `jsonify()` 处理包含 numpy 数据的结果。

## ROI 多边形框选

前端：HTML5 Canvas 覆盖在图像上，点击添加顶点，双击闭合。多边形坐标（按原图像素计）通过 `params.roi_polygon` 传给后端。后端：`cv2.fillPoly` 生成白色遮罩，多边形外像素设为白色（255,255,255）再分析，结果图上用蓝色边框绘出区域。

## 端口管理

多个 Flask 实例容易堆积在端口 5000，因为通过 bash/Cygwin 的 taskkill 杀不干净。`launch.py` 现在启动前自动清理端口 5000 上的旧进程。如果线上 API 返回 500 但 test_client 返回 200，说明旧进程在跑旧代码——杀干净 Python 进程后重试。
