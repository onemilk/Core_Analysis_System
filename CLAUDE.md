# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 核心规则

- **所有生成的代码必须带中文注释**——函数、类、关键逻辑都要用中文说明用途
- **每次用户说结束工作时**——必须重新打包 .exe、确保 GitHub 已同步、保存进度到 memory 文件
- **始终使用中文回复**

## 项目概述

基于 Flask + pywebview 的岩心图像分析桌面应用。原生 Windows 窗口，Python 3.12 引擎，支持孔洞/裂缝/粒度分析。

## 常用命令

```bash
# 运行所有测试
D:/python/python312/python.exe -m pytest tests/ -q

# 启动开发服务器
D:/python/python312/python.exe launch.py

# 打包成 exe（结束时必执行）
D:/python/python312/python.exe -m PyInstaller --name="CoreAnalysisSystem" --onefile --windowed --add-data="templates;templates" --add-data="static;static" --add-data="knowledge.json;." --add-data="best_model.pth;." --add-data="ohters;ohters" --hidden-import=scipy.ndimage --hidden-import=skimage.feature --hidden-import=skimage.segmentation --hidden-import=webview --hidden-import=crackawarenet --hidden-import=unet_model --hidden-import=engine.fracture_dl_model launch.py

# 结束工作流程
1. D:/python/python312/python.exe -m pytest tests/ -q
2. git add -A && git commit -m "描述改动"
3. git push origin master
4. 重新打包 exe
5. 更新 memory 文件
```

## 架构

```
app.py              Flask 路由（/api/analyze, /api/report/generate, /api/knowledge）
launch.py           启动脚本——杀旧进程 → Flask 线程 → pywebview 原生窗口
engine/             分析引擎（纯 Python，不依赖 Flask）
  hole_analyzer.py     灰度阈值 + Otsu + 轮廓提取
  fracture_analyzer.py  自适应阈值 + 距离变换 + 狭长度过滤 + 深度学习融合
  fracture_dl_model.py  CrackAwareNet/Attention U-Net 单例加载器（80MB权重）
  grain_analyzer.py     Otsu + 分水岭 + peak_local_max + D10/D50/D90
  report_generator.py   Jinja2 + matplotlib 生成 HTML 报告
ohters/             深度学习模型定义（CrackAwareNet + Attention U-Net）
best_model.pth      预训练权重文件（80MB）
templates/          Jinja2 模板（index.html, knowledge.html, hole/fracture/grain_report.html）
static/app.js       前端逻辑——上传、API 调用、Chart.js、ROI 多边形
report/             报告输出目录
```

## JSON 序列化

Numpy 类型（float32、int64）不能直接 JSON 序列化。引擎输出已显式转为 Python 原生类型。API 使用 `json.dumps()` 兜底，不要直接用 `jsonify()` 处理含 numpy 的数据。

## ROI 多边形框选

前端 Canvas 叠加在图像上，点击添加顶点，双击闭合。坐标通过 `params.roi_polygon` 传给后端。后端 `cv2.fillPoly` 生成遮罩，多边形外像素设为白色（255,255,255）。

## 报告生成

使用旧项目的 Jinja2 + matplotlib 方案。前端 POST 分析数据到 `/api/report/generate`，服务端调用 `report_generator.py` 生成完整 HTML 报告，保存到 `report/` 目录并用 `webbrowser.open()` 打开。

## 端口管理

Flask 在 `launch.py` 中以 `daemon=False` 线程运行，配合 pywebview 原生窗口。启动前自动清理端口 5000 旧进程。如 API 返回 500 但 test_client 返回 200，说明旧进程在跑旧代码。
