let currentImage = null, currentType = 'hole', resultData = null, imageData = null, chart = null;
const mainImg = document.getElementById('mainImage');
const roiCanvas = document.getElementById('roiCanvas');
const roiCtx = roiCanvas.getContext('2d');
const clearRoiBtn = document.getElementById('btnClearROI');
let roiMode = false, roiPoints = [], roiPolygon = null;

document.getElementById('btnOpen').onclick = () => document.getElementById('fileInput').click();
document.getElementById('fileInput').onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = ev => { currentImage = ev.target.result; mainImg.src = currentImage; };
  reader.readAsDataURL(f);
};

// Auto-fit ROI canvas when image loads
mainImg.onload = () => {
  roiCanvas.width = mainImg.clientWidth || mainImg.naturalWidth || 400;
  roiCanvas.height = mainImg.clientHeight || mainImg.naturalHeight || 300;
};

['btnHole','btnFracture','btnGrain'].forEach(id => {
  document.getElementById(id).onclick = function() {
    document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    currentType = this.id === 'btnHole' ? 'hole' : this.id === 'btnFracture' ? 'fracture' : 'grain';
    document.getElementById('holeParams').style.display = (currentType==='hole'||currentType==='fracture')?'':'none';
    document.getElementById('grainParams').style.display = currentType==='grain'?'':'none';
  };
});
document.getElementById('blockSize').oninput = function() { document.getElementById('blockSizeVal').textContent = this.value; };

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

async function runAnalysis() {
  if (!currentImage) { alert('请先打开图像'); return; }
  if (roiMode) { alert('请双击闭合多边形后再分析'); return; }
  const params = {};
  if (currentType === 'hole' || currentType === 'fracture') {
    params.threshold = +document.getElementById('threshold').value;
    params.min_area = +document.getElementById('minArea').value;
    params.max_area = +document.getElementById('maxArea').value;
    params.scale_mm_per_px = +document.getElementById('scale').value;
  } else if (currentType === 'grain') {
    params.min_area = +document.getElementById('grainMinArea').value;
    params.block_size = +document.getElementById('blockSize').value;
    params.scale = +document.getElementById('grainScale').value;
  }
  if (roiPolygon && roiPolygon.length >= 3) params.roi_polygon = roiPolygon;

  try {
    const res = await fetch('/api/analyze', {method:'POST',headers:{'Content-Type':'application/json'}, body:JSON.stringify({image:currentImage,type:currentType,params})});
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    resultData = data;
    imageData = data.images;
    mainImg.src = data.images.result || data.images.gray || currentImage;
    let s = '';
    for (const [k,v] of Object.entries(data.summary)) {
      if (k === 'diameters' || k === 'size_distribution') continue;
      s += '<tr><td><b>'+k+'</b></td><td>'+(typeof v==='number'?v.toFixed(2):v)+'</td></tr>';
    }
    document.getElementById('summary').innerHTML = '<table>'+s+'</table>';
    if (data.summary.diameters && data.summary.diameters.length > 0) {
      const ctx = document.getElementById('chart').getContext('2d');
      if (chart) chart.destroy();
      chart = new Chart(ctx, {type:'bar', data:{labels:data.summary.diameters.map((_,i)=>'#'+(i+1)), datasets:[{label:'直径(mm)',data:data.summary.diameters}]}, options:{responsive:true,maintainAspectRatio:false}});
    }
  } catch(e) { alert('分析请求失败: '+e.message); }
}
document.getElementById('btnAnalyze').onclick = runAnalysis;

document.getElementById('btnExportJSON').onclick = () => {
  if (!resultData) return;
  const blob = new Blob([JSON.stringify(resultData, null, 2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'analysis.json'; a.click();
};

document.getElementById('btnExportCSV').onclick = () => {
  if (!resultData || !resultData.summary) return;
  let csv = 'Key,Value\n' + Object.entries(resultData.summary).map(([k,v]) => `"${k}","${v}"`).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'analysis.csv'; a.click();
};

// ── ROI Polygon Selection ──
document.getElementById('btnROI').onclick = () => {
  roiMode = !roiMode;
  if (roiMode) {
    if (!currentImage) return alert('请先打开图像');
    roiCanvas.width = mainImg.clientWidth || mainImg.naturalWidth || 400;
    roiCanvas.height = mainImg.clientHeight || mainImg.naturalHeight || 300;
    roiCanvas.style.display = 'block';
    roiPoints = [];
    roiPolygon = null;
    mainImg.style.outline = '3px solid orange';
    document.getElementById('btnROI').textContent = '📐 点击岩石边缘(双击闭合)';
    document.getElementById('btnROI').style.background = '#e74c3c';
  } else {
    exitRoiMode();
  }
};

roiCanvas.ondblclick = (e) => {
  if (!roiMode || roiPoints.length < 3) return;
  roiPolygon = [...roiPoints];
  drawRoiPreview();
  document.getElementById('btnROI').textContent = '📐 框选区域';
  document.getElementById('btnROI').style.background = '#27ae60';
  clearRoiBtn.style.display = 'inline-block';
  roiMode = false;
  roiCanvas.style.display = 'none';
};

roiCanvas.onclick = (e) => {
  if (!roiMode) return;
  const rect = roiCanvas.getBoundingClientRect();
  const sx = mainImg.clientWidth;
  const sy = mainImg.clientHeight;
  const nx = mainImg.naturalWidth;
  const ny = mainImg.naturalHeight;
  const x = (e.clientX - rect.left) * nx / sx;
  const y = (e.clientY - rect.top) * ny / sy;
  roiPoints.push({x, y});
  drawRoiPreview();
};

function drawRoiPreview() {
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  if (roiPoints.length < 2) return;
  roiCtx.beginPath();
  roiCtx.moveTo(roiPoints[0].x * roiCanvas.width / mainImg.naturalWidth, roiPoints[0].y * roiCanvas.height / mainImg.naturalHeight);
  for (let i = 1; i < roiPoints.length; i++) {
    roiCtx.lineTo(roiPoints[i].x * roiCanvas.width / mainImg.naturalWidth, roiPoints[i].y * roiCanvas.height / mainImg.naturalHeight);
  }
  if (roiPolygon) roiCtx.closePath();
  roiCtx.strokeStyle = '#00ff00';
  roiCtx.lineWidth = 2;
  roiCtx.stroke();
  roiCtx.fillStyle = 'rgba(0,255,0,0.1)';
  if (roiPolygon) roiCtx.fill();
  // Draw vertex dots
  (roiPolygon || roiPoints).forEach(p => {
    const cx = p.x * roiCanvas.width / mainImg.naturalWidth;
    const cy = p.y * roiCanvas.height / mainImg.naturalHeight;
    roiCtx.beginPath();
    roiCtx.arc(cx, cy, 4, 0, Math.PI*2);
    roiCtx.fillStyle = '#00ff00';
    roiCtx.fill();
  });
}

clearRoiBtn.onclick = () => {
  roiPolygon = null;
  roiPoints = [];
  clearRoiBtn.style.display = 'none';
  document.getElementById('btnROI').style.background = '#3498db';
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
};

function exitRoiMode() {
  roiMode = false;
  roiCanvas.style.display = 'none';
  mainImg.style.outline = '';
  document.getElementById('btnROI').textContent = '📐 框选区域';
  document.getElementById('btnROI').style.background = '#3498db';
}

document.getElementById('btnKnowledge').onclick = () => window.open('/knowledge', '_blank');
document.getElementById('btnReport').onclick = () => {
  if (!resultData) return alert('请先完成分析');
  const d = resultData, s = d.summary || {};
  const now = new Date().toLocaleString();
  const css = '<style>body{font-family:"Microsoft YaHei",sans-serif;padding:30px;color:#333;max-width:900px;margin:0 auto}'+
    'h1{text-align:center;font-size:20px;border-bottom:2px solid #2c3e50;padding-bottom:12px}'+
    'h2{font-size:16px;border-bottom:1px solid #ccc;margin-top:28px;padding-bottom:6px}'+
    'table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}'+
    'th,td{border:1px solid #999;padding:6px 10px;text-align:left}'+
    'th{background:#2c3e50;color:#fff} tr:nth-child(even){background:#f5f5f5}'+
    '.info td{border:none;padding:4px 10px} .chart{text-align:center;margin:20px 0}'+
    '.chart img{max-width:100%;border:1px solid #ddd}'+
    '.footer{text-align:center;color:#999;margin-top:30px;font-size:12px}</style>';
  let html = '<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8><title>岩心分析报告</title>'+css+'</head><body>';

  // ── Hole Report ──
  if (currentType === 'hole') {
    html += '<h1>碳酸盐岩岩心孔洞分析报告</h1>';
    html += '<h2>基础信息</h2><table class=info><tr><td><b>分析类型</b></td><td>孔洞分析</td><td><b>生成时间</b></td><td>'+now+'</td></tr></table>';
    html += '<h2>一、孔洞检测统计</h2><table><tr><th>指标</th><th>值</th><th>单位</th></tr>';
    html += statRow('孔洞总数', s.hole_count, '个');
    html += statRow('孔洞总面积', s.total_area, 'mm²');
    html += statRow('平均面积', s.avg_area, 'mm²');
    html += statRow('面孔率', s.porosity_percent, '%');
    html += statRow('平均等效直径', s.avg_diameter_mm, 'mm');
    html += statRow('最大等效直径', s.max_diameter_mm, 'mm');
    html += statRow('最小等效直径', s.min_diameter_mm, 'mm');
    html += '</table>';
    if (s.size_distribution) {
      html += '<h2>二、孔洞大小分布</h2><table><tr><th>分类</th><th>孔径范围</th><th>数量</th><th>占比</th></tr>';
      const cats = ['大洞(>10mm)','中洞(5-10mm)','小洞(1-5mm)','针孔(<1mm)'];
      const ranges = ['>10mm','5-10mm','1-4.9mm','<1mm'];
      const total = s.hole_count || 1;
      cats.forEach((c,i) => { html += '<tr><td>'+c.replace(/\(.*\)/,'')+'</td><td>'+ranges[i]+'</td><td>'+(s.size_distribution[c]||0)+'</td><td>'+((s.size_distribution[c]||0)/total*100).toFixed(1)+'%</td></tr>'; });
      html += '</table>';
    }
    if (d.results && d.results.length > 0) {
      html += '<h2>三、孔洞明细</h2><table><tr><th>序号</th><th>面积(mm²)</th><th>等效直径(mm)</th><th>圆形度</th></tr>';
      d.results.forEach((r,i) => { html += '<tr><td>'+(i+1)+'</td><td>'+r.diameter_mm?r.area_mm2.toFixed(4):r.area_mm2.toFixed(4)+'</td><td>'+r.diameter_mm.toFixed(4)+'</td><td>'+r.circularity.toFixed(4)+'</td></tr>'; });
      html += '</table>';
    }
  }
  // ── Fracture Report ──
  else if (currentType === 'fracture') {
    html += '<h1>碳酸盐岩岩心裂缝分析报告</h1>';
    html += '<h2>基础信息</h2><table class=info><tr><td><b>分析类型</b></td><td>裂缝分析</td><td><b>生成时间</b></td><td>'+now+'</td></tr></table>';
    html += '<h2>一、裂缝检测统计</h2><table><tr><th>指标</th><th>值</th><th>单位</th></tr>';
    html += statRow('裂缝总条数', s.crack_count, '条');
    html += statRow('裂缝总面积', s.total_area, 'px²');
    html += statRow('平均宽度', s.avg_width, 'px');
    html += statRow('最大宽度', s.max_width, 'px');
    html += statRow('平均长度', s.avg_length, 'px');
    html += statRow('最大长度', s.max_length, 'px');
    html += '</table>';
    if (d.results && d.results.length > 0) {
      html += '<h2>二、裂缝明细</h2><table><tr><th>序号</th><th>长度(px)</th><th>宽度(px)</th><th>面积(px²)</th><th>实体度</th></tr>';
      d.results.forEach((r,i) => { html += '<tr><td>'+(i+1)+'</td><td>'+r.length_px.toFixed(2)+'</td><td>'+r.width_px.toFixed(2)+'</td><td>'+r.area_px.toFixed(2)+'</td><td>'+r.solidity.toFixed(4)+'</td></tr>'; });
      html += '</table>';
    }
  }
  // ── Grain Report ──
  else if (currentType === 'grain') {
    html += '<h1>砾岩岩心粒度分析报告</h1>';
    html += '<h2>基础信息</h2><table class=info><tr><td><b>分析类型</b></td><td>粒度分析</td><td><b>生成时间</b></td><td>'+now+'</td></tr></table>';
    html += '<h2>一、粒度检测统计</h2><table><tr><th>指标</th><th>值</th><th>单位</th></tr>';
    html += statRow('颗粒总数', s.total_count, '个');
    html += statRow('平均粒径', s.avg_diameter_mm, 'mm');
    html += statRow('中值粒径 (D50)', s.d50_mm, 'mm');
    html += statRow('D10', s.d10_mm, 'mm');
    html += statRow('D90', s.d90_mm, 'mm');
    html += statRow('标准偏差', s.std_dev_mm, 'mm');
    html += statRow('最大粒径', s.max_diameter_mm, 'mm');
    html += statRow('最小粒径', s.min_diameter_mm, 'mm');
    html += '</table>';
    if (s.size_distribution) {
      html += '<h2>二、粒度分布 (Udden-Wentworth)</h2><table><tr><th>粒级</th><th>粒径范围</th><th>数量</th><th>占比</th></tr>';
      const cats = ['砾','砂','粉砂','泥'], ranges = ['>2mm','0.0625-2mm','0.0039-0.0625mm','<0.0039mm'];
      const total = s.total_count || 1;
      cats.forEach((c,i) => { html += '<tr><td>'+c+'</td><td>'+ranges[i]+'</td><td>'+(s.size_distribution[c]||0)+'</td><td>'+((s.size_distribution[c]||0)/total*100).toFixed(1)+'%</td></tr>'; });
      html += '</table>';
    }
    if (d.results && d.results.length > 0) {
      html += '<h2>三、颗粒明细</h2><table><tr><th>序号</th><th>直径(mm)</th><th>Feret长轴(mm)</th><th>Feret短轴(mm)</th><th>圆度</th><th>粒级</th></tr>';
      d.results.forEach((r,i) => { html += '<tr><td>'+(i+1)+'</td><td>'+r.d_mm.toFixed(4)+'</td><td>'+r.feret_long.toFixed(4)+'</td><td>'+r.feret_short.toFixed(4)+'</td><td>'+r.circularity.toFixed(4)+'</td><td>'+r.size+'</td></tr>'; });
      html += '</table>';
    }
  }

  html += '<h2>附图</h2>';
  if (d.images && d.images.result) html += '<div class=chart><img src="'+d.images.result+'"><p>结果标记图</p></div>';
  if (d.images && d.images.binary) html += '<div class=chart><img src="'+d.images.binary+'"><p>二值化图</p></div>';

  // Chart canvas for histogram
  if (s.diameters && s.diameters.length > 0) {
    html += '<h2>粒径分布直方图</h2><div class=chart><canvas id=rptChart width=800 height=400></canvas></div>';
  }
  html += '<div class=footer>岩心孔洞裂缝分析系统 v1.0</div>';
  html += '<script src=\"https://cdn.jsdelivr.net/npm/chart.js@4\"></script>';
  html += '<script>var _diameters='+JSON.stringify(s.diameters||[])+';';
  html += 'var _labels=_diameters.map((_,i)=>\"#\"+(i+1));';
  html += 'new Chart(document.getElementById(\"rptChart\"),{type:\"bar\",data:{labels:_labels,datasets:[{label:\"直径(mm)\",data:_diameters}]},options:{responsive:true}});';
  html += '</script>';
  html += '</body></html>';
  const w = window.open('', '_blank');
  w.document.write(html);
  w.document.close();
};
function statRow(label, val, unit) {
  const v = val != null ? (typeof val === 'number' ? val.toFixed(2) : val) : '—';
  return '<tr><td><b>'+label+'</b></td><td>'+v+'</td><td>'+unit+'</td></tr>';
}
