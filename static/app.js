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
  let md = '';

  // ── Hole Report (Markdown) ──
  if (currentType === 'hole') {
    md += '# 碳酸盐岩岩心孔洞分析报告\n\n';
    md += '> 生成时间: ' + now + '\n\n';
    md += '## 一、孔洞检测统计\n\n';
    md += '| 指标 | 值 | 单位 |\n|------|-----|------|\n';
    md += '| 孔洞总数 | ' + (s.hole_count||0) + ' | 个 |\n';
    md += '| 孔洞总面积 | ' + fmt(s.total_area) + ' | mm² |\n';
    md += '| 平均面积 | ' + fmt(s.avg_area) + ' | mm² |\n';
    md += '| 面孔率 | ' + fmt(s.porosity_percent) + ' | % |\n';
    md += '| 平均等效直径 | ' + fmt(s.avg_diameter_mm) + ' | mm |\n';
    md += '| 最大等效直径 | ' + fmt(s.max_diameter_mm) + ' | mm |\n';
    md += '| 最小等效直径 | ' + fmt(s.min_diameter_mm) + ' | mm |\n\n';
    if (s.size_distribution) {
      md += '## 二、孔洞大小分布\n\n';
      md += '| 分类 | 孔径范围 | 数量 | 占比 |\n|------|----------|------|------|\n';
      const cats = ['大洞(>10mm)','中洞(5-10mm)','小洞(1-5mm)','针孔(<1mm)'];
      const ranges = ['>10mm','5-10mm','1-4.9mm','<1mm'];
      const total = s.hole_count || 1;
      cats.forEach((c,i) => { md += '| '+c.replace(/\(.*\)/,'')+' | '+ranges[i]+' | '+(s.size_distribution[c]||0)+' | '+((s.size_distribution[c]||0)/total*100).toFixed(1)+'% |\n'; });
      md += '\n';
    }
    if (d.results && d.results.length > 0) {
      md += '## 三、孔洞明细\n\n';
      md += '| 序号 | 面积(mm²) | 等效直径(mm) | 圆形度 |\n|------|-----------|-------------|--------|\n';
      d.results.forEach((r,i) => { md += '| '+(i+1)+' | '+r.area_mm2.toFixed(4)+' | '+r.diameter_mm.toFixed(4)+' | '+r.circularity.toFixed(4)+' |\n'; });
      md += '\n';
    }
  }
  // ── Fracture Report (Markdown) ──
  else if (currentType === 'fracture') {
    md += '# 碳酸盐岩岩心裂缝分析报告\n\n';
    md += '> 生成时间: ' + now + '\n\n';
    md += '## 一、裂缝检测统计\n\n';
    md += '| 指标 | 值 | 单位 |\n|------|-----|------|\n';
    md += '| 裂缝总条数 | ' + (s.crack_count||0) + ' | 条 |\n';
    md += '| 裂缝总面积 | ' + fmt(s.total_area) + ' | px² |\n';
    md += '| 平均宽度 | ' + fmt(s.avg_width) + ' | px |\n';
    md += '| 最大宽度 | ' + fmt(s.max_width) + ' | px |\n';
    md += '| 平均长度 | ' + fmt(s.avg_length) + ' | px |\n';
    md += '| 最大长度 | ' + fmt(s.max_length) + ' | px |\n\n';
    if (d.results && d.results.length > 0) {
      md += '## 二、裂缝明细\n\n';
      md += '| 序号 | 长度(px) | 宽度(px) | 面积(px²) | 实体度 |\n|------|----------|----------|-----------|--------|\n';
      d.results.forEach((r,i) => { md += '| '+(i+1)+' | '+r.length_px.toFixed(2)+' | '+r.width_px.toFixed(2)+' | '+r.area_px.toFixed(2)+' | '+r.solidity.toFixed(4)+' |\n'; });
      md += '\n';
    }
  }
  // ── Grain Report (Markdown) ──
  else if (currentType === 'grain') {
    md += '# 砾岩岩心粒度分析报告\n\n';
    md += '> 生成时间: ' + now + '\n\n';
    md += '## 一、粒度检测统计\n\n';
    md += '| 指标 | 值 | 单位 |\n|------|-----|------|\n';
    md += '| 颗粒总数 | ' + (s.total_count||0) + ' | 个 |\n';
    md += '| 平均粒径 | ' + fmt(s.avg_diameter_mm) + ' | mm |\n';
    md += '| 中值粒径(D50) | ' + fmt(s.d50_mm) + ' | mm |\n';
    md += '| D10 | ' + fmt(s.d10_mm) + ' | mm |\n';
    md += '| D90 | ' + fmt(s.d90_mm) + ' | mm |\n';
    md += '| 标准偏差 | ' + fmt(s.std_dev_mm) + ' | mm |\n';
    md += '| 最大粒径 | ' + fmt(s.max_diameter_mm) + ' | mm |\n';
    md += '| 最小粒径 | ' + fmt(s.min_diameter_mm) + ' | mm |\n\n';
    if (s.size_distribution) {
      md += '## 二、粒度分布 (Udden-Wentworth)\n\n';
      md += '| 粒级 | 粒径范围 | 数量 | 占比 |\n|------|----------|------|------|\n';
      const cats = ['砾','砂','粉砂','泥'], ranges = ['>2mm','0.0625-2mm','0.0039-0.0625mm','<0.0039mm'];
      const total = s.total_count || 1;
      cats.forEach((c,i) => { md += '| '+c+' | '+ranges[i]+' | '+(s.size_distribution[c]||0)+' | '+((s.size_distribution[c]||0)/total*100).toFixed(1)+'% |\n'; });
      md += '\n';
    }
    if (d.results && d.results.length > 0) {
      md += '## 三、颗粒明细\n\n';
      md += '| 序号 | 直径(mm) | Feret长轴(mm) | Feret短轴(mm) | 圆度 | 粒级 |\n|------|----------|---------------|---------------|------|------|\n';
      d.results.forEach((r,i) => { md += '| '+(i+1)+' | '+r.d_mm.toFixed(4)+' | '+r.feret_long.toFixed(4)+' | '+r.feret_short.toFixed(4)+' | '+r.circularity.toFixed(4)+' | '+r.size+' |\n'; });
      md += '\n';
    }
  }

  md += '---\n*岩心孔洞裂缝分析系统 v1.0*\n';
  fetch('/api/report/save', {method:'POST', body: md}).then(r => r.json()).then(d => {
    if (d.status === 'ok') alert('报告已保存到: ' + d.path);
    else alert('保存失败');
  });
};
function fmt(v) { return v != null ? (typeof v === 'number' ? v.toFixed(2) : v) : '—'; }
