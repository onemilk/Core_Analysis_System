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

async function runAnalysis() {
  if (!currentImage) { alert('请先打开图像'); return; }
  if (roiMode) { alert('请双击闭合多边形后再分析'); return; }
  const params = {};
  if (currentType === 'hole') params.threshold = +document.getElementById('threshold').value;
  params.min_area = +document.getElementById('minArea').value;
  params.max_area = +document.getElementById('maxArea').value;
  params.scale_mm_per_px = +document.getElementById('scale').value;
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
  const d = resultData;
  const s = d.summary || {};
  let html = '<!DOCTYPE html><html lang=zh-CN><head><meta charset=UTF-8><title>分析报告</title>';
  html += '<style>body{font-family:"Microsoft YaHei",sans-serif;padding:20px} h1{text-align:center}';
  html += 'table{border-collapse:collapse;width:100%;margin:8px 0} th,td{border:1px solid #666;padding:6px} th{background:#f0f0f0}</style></head><body>';
  html += '<h1>岩心分析报告</h1><h2>统计摘要</h2><table>';
  for (const [k,v] of Object.entries(s)) {
    if (k === 'diameters' || k === 'size_distribution') continue;
    html += '<tr><td><b>'+k+'</b></td><td>'+(typeof v==='number'?v.toFixed(2):JSON.stringify(v))+'</td></tr>';
  }
  html += '</table>';
  if (d.images && d.images.result) {
    html += '<h2>结果图</h2><img src="'+d.images.result+'" style="max-width:100%">';
  }
  html += '<p style="color:#999;margin-top:20px">生成时间: '+new Date().toLocaleString()+'</p></body></html>';
  const w = window.open('', '_blank');
  w.document.write(html);
  w.document.close();
};
