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

  const res = await fetch('/api/analyze', {method:'POST',headers:{'Content-Type':'application/json'}, body:JSON.stringify({image:currentImage,type:currentType,params})});
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
    chart = new Chart(ctx, {type:'bar', data:{labels:data.summary.diameters.map((_,i)=>`#${i+1}`), datasets:[{label:'直径(mm)',data:data.summary.diameters}]}, options:{responsive:true,maintainAspectRatio:false}});
  }
};

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
