// =============================================================
// FIELD NOTES ON LEAF PATHOLOGY — frontend logic
// =============================================================

// ---------- OVERVIEW: populate specimen cards ----------
async function loadClasses() {
  try {
    const res = await fetch('/api/classes');
    const data = await res.json();
    const container = document.getElementById('specimen-cards');
    container.innerHTML = '';

    Object.entries(data).forEach(([crop, conditions]) => {
      const card = document.createElement('div');
      card.className = 'specimen-card';
      card.innerHTML = `
        <h3>${crop}</h3>
        <span class="count">${conditions.length} conditions catalogued</span>
        <ul>${conditions.map(c => `<li>${c}</li>`).join('')}</ul>
      `;
      container.appendChild(card);
    });
  } catch (e) {
    console.error('Could not load specimen classes', e);
  }
}

// ---------- DIAGNOSIS STATION: upload + predict ----------
const tray = document.getElementById('upload-tray');
const trayInner = document.getElementById('tray-inner');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');

const resultPlaceholder = document.getElementById('result-placeholder');
const loadingState = document.getElementById('loading-state');
const stampResult = document.getElementById('stamp-result');
const stampCrop = document.getElementById('stamp-crop');
const stampDisease = document.getElementById('stamp-disease');
const stampConfidence = document.getElementById('stamp-confidence');
const primaryStamp = document.getElementById('primary-stamp');
const altReadings = document.getElementById('alt-readings');

tray.addEventListener('click', () => fileInput.click());

tray.addEventListener('dragover', (e) => {
  e.preventDefault();
  tray.classList.add('dragover');
});
tray.addEventListener('dragleave', () => tray.classList.remove('dragover'));
tray.addEventListener('drop', (e) => {
  e.preventDefault();
  tray.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please choose an image file.');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    trayInner.hidden = true;
  };
  reader.readAsDataURL(file);

  runPrediction(file);
}

async function runPrediction(file) {
  resultPlaceholder.hidden = true;
  stampResult.hidden = true;
  loadingState.hidden = false;

  const formData = new FormData();
  formData.append('image', file);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      throw new Error(data.error);
    }

    renderStamp(data.predictions);
  } catch (err) {
    console.error(err);
    loadingState.hidden = true;
    resultPlaceholder.hidden = false;
    resultPlaceholder.innerHTML = `<p>Reading failed.</p><p class="muted">${err.message || 'Check the server console.'}</p>`;
  }
}

function renderStamp(predictions) {
  loadingState.hidden = true;
  stampResult.hidden = false;

  const top = predictions[0];
  stampCrop.textContent = top.crop;
  stampDisease.textContent = top.disease;
  stampConfidence.textContent = `${top.confidence}% confidence`;
  primaryStamp.classList.toggle('healthy', top.is_healthy);

  // Retrigger stamp animation on repeated uploads
  primaryStamp.style.animation = 'none';
  primaryStamp.offsetHeight; // force reflow
  primaryStamp.style.animation = '';

  altReadings.innerHTML = predictions.slice(1).map(p => `
    <div class="alt-reading-row">
      <span>${p.crop} — ${p.disease}</span>
      <span>${p.confidence}%</span>
    </div>
  `).join('');
}

// ---------- COMPARISON PLATES: table + hand-drawn bars ----------
async function loadComparison() {
  try {
    const res = await fetch('/api/comparison');
    const data = await res.json();
    const results = data.results || [];

    const tbody = document.getElementById('metrics-body');
    tbody.innerHTML = results.map(r => `
      <tr>
        <td>${r.model}</td>
        <td>${(r.accuracy * 100).toFixed(2)}%</td>
        <td>${(r.macro_f1 * 100).toFixed(2)}%</td>
        <td>${(r.macro_precision * 100).toFixed(2)}%</td>
        <td>${(r.macro_recall * 100).toFixed(2)}%</td>
      </tr>
    `).join('');

    const barsWrap = document.getElementById('bars-wrap');
    barsWrap.innerHTML = results.map(r => {
      const pct = (r.macro_f1 * 100).toFixed(1);
      const isDL = r.model.toLowerCase().includes('resnet');
      return `
        <div class="bar-row">
          <span class="bar-label">${r.model}</span>
          <div class="bar-track"><div class="bar-fill ${isDL ? 'dl' : ''}" style="width:0%" data-target="${pct}"></div></div>
          <span class="bar-value">${pct}%</span>
        </div>
      `;
    }).join('');

    // animate bars in after paint
    requestAnimationFrame(() => {
      document.querySelectorAll('.bar-fill').forEach(el => {
        el.style.width = el.dataset.target + '%';
      });
    });
  } catch (e) {
    console.error('Could not load comparison data', e);
  }
}

loadClasses();
loadComparison();
