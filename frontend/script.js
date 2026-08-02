// Empty string = same origin. Flask serves this frontend and the API together,
// so this only needs overriding if you split them into separate servers.
const API_BASE_URL = window.API_BASE_URL || '';

const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const dropZoneText = document.getElementById('drop-zone-text');
const preview = document.getElementById('preview');
const detectButton = document.getElementById('detect-button');
const errorEl = document.getElementById('error');
const resultEl = document.getElementById('result');
const resultBadge = document.getElementById('result-badge');
const probabilitiesEl = document.getElementById('probabilities');

const LABELS = {
  with_mask: 'Mask detected',
  without_mask: 'No mask detected',
  partial_mask: 'Mask worn incorrectly',
};

let selectedFile = null;

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.hidden = true;
}

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please choose an image file.');
    return;
  }
  clearError();
  selectedFile = file;
  detectButton.disabled = false;
  resultEl.hidden = true;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropZoneText.hidden = true;
  };
  reader.readAsDataURL(file);
}

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = '#7c8cff';
});
dropZone.addEventListener('dragleave', () => {
  dropZone.style.borderColor = '';
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = '';
  handleFile(e.dataTransfer.files[0]);
});

detectButton.addEventListener('click', async () => {
  if (!selectedFile) return;

  clearError();
  detectButton.disabled = true;
  detectButton.textContent = 'Detecting...';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      showError(data.error || 'Something went wrong.');
      resultEl.hidden = true;
      return;
    }

    renderResult(data);
  } catch (err) {
    showError('Could not reach the detection service. Is the backend running?');
  } finally {
    detectButton.disabled = false;
    detectButton.textContent = 'Detect';
  }
});

function renderResult(data) {
  resultBadge.textContent = `${LABELS[data.label] || data.label} (${Math.round(data.confidence * 100)}%)`;
  resultBadge.className = `result-badge ${data.label}`;

  probabilitiesEl.innerHTML = '';
  const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
  for (const [label, prob] of entries) {
    const row = document.createElement('div');
    row.className = 'probability-row';

    const name = document.createElement('span');
    name.textContent = label.replace('_', ' ');

    const track = document.createElement('div');
    track.className = 'probability-bar-track';
    const fill = document.createElement('div');
    fill.className = 'probability-bar-fill';
    fill.style.width = `${Math.round(prob * 100)}%`;
    track.appendChild(fill);

    const pct = document.createElement('span');
    pct.textContent = `${Math.round(prob * 100)}%`;

    row.append(name, track, pct);
    probabilitiesEl.appendChild(row);
  }

  resultEl.hidden = false;
}
