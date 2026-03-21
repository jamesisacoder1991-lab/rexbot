const manifestGrid = document.getElementById('manifestGrid');
const toolSelect = document.getElementById('toolSelect');
const toolArgs = document.getElementById('toolArgs');
const toolOutput = document.getElementById('toolOutput');
const chatLog = document.getElementById('chatLog');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');

const platformValue = document.getElementById('platformValue');
const pythonValue = document.getElementById('pythonValue');
const cpuValue = document.getElementById('cpuValue');
const memoryValue = document.getElementById('memoryValue');

const statusClass = (status) => {
  if (status.startsWith('working')) return status.includes('partial') ? 'partial' : 'working';
  if (status.includes('disabled')) return 'disabled';
  return 'prototype';
};

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  return response.json();
}

async function loadManifest() {
  const manifest = await fetchJSON('/api/manifest');
  manifestGrid.innerHTML = manifest.map((item) => `
    <article class="manifest-card">
      <h3>${item.item}. ${item.name}</h3>
      <p>${item.tool}</p>
      <span class="tag ${statusClass(item.status)}">${item.status}</span>
    </article>
  `).join('');
}

async function loadTools() {
  const tools = await fetchJSON('/api/tools');
  toolSelect.innerHTML = tools.map((tool) => `<option value="${tool.name}">${tool.name}</option>`).join('');
}

async function loadSystemInfo() {
  const info = await fetchJSON('/api/system-info');
  platformValue.textContent = info.platform ?? '—';
  pythonValue.textContent = info.python ?? '—';
  cpuValue.textContent = info.cpu_count ?? '—';
  memoryValue.textContent = info.memory_mb ?? '—';
}

function appendChat(who, text) {
  const entry = document.createElement('div');
  entry.className = 'chat-entry';
  entry.innerHTML = `<div class="who">${who}</div><div class="text"></div>`;
  entry.querySelector('.text').textContent = text;
  chatLog.appendChild(entry);
  chatLog.scrollTop = chatLog.scrollHeight;
}

document.getElementById('refreshManifest').addEventListener('click', loadManifest);
document.getElementById('refreshTools').addEventListener('click', loadTools);
document.getElementById('resetChat').addEventListener('click', async () => {
  await fetchJSON('/api/reset-chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  chatLog.innerHTML = '';
});

document.getElementById('toolForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const payload = {
      tool_name: toolSelect.value,
      arguments: toolArgs.value.trim() ? JSON.parse(toolArgs.value) : {},
    };
    const result = await fetchJSON('/api/tool', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    toolOutput.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    toolOutput.textContent = `Tool UI error: ${error.message}`;
  }
});

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  appendChat('user', message);
  chatInput.value = '';
  const result = await fetchJSON('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  appendChat('rexbot', result.ok ? result.reply : result.error);
});

await Promise.all([loadManifest(), loadTools(), loadSystemInfo()]);
