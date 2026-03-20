const manifestGrid = document.getElementById('manifestGrid');
const toolSelect = document.getElementById('toolSelect');
const toolArgs = document.getElementById('toolArgs');
const toolOutput = document.getElementById('toolOutput');
const toolActivity = document.getElementById('toolActivity');
const chatLog = document.getElementById('chatLog');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatHint = document.getElementById('chatHint');
const promptChips = document.getElementById('promptChips');

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

async function loadDashboardState() {
  const state = await fetchJSON('/api/dashboard-state');
  chatHint.textContent = state.chat_enabled
    ? 'Chat is live. Ask normally and Rexbot will choose tools automatically.'
    : 'Chat is disabled on the server. Start the dashboard with --enable-chat to use natural-language requests.';
  promptChips.innerHTML = state.example_prompts
    .map((prompt) => `<button type="button" class="chip">${prompt}</button>`)
    .join('');
  promptChips.querySelectorAll('.chip').forEach((button) => {
    button.addEventListener('click', () => {
      chatInput.value = button.textContent;
      chatInput.focus();
    });
  });
}

async function loadManifest() {
  const manifest = await fetchJSON('/api/manifest');
  manifestGrid.innerHTML = manifest.slice(0, 8).map((item) => `
    <article class="manifest-card">
      <div class="manifest-copy">
        <h3>${item.name}</h3>
        <p>${item.tool}</p>
      </div>
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
  const entry = document.createElement('article');
  entry.className = `chat-entry ${who}`;
  entry.innerHTML = `<div class="who">${who === 'user' ? 'You' : 'Rexbot'}</div><div class="text"></div>`;
  entry.querySelector('.text').textContent = text;
  chatLog.appendChild(entry);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderToolActivity(events = []) {
  if (!events.length) {
    toolActivity.className = 'activity-list empty-state';
    toolActivity.textContent = 'Ask Rexbot to see the tools it picks.';
    return;
  }

  toolActivity.className = 'activity-list';
  toolActivity.innerHTML = events.map((event) => {
    if (event.event === 'tool_requested') {
      return `
        <article class="activity-item">
          <span class="activity-type requested">planned</span>
          <div>
            <strong>${event.tool}</strong>
            <p>${event.decision}</p>
          </div>
        </article>
      `;
    }
    if (event.event === 'tool_reviewed') {
      return `
        <article class="activity-item">
          <span class="activity-type reviewed">review</span>
          <div>
            <strong>${event.tool}</strong>
            <p>${event.review_message}</p>
          </div>
        </article>
      `;
    }
    return `
      <article class="activity-item">
        <span class="activity-type completed">done</span>
        <div>
          <strong>${event.tool}</strong>
          <p>Completed for ${Array.isArray(event.targets) ? event.targets.join(', ') : event.target ?? 'request'}.</p>
        </div>
      </article>
    `;
  }).join('');
}

document.getElementById('primaryExample').addEventListener('click', () => {
  chatInput.value = 'Organize my downloads by file type.';
  chatInput.focus();
});

document.getElementById('refreshManifest').addEventListener('click', loadManifest);
document.getElementById('refreshTools').addEventListener('click', loadTools);
document.getElementById('resetChat').addEventListener('click', async () => {
  await fetchJSON('/api/reset-chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  chatLog.innerHTML = '';
  renderToolActivity([]);
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
  renderToolActivity(result.tool_events ?? []);
});

await Promise.all([loadDashboardState(), loadManifest(), loadTools(), loadSystemInfo()]);
