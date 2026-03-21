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
const helperText = document.getElementById('helperText');
const sendButton = document.getElementById('sendButton');
const modeValue = document.getElementById('modeValue');
const approvedRoots = document.getElementById('approvedRoots');
const toolRunner = document.querySelector('.tool-panel');
const toolRunnerToggle = document.getElementById('toolRunnerToggle');
let chatEnabled = false;

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
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error ?? `Request failed with status ${response.status}`);
  }
  return payload;
}

function setComposerState({ enabled, busy = false }) {
  chatInput.disabled = !enabled || busy;
  sendButton.disabled = !enabled || busy;
  sendButton.textContent = busy ? 'Working…' : 'Send';
}

function renderApprovedRoots(roots = []) {
  approvedRoots.innerHTML = roots.length
    ? roots.map((root) => `<span class="root-pill">${root}</span>`).join('')
    : '—';
}

async function loadDashboardState() {
  const state = await fetchJSON('/api/dashboard-state');
  chatEnabled = Boolean(state.chat_enabled);
  modeValue.textContent = state.mode ?? '—';
  renderApprovedRoots(state.approved_roots ?? []);
  chatHint.textContent = chatEnabled
    ? 'Chat is live. Ask normally and Rexbot will choose tools automatically.'
    : 'Chat is disabled on the server. Start the dashboard with --enable-chat to use natural-language requests.';
  helperText.textContent = chatEnabled
    ? 'Rexbot will automatically choose tools when chat is enabled.'
    : 'Enable chat on the server to test natural-language requests.';
  setComposerState({ enabled: chatEnabled });

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

function clearEmptyChat() {
  if (chatLog.classList.contains('empty-chat')) {
    chatLog.innerHTML = '';
    chatLog.classList.remove('empty-chat');
  }
}

function appendChat(who, text) {
  clearEmptyChat();
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

function resetChatView() {
  chatLog.classList.add('empty-chat');
  chatLog.innerHTML = `
    <div class="empty-chat-card">
      <strong>Ready to test</strong>
      <p>Ask for a real task and Rexbot will choose the smallest useful tools automatically.</p>
    </div>
  `;
}

document.getElementById('primaryExample').addEventListener('click', () => {
  chatInput.value = 'Organize my downloads by file type.';
  chatInput.focus();
});

document.getElementById('refreshManifest').addEventListener('click', loadManifest);
document.getElementById('refreshTools').addEventListener('click', loadTools);
document.getElementById('resetChat').addEventListener('click', async () => {
  await fetchJSON('/api/reset-chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  resetChatView();
  renderToolActivity([]);
});

toolRunner.addEventListener('toggle', () => {
  toolRunnerToggle.textContent = toolRunner.open ? 'Close' : 'Open';
});

document.getElementById('toolForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  toolOutput.textContent = 'Running tool…';
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
  if (!message || sendButton.disabled) return;

  appendChat('user', message);
  chatInput.value = '';
  setComposerState({ enabled: true, busy: true });
  try {
    const result = await fetchJSON('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    appendChat('rexbot', result.reply);
    renderToolActivity(result.tool_events ?? []);
  } catch (error) {
    appendChat('rexbot', error.message);
  } finally {
    setComposerState({ enabled: chatEnabled, busy: false });
  }
});

resetChatView();
await Promise.all([loadDashboardState(), loadManifest(), loadTools(), loadSystemInfo()]);
