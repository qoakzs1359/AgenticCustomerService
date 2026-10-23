/**
 * 人工客服后台 —— WebSocket 客户端
 */
const agentIdEl = document.getElementById('agentId');
const statusText = document.getElementById('statusText');
const queueBadge = document.getElementById('queueBadge');
const queueList = document.getElementById('queueList');
const messageArea = document.getElementById('messageArea');
const inputBox = document.getElementById('inputBox');
const sendBtn = document.getElementById('sendBtn');
const closeBtn = document.getElementById('closeBtn');

let ws = null;
let agentId = null;
let currentSession = null;

// ====== 队列渲染 ======
function renderQueue(sessions) {
  queueBadge.textContent = sessions.length;
  if (sessions.length === 0) {
    queueList.innerHTML = '<div class="queue-empty">暂无待接入会话<br>有用户点【转人工】后会在这里出现</div>';
    return;
  }
  queueList.innerHTML = sessions.map(sid => `
    <div class="queue-item" data-sid="${sid}">
      🔔 用户请求转接人工
      <div class="q-sid">${sid}</div>
      <button class="btn-take" onclick="takeSession('${sid}')">▶ 立即接入</button>
    </div>
  `).join('');
}

// ====== 接管会话 ======
function takeSession(sessionId) {
  if (!ws) return;
  ws.send(JSON.stringify({ type: 'take_session', session_id: sessionId }));
}

// ====== 选择当前会话 ======
function selectSession(sessionId) {
  currentSession = sessionId;
  inputBox.disabled = false;
  sendBtn.disabled = false;
  closeBtn.style.display = 'inline-block';
  statusText.textContent = `正在服务 ${sessionId}`;
  messageArea.innerHTML = '';
  inputBox.focus();
}

// ====== 发送消息 ======
function sendMessage() {
  const text = inputBox.value.trim();
  if (!text || !ws || !currentSession) return;
  ws.send(JSON.stringify({ type: 'send', session_id: currentSession, text }));
  inputBox.value = '';
  // 把自己的消息显示在气泡里
  appendMessage(text, 'bot');
}

// ====== 消息渲染 ======
function appendMessage(text, sender = 'user', extraClass = '') {
  const div = document.createElement('div');
  div.className = `msg ${sender} ${extraClass}`;
  div.innerHTML = `<div class="msg-content"></div>`;
  div.querySelector('.msg-content').textContent = text;
  messageArea.appendChild(div);
  messageArea.scrollTop = messageArea.scrollHeight;
}

// ====== 关闭会话 ======
function closeSession() {
  if (!currentSession || !ws) return;
  ws.send(JSON.stringify({ type: 'close_session', session_id: currentSession }));
  currentSession = null;
  inputBox.disabled = true;
  sendBtn.disabled = true;
  closeBtn.style.display = 'none';
  statusText.textContent = '等待接入用户...';
  messageArea.innerHTML = '<div class="agent-tooltip">✅ 会话已关闭</div>';
}

// ====== WebSocket 连接 ======
function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws/agent`);

  ws.onopen = () => {
    console.log('[Agent WS] 连接成功');
  };

  ws.onmessage = (event) => {
    try {
      const { type, data } = JSON.parse(event.data);
      handleMessage(type, data);
    } catch (e) {
      console.error('[Agent WS] 消息解析失败:', e);
    }
  };

  ws.onclose = () => {
    console.log('[Agent WS] 连接关闭，3秒后重连...');
    ws = null;
    setTimeout(connect, 3000);
  };

  ws.onerror = (err) => console.error('[Agent WS] 错误:', err);
}

function handleMessage(type, data) {
  switch (type) {
    case 'session_start':
      agentId = data.agent_id;
      agentIdEl.textContent = `[${agentId}]`;
      break;

    case 'queue_update':
      renderQueue(data);
      break;

    case 'new_handoff':
      // 有新的用户请求转接 —— 刷新队列
      // server 会单独发 queue_update，这里不需要额外处理
      break;

    case 'session_taken':
      selectSession(data.session_id);
      break;

    case 'user_message':
      if (currentSession === data.session_id) {
        appendMessage(data.text, 'user');
      } else {
        // 不是当前会话 —— 给个提示
        console.log(`[Agent] 收到其他会话消息: ${data.session_id}`);
      }
      break;

    case 'session_closed':
      if (currentSession === data.session_id) {
        closeSession();
      }
      break;
  }
}

// ====== 事件绑定 ======
sendBtn.addEventListener('click', sendMessage);
inputBox.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
closeBtn.addEventListener('click', closeSession);

// ====== 启动 ======
connect();
