/**
 * 用户聊天前端 —— WebSocket 客户端
 */
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const messageArea = document.getElementById('messageArea');
const inputBox = document.getElementById('inputBox');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');

let ws = null;
let sessionId = null;
let isTyping = false;

// ====== 状态提示 ======
function setStatus(state) {
  statusDot.className = 'status-dot';
  if (state === 'connecting') {
    statusText.textContent = '连接中...';
  } else if (state === 'online') {
    statusDot.classList.add('online');
    statusText.textContent = '智能客服在线';
  } else if (state === 'human') {
    statusDot.classList.add('online');
    statusText.textContent = '人工客服在线';
  } else {
    statusDot.classList.add('offline');
    statusText.textContent = '已断开';
  }
}

// ====== 消息渲染 ======
function appendMessage(text, sender = 'bot', extraClass = '') {
  const div = document.createElement('div');
  div.className = `msg ${sender} ${extraClass}`;
  div.innerHTML = `<div class="msg-content"></div>`;
  div.querySelector('.msg-content').textContent = text;
  messageArea.appendChild(div);
  messageArea.scrollTop = messageArea.scrollHeight;
}

function removeTyping() {
  document.querySelectorAll('.msg.typing').forEach(el => el.remove());
}

function showHandoffNotice(text) {
  const div = document.createElement('div');
  div.className = 'handoff-notice';
  div.textContent = text;
  messageArea.appendChild(div);
  messageArea.scrollTop = messageArea.scrollHeight;
}

// ====== 发送消息 ======
async function sendMessage(text) {
  if (!text || !ws) return;

  appendMessage(text, 'user');
  inputBox.value = '';

  // 显示"正在输入"
  appendMessage('客服正在回复...', 'typing');

  ws.send(JSON.stringify({ type: 'chat', text }));
}

// ====== WebSocket 连接 ======
function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws/user`);

  setStatus('connecting');

  ws.onopen = () => {
    console.log('[WS] 连接成功');
    setStatus('online');
  };

  ws.onmessage = (event) => {
    try {
      const { type, data } = JSON.parse(event.data);
      handleMessage(type, data);
    } catch (e) {
      console.error('[WS] 消息解析失败:', e, event.data);
    }
  };

  ws.onclose = () => {
    console.log('[WS] 连接关闭，3秒后重连...');
    setStatus('offline');
    ws = null;
    setTimeout(connect, 3000);
  };

  ws.onerror = (err) => {
    console.error('[WS] 错误:', err);
  };
}

function handleMessage(type, data) {
  switch (type) {
    case 'session_start':
      sessionId = data.session_id;
      console.log('[WS] 会话 ID:', sessionId);
      break;

    case 'reply':
      removeTyping();
      appendMessage(data.reply, 'bot');
      break;

    case 'handoff':
      removeTyping();
      showHandoffNotice('🔄 正在为您转接人工客服，请稍候...');
      break;

    case 'agent_connected':
      setStatus('human');
      showHandoffNotice(`✅ 人工客服已接入，正在为您服务`);
      break;

    case 'agent_typing':
      // 人工客服正在输入 —— 不重复显示 typing
      break;

    case 'agent_message':
      removeTyping();
      appendMessage(data.text, 'bot', 'human-agent');
      break;

    case 'session_closed':
      showHandoffNotice('会话已结束，您可以继续发送问题');
      setStatus('online');
      break;

    case 'error':
      removeTyping();
      appendMessage(`⚠️ ${data}`, 'bot');
      break;
  }
}

// ====== 事件绑定 ======
sendBtn.addEventListener('click', () => {
  sendMessage(inputBox.value.trim());
});

inputBox.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(inputBox.value.trim());
  }
});

document.querySelectorAll('.quick-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    sendMessage(btn.dataset.text);
  });
});

clearBtn.addEventListener('click', () => {
  messageArea.innerHTML = '';
});

// ====== 启动 ======
connect();
