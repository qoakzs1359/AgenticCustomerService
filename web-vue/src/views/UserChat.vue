<template>
  <div class="user-chat-wrapper">
    <div class="chat-container">
      <!-- 顶部店铺信息 -->
      <el-card class="chat-header" shadow="never">
        <div class="shop-info">
          <div class="avatar">🛍️</div>
          <div class="shop-meta">
            <div class="shop-name">好物优选旗舰店</div>
            <div class="shop-status">
              <el-tag :type="statusTagType" size="small" effect="light" round>
                <span class="status-dot" :class="statusClass"></span>
                {{ statusText }}
              </el-tag>
            </div>
          </div>
        </div>
        <el-button :icon="Delete" circle size="small" text @click="clearMessages" />
      </el-card>

      <!-- 消息列表 -->
      <div ref="messageArea" class="message-area">
        <div v-if="messages.length === 0" class="welcome-message">
          <div class="welcome-icon">👋</div>
          <div>
            <strong>你好呀~ 我是智能客服小助手</strong><br>
            有任何关于商品、订单、物流的问题都可以问我哦！<br>
            需要人工服务请发送【转人工】
          </div>
        </div>

        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          :class="['msg', msg.role, { 'human-agent': msg.fromHuman }]"
        >
          <div class="msg-content">{{ msg.content }}</div>
        </div>

        <div v-if="agentTyping" class="msg bot typing">
          <div class="msg-content">客服正在回复中...</div>
        </div>

        <el-alert
          v-if="handoffNotice"
          :title="handoffNotice"
          type="warning"
          :closable="false"
          show-icon
          class="handoff-notice"
        />
      </div>

      <!-- 快捷提问 -->
      <div class="quick-questions">
        <el-button size="small" @click="sendQuick('我的订单到哪了？订单号 DD20261022001')">📦 查订单</el-button>
        <el-button size="small" @click="sendQuick('快递到哪了？快递号 SF1234567890')">🚚 查物流</el-button>
        <el-button size="small" @click="sendQuick('P10086 这个商品有货吗？多少钱？')">💰 查商品</el-button>
        <el-button size="small" type="warning" @click="sendQuick('转人工')">👤 转人工</el-button>
      </div>

      <!-- 输入区 -->
      <div class="input-area">
        <el-input
          v-model="inputText"
          type="textarea"
          :rows="2"
          placeholder="输入消息，按 Enter 发送，Shift+Enter 换行"
          resize="none"
          @keydown.enter.exact.prevent="doSend"
        />
        <el-button type="primary" :icon="Promotion" size="large" @click="doSend">发送</el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onUnmounted } from 'vue'
import { useSocket } from '../composables/useSocket'
import { ElMessage } from 'element-plus'
import { Delete, Promotion } from '@element-plus/icons-vue'

const messages = ref([])
const inputText = ref('')
const agentTyping = ref(false)
const handoffNotice = ref('')
const sessionId = ref('')
const messageArea = ref(null)

// ===== 状态 =====
const statusIsHuman = ref(false)
const statusText = computed(() => {
  if (!wsState.value) return '连接中...'
  return statusIsHuman.value ? '人工客服在线' : '智能客服在线'
})
const statusClass = computed(() => wsState.value ? (statusIsHuman.value ? 'human' : 'online') : 'offline')
const statusTagType = computed(() => wsState.value ? 'success' : 'danger')

function handleMessage(type, data) {
  switch (type) {
    case 'session_start':
      sessionId.value = data.session_id
      break
    case 'reply':
      agentTyping.value = false
      messages.value.push({ role: 'bot', content: data.reply })
      break
    case 'handoff':
      agentTyping.value = false
      handoffNotice.value = '🔄 正在为您转接人工客服，请稍候...'
      break
    case 'agent_connected':
      statusIsHuman.value = true
      handoffNotice.value = '✅ 人工客服已接入，正在为您服务'
      break
    case 'agent_typing':
      // 人工客服正在输入
      break
    case 'agent_message':
      agentTyping.value = false
      messages.value.push({ role: 'bot', content: data.text, fromHuman: true })
      break
    case 'session_closed':
      handoffNotice.value = '会话已结束，您可以继续发送问题'
      statusIsHuman.value = false
      break
    case 'error':
      agentTyping.value = false
      ElMessage.error(data)
      break
  }
  scrollToBottom()
}

const { connected: wsState, connect, send, disconnect } = useSocket('/ws/user', handleMessage)

function doSend() {
  const text = inputText.value.trim()
  if (!text) return
  if (!wsState.value) {
    ElMessage.warning('连接未就绪，请稍候')
    return
  }
  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  agentTyping.value = true
  handoffNotice.value = ''
  send('chat', { text })
  scrollToBottom()
}

function sendQuick(text) {
  inputText.value = text
  doSend()
}

function clearMessages() {
  messages.value = []
  handoffNotice.value = ''
  statusIsHuman.value = false
}

function scrollToBottom() {
  nextTick(() => {
    if (messageArea.value) {
      messageArea.value.scrollTop = messageArea.value.scrollHeight
    }
  })
}

// 启动连接
connect()
onUnmounted(() => disconnect())
</script>

<style scoped>
.user-chat-wrapper {
  min-height: 100%;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 20px;
  background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
}

.chat-container {
  width: 420px;
  height: 700px;
  display: flex;
  flex-direction: column;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0,0,0,0.15);
}

.chat-container .chat-header {
  border-radius: 0;
  background: linear-gradient(135deg, #ff6a00 0%, #ee0979 100%);
  border: none;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
}

.chat-container .chat-header :deep(.el-card__body) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0;
  width: 100%;
  background: transparent;
  color: #fff;
}

.shop-info { display: flex; align-items: center; gap: 10px; }
.avatar {
  width: 40px; height: 40px;
  background: rgba(255,255,255,0.2);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
}
.shop-name { font-weight: 600; font-size: 15px; }

.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 4px;
}
.status-dot.online { background: #4caf50; }
.status-dot.human { background: #2196f3; }
.status-dot.offline { background: #f44336; }

.message-area {
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  background: #f5f5f7;
}

.welcome-message {
  background: #fff;
  border-radius: 12px;
  padding: 14px;
  text-align: center;
  font-size: 13px;
  color: #555;
  line-height: 1.8;
  margin-bottom: 12px;
}
.welcome-icon { font-size: 32px; margin-bottom: 8px; }

.msg { margin-bottom: 14px; display: flex; }
.msg.user { justify-content: flex-end; }
.msg-content {
  max-width: 75%;
  padding: 10px 14px;
  border-radius: 16px;
  font-size: 14px;
  line-height: 1.6;
  word-break: break-all;
  white-space: pre-wrap;
}
.msg.bot .msg-content {
  background: #fff;
  color: #333;
  border-top-left-radius: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.msg.user .msg-content {
  background: linear-gradient(135deg, #ff6a00, #ee0979);
  color: #fff;
  border-top-right-radius: 4px;
}
.msg.bot.human-agent .msg-content {
  background: #e3f2fd;
  border-left: 3px solid #2196f3;
}
.msg.bot.typing .msg-content {
  background: #fff;
  color: #999;
  font-style: italic;
}

.handoff-notice {
  margin: 8px 0;
}

.quick-questions {
  padding: 8px 12px;
  background: #fff;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  border-top: 1px solid #eee;
}

.input-area {
  padding: 12px;
  background: #fff;
  border-top: 1px solid #eee;
  display: flex;
  gap: 10px;
  align-items: flex-end;
}
</style>
