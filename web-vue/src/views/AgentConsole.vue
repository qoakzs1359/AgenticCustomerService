<template>
  <div class="agent-layout">
    <!-- 左侧队列 -->
    <el-card class="queue-panel" shadow="hover">
      <template #header>
        <div class="queue-header">
          <span class="queue-title">📋 待接入队列</span>
          <el-tag v-if="queueList.length" type="danger" effect="dark" round>{{ queueList.length }}</el-tag>
        </div>
      </template>

      <div v-if="queueList.length === 0" class="queue-empty">
        暂无待接入会话<br>有用户点【转人工】后会在这里出现
      </div>

      <div
        v-for="sid in queueList"
        :key="sid"
        class="queue-item"
        @click="takeSession(sid)"
      >
        <div class="q-top">🔔 用户请求转接人工</div>
        <div class="q-sid">{{ sid }}</div>
        <el-button size="small" type="primary" @click.stop="takeSession(sid)">▶ 立即接入</el-button>
      </div>
    </el-card>

    <!-- 右侧聊天区 -->
    <el-card class="chat-panel" shadow="hover">
      <template #header>
        <div class="chat-header">
          <div class="header-left">
            <div class="avatar">🧑‍💼</div>
            <div class="header-meta">
              <div class="header-name">
                人工坐席 <span class="agent-id">[{{ agentId || '--' }}]</span>
              </div>
              <div class="header-sub">{{ sessionInfo }}</div>
            </div>
          </div>
          <el-button
            v-if="currentSession"
            type="danger"
            size="small"
            :icon="SwitchButton"
            plain
            @click="closeSession"
          >
            关闭会话
          </el-button>
        </div>
      </template>

      <!-- 提示 -->
      <el-alert
        v-if="!currentSession"
        title="👈 从左侧队列点击接入用户会话"
        type="info"
        :closable="false"
        show-icon
        class="tip-alert"
      />

      <!-- 消息列表 -->
      <div v-show="currentSession" ref="messageArea" class="message-area">
        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          :class="['msg', msg.role]"
        >
          <div class="msg-content">{{ msg.content }}</div>
        </div>
      </div>

      <!-- 输入区 -->
      <div class="input-area">
        <el-input
          v-model="inputText"
          type="textarea"
          :rows="2"
          placeholder="选择会话后输入回复...（Enter 发送）"
          resize="none"
          :disabled="!currentSession"
          @keydown.enter.exact.prevent="doSend"
        />
        <el-button
          type="primary"
          :icon="Promotion"
          size="large"
          :disabled="!currentSession"
          @click="doSend"
        >发送</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onUnmounted } from 'vue'
import { useSocket } from '../composables/useSocket'
import { ElMessage } from 'element-plus'
import { Promotion, SwitchButton } from '@element-plus/icons-vue'

const queueList = ref([])
const agentId = ref('')
const currentSession = ref('')
const messages = ref([])
const inputText = ref('')
const messageArea = ref(null)

const sessionInfo = computed(() => {
  if (!currentSession.value) return '等待接入用户...'
  return `正在服务 ${currentSession.value}`
})

function handleMessage(type, data) {
  switch (type) {
    case 'session_start':
      agentId.value = data.agent_id
      break
    case 'queue_update':
      queueList.value = data || []
      break
    case 'new_handoff':
      // queue_update 会同步刷新，这里不用额外处理
      break
    case 'session_taken':
      currentSession.value = data.session_id
      messages.value = []
      ElMessage.success(`已接入会话 ${data.session_id}`)
      break
    case 'user_message':
      if (currentSession.value === data.session_id) {
        messages.value.push({ role: 'user', content: data.text })
        scrollToBottom()
      }
      break
    case 'session_closed':
      if (currentSession.value === data.session_id) {
        currentSession.value = ''
        messages.value = []
        ElMessage.info('会话已关闭')
      }
      break
  }
}

const { connect, send, disconnect } = useSocket('/ws/agent', handleMessage)

function takeSession(sid) {
  send('take_session', { session_id: sid })
}

function doSend() {
  const text = inputText.value.trim()
  if (!text || !currentSession.value) return
  send('send', { session_id: currentSession.value, text })
  messages.value.push({ role: 'bot', content: text })
  inputText.value = ''
  scrollToBottom()
}

function closeSession() {
  if (!currentSession.value) return
  send('close_session', { session_id: currentSession.value })
  currentSession.value = ''
  messages.value = []
}

function scrollToBottom() {
  nextTick(() => {
    if (messageArea.value) {
      messageArea.value.scrollTop = messageArea.value.scrollHeight
    }
  })
}

connect()
onUnmounted(() => disconnect())
</script>

<script>
export default { name: 'AgentConsole' }
</script>

<style scoped>
.agent-layout {
  display: flex;
  gap: 20px;
  padding: 20px;
  min-height: 100vh;
  background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
}

/* 左侧队列 */
.queue-panel {
  width: 280px;
  height: calc(100vh - 40px);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
}

.queue-panel :deep(.el-card__body) {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.queue-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.queue-title { font-size: 15px; font-weight: 700; }

.queue-empty {
  color: #999;
  font-size: 13px;
  text-align: center;
  padding: 40px 0;
  line-height: 1.8;
}

.queue-item {
  background: #f5f5f5;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.2s;
  border-left: 3px solid transparent;
}
.queue-item:hover {
  background: #fff3e0;
  border-left-color: #ff6a00;
}
.q-top { font-size: 12px; color: #333; }
.q-sid { font-size: 11px; color: #999; margin: 4px 0 8px; }

/* 右侧聊天 */
.chat-panel {
  flex: 1;
  height: calc(100vh - 40px);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.header-left { display: flex; align-items: center; gap: 12px; }
.avatar {
  width: 42px; height: 42px;
  background: #e3f2fd;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
}
.header-name { font-weight: 600; font-size: 15px; }
.agent-id { font-size: 12px; color: #666; font-weight: normal; }
.header-sub { font-size: 12px; color: #1976d2; }

.tip-alert { margin-bottom: 8px; }

.message-area {
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  background: #f5f5f7;
  border-radius: 8px;
  margin: 8px 0;
}

.msg { margin-bottom: 12px; display: flex; }
.msg.user { justify-content: flex-start; }
.msg.bot { justify-content: flex-end; }
.msg-content {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
  word-break: break-all;
  white-space: pre-wrap;
}
.msg.user .msg-content {
  background: #fff;
  color: #333;
  border-top-left-radius: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.msg.bot .msg-content {
  background: linear-gradient(135deg, #1976d2, #42a5f5);
  color: #fff;
  border-top-right-radius: 4px;
}

.input-area {
  padding-top: 8px;
  display: flex;
  gap: 10px;
  align-items: flex-end;
}
</style>
