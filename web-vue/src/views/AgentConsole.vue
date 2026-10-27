<template>
  <div class="agent-layout">
    <!-- 左侧面板（待接入 + 进行中） -->
    <el-card class="side-panel" shadow="hover">
      <el-tabs v-model="sideTab" class="side-tabs">
        <!-- ====== Tab 1: 待接入队列 ====== -->
        <el-tab-pane name="queue">
          <template #label>
            📋 待接入
            <el-tag v-if="queueList.length" type="danger" effect="dark" round size="small">
              {{ queueList.length }}
            </el-tag>
          </template>

          <div v-if="queueList.length === 0" class="panel-empty">
            暂无待接入<br>用户点【转人工】后会在这里出现
          </div>

          <div
            v-for="sid in queueList"
            :key="sid"
            class="queue-item"
            @click="takeSession(sid)"
          >
            <div class="q-top">🔔 新转接请求</div>
            <div class="q-sid">{{ sid }}</div>
            <el-button size="small" type="primary" @click.stop="takeSession(sid)">
              ▶ 立即接入
            </el-button>
          </div>
        </el-tab-pane>

        <!-- ====== Tab 2: 进行中的会话 ====== -->
        <el-tab-pane name="active">
          <template #label>
            💬 进行中
            <el-tag v-if="activeSessions.length" type="primary" effect="dark" round size="small">
              {{ activeSessions.length }}
            </el-tag>
          </template>

          <div v-if="activeSessions.length === 0" class="panel-empty">
            暂无可服务会话<br>接入用户后会出现在这里
          </div>

          <div
            v-for="s in activeSessions"
            :key="s.session_id"
            :class="['active-item', { active: currentSession === s.session_id }]"
            @click="selectActiveSession(s.session_id)"
          >
            <div class="q-top">💬 {{ s.session_id }}</div>
            <div class="q-sub">{{ s.history?.length || 0 }} 条消息</div>
          </div>
        </el-tab-pane>
      </el-tabs>
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
        :title="activeSessions.length ? '👈 从左侧「进行中」选择一个会话' : '👈 从左侧「待接入」接入用户会话'"
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

const sideTab = ref('queue')                    // 左侧 Tab 切换
const queueList = ref([])                       // 待接入队列
const activeSessions = ref([])                  // 已接入会话: [{session_id, history: [{role, content}]}]
const sessionMessageMap = ref({})               // 每个会话独立的消息缓存: {session_id: [{role, content}]}

const agentId = ref('')
const currentSession = ref('')                  // 当前选中的 session_id
const messages = ref([])                        // 当前会话的消息（派生自 sessionMessageMap[currentSession]）
const inputText = ref('')
const messageArea = ref(null)

const sessionInfo = computed(() => {
  if (!currentSession.value) {
    if (activeSessions.value.length > 0) return `有 ${activeSessions.value.length} 个会话进行中，请选择`
    return '等待接入用户...'
  }
  return `正在服务 ${currentSession.value}`
})

// 当前会话消息变化 → 更新视图
function loadMessagesOf(sid) {
  messages.value = sessionMessageMap.value[sid] || []
  nextTick(scrollToBottom)
}

function selectActiveSession(sid) {
  currentSession.value = sid
  loadMessagesOf(sid)
}

function updateActiveList(newList) {
  // 确保每个 active session 的 history 同步进 sessionMessageMap
  for (const s of newList) {
    if (!sessionMessageMap.value[s.session_id] && s.history?.length) {
      sessionMessageMap.value[s.session_id] = [...s.history]
    }
  }
  activeSessions.value = newList
}

function handleMessage(type, data) {
  switch (type) {
    case 'session_start':
      agentId.value = data.agent_id
      break

    case 'agent_sessions':
      // 客服上线 / close_session 后，后端推送最新的 active 列表
      updateActiveList(data || [])
      break

    case 'queue_update':
      queueList.value = data || []
      break

    case 'new_handoff':
      if (data.session_id && !queueList.value.includes(data.session_id)) {
        queueList.value.push(data.session_id)
      }
      break

    case 'session_taken': {
      const sid = data.session_id
      const history = data.history || []
      // 加入 active 列表
      if (!activeSessions.value.find(s => s.session_id === sid)) {
        activeSessions.value.push({ session_id: sid, history })
      }
      // 把历史存进 sessionMessageMap
      sessionMessageMap.value[sid] = [...history]
      // 自动选中刚接入的会话
      currentSession.value = sid
      loadMessagesOf(sid)
      sideTab.value = 'active'
      ElMessage.success(`已接入会话 ${sid}`)
      break
    }

    case 'user_message': {
      // 用户发的消息（被人工接管后直接转发过来的）
      const sid = data.session_id
      if (!sessionMessageMap.value[sid]) {
        sessionMessageMap.value[sid] = []
      }
      sessionMessageMap.value[sid].push({ role: 'user', content: data.text })
      // 如果这是当前选中的会话 → 刷新视图
      if (currentSession.value === sid) {
        loadMessagesOf(sid)
      }
      break
    }

    case 'session_closed': {
      const sid = data.session_id
      // 从 active 列表移除
      activeSessions.value = activeSessions.value.filter(s => s.session_id !== sid)
      delete sessionMessageMap.value[sid]
      // 如果是当前选中的会话 → 清空并切换
      if (currentSession.value === sid) {
        currentSession.value = ''
        messages.value = []
        // 自动切到下一个 active 会话
        if (activeSessions.value.length > 0) {
          selectActiveSession(activeSessions.value[0].session_id)
        }
        ElMessage.info('会话已关闭')
      }
      break
    }
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
  // 先本地乐观更新
  if (!sessionMessageMap.value[currentSession.value]) {
    sessionMessageMap.value[currentSession.value] = []
  }
  sessionMessageMap.value[currentSession.value].push({ role: 'bot', content: text })
  inputText.value = ''
  loadMessagesOf(currentSession.value)
}

function closeSession() {
  if (!currentSession.value) return
  const sid = currentSession.value
  send('close_session', { session_id: sid })
  // optimistic: 从 active 列表移除
  activeSessions.value = activeSessions.value.filter(s => s.session_id !== sid)
  delete sessionMessageMap.value[sid]
  currentSession.value = ''
  messages.value = []
  // 自动切到下一个
  if (activeSessions.value.length > 0) {
    selectActiveSession(activeSessions.value[0].session_id)
  }
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

<style scoped>
.agent-layout {
  display: flex;
  gap: 20px;
  padding: 20px;
  min-height: 100vh;
  background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
}

/* ====== 左侧面板（Tab 切换）====== */
.side-panel {
  width: 300px;
  height: calc(100vh - 40px);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
}

.side-panel :deep(.el-card__body) {
  flex: 1;
  overflow-y: auto;
  padding: 0;
}

.side-tabs :deep(.el-tabs__header) {
  margin: 0;
  padding: 8px 8px 0;
  background: #fafafa;
}
.side-tabs :deep(.el-tabs__content) {
  padding: 8px;
}

.panel-empty {
  color: #999;
  font-size: 13px;
  text-align: center;
  padding: 40px 0;
  line-height: 1.8;
}

.queue-item, .active-item {
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
.active-item:hover {
  background: #e3f2fd;
  border-left-color: #1976d2;
}
.active-item.active {
  background: #1976d2;
  color: #fff;
  border-left-color: #0d47a1;
}
.active-item.active .q-sub {
  color: rgba(255,255,255,0.7);
}

.q-top { font-size: 12px; color: #333; }
.active-item.active .q-top { color: #fff; }
.q-sid { font-size: 11px; color: #999; margin: 4px 0 8px; }
.q-sub { font-size: 11px; color: #888; margin-top: 4px; }

/* ====== 右侧聊天 ====== */
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
