/**
 * WebSocket 连接管理 composable
 * 参数:
 *   endpoint: '/ws/user' 或 '/ws/agent'
 *   onMessageHandler: 一个函数，收到 {type, data} 消息时调用
 */
import { ref, onUnmounted } from 'vue'

export function useSocket(endpoint, onMessageHandler) {
  const connected = ref(false)
  const url = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${endpoint}`

  let ws = null
  let reconnectTimer = null
  let manualClose = false

  function connect() {
    if (manualClose) return
    ws = new WebSocket(url)

    ws.onopen = () => {
      connected.value = true
      console.log(`[WS] ${endpoint} 连接成功`)
    }

    ws.onmessage = (event) => {
      try {
        const { type, data } = JSON.parse(event.data)
        onMessageHandler?.(type, data)
      } catch (e) {
        console.error('[WS] 消息解析失败:', e)
      }
    }

    ws.onclose = () => {
      connected.value = false
      ws = null
      if (!manualClose) {
        reconnectTimer = setTimeout(connect, 3000)
      }
    }

    ws.onerror = (err) => {
      console.error('[WS] 错误:', err)
    }
  }

  function send(type, data) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false
    ws.send(JSON.stringify({ type, ...data }))
    return true
  }

  function disconnect() {
    manualClose = true
    if (reconnectTimer) clearTimeout(reconnectTimer)
    ws?.close()
  }

  onUnmounted(() => {
    disconnect()
  })

  return { connected, connect, send, disconnect }
}
