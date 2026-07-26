import { ref } from 'vue'
import type { UIMessage, SourceDocument } from '../types'
import { streamChat, getChatHistory } from '../api/chat'

export interface SessionSummary {
  session_id: string
  preview: string
  message_count: number
  updated_at: number
}

export function useChat() {
  const messages = ref<UIMessage[]>([])
  const sending = ref(false)
  const sessionId = ref(generateSessionId())
  const error = ref<string | null>(null)

  // 会话列表（本地维护）
  const sessions = ref<SessionSummary[]>([])

  // Agent 选项
  const enableWebSearch = ref(false)
  const enableReflection = ref(true)

  // 流式状态
  const streamingContent = ref('')
  const streamSources = ref<SourceDocument[]>([])
  const streamAgentPath = ref<string[]>([])
  const isStreaming = ref(false)

  // 幻觉检测结果 — 独立 ref，key 为消息 ID
  const hallucinationResults = ref<Record<string, { passed: boolean; faithfulness: number }>>({})

  function generateSessionId(): string {
    return crypto.randomUUID().slice(0, 8)
  }

  function addMessage(msg: UIMessage) {
    messages.value.push(msg)
  }

  function updateSessionPreview() {
    const userMsgs = messages.value.filter(m => m.role === 'user')
    const preview = userMsgs.length > 0
      ? userMsgs[userMsgs.length - 1].content.slice(0, 40) + (userMsgs[userMsgs.length - 1].content.length > 40 ? '...' : '')
      : '空会话'

    const existing = sessions.value.find(s => s.session_id === sessionId.value)
    if (existing) {
      existing.preview = preview
      existing.message_count = messages.value.length
      existing.updated_at = Date.now()
      sessions.value.sort((a, b) => b.updated_at - a.updated_at)
    } else if (messages.value.length > 0) {
      sessions.value.unshift({
        session_id: sessionId.value,
        preview,
        message_count: messages.value.length,
        updated_at: Date.now(),
      })
    }
  }

  async function loadSession(id: string) {
    if (sending.value) return

    error.value = null
    sessionId.value = id
    messages.value = []

    try {
      const result = await getChatHistory(id)
      if (result && result.messages.length > 0) {
        messages.value = result.messages.map(msg => ({
          id: crypto.randomUUID(),
          role: msg.role,
          content: msg.content,
          timestamp: Date.now(),
        }))
        updateSessionPreview()
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '加载会话历史失败'
    }
  }

  function newSession() {
    sessionId.value = generateSessionId()
    messages.value = []
    error.value = null
    streamingContent.value = ''
    streamSources.value = []
    streamAgentPath.value = []
  }

  async function send(query: string) {
    if (!query.trim() || sending.value) return

    error.value = null

    const userMsg: UIMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: Date.now(),
    }
    addMessage(userMsg)

    sending.value = true
    isStreaming.value = true
    streamingContent.value = ''
    streamSources.value = []
    streamAgentPath.value = []

    const assistantMsg: UIMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isStreaming: true,
    }
    addMessage(assistantMsg)
    const assistantId = assistantMsg.id
    const msgIndex = messages.value.length - 1

    // 等待浏览器渲染空消息气泡 + 光标
    await new Promise(r => requestAnimationFrame(r))

    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null

    try {
      const response = await streamChat({
        query,
        session_id: sessionId.value,
        enable_web_search: enableWebSearch.value,
        enable_reflection: enableReflection.value,
        stream: true,
      })

      reader = response.body?.getReader() ?? null
      if (!reader) throw new Error('无法获取响应流')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        buffer = buffer.replace(/\r\n/g, '\n')
        const events = buffer.split('\n\n')
        buffer = events.pop() || ''

        for (const rawEvent of events) {
          if (!rawEvent.trim()) continue

          const lines = rawEvent.split('\n')
          let eventType = 'token'
          const dataLines: string[] = []

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim()
            } else if (line.startsWith('data: ')) {
              dataLines.push(line.slice(6))
            }
          }

          const data = dataLines.join('\n')

          switch (eventType) {
            case 'source':
              try {
                streamSources.value = JSON.parse(data)
                messages.value[msgIndex].sources = streamSources.value
              } catch {}
              break
            case 'path':
              try {
                streamAgentPath.value = JSON.parse(data)
                messages.value[msgIndex].agent_path = streamAgentPath.value
              } catch {}
              break
            case 'token':
              streamingContent.value += data
              messages.value[msgIndex].content = streamingContent.value
              break
            case 'done':
              messages.value[msgIndex].isStreaming = false
              break
            case 'hallucination':
              try {
                const hr = JSON.parse(data)
                hallucinationResults.value = {
                  ...hallucinationResults.value,
                  [assistantId]: { passed: hr.passed, faithfulness: hr.faithfulness },
                }
              } catch {}
              break
            case 'error':
              try {
                const err = JSON.parse(data)
                error.value = err.detail || '流式对话出错'
              } catch {
                error.value = data || '流式对话出错'
              }
              messages.value[msgIndex].isStreaming = false
              break
          }

          await new Promise(r => setTimeout(r, 0))
        }
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '流式对话失败'
      messages.value[msgIndex].isStreaming = false
    } finally {
      if (reader) {
        try { reader.releaseLock() } catch {}
      }
      isStreaming.value = false
      sending.value = false
    }

    updateSessionPreview()
  }

  return {
    messages,
    sending,
    sessionId,
    error,
    sessions,
    enableWebSearch,
    enableReflection,
    streamingContent,
    streamSources,
    streamAgentPath,
    isStreaming,
    hallucinationResults,
    send,
    loadSession,
    newSession,
  }
}
