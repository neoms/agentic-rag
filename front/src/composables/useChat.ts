import { ref } from 'vue'
import type { UIMessage, ChatMode, SourceDocument, AgenticChatResponse } from '../types'
import { simpleChat, agenticChat, streamChat, getChatHistory } from '../api/chat'

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
  const mode = ref<ChatMode>('simple')
  const error = ref<string | null>(null)

  // 会话列表（本地维护）
  const sessions = ref<SessionSummary[]>([])

  // Agent 模式选项
  const enableWebSearch = ref(false)
  const enableReflection = ref(true)

  // 流式状态
  const streamingContent = ref('')
  const streamSources = ref<SourceDocument[]>([])
  const streamAgentPath = ref<string[]>([])
  const isStreaming = ref(false)

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
      // 重新排序：最新在前
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

    if (mode.value === 'stream') {
      await sendStream(query)
    } else {
      await sendNormal(query)
    }

    updateSessionPreview()
  }

  async function sendNormal(query: string) {
    sending.value = true

    try {
      let response
      if (mode.value === 'simple') {
        response = await simpleChat({
          query,
          session_id: sessionId.value,
        })
      } else {
        response = await agenticChat({
          query,
          session_id: sessionId.value,
          enable_web_search: enableWebSearch.value,
          enable_reflection: enableReflection.value,
        })
      }

      if (response) {
        const agenticResponse = response as AgenticChatResponse
        const assistantMsg: UIMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response.answer,
          timestamp: Date.now(),
          sources: response.sources,
          reflection_count: response.reflection_count,
          agent_path: 'agent_path' in response ? agenticResponse.agent_path : undefined,
          tool_calls: 'tool_calls' in response ? agenticResponse.tool_calls : undefined,
        }
        addMessage(assistantMsg)
        sessionId.value = response.session_id
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '对话请求失败'
    } finally {
      sending.value = false
    }
  }

  async function sendStream(query: string) {
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
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('event: ')) continue

          if (line.startsWith('data: ')) {
            const eventType = getCurrentEventType(lines, line)
            const data = line.slice(6)

            switch (eventType) {
              case 'source':
                try {
                  streamSources.value = JSON.parse(data)
                  assistantMsg.sources = streamSources.value
                } catch {}
                break
              case 'path':
                try {
                  streamAgentPath.value = JSON.parse(data)
                  assistantMsg.agent_path = streamAgentPath.value
                } catch {}
                break
              case 'token':
                streamingContent.value += data
                assistantMsg.content = streamingContent.value
                break
              case 'done':
                assistantMsg.isStreaming = false
                break
              case 'error':
                try {
                  const err = JSON.parse(data)
                  error.value = err.detail || '流式对话出错'
                } catch {
                  error.value = data || '流式对话出错'
                }
                assistantMsg.isStreaming = false
                break
            }
          }
        }
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '流式对话失败'
      assistantMsg.isStreaming = false
    } finally {
      if (reader) {
        try { reader.releaseLock() } catch {}
      }
      isStreaming.value = false
      sending.value = false
    }
  }

  function getCurrentEventType(lines: string[], currentLine: string): string {
    const idx = lines.indexOf(currentLine)
    if (idx > 0 && lines[idx - 1].startsWith('event: ')) {
      return lines[idx - 1].slice(7).trim()
    }
    for (let i = idx - 1; i >= 0; i--) {
      if (lines[i].startsWith('event: ')) {
        return lines[i].slice(7).trim()
      }
    }
    return 'token'
  }

  return {
    messages,
    sending,
    sessionId,
    mode,
    error,
    sessions,
    enableWebSearch,
    enableReflection,
    streamingContent,
    streamSources,
    streamAgentPath,
    isStreaming,
    send,
    loadSession,
    newSession,
  }
}
