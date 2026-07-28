import { ref, watch } from 'vue'
import type { UIMessage, SourceDocument } from '../types'
import { streamChat, getChatHistory } from '../api/chat'
import * as flow from './agentFlowState'

export interface SessionSummary {
  session_id: string
  preview: string
  message_count: number
  updated_at: number
}

const LS_KEY_SESSIONS = 'agentic-rag-sessions'
const LS_KEY_MESSAGES_PREFIX = 'agentic-rag-msgs-'
const LS_KEY_LAST_SESSION = 'agentic-rag-last-sid'
const LS_KEY_HALLUCINATION = 'agentic-rag-hallucination'

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

function saveJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {}
}

function loadSessionIds(): string[] {
  return loadJson<string[]>(LS_KEY_SESSIONS, [])
}

function saveSessionIds(ids: string[]): void {
  saveJson(LS_KEY_SESSIONS, ids)
}

function loadMessages(sid: string): UIMessage[] {
  return loadJson<UIMessage[]>(`${LS_KEY_MESSAGES_PREFIX}${sid}`, [])
}

function saveMessages(sid: string, msgs: UIMessage[]): void {
  saveJson(`${LS_KEY_MESSAGES_PREFIX}${sid}`, msgs)
}

function removeMessages(sid: string): void {
  try { localStorage.removeItem(`${LS_KEY_MESSAGES_PREFIX}${sid}`) } catch {}
}

export function useChat() {
  // —— 初始化 ——
  const savedSessionIds = loadSessionIds()
  const lastSid = localStorage.getItem(LS_KEY_LAST_SESSION)

  // 恢复上次活跃会话（如果 session 列表中还存在）
  const initSid = (lastSid && savedSessionIds.includes(lastSid))
    ? lastSid
    : generateSessionId()

  const initMessages = loadMessages(initSid)
  const initSessions = buildSessionSummaries(savedSessionIds)

  const messages = ref<UIMessage[]>(initMessages)
  const sending = ref(false)
  const sessionId = ref(initSid)
  const error = ref<string | null>(null)
  const sessions = ref<SessionSummary[]>(initSessions)

  // Agent 选项：直接使用共享 flow 状态，ChatInput 也是同一份引用
  const enableWebSearch = flow.enableWebSearch
  const enableReflection = flow.enableReflection
  const enableRerank = flow.enableRerank
  const enableGradeDocuments = flow.enableGradeDocuments
  const enableTransformQuery = flow.enableTransformQuery

  // 流式状态
  const streamingContent = ref('')
  const streamSources = ref<SourceDocument[]>([])
  const streamAgentPath = ref<string[]>([])
  const isStreaming = ref(false)

  // 幻觉检测结果 — 独立 ref，key 为消息 ID
  const hallucinationResults = ref<Record<string, { passed: boolean; faithfulness: number }>>(
    loadJson<Record<string, { passed: boolean; faithfulness: number }>>(LS_KEY_HALLUCINATION, {}),
  )

  // —— localStorage 同步 ——
  // 消息变化 → 持久化
  watch(messages, (val) => {
    saveMessages(sessionId.value, val)
    syncSessionIds()
  }, { deep: true })

  // sessionId 变化 → 持久化
  watch(sessionId, (val) => {
    if (val) localStorage.setItem(LS_KEY_LAST_SESSION, val)
  })

  // sessions 变化 → 持久化 id 列表
  watch(sessions, (val) => {
    const ids = val.map(s => s.session_id)
    saveSessionIds(ids)
  }, { deep: true })

  // 幻觉结果变化 → 持久化
  watch(hallucinationResults, (val) => {
    saveJson(LS_KEY_HALLUCINATION, val)
  }, { deep: true })

  // —— 辅助函数 ——

  function generateSessionId(): string {
    return crypto.randomUUID().slice(0, 8)
  }

  function buildSessionSummaries(ids: string[]): SessionSummary[] {
    return ids.map(sid => {
      const msgs = loadMessages(sid)
      const userMsgs = msgs.filter(m => m.role === 'user')
      const preview = userMsgs.length > 0
        ? userMsgs[userMsgs.length - 1].content.slice(0, 40) + (userMsgs[userMsgs.length - 1].content.length > 40 ? '...' : '')
        : '空会话'
      return {
        session_id: sid,
        preview,
        message_count: msgs.length,
        updated_at: msgs.length > 0 ? (msgs[msgs.length - 1].timestamp || 0) : 0,
      }
    }).sort((a, b) => b.updated_at - a.updated_at)
  }

  function syncSessionIds() {
    const currentIds = loadSessionIds()
    if (!currentIds.includes(sessionId.value)) {
      currentIds.push(sessionId.value)
      saveSessionIds(currentIds)
    }
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

    // 先从 localStorage 恢复（瞬时），再从后端刷新（保证最新）
    const cached = loadMessages(id)
    if (cached.length > 0) {
      messages.value = cached
      updateSessionPreview()
      return
    }

    // 本地没有缓存，从后端拉取
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
    const sid = generateSessionId()
    sessionId.value = sid
    messages.value = []
    error.value = null
    streamingContent.value = ''
    streamSources.value = []
    streamAgentPath.value = []
    // 确保新 session 也初始化一份空的 localStorage 条目
    saveMessages(sid, [])
  }

  function deleteSession(id: string) {
    // 从内存中移除
    sessions.value = sessions.value.filter(s => s.session_id !== id)
    removeMessages(id)

    // 如果删除的是当前会话，切换到最新会话或新建
    if (sessionId.value === id) {
      if (sessions.value.length > 0) {
        loadSession(sessions.value[0].session_id)
      } else {
        newSession()
      }
    }
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
    flow.currentNode.value = null
    flow.completedNodes.value = []

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
        enable_rerank: enableRerank.value,
        enable_grade_documents: enableGradeDocuments.value,
        enable_transform_query: enableTransformQuery.value,
        enable_bm25: flow.enableBm25.value,
        enable_hyde: flow.enableHyde.value,
        enable_multi_query: flow.enableMultiQuery.value,
        enable_kg: flow.enableKg.value,
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
            case 'node_start':
              // 节点开始执行 → 设置活跃节点，如果之前已完成则移除（处理 generate 重新激活等场景）
              flow.currentNode.value = data
              flow.completedNodes.value = flow.completedNodes.value.filter(n => n !== data)
              break
            case 'node_step':
              // 节点执行完成 → 标记为已完成
              flow.completedNodes.value = [...flow.completedNodes.value, data]
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
              // 将手动激活的 generate 补回已完成列表（始终执行）
              if (!flow.completedNodes.value.includes('generate')) {
                flow.completedNodes.value = [...flow.completedNodes.value, 'generate']
              }
              // check_hallucination 仅在自反思开启且实际运行时补回
              if (enableReflection.value && !flow.completedNodes.value.includes('check_hallucination')) {
                flow.completedNodes.value = [...flow.completedNodes.value, 'check_hallucination']
              }
              flow.currentNode.value = null
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
              flow.currentNode.value = null
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
    streamingContent,
    streamSources,
    streamAgentPath,
    isStreaming,
    hallucinationResults,
    send,
    loadSession,
    newSession,
    deleteSession,
  }
}
