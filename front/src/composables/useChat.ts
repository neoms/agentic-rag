import { ref } from 'vue'
import type { UIMessage, SourceDocument, CitationInfo, ChatHistoryMessage } from '../types'
import { streamChat, getChatHistory, deleteChatHistory, listChatSessions } from '../api/chat'
import * as flow from './agentFlowState'

export interface SessionSummary {
  session_id: string
  preview: string
  message_count: number
  updated_at: number // ms
}

export function useChat() {
  // —— 会话/消息状态：以数据库为唯一数据源，不再写 localStorage ——
  const messages = ref<UIMessage[]>([])
  const sending = ref(false)
  const sessionId = ref('')
  const error = ref<string | null>(null)
  const sessions = ref<SessionSummary[]>([])
  const loading = ref(true)

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

  // 幻觉检测结果 — 仅内存态（SSE 实时写入；刷新后从历史接口按消息恢复）
  const hallucinationResults = ref<Record<string, { passed: boolean; faithfulness: number }>>({})

  // 引文标注元数据
  const streamCitations = ref<Record<string, CitationInfo>>({})

  // —— 辅助函数 ——

  function generateSessionId(): string {
    return crypto.randomUUID().slice(0, 8)
  }

  async function refreshSessions(): Promise<void> {
    try {
      const res = await listChatSessions()
      sessions.value = res.sessions.map(s => ({
        session_id: s.session_id,
        preview: s.preview,
        message_count: s.message_count,
        updated_at: Math.round(s.updated_at * 1000),
      }))
    } catch (e) {
      error.value = e instanceof Error ? e.message : '加载会话列表失败'
    }
  }

  function buildMessagesFromHistory(msgs: ChatHistoryMessage[]): UIMessage[] {
    const built: UIMessage[] = []
    const hMap: Record<string, { passed: boolean; faithfulness: number }> = {}
    for (const m of msgs) {
      const id = crypto.randomUUID()
      const ui: UIMessage = { id, role: m.role, content: m.content, timestamp: Date.now() }
      if (m.role === 'assistant' && m.hallucination) {
        ui.hallucination_passed = m.hallucination.passed
        hMap[id] = { passed: m.hallucination.passed, faithfulness: m.hallucination.faithfulness }
      }
      built.push(ui)
    }
    hallucinationResults.value = hMap
    return built
  }

  async function loadSession(id: string) {
    if (sending.value) return

    error.value = null
    sessionId.value = id
    messages.value = []
    try {
      const result = await getChatHistory(id)
      messages.value = buildMessagesFromHistory(result.messages)
    } catch (e) {
      error.value = e instanceof Error ? e.message : '加载会话历史失败'
    }
  }

  async function init() {
    try {
      await refreshSessions()
      if (sessions.value.length > 0) {
        await loadSession(sessions.value[0].session_id)
      } else {
        newSession()
      }
    } finally {
      loading.value = false
    }
  }

  function newSession() {
    sessionId.value = generateSessionId()
    messages.value = []
    error.value = null
    streamingContent.value = ''
    streamSources.value = []
    streamAgentPath.value = []
    streamCitations.value = {}
    hallucinationResults.value = {}
    flow.currentNode.value = null
    flow.completedNodes.value = []
    flow.skippedNodes.value = []
    flow.nodeDataMap.value = {}
  }

  async function deleteSession(id: string) {
    // 先删除后端持久化历史，失败则中止本地删除，保持前后端一致
    try {
      await deleteChatHistory(id)
    } catch (e) {
      error.value = e instanceof Error ? e.message : '删除会话历史失败'
      return
    }

    // 从内存中移除
    sessions.value = sessions.value.filter(s => s.session_id !== id)

    // 如果删除的是当前会话，切换到最新会话或新建
    if (sessionId.value === id) {
      if (sessions.value.length > 0) {
        await loadSession(sessions.value[0].session_id)
      } else {
        newSession()
      }
    }
  }

  async function send(query: string) {
    if (!query.trim() || sending.value) return
    if (!sessionId.value) newSession()

    error.value = null

    const userMsg: UIMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: Date.now(),
    }
    messages.value.push(userMsg)

    sending.value = true
    isStreaming.value = true
    streamingContent.value = ''
    streamSources.value = []
    streamAgentPath.value = []
    streamCitations.value = {}
    flow.currentNode.value = null
    flow.completedNodes.value = []
    flow.skippedNodes.value = []
    flow.nodeDataMap.value = {}          // 清空上一轮节点 I/O 数据

    const assistantMsg: UIMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isStreaming: true,
    }
    messages.value.push(assistantMsg)
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
                // 供流程图使用：根据执行路径计算运行时跳过的节点
                const skipped: string[] = []
                for (const p of streamAgentPath.value) {
                  // analyze_kg_intent (disabled/empty kg/error) → kg_retrieve skipped
                  if (p.startsWith('analyze_kg_intent') && (p.includes('disabled') || p.includes('empty kg') || p.includes('error'))) {
                    skipped.push('kg_retrieve')
                  }
                }
                flow.skippedNodes.value = skipped
              } catch {}
              break
            case 'citations':
              try {
                const parsedCitations = JSON.parse(data)
                streamCitations.value = parsedCitations
                // 用完整对象替换触发响应式更新，确保子组件 MessageBubble
                // 的 props.citations 能收到变更
                if (messages.value[msgIndex]) {
                  messages.value[msgIndex] = {
                    ...messages.value[msgIndex],
                    citations: parsedCitations,
                  }
                }
              } catch (e) {
                console.error('Failed to parse citations:', e)
              }
              break
            case 'token':
              streamingContent.value += data
              messages.value[msgIndex].content = streamingContent.value
              break
            case 'done':
              messages.value[msgIndex].isStreaming = false
              // 缓存命中时本次只走 cache_lookup → cache_replay，主链并未执行
              const isCacheHit = streamAgentPath.value.includes('cache_replay')
              // generate_simple/complex 是图内节点，node_step 已由 updates 模式在
              // astream 循环中发出。仅当意外缺失时，根据执行路径补全实际执行的节点
              const genNode = streamAgentPath.value.find(p => p.startsWith('generate_'))
              if (genNode && !flow.completedNodes.value.includes(genNode)) {
                flow.completedNodes.value = [...flow.completedNodes.value, genNode]
              }
              // check_hallucination 仅在自反思开启且实际运行时补回（缓存命中时跳过）
              if (enableReflection.value && !isCacheHit && !flow.completedNodes.value.includes('check_hallucination')) {
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
            case 'node_data':
              try {
                flow.nodeDataMap.value = JSON.parse(data)
              } catch (e) {
                console.error('Failed to parse node_data:', e)
              }
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

    // 结束后从数据库刷新会话列表（预览/消息数/排序以数据库为准）
    await refreshSessions()
  }

  // 启动时从数据库加载会话列表与最近会话
  void init()

  return {
    messages,
    sending,
    sessionId,
    error,
    sessions,
    loading,
    streamingContent,
    streamSources,
    isStreaming,
    streamCitations,
    hallucinationResults,
    send,
    loadSession,
    newSession,
    deleteSession,
  }
}
