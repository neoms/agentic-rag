"""对话记忆管理器 - 支持会话隔离、窗口限制（不依赖已废弃的 langchain.memory）"""

import logging
from dataclasses import dataclass, field
from src.config.settings import settings
from src.store.state_store import get_runtime_state_store

logger = logging.getLogger(__name__)


@dataclass
class SessionMemory:
    """单个会话的记忆存储"""
    messages: list[dict] = field(default_factory=list)  # [{"role": "user"|"assistant", "content": "..."}]

    def add_message(self, role: str, content: str, hallucination: dict | None = None):
        msg: dict = {"role": role, "content": content}
        if hallucination is not None:
            msg["hallucination"] = hallucination
        self.messages.append(msg)

    def get_messages(self) -> list[dict]:
        return self.messages

    def clear(self):
        self.messages.clear()


class MemoryManager:
    """多会话对话记忆管理器"""

    def __init__(self):
        self._sessions: dict[str, SessionMemory] = {}
        self._window_size = settings.memory_window_size
        self._store = get_runtime_state_store()
        logger.info("MemoryManager 初始化: window_size=%d", self._window_size)

    def _get_session(self, session_id: str) -> SessionMemory:
        """获取或创建指定会话"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory()
            # 从持久化存储懒加载历史（重启后恢复）
            try:
                for msg in self._store.get_messages(session_id):
                    self._sessions[session_id].add_message(
                        msg["role"], msg["content"], msg.get("hallucination")
                    )
                logger.info(
                    "会话 %s: 从持久化恢复 %d 条消息",
                    session_id,
                    len(self._sessions[session_id].get_messages()),
                )
            except Exception as e:
                logger.warning("会话历史恢复失败（不影响使用）: %s", e)
        return self._sessions[session_id]

    def get_history(self, session_id: str) -> list[dict]:
        """获取会话历史消息列表（受窗口大小限制，最近 N 轮）"""
        session = self._sessions.get(session_id)
        if session is None:
            session = self._get_session(session_id)
            if not session.get_messages():
                return []
        # 窗口限制: 最近 N 轮对话 = N*2 条消息（一问一答）
        all_msgs = session.get_messages()
        max_msgs = self._window_size * 2
        logger.info("获取历史: session=%s, total=%d, max=%d", session_id, len(all_msgs), max_msgs)
        if len(all_msgs) > max_msgs:
            return all_msgs[-max_msgs:]
        return all_msgs

    def add_interaction(self, session_id: str, user_query: str, assistant_answer: str):
        """记录一轮对话"""
        session = self._get_session(session_id)
        session.add_message("user", user_query)
        session.add_message("assistant", assistant_answer)
        # 同步持久化（失败不影响对话主流程）
        try:
            self._store.add_message(session_id, "user", user_query)
            self._store.add_message(session_id, "assistant", assistant_answer)
        except Exception as e:
            logger.warning("会话历史持久化失败（不影响对话）: %s", e)
        logger.info("记录对话: session=%s, query_len=%d, answer_len=%d, total_msgs=%d",
                     session_id, len(user_query), len(assistant_answer), len(session.messages))

    def add_hallucination_result(
        self, session_id: str, passed: bool, faithfulness: float
    ):
        """记录幻觉检测结果：更新内存 + 持久化到最近一条 assistant 消息"""
        result = {"passed": bool(passed), "faithfulness": round(float(faithfulness), 1)}
        session = self._get_session(session_id)
        # 更新内存中最近一条 assistant 消息
        for msg in reversed(session.messages):
            if msg["role"] == "assistant":
                msg["hallucination"] = result
                break
        # 同步持久化（失败不影响对话主流程）
        try:
            self._store.set_hallucination(session_id, result)
        except Exception as e:
            logger.warning("幻觉结果持久化失败（不影响对话）: %s", e)
        logger.info("记录幻觉结果: session=%s, passed=%s, faithfulness=%.1f",
                    session_id, passed, faithfulness)

    def clear(self, session_id: str):
        """清除指定会话记忆"""
        self._sessions.pop(session_id, None)
        try:
            self._store.clear_messages(session_id)
        except Exception as e:
            logger.warning("会话历史清理失败: %s", e)
        logger.info("清除会话: %s", session_id)

    def get_chat_history_string(self, session_id: str) -> str:
        """获取会话历史的字符串表示，用于注入 Prompt"""
        history = self.get_history(session_id)
        if not history:
            return ""
        lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    @property
    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def list_sessions(self) -> list[dict]:
        """返回全部会话摘要（按最近活跃时间倒序）"""
        try:
            return self._store.list_sessions()
        except Exception as e:
            logger.warning("会话列表加载失败: %s", e)
            return []


# 全局单例
memory_manager = MemoryManager()
