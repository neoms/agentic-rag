"""对话记忆管理器 - 支持会话隔离、窗口限制（不依赖已废弃的 langchain.memory）"""

import logging
from dataclasses import dataclass, field
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SessionMemory:
    """单个会话的记忆存储"""
    messages: list[dict] = field(default_factory=list)  # [{"role": "user"|"assistant", "content": "..."}]

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return self.messages

    def clear(self):
        self.messages.clear()


class MemoryManager:
    """多会话对话记忆管理器"""

    def __init__(self):
        self._sessions: dict[str, SessionMemory] = {}
        self._window_size = settings.memory_window_size
        logger.info("MemoryManager 初始化: window_size=%d", self._window_size)

    def _get_session(self, session_id: str) -> SessionMemory:
        """获取或创建指定会话"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory()
            logger.info("创建新会话: %s", session_id)
        return self._sessions[session_id]

    def get_history(self, session_id: str) -> list[dict]:
        """获取会话历史消息列表（受窗口大小限制，最近 N 轮）"""
        session = self._sessions.get(session_id)
        if session is None:
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
        logger.info("记录对话: session=%s, query_len=%d, answer_len=%d, total_msgs=%d",
                     session_id, len(user_query), len(assistant_answer), len(session.messages))

    def clear(self, session_id: str):
        """清除指定会话记忆"""
        self._sessions.pop(session_id, None)
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


# 全局单例
memory_manager = MemoryManager()
