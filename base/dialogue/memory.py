"""
对话记忆管理模块，提供对话历史的存储和检索功能
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


class Memory(ABC):
    """记忆抽象基类"""
    
    @abstractmethod
    async def add(self, message: Message) -> None:
        """添加消息到记忆"""
        pass
    
    @abstractmethod
    async def get_recent(self, k: int) -> List[Message]:
        """获取最近k条消息"""
        pass
    
    @abstractmethod
    async def search(self, query: str) -> List[Message]:
        """搜索相关消息"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """清空记忆"""
        pass


class ShortTermMemory(Memory):
    """短期记忆，保存最近的对话"""
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages: List[Message] = []
    
    async def add(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    async def get_recent(self, k: int) -> List[Message]:
        return self.messages[-k:]
    
    async def search(self, query: str) -> List[Message]:
        # 简单实现：返回包含查询词的消息
        return [msg for msg in self.messages if query.lower() in msg.content.lower()]
    
    async def clear(self) -> None:
        self.messages.clear()


class LongTermMemory(Memory):
    """长期记忆，使用向量存储保存历史对话"""
    
    def __init__(self, vector_store: Any):
        self.vector_store = vector_store
    
    async def add(self, message: Message) -> None:
        # 实现向量存储的添加逻辑
        pass
    
    async def get_recent(self, k: int) -> List[Message]:
        # 实现向量存储的检索逻辑
        pass
    
    async def search(self, query: str) -> List[Message]:
        # 实现向量存储的语义搜索
        pass
    
    async def clear(self) -> None:
        # 实现向量存储的清空逻辑
        pass


class MemoryConfig(BaseModel):
    """记忆配置"""
    use_short_term: bool = True
    use_long_term: bool = True
    short_term_size: int = 10
    long_term_threshold: float = 0.7  # 长期记忆的相关性阈值
    extra_params: Dict[str, Any] = {} 