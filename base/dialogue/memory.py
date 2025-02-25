"""
对话记忆管理模块，提供对话历史的存储和检索功能
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic

from pydantic import BaseModel, Field


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
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典格式"""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建消息"""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            metadata=data.get("metadata", {})
        )


T = TypeVar('T')

class Memory(Generic[T], ABC):
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
    async def search(self, query: str, limit: int = 5) -> List[Message]:
        """搜索相关消息
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """清空记忆"""
        pass
    
    @abstractmethod
    async def get_formatted_history(self, formatter: Optional[Callable[[Message], str]] = None) -> str:
        """获取格式化的历史记录
        
        Args:
            formatter: 自定义格式化函数，如果为None则使用默认格式化
        """
        pass
    
    @abstractmethod
    async def save(self, path: str) -> None:
        """保存记忆到文件
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    async def load(self, path: str) -> None:
        """从文件加载记忆
        
        Args:
            path: 加载路径
        """
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
        return self.messages[-min(k, len(self.messages)):]
    
    async def search(self, query: str, limit: int = 5) -> List[Message]:
        # 简单实现：返回包含查询词的消息
        results = [msg for msg in self.messages if query.lower() in msg.content.lower()]
        return results[:limit]
    
    async def clear(self) -> None:
        self.messages.clear()
    
    async def get_formatted_history(self, formatter: Optional[Callable[[Message], str]] = None) -> str:
        """获取格式化的历史记录"""
        if not self.messages:
            return ""
        
        if formatter:
            return "\n".join(formatter(msg) for msg in self.messages)
        
        # 默认格式化
        formatted = []
        for msg in self.messages:
            role_name = {
                MessageRole.SYSTEM: "系统",
                MessageRole.USER: "用户",
                MessageRole.ASSISTANT: "助手",
                MessageRole.FUNCTION: "函数"
            }.get(msg.role, str(msg.role))
            formatted.append(f"{role_name}: {msg.content}")
        
        return "\n".join(formatted)
    
    async def save(self, path: str) -> None:
        """保存记忆到文件"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([msg.to_dict() for msg in self.messages], f, ensure_ascii=False, indent=2)
    
    async def load(self, path: str) -> None:
        """从文件加载记忆"""
        import json
        import os
        if not os.path.exists(path):
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.messages = [Message.from_dict(item) for item in data]


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
    use_long_term: bool = False
    short_term_size: int = 10
    long_term_threshold: float = 0.7  # 长期记忆的相关性阈值
    save_path: Optional[str] = None  # 记忆保存路径
    extra_params: Dict[str, Any] = Field(default_factory=dict) 