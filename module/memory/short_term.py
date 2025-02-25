"""
短期记忆实现，基于内存列表存储最近的对话消息
"""
import json
import os
from typing import Callable, Dict, List, Optional, Any

from base.dialogue.memory import Memory, Message, MessageRole


class ShortTermMemory(Memory):
    """短期记忆，保存最近的对话"""
    
    def __init__(self, max_messages: int = 10):
        """初始化短期记忆
        
        Args:
            max_messages: 最大保存的消息数量
        """
        self.max_messages = max_messages
        self.messages: List[Message] = []
    
    async def add(self, message: Message) -> None:
        """添加消息到记忆
        
        Args:
            message: 要添加的消息
        """
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    async def get_recent(self, k: int) -> List[Message]:
        """获取最近k条消息
        
        Args:
            k: 要获取的消息数量
            
        Returns:
            最近的k条消息列表
        """
        return self.messages[-min(k, len(self.messages)):]
    
    async def search(self, query: str, limit: int = 5) -> List[Message]:
        """搜索包含查询词的消息
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            匹配的消息列表
        """
        results = [msg for msg in self.messages if query.lower() in msg.content.lower()]
        return results[:limit]
    
    async def clear(self) -> None:
        """清空记忆"""
        self.messages.clear()
    
    async def get_formatted_history(self, formatter: Optional[Callable[[Message], str]] = None) -> str:
        """获取格式化的历史记录
        
        Args:
            formatter: 自定义格式化函数，如果为None则使用默认格式化
            
        Returns:
            格式化的历史记录字符串
        """
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
        """保存记忆到文件
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([msg.to_dict() for msg in self.messages], f, ensure_ascii=False, indent=2)
    
    async def load(self, path: str) -> None:
        """从文件加载记忆
        
        Args:
            path: 加载路径
        """
        if not os.path.exists(path):
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.messages = [Message.from_dict(item) for item in data]
    
    def __len__(self) -> int:
        """获取记忆中的消息数量"""
        return len(self.messages) 