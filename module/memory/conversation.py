"""
对话记忆实现，专门用于管理对话历史
"""
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from base.dialogue.memory import Memory, Message, MessageRole


class ConversationMemory(Memory):
    """对话记忆类，用于管理对话历史"""
    
    def __init__(self, max_turns: int = 5):
        """初始化对话记忆
        
        Args:
            max_turns: 最大保存的对话轮次
        """
        self.max_turns = max_turns
        self.messages: List[Message] = []
    
    async def add(self, message: Message) -> None:
        """添加消息到记忆
        
        Args:
            message: 要添加的消息
        """
        self.messages.append(message)
        self._truncate_history()
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加用户消息
        
        Args:
            content: 用户消息内容
            metadata: 元数据
        """
        message = Message(
            role=MessageRole.USER,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self._truncate_history()
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加助手消息
        
        Args:
            content: 助手消息内容
            metadata: 元数据
        """
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self._truncate_history()
    
    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加系统消息
        
        Args:
            content: 系统消息内容
            metadata: 元数据
        """
        message = Message(
            role=MessageRole.SYSTEM,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self._truncate_history()
    
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
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取对话历史
        
        Returns:
            对话历史列表
        """
        return [msg.to_dict() for msg in self.messages]
    
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
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """获取适用于LLM的消息格式
        
        Returns:
            适用于LLM的消息列表
        """
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.messages
        ]
    
    async def save(self, path: str) -> None:
        """保存记忆到文件
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "max_turns": self.max_turns,
                "messages": [msg.to_dict() for msg in self.messages]
            }, f, ensure_ascii=False, indent=2)
    
    async def load(self, path: str) -> None:
        """从文件加载记忆
        
        Args:
            path: 加载路径
        """
        if not os.path.exists(path):
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.max_turns = data.get("max_turns", self.max_turns)
            self.messages = [Message.from_dict(item) for item in data.get("messages", [])]
    
    def _truncate_history(self) -> None:
        """截断历史，保持在最大轮次内"""
        # 计算用户和助手消息的对数
        user_assistant_pairs = 0
        roles_seen = set()
        
        for msg in reversed(self.messages):
            if msg.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                roles_seen.add(msg.role)
                if len(roles_seen) == 2:  # 找到一对用户和助手消息
                    user_assistant_pairs += 1
                    roles_seen.clear()
                    
                    if user_assistant_pairs >= self.max_turns:
                        # 保留系统消息和最近的max_turns对消息
                        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
                        recent_messages = self.messages[-2 * self.max_turns:]
                        self.messages = system_messages + [
                            msg for msg in recent_messages 
                            if msg.role != MessageRole.SYSTEM
                        ]
                        break 