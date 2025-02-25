"""
Redis记忆实现，用于分布式场景
"""
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from base.dialogue.memory import Memory, Message, MessageRole


class RedisMemory(Memory):
    """Redis记忆，用于分布式场景"""
    
    def __init__(
        self,
        redis_client: Any,
        key_prefix: str = "memory:",
        max_messages: int = 100,
        ttl: int = 3600 * 24 * 7  # 默认一周过期
    ):
        """初始化Redis记忆
        
        Args:
            redis_client: Redis客户端实例
            key_prefix: 键前缀
            max_messages: 最大消息数量
            ttl: 过期时间（秒）
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.max_messages = max_messages
        self.ttl = ttl
        self.list_key = f"{key_prefix}messages"
    
    async def add(self, message: Message) -> None:
        """添加消息到记忆
        
        Args:
            message: 要添加的消息
        """
        # 将消息序列化为JSON
        message_json = json.dumps(message.to_dict(), ensure_ascii=False)
        
        # 使用Redis事务保证原子性
        async with self.redis.pipeline(transaction=True) as pipe:
            # 添加消息到列表头部
            await pipe.lpush(self.list_key, message_json)
            # 保持列表长度不超过max_messages
            await pipe.ltrim(self.list_key, 0, self.max_messages - 1)
            # 设置过期时间
            await pipe.expire(self.list_key, self.ttl)
            # 执行事务
            await pipe.execute()
    
    async def get_recent(self, k: int) -> List[Message]:
        """获取最近k条消息
        
        Args:
            k: 要获取的消息数量
            
        Returns:
            最近的k条消息列表
        """
        # 获取最近的k条消息
        messages_json = await self.redis.lrange(self.list_key, 0, k - 1)
        
        # 解析消息
        messages = []
        for msg_json in messages_json:
            try:
                msg_dict = json.loads(msg_json)
                messages.append(Message.from_dict(msg_dict))
            except Exception as e:
                print(f"解析消息失败: {e}")
        
        # 返回消息，注意Redis LRANGE返回的顺序是从新到旧
        return messages
    
    async def search(self, query: str, limit: int = 5) -> List[Message]:
        """搜索相关消息
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            相关的消息列表
        """
        # 获取所有消息
        all_messages_json = await self.redis.lrange(self.list_key, 0, -1)
        
        # 解析消息并搜索
        matching_messages = []
        for msg_json in all_messages_json:
            try:
                msg_dict = json.loads(msg_json)
                if query.lower() in msg_dict["content"].lower():
                    matching_messages.append(Message.from_dict(msg_dict))
                    if len(matching_messages) >= limit:
                        break
            except Exception as e:
                print(f"解析消息失败: {e}")
        
        return matching_messages
    
    async def clear(self) -> None:
        """清空记忆"""
        await self.redis.delete(self.list_key)
    
    async def get_formatted_history(self, formatter: Optional[Callable[[Message], str]] = None) -> str:
        """获取格式化的历史记录
        
        Args:
            formatter: 自定义格式化函数，如果为None则使用默认格式化
            
        Returns:
            格式化的历史记录字符串
        """
        # 获取最近的消息
        messages = await self.get_recent(10)
        
        if not messages:
            return ""
        
        if formatter:
            return "\n".join(formatter(msg) for msg in messages)
        
        # 默认格式化
        formatted = []
        for msg in messages:
            role_name = {
                MessageRole.SYSTEM: "系统",
                MessageRole.USER: "用户",
                MessageRole.ASSISTANT: "助手",
                MessageRole.FUNCTION: "函数"
            }.get(msg.role, str(msg.role))
            formatted.append(f"{role_name}: {msg.content}")
        
        return "\n".join(formatted)
    
    async def save(self, path: str) -> None:
        """保存记忆配置到文件
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config = {
            "key_prefix": self.key_prefix,
            "max_messages": self.max_messages,
            "ttl": self.ttl,
            "list_key": self.list_key
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    async def load(self, path: str) -> None:
        """从文件加载记忆配置
        
        Args:
            path: 加载路径
        """
        if not os.path.exists(path):
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.key_prefix = config.get("key_prefix", self.key_prefix)
            self.max_messages = config.get("max_messages", self.max_messages)
            self.ttl = config.get("ttl", self.ttl)
            self.list_key = config.get("list_key", self.list_key)