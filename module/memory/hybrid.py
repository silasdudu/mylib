"""
混合记忆实现，结合短期和长期记忆
"""
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

from base.dialogue.memory import Memory, Message, MessageRole
from .short_term import ShortTermMemory
from .long_term import LongTermMemory


class HybridMemory(Memory):
    """混合记忆，结合短期和长期记忆"""
    
    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        long_term_threshold: float = 0.7
    ):
        """初始化混合记忆
        
        Args:
            short_term: 短期记忆实例
            long_term: 长期记忆实例
            long_term_threshold: 长期记忆相关性阈值
        """
        self.short_term = short_term
        self.long_term = long_term
        self.long_term_threshold = long_term_threshold
    
    async def add(self, message: Message) -> None:
        """添加消息到记忆
        
        Args:
            message: 要添加的消息
        """
        # 同时添加到短期和长期记忆
        await self.short_term.add(message)
        await self.long_term.add(message)
    
    async def get_recent(self, k: int) -> List[Message]:
        """获取最近k条消息
        
        Args:
            k: 要获取的消息数量
            
        Returns:
            最近的k条消息列表
        """
        # 优先从短期记忆获取
        return await self.short_term.get_recent(k)
    
    async def search(self, query: str, limit: int = 5) -> List[Message]:
        """搜索相关消息
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            相关的消息列表
        """
        # 先从短期记忆搜索
        short_term_results = await self.short_term.search(query, limit)
        
        # 如果短期记忆结果不足，再从长期记忆搜索
        if len(short_term_results) < limit:
            long_term_results = await self.long_term.search(
                query, 
                limit=limit - len(short_term_results)
            )
            
            # 合并结果，去重
            seen_contents = {msg.content for msg in short_term_results}
            for msg in long_term_results:
                if msg.content not in seen_contents:
                    short_term_results.append(msg)
                    seen_contents.add(msg.content)
        
        return short_term_results
    
    async def clear(self) -> None:
        """清空记忆"""
        await self.short_term.clear()
        await self.long_term.clear()
    
    async def get_formatted_history(self, formatter: Optional[Callable[[Message], str]] = None) -> str:
        """获取格式化的历史记录
        
        Args:
            formatter: 自定义格式化函数，如果为None则使用默认格式化
            
        Returns:
            格式化的历史记录字符串
        """
        # 优先使用短期记忆的格式化历史
        return await self.short_term.get_formatted_history(formatter)
    
    async def save(self, path: str) -> None:
        """保存记忆到文件
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存短期记忆
        short_term_path = f"{path}.short_term"
        await self.short_term.save(short_term_path)
        
        # 保存长期记忆
        long_term_path = f"{path}.long_term"
        await self.long_term.save(long_term_path)
        
        # 保存混合记忆配置
        config = {
            "long_term_threshold": self.long_term_threshold,
            "short_term_path": short_term_path,
            "long_term_path": long_term_path
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    async def load(self, path: str) -> None:
        """从文件加载记忆
        
        Args:
            path: 加载路径
        """
        if not os.path.exists(path):
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.long_term_threshold = config.get("long_term_threshold", self.long_term_threshold)
        
        # 加载短期记忆
        short_term_path = config.get("short_term_path", f"{path}.short_term")
        if os.path.exists(short_term_path):
            await self.short_term.load(short_term_path)
        
        # 加载长期记忆
        long_term_path = config.get("long_term_path", f"{path}.long_term")
        if os.path.exists(long_term_path):
            await self.long_term.load(long_term_path) 