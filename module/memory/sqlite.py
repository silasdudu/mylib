"""
SQLite记忆实现，用于本地持久化
"""
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from base.dialogue.memory import Memory, Message, MessageRole


class SQLiteMemory(Memory):
    """SQLite记忆，用于本地持久化"""
    
    def __init__(
        self,
        db_path: str,
        table_name: str = "memory",
        max_messages: int = 1000
    ):
        """初始化SQLite记忆
        
        Args:
            db_path: 数据库路径
            table_name: 表名
            max_messages: 最大消息数量
        """
        self.db_path = db_path
        self.table_name = table_name
        self.max_messages = max_messages
        self._init_db()
    
    def _init_db(self) -> None:
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建消息表
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建索引
        cursor.execute(f'''
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
        ON {self.table_name} (timestamp)
        ''')
        
        cursor.execute(f'''
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_role 
        ON {self.table_name} (role)
        ''')
        
        conn.commit()
        conn.close()
    
    async def add(self, message: Message) -> None:
        """添加消息到记忆
        
        Args:
            message: 要添加的消息
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 插入消息
        cursor.execute(
            f"INSERT INTO {self.table_name} (role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
            (
                message.role.value,
                message.content,
                message.timestamp.isoformat(),
                json.dumps(message.metadata, ensure_ascii=False) if message.metadata else None
            )
        )
        
        # 如果超过最大消息数量，删除旧消息
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        
        if count > self.max_messages:
            # 删除最旧的消息，保留最新的max_messages条
            cursor.execute(f'''
            DELETE FROM {self.table_name}
            WHERE id IN (
                SELECT id FROM {self.table_name}
                ORDER BY timestamp ASC
                LIMIT {count - self.max_messages}
            )
            ''')
        
        conn.commit()
        conn.close()
    
    async def get_recent(self, k: int) -> List[Message]:
        """获取最近k条消息
        
        Args:
            k: 要获取的消息数量
            
        Returns:
            最近的k条消息列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查询最近的k条消息
        cursor.execute(
            f"SELECT role, content, timestamp, metadata FROM {self.table_name} ORDER BY timestamp DESC LIMIT ?",
            (k,)
        )
        
        messages = []
        for row in cursor.fetchall():
            role, content, timestamp_str, metadata_str = row
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            messages.append(Message(
                role=MessageRole(role),
                content=content,
                timestamp=datetime.fromisoformat(timestamp_str),
                metadata=metadata
            ))
        
        conn.close()
        
        # 返回按时间正序排列的消息
        return list(reversed(messages))
    
    async def search(self, query: str, limit: int = 5) -> List[Message]:
        """搜索相关消息
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            相关的消息列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 使用LIKE进行简单搜索
        cursor.execute(
            f"SELECT role, content, timestamp, metadata FROM {self.table_name} WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit)
        )
        
        messages = []
        for row in cursor.fetchall():
            role, content, timestamp_str, metadata_str = row
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            messages.append(Message(
                role=MessageRole(role),
                content=content,
                timestamp=datetime.fromisoformat(timestamp_str),
                metadata=metadata
            ))
        
        conn.close()
        return messages
    
    async def clear(self) -> None:
        """清空记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"DELETE FROM {self.table_name}")
        
        conn.commit()
        conn.close()
    
    async def get_formatted_history(self, formatter: Optional[Callable[[Message], str]] = None) -> str:
        """获取格式化的历史记录
        
        Args:
            formatter: 自定义格式化函数，如果为None则使用默认格式化
            
        Returns:
            格式化的历史记录字符串
        """
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
            "db_path": self.db_path,
            "table_name": self.table_name,
            "max_messages": self.max_messages
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
            self.table_name = config.get("table_name", self.table_name)
            self.max_messages = config.get("max_messages", self.max_messages)
            
            # 如果数据库路径变更，需要重新初始化
            new_db_path = config.get("db_path", self.db_path)
            if new_db_path != self.db_path:
                self.db_path = new_db_path
                self._init_db() 