"""
文档分块模块，支持将文档切分为适合检索的小块
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .document import Document, DocumentType
from ..core.logging import AsyncLogger, LogLevel


class ChunkMetadata(BaseModel):
    """分块元数据"""
    chunk_id: str
    doc_id: str
    start_char: int
    end_char: int
    text_len: int
    extra: Dict[str, Any] = {}


@dataclass
class Chunk:
    """文档分块"""
    text: str
    metadata: ChunkMetadata


class ChunkerConfig(BaseModel):
    """分块器配置基类"""
    extra_params: Dict[str, Any] = {}  # 额外的特定实现参数


class Chunker(ABC):
    """分块器抽象基类"""
    
    def __init__(self, config: ChunkerConfig, logger: Optional[AsyncLogger] = None):
        """初始化分块器
        
        Args:
            config: 分块器配置
            logger: 可选的日志记录器
        """
        self.config = config
        self.logger = logger
    
    async def _log(self, level: LogLevel, message: str):
        """内部日志记录方法
        
        Args:
            level: 日志级别
            message: 日志消息
        """
        if self.logger:
            await self.logger.log(level, message)
    
    @abstractmethod
    async def split(self, document: Document) -> List[Chunk]:
        """将文档分割为块"""
        pass
    
    @abstractmethod
    async def split_batch(self, documents: List[Document]) -> List[List[Chunk]]:
        """批量分割文档"""
        pass
    
    @abstractmethod
    async def merge(self, chunks: List[Chunk]) -> Chunk:
        """将块合并为文档"""
        pass


class ChunkerRegistry:
    """分块器注册表"""
    
    def __init__(self):
        self._chunkers: Dict[DocumentType, Chunker] = {}
        
    def register(
        self,
        doc_type: DocumentType,
        chunker: Chunker
    ) -> None:
        """注册分块器"""
        self._chunkers[doc_type] = chunker
        
    def get_chunker(
        self,
        doc_type: DocumentType
    ) -> Optional[Chunker]:
        """获取分块器"""
        return self._chunkers.get(doc_type)
        
    def list_supported_types(self) -> List[DocumentType]:
        """列出支持的文档类型"""
        return list(self._chunkers.keys()) 