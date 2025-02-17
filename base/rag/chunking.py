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
    """分块器配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    extra_params: Dict[str, Any] = {}


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


class TextChunker(Chunker):
    """文本分块器"""
    
    async def split(self, document: Document) -> List[Chunk]:
        """按字符数分割文本"""
        if document.metadata.doc_type != DocumentType.TEXT:
            raise ValueError("TextChunker只支持文本文档")
            
        text = await document.to_text()
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = min(start + self.config.chunk_size, len(text))
            
            # 如果不是最后一块，尝试在单词边界处分割
            if end < len(text):
                # 向后查找最近的分隔符
                while end > start and not text[end].isspace():
                    end -= 1
                if end == start:  # 如果没找到合适的分割点
                    end = start + self.config.chunk_size
                    
            chunk_content = text[start:end]
            
            # 创建分块元数据
            metadata = ChunkMetadata(
                chunk_id=f"{document.metadata.doc_id}_{start}_{end}",
                doc_id=document.metadata.doc_id,
                start_char=start,
                end_char=end,
                text_len=len(chunk_content)
            )
            
            chunks.append(Chunk(chunk_content, metadata))
            
            # 更新起始位置，考虑重叠
            start = end - self.config.chunk_overlap
            
        return chunks
    
    async def split_batch(self, documents: List[Document]) -> List[List[Chunk]]:
        """批量分割文本文档"""
        results = []
        for doc in documents:
            chunks = await self.split(doc)
            results.append(chunks)
        return results
    
    async def merge(self, chunks: List[Chunk]) -> Chunk:
        """将文本块合并为文档"""
        # 按照起始位置排序
        sorted_chunks = sorted(chunks, key=lambda x: x.metadata.start_char)
        
        # 合并文本
        merged_text = ""
        last_end = 0
        
        for chunk in sorted_chunks:
            if chunk.metadata.start_char > last_end:
                # 处理缺失的部分
                merged_text += " " * (chunk.metadata.start_char - last_end)
            merged_text += chunk.text
            last_end = chunk.metadata.end_char
            
        # 创建新的文档
        from .document import TextDocument, DocumentMetadata
        metadata = DocumentMetadata(
            doc_id=chunks[0].metadata.doc_id,
            doc_type=DocumentType.TEXT,
            source="merged",
            created_at="",  # 需要设置实际时间
        )
        
        return Chunk(merged_text, ChunkMetadata(
            chunk_id=f"{chunks[0].metadata.doc_id}_merged",
            doc_id=chunks[0].metadata.doc_id,
            start_char=0,
            end_char=len(merged_text),
            text_len=len(merged_text),
            extra={}
        ))


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