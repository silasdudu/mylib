"""
文本分块器实现
"""
from typing import List, Optional, Dict, Any
from pydantic import Field
from base.rag.chunking import Chunker, ChunkerConfig, Chunk, ChunkMetadata
from base.rag.document import Document, DocumentType
from base.core.logging import AsyncLogger, LogLevel


class TextChunkerConfig(ChunkerConfig):
    """文本分块器配置"""
    chunk_size: int = Field(default=1000, description="每个块的目标大小(字符数)")
    chunk_overlap: int = Field(default=200, description="相邻块之间的重叠大小(字符数)")
    min_chunk_size: int = Field(default=100, description="块的最小大小")
    max_chunk_size: int = Field(default=2000, description="块的最大大小")
    split_by: str = Field(default="char", description="分块方式: char(按字符), sentence(按句子), paragraph(按段落)")


class TextChunker(Chunker):
    """文本分块器"""
    
    def __init__(self, config: TextChunkerConfig, logger: Optional[AsyncLogger] = None):
        """初始化文本分块器
        
        Args:
            config: 分块器配置
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
    
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
            
        # 创建新的元数据
        metadata = ChunkMetadata(
            chunk_id=f"{chunks[0].metadata.doc_id}_merged",
            doc_id=chunks[0].metadata.doc_id,
            start_char=0,
            end_char=len(merged_text),
            text_len=len(merged_text)
        )
        
        return Chunk(merged_text, metadata)