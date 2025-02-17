"""
基于长度的文本切分器
"""
from typing import List, Optional

from base.core.logging import AsyncLogger, LogLevel
from base.rag.chunking import Chunk, ChunkMetadata, Chunker, ChunkerConfig
from base.rag.document import Document


class LengthChunker(Chunker):
    """基于长度的文本切分器，将文本按照固定长度进行切分"""
    
    def __init__(
        self,
        config: ChunkerConfig,
        logger: Optional[AsyncLogger] = None
    ):
        """初始化切分器
        
        Args:
            config: 切分器配置，包含 chunk_size 和 chunk_overlap
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
    
    async def split(self, document: Document) -> List[Chunk]:
        """将文档切分为固定长度的文本块
        
        Args:
            document: 要切分的文档
            
        Returns:
            切分后的文本块列表
        """
        await self._log(LogLevel.INFO, f"开始按长度切分文档: {document.metadata.doc_id}")
        
        # 获取文本内容
        text = await document.to_text()
        if not text:
            await self._log(LogLevel.WARNING, "文档内容为空")
            return []
            
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = min(start + self.config.chunk_size, len(text))
            
            # 如果不是最后一个块，尝试在单词边界处切分
            if end < len(text):
                # 向后查找最近的空格
                while end > start and not text[end - 1].isspace():
                    end -= 1
                # 如果没找到空格，就强制在原位置切分
                if end == start:
                    end = start + self.config.chunk_size
            
            # 提取当前块的文本
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # 只添加非空块
                # 创建块元数据
                metadata = ChunkMetadata(
                    chunk_id=f"{document.metadata.doc_id}_chunk_{chunk_id}",
                    doc_id=document.metadata.doc_id,
                    start_char=start,
                    end_char=end,
                    text_len=len(chunk_text)
                )
                
                # 创建并添加块
                chunk = Chunk(text=chunk_text, metadata=metadata)
                chunks.append(chunk)
                chunk_id += 1
                
                await self._log(LogLevel.DEBUG, f"创建块 {metadata.chunk_id}, 长度: {metadata.text_len}")
            
            # 移动到下一个起始位置，考虑重叠
            start = end - self.config.chunk_overlap
            if start < 0:
                start = 0
        
        await self._log(LogLevel.INFO, f"文档切分完成，共生成 {len(chunks)} 个块")
        return chunks

    async def split_batch(self, documents: List[Document]) -> List[List[Chunk]]:
        """批量处理多个文档
        
        Args:
            documents: 要处理的文档列表
            
        Returns:
            每个文档对应的文本块列表
        """
        await self._log(LogLevel.INFO, f"开始批量处理 {len(documents)} 个文档")
        results = []
        
        for doc in documents:
            try:
                chunks = await self.split(doc)
                results.append(chunks)
                await self._log(LogLevel.DEBUG, f"文档 {doc.metadata.doc_id} 生成了 {len(chunks)} 个块")
            except Exception as e:
                await self._log(LogLevel.ERROR, f"处理文档 {doc.metadata.doc_id} 时出错: {str(e)}")
                results.append([])
                
        await self._log(LogLevel.INFO, "批量处理完成")
        return results
        
    async def merge(self, chunks: List[Chunk]) -> Chunk:
        """合并多个文本块
        
        Args:
            chunks: 要合并的文本块列表
            
        Returns:
            合并后的文本块
            
        Raises:
            ValueError: 当块列表为空时抛出
        """
        if not chunks:
            raise ValueError("无法合并空的块列表")
            
        # 按照块的顺序合并文本
        merged_text = ' '.join(chunk.text for chunk in chunks)
        
        # 使用第一个块的文档ID
        doc_id = chunks[0].metadata.doc_id
        
        # 创建新的元数据
        metadata = ChunkMetadata(
            chunk_id=f"{doc_id}_merged_{chunks[0].metadata.chunk_id}",
            doc_id=doc_id,
            start_char=chunks[0].metadata.start_char,
            end_char=chunks[-1].metadata.end_char,
            text_len=len(merged_text)
        )
        
        return Chunk(text=merged_text, metadata=metadata) 