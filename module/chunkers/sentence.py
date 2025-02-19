"""
基于句子的文本切分器
"""
import re
from typing import List, Optional

from base.core.logging import AsyncLogger, LogLevel
from base.rag.chunking import Chunk, ChunkMetadata, Chunker, ChunkerConfig
from base.rag.document import Document


class SentenceChunker(Chunker):
    """基于句子的文本切分器，将文本按照句子边界进行切分"""
    
    def __init__(
        self,
        config: ChunkerConfig,
        sentence_end_chars: str = '.!?。！？',
        min_sentence_length: int = 10,
        logger: Optional[AsyncLogger] = None
    ):
        """初始化切分器
        
        Args:
            config: 切分器配置
            sentence_end_chars: 句子结束标记字符
            min_sentence_length: 最小句子长度，小于此长度的句子将与相邻句子合并
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
        self.sentence_end_chars = sentence_end_chars
        self.min_sentence_length = min_sentence_length
        # 构建句子分隔正则表达式
        self.sentence_pattern = f'([{re.escape(sentence_end_chars)}]\\s+)'
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子
        
        Args:
            text: 要分割的文本
            
        Returns:
            句子列表
        """
        # 使用正则表达式分割句子
        sentences = re.split(self.sentence_pattern, text)
        # 合并句子和其结束标记
        merged_sentences = []
        current_sentence = ''
        
        for part in sentences:
            current_sentence += part
            # 如果当前部分以句子结束标记结尾
            if any(part.strip().endswith(char) for char in self.sentence_end_chars):
                if len(current_sentence.strip()) >= self.min_sentence_length:
                    merged_sentences.append(current_sentence.strip())
                    current_sentence = ''
                    
        # 处理最后一个句子
        if current_sentence.strip():
            merged_sentences.append(current_sentence.strip())
            
        return merged_sentences
    
    async def split(self, document: Document) -> List[Chunk]:
        """将文档切分为句子块
        
        Args:
            document: 要切分的文档
            
        Returns:
            切分后的文本块列表
        """
        await self._log(LogLevel.INFO, f"开始按句子切分文档: {document.metadata.doc_id}")
        
        # 获取文本内容
        text = await document.to_text()
        if not text:
            await self._log(LogLevel.WARNING, "文档内容为空")
            return []
            
        # 分割句子
        sentences = self._split_into_sentences(text)
        await self._log(LogLevel.DEBUG, f"初步分割得到 {len(sentences)} 个句子")
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 如果当前块加上新句子超过最大长度，创建新块
            if current_length + sentence_length > self.config.chunk_size and current_chunk:
                # 创建当前块
                chunk_text = ' '.join(current_chunk)
                metadata = ChunkMetadata(
                    chunk_id=f"{document.metadata.doc_id}_chunk_{chunk_id}",
                    doc_id=document.metadata.doc_id,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    text_len=len(chunk_text)
                )
                
                chunk = Chunk(text=chunk_text, metadata=metadata)
                chunks.append(chunk)
                await self._log(LogLevel.DEBUG, f"创建块 {metadata.chunk_id}, 长度: {metadata.text_len}")
                
                # 重置当前块，考虑重叠
                if self.config.chunk_overlap > 0:
                    # 保留最后一个句子作为重叠
                    current_chunk = current_chunk[-1:]
                    current_length = len(current_chunk[-1]) if current_chunk else 0
                else:
                    current_chunk = []
                    current_length = 0
                
                chunk_id += 1
                start_char = start_char + len(chunk_text) - current_length
            
            # 添加新句子到当前块
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # 处理最后一个块
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            metadata = ChunkMetadata(
                chunk_id=f"{document.metadata.doc_id}_chunk_{chunk_id}",
                doc_id=document.metadata.doc_id,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                text_len=len(chunk_text)
            )
            
            chunk = Chunk(text=chunk_text, metadata=metadata)
            chunks.append(chunk)
            await self._log(LogLevel.DEBUG, f"创建最后一个块 {metadata.chunk_id}, 长度: {metadata.text_len}")
        
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