"""
基于语义的文本切分器，使用语言模型来识别语义边界
"""
import re
from typing import List, Optional
import numpy as np

from base.core.logging import AsyncLogger, LogLevel
from base.rag.chunking import Chunk, ChunkMetadata, Chunker, ChunkerConfig
from base.rag.document import Document
from base.model.embedding import DenseEmbeddingModel, EmbeddingConfig


class SemanticChunker(Chunker):
    """基于语义的文本切分器，使用语言模型来识别语义边界"""
    
    def __init__(
        self,
        config: ChunkerConfig,
        embedding_model: Optional[DenseEmbeddingModel] = None,
        similarity_threshold: float = 0.5,
        logger: Optional[AsyncLogger] = None
    ):
        """初始化切分器
        
        Args:
            config: 切分器配置
            embedding_model: 嵌入模型实例，如果不提供则创建默认模型
            similarity_threshold: 语义相似度阈值，用于判断是否应该切分
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
        self.similarity_threshold = similarity_threshold
        
        # 初始化嵌入模型
        if embedding_model is None:
            self.embedding_model = DenseEmbeddingModel(
                config=EmbeddingConfig(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    dimension=384,
                    normalize=True
                )
            )
        else:
            self.embedding_model = embedding_model
    
    async def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """计算两个嵌入向量之间的余弦相似度
        
        Args:
            emb1: 第一个嵌入向量
            emb2: 第二个嵌入向量
            
        Returns:
            相似度分数
        """
        # 转换为numpy数组
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        # 计算余弦相似度
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    async def _find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
        """找出语义边界的位置
        
        Args:
            sentences: 句子列表
            
        Returns:
            边界位置的列表
        """
        if len(sentences) <= 1:
            return []
            
        # 获取所有句子的embeddings
        embeddings = await self.embedding_model.embed_texts(sentences)
        boundaries = []
        
        # 计算相邻句子之间的相似度
        for i in range(len(sentences) - 1):
            similarity = await self._calculate_similarity(
                embeddings[i].vector,
                embeddings[i + 1].vector
            )
            # 如果相似度低于阈值，认为是语义边界
            if similarity < self.similarity_threshold:
                boundaries.append(i + 1)
                
        return boundaries
    
    async def split(self, document: Document) -> List[Chunk]:
        """基于语义边界切分文档
        
        Args:
            document: 要切分的文档
            
        Returns:
            切分后的文本块列表
        """
        await self._log(LogLevel.INFO, f"开始按语义切分文档: {document.metadata.doc_id}")
        
        # 获取文本内容
        text = await document.to_text()
        if not text:
            await self._log(LogLevel.WARNING, "文档内容为空")
            return []
            
        # 首先按句号、问号、感叹号等分割成句子
        sentence_pattern = r'[.!?。！？]+\s*'
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
        
        await self._log(LogLevel.DEBUG, f"初步分割得到 {len(sentences)} 个句子")
        
        # 找出语义边界
        boundaries = await self._find_semantic_boundaries(sentences)
        await self._log(LogLevel.DEBUG, f"找到 {len(boundaries)} 个语义边界")
        
        # 根据语义边界和最大长度限制创建块
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        start_char = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # 如果当前块加上新句子超过最大长度，或者遇到语义边界，创建新块
            if ((current_length + sentence_length > self.config.chunk_size and current_chunk) or
                i in boundaries):
                if current_chunk:
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
                    
                    # 重置当前块
                    current_chunk = []
                    current_length = 0
                    chunk_id += 1
                    start_char = start_char + len(chunk_text)
            
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