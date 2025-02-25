"""
医疗检索系统模块
"""
from typing import Dict, Any, List, Optional

from module.models.embedding.custom import CustomEmbedding
from module.vectordbs.chroma import ChromaVectorDB
from module.chunkers.text import Chunk as TextChunk
from module.retrievers.vector import VectorRetriever, VectorRetrieverConfig
from module.retrievers.vector import SearchResult as RetrievalResult


class MedicalRAGSystem:
    """医疗RAG系统，用于文档检索"""
    
    def __init__(self, embedding_model: CustomEmbedding, vector_db: ChromaVectorDB = None):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        
        # 初始化向量检索器（仅作为备用）
        retriever_config = VectorRetrieverConfig(
            distance_metric="cosine",
            normalize_vectors=True,
            cache_vectors=True
        )
        self.retriever = VectorRetriever(config=retriever_config)
    
    async def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """添加文档到检索系统"""
        # 创建文本块
        chunk = TextChunk(text=text, metadata=metadata or {})
        
        # 生成嵌入向量
        embedding = await self.embedding_model.embed_chunk(chunk)
        
        # 优先添加到向量数据库（持久化存储）
        if self.vector_db:
            await self.vector_db.add_vectors([embedding.embedding], [chunk])
            print(f"已将文档添加到持久化向量数据库: {chunk.metadata.get('chunk_id', '未知ID')}")
        else:
            # 如果没有向量数据库，则使用内存型检索器作为备用
            await self.retriever.index([chunk], [embedding])
            print(f"警告：使用内存型检索器存储文档（不会持久化）: {chunk.metadata.get('chunk_id', '未知ID')}")
    
    async def search(self, query: str, limit: int = 5) -> List[RetrievalResult]:
        """搜索相关文档"""
        # 优先使用持久化的向量数据库
        if self.vector_db:
            # 生成查询向量
            query_embedding = await self.embedding_model.embed(query)
            
            # 使用向量数据库搜索
            results = await self.vector_db.search(
                query_vector=query_embedding.vector,
                top_k=limit
            )
            
            if results and len(results) > 0:
                print(f"从持久化向量数据库检索到 {len(results)} 条结果")
                return results
        
        # 如果向量数据库不可用或没有结果，则尝试使用内存型检索器
        if hasattr(self.retriever, '_chunks') and len(self.retriever._chunks) > 0:
            results = await self.retriever.search(query, top_k=limit)
            print(f"从内存型检索器检索到 {len(results)} 条结果")
            return results
        
        # 如果都没有结果，返回空列表
        print("未找到相关文档")
        return [] 