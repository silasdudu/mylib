"""
基于内存的简单向量数据库实现
"""
from typing import List, Optional, Dict, Any
import numpy as np

from base.rag.vectordb import VectorDB, VectorDBConfig, SearchResult
from base.rag.chunking import Chunk
from base.core.logging import LogLevel


class MemoryVectorDB(VectorDB):
    """基于内存的向量数据库，将向量和文档块存储在内存中"""
    
    def __init__(self, config: VectorDBConfig, **kwargs):
        """初始化内存向量数据库
        
        Args:
            config: 数据库配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self._vectors = None  # 向量数组
        self._chunks = {}  # 文档块字典，键为chunk_id
        self._chunk_ids = []  # 保持插入顺序的chunk_id列表
    
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        self._vectors = self._validate_vectors(vectors)
        
        # 存储文档块
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        self._chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        
        self._index_ready = True
        await self._log(LogLevel.INFO, "索引创建完成")
    
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"添加向量，数量: {len(vectors)}")
        
        # 验证向量格式
        new_vectors = self._validate_vectors(vectors)
        
        # 如果是第一次添加，直接创建索引
        if self._vectors is None:
            await self.create_index(vectors, chunks)
            return
            
        # 添加到现有索引
        self._vectors = np.vstack([self._vectors, new_vectors])
        
        # 添加文档块
        for chunk in chunks:
            chunk_id = chunk.metadata.chunk_id
            self._chunks[chunk_id] = chunk
            self._chunk_ids.append(chunk_id)
            
        await self._log(LogLevel.INFO, "向量添加完成")
    
    async def delete_vectors(self, chunk_ids: List[str]) -> None:
        """从索引中删除向量
        
        Args:
            chunk_ids: 要删除的文档块ID列表
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return
            
        await self._log(LogLevel.INFO, f"删除向量，数量: {len(chunk_ids)}")
        
        # 找出要删除的索引
        indices_to_delete = []
        for i, chunk_id in enumerate(self._chunk_ids):
            if chunk_id in chunk_ids:
                indices_to_delete.append(i)
                self._chunks.pop(chunk_id, None)
                
        # 更新向量数组和chunk_id列表
        if indices_to_delete:
            mask = np.ones(len(self._chunk_ids), dtype=bool)
            mask[indices_to_delete] = False
            self._vectors = self._vectors[mask]
            self._chunk_ids = [
                chunk_id for i, chunk_id in enumerate(self._chunk_ids)
                if i not in indices_to_delete
            ]
            
        await self._log(LogLevel.INFO, "向量删除完成")
    
    async def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量，如果不指定则使用配置中的值
            filter_params: 可选的过滤参数
            
        Returns:
            搜索结果列表
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return []
            
        # 验证查询向量
        query_vector = self._validate_vectors([query_vector])[0]
        
        # 计算相似度
        scores = self._compute_similarity(query_vector, self._vectors)
        
        # 获取top_k结果
        k = top_k or self.config.top_k
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks[chunk_id]
            results.append(SearchResult(
                chunk=chunk,
                score=float(scores[idx])
            ))
            
        return results
    
    async def batch_search(
        self,
        query_vectors: List[List[float]],
        top_k: Optional[int] = None,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[List[SearchResult]]:
        """批量搜索最相似的向量
        
        Args:
            query_vectors: 查询向量列表
            top_k: 每个查询返回的结果数量
            filter_params: 可选的过滤参数
            
        Returns:
            每个查询对应的搜索结果列表
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return []
            
        results = []
        for query_vector in query_vectors:
            batch_results = await self.search(query_vector, top_k, filter_params)
            results.append(batch_results)
            
        return results
    
    async def save(self, path: str) -> None:
        """保存索引到文件
        
        Args:
            path: 保存路径
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return
            
        await self._log(LogLevel.INFO, f"保存索引到: {path}")
        
        # 保存为npz文件
        np.savez(
            path,
            vectors=self._vectors,
            chunk_ids=np.array(self._chunk_ids),
            chunks=np.array([chunk.json() for chunk in self._chunks.values()])
        )
        
        await self._log(LogLevel.INFO, "索引保存完成")
    
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        await self._log(LogLevel.INFO, f"从文件加载索引: {path}")
        
        # 加载npz文件
        data = np.load(path, allow_pickle=True)
        self._vectors = data["vectors"]
        self._chunk_ids = data["chunk_ids"].tolist()
        
        # 恢复文档块
        chunks_data = data["chunks"]
        self._chunks = {}
        for chunk_data in chunks_data:
            chunk = Chunk.parse_raw(chunk_data.tolist())
            self._chunks[chunk.metadata.chunk_id] = chunk
            
        self._index_ready = True
        await self._log(LogLevel.INFO, "索引加载完成") 