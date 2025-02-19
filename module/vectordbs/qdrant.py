"""
基于 Qdrant 的向量数据库实现
"""
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams

from base.rag.vectordb import VectorDB, VectorDBConfig, SearchResult
from base.rag.chunking import Chunk
from base.core.logging import LogLevel


class QdrantVectorDBConfig(VectorDBConfig):
    """Qdrant 向量数据库配置"""
    location: Optional[str] = None  # 本地存储路径，如果为None则使用内存存储
    url: Optional[str] = None  # Qdrant 服务器地址
    port: int = 6333  # Qdrant 服务器端口
    collection_name: str = "document_chunks"  # 集合名称
    api_key: Optional[str] = None  # API密钥
    prefer_grpc: bool = True  # 是否优先使用gRPC
    distance: str = "Cosine"  # 距离类型：Cosine, Euclid, Dot
    replication_factor: int = 1  # 副本数量
    write_consistency_factor: int = 1  # 写一致性因子
    on_disk_payload: bool = True  # 是否将payload存储在磁盘上


class QdrantVectorDB(VectorDB):
    """基于 Qdrant 的向量数据库实现"""
    
    def __init__(self, config: QdrantVectorDBConfig, **kwargs):
        """初始化 Qdrant 向量数据库
        
        Args:
            config: 数据库配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: QdrantVectorDBConfig = config
        
        # 创建客户端
        client_kwargs = {
            "prefer_grpc": self.config.prefer_grpc
        }
        
        if self.config.url:
            client_kwargs.update({
                "url": self.config.url,
                "port": self.config.port
            })
            if self.config.api_key:
                client_kwargs["api_key"] = self.config.api_key
        else:
            client_kwargs["location"] = self.config.location or ":memory:"
            
        self._client = QdrantClient(**client_kwargs)
        self._chunks = {}
    
    def _get_distance(self) -> Distance:
        """获取距离类型
        
        Returns:
            Qdrant距离类型
        """
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        return distance_map[self.config.distance.lower()]
    
    def _prepare_payload(self, chunk: Chunk) -> Dict[str, Any]:
        """准备要存储的payload
        
        Args:
            chunk: 文档块
            
        Returns:
            Qdrant payload
        """
        return {
            "chunk_id": chunk.metadata.chunk_id,
            "doc_id": chunk.metadata.doc_id,
            "text": chunk.text,
            "metadata": chunk.metadata.model_dump()
        }
    
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建 Qdrant 索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 如果集合已存在，先删除
        try:
            self._client.delete_collection(self.config.collection_name)
        except Exception:
            pass
            
        # 创建集合
        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.config.dimension,
                distance=self._get_distance()
            ),
            replication_factor=self.config.replication_factor,
            write_consistency_factor=self.config.write_consistency_factor,
            on_disk_payload=self.config.on_disk_payload
        )
        
        # 准备数据
        points = [
            models.PointStruct(
                id=i,
                vector=vector.tolist(),
                payload=self._prepare_payload(chunk)
            )
            for i, (vector, chunk) in enumerate(zip(vectors_array, chunks))
        ]
        
        # 批量插入数据
        self._client.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
        
        # 存储文档块映射
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        
        self._index_ready = True
        await self._log(LogLevel.INFO, "Qdrant 索引创建完成")
    
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        if not self._index_ready:
            await self.create_index(vectors, chunks)
            return
            
        await self._log(LogLevel.INFO, f"添加向量到 Qdrant 索引，数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 获取当前最大ID
        current_points = self._client.scroll(
            collection_name=self.config.collection_name,
            limit=1,
            with_payload=False,
            with_vectors=False
        )[0]
        next_id = len(current_points)
        
        # 准备数据
        points = [
            models.PointStruct(
                id=next_id + i,
                vector=vector.tolist(),
                payload=self._prepare_payload(chunk)
            )
            for i, (vector, chunk) in enumerate(zip(vectors_array, chunks))
        ]
        
        # 批量插入数据
        self._client.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
        
        # 更新文档块映射
        for chunk in chunks:
            self._chunks[chunk.metadata.chunk_id] = chunk
            
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
        
        # 构建过滤条件
        filter_selector = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchAny(any=chunk_ids)
                )
            ]
        )
        
        # 执行删除
        self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=filter_selector
        )
        
        # 更新文档块映射
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
            
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
        query_array = self._validate_vectors([query_vector])
        
        # 构建过滤条件
        filter_selector = None
        if filter_params:
            conditions = []
            for key, value in filter_params.items():
                if isinstance(value, (list, tuple)):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            filter_selector = models.Filter(must=conditions)
            
        # 执行搜索
        k = top_k or self.config.top_k
        results = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=query_array[0].tolist(),
            limit=k,
            query_filter=filter_selector
        )
        
        # 构建结果
        search_results = []
        for hit in results:
            chunk_id = hit.payload["chunk_id"]
            if chunk_id in self._chunks:
                chunk = self._chunks[chunk_id]
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=float(hit.score)
                ))
                
        return search_results
    
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
            
        # 验证查询向量
        query_array = self._validate_vectors(query_vectors)
        
        # 构建过滤条件
        filter_selector = None
        if filter_params:
            conditions = []
            for key, value in filter_params.items():
                if isinstance(value, (list, tuple)):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            filter_selector = models.Filter(must=conditions)
            
        # 执行批量搜索
        k = top_k or self.config.top_k
        results = self._client.search_batch(
            collection_name=self.config.collection_name,
            queries=[
                models.SearchRequest(
                    vector=vector.tolist(),
                    limit=k,
                    filter=filter_selector
                )
                for vector in query_array
            ]
        )
        
        # 构建结果
        batch_results = []
        for batch_hits in results:
            query_results = []
            for hit in batch_hits:
                chunk_id = hit.payload["chunk_id"]
                if chunk_id in self._chunks:
                    chunk = self._chunks[chunk_id]
                    query_results.append(SearchResult(
                        chunk=chunk,
                        score=float(hit.score)
                    ))
            batch_results.append(query_results)
            
        return batch_results
    
    async def save(self, path: str) -> None:
        """保存索引到文件
        
        Args:
            path: 保存路径
            
        Note:
            如果使用本地存储，Qdrant 会自动持久化。
            这里只保存文档块映射等客户端状态。
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return
            
        await self._log(LogLevel.INFO, f"保存 Qdrant 客户端状态到: {path}")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存客户端状态
        state = {
            "collection_name": self.config.collection_name,
            "chunks": {
                chunk_id: chunk.model_dump()
                for chunk_id, chunk in self._chunks.items()
            }
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            
        await self._log(LogLevel.INFO, "客户端状态保存完成")
    
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 状态文件路径
            
        Note:
            如果使用本地存储，Qdrant 会自动从持久化目录加载数据。
            这里只加载文档块映射等客户端状态。
        """
        await self._log(LogLevel.INFO, f"从文件加载 Qdrant 客户端状态: {path}")
        
        # 加载客户端状态
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # 检查集合是否存在
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if state["collection_name"] in collection_names:
            # 恢复文档块映射
            self._chunks = {
                chunk_id: Chunk.parse_obj(chunk_data)
                for chunk_id, chunk_data in state["chunks"].items()
            }
            
            self._index_ready = True
            await self._log(LogLevel.INFO, "客户端状态加载完成")
        else:
            await self._log(
                LogLevel.ERROR,
                f"集合 {state['collection_name']} 不存在，请先创建集合"
            ) 