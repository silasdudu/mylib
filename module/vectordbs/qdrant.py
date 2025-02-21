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
from base.rag.chunking import Chunk, ChunkMetadata
from base.core.logging import LogLevel


class QdrantVectorDBConfig(VectorDBConfig):
    """Qdrant 向量数据库配置"""
    host: str = "localhost"  # Qdrant 服务器地址
    port: int = 6333  # Qdrant 服务器端口
    grpc_port: int = 6334  # gRPC 端口
    api_key: Optional[str] = None  # API 密钥
    collection_name: str = "document_chunks"  # 集合名称
    distance: str = "Cosine"  # 距离度量：Cosine, Euclid, Dot
    on_disk: bool = False  # 是否使用磁盘存储
    replication_factor: int = 1  # 副本数量
    write_consistency_factor: int = 1  # 写一致性因子
    index_params: Dict[str, Any] = {}  # 额外的索引参数


class QdrantVectorDB(VectorDB):
    """基于 Qdrant 的向量数据库实现"""
    
    def __init__(self, config: QdrantVectorDBConfig, logger: Optional[AsyncLogger] = None):
        """初始化 Qdrant 向量数据库
        
        Args:
            config: 数据库配置
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
        self.config: QdrantVectorDBConfig = config
        
        # 初始化客户端
        self._client = QdrantClient(
            host=config.host,
            port=config.port,
            grpc_port=config.grpc_port,
            api_key=config.api_key,
            prefer_grpc=True
        )
        
        # 初始化集合
        self._get_or_create_collection()
        
    def _get_or_create_collection(self) -> None:
        """获取或创建集合"""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.config.collection_name for c in collections)
        
        if not exists:
            # 创建集合
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimension,
                    distance=Distance[self.config.distance]
                ),
                on_disk=self.config.on_disk,
                replication_factor=self.config.replication_factor,
                write_consistency_factor=self.config.write_consistency_factor,
                **self.config.index_params
            )
            
    def _chunk_to_payload(self, chunk: Chunk) -> Dict[str, Any]:
        """将 Chunk 对象转换为 payload"""
        return {
            "chunk_id": chunk.metadata.chunk_id,
            "doc_id": chunk.metadata.doc_id,
            "text": chunk.text,
            "start_char": chunk.metadata.start_char,
            "end_char": chunk.metadata.end_char,
            "text_len": chunk.metadata.text_len,
            "extra": chunk.metadata.extra
        }
        
    def _payload_to_chunk(self, payload: Dict[str, Any]) -> Chunk:
        """将 payload 转换为 Chunk 对象"""
        metadata = ChunkMetadata(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            start_char=payload["start_char"],
            end_char=payload["end_char"],
            text_len=payload["text_len"],
            extra=payload["extra"]
        )
        return Chunk(text=payload["text"], metadata=metadata)
        
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 准备数据
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors_array)):
            points.append(models.PointStruct(
                id=i,
                vector=vector.tolist(),
                payload=self._chunk_to_payload(chunk)
            ))
            
        # 批量插入数据
        self._client.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
        
        self._index_ready = True
        await self._log(LogLevel.INFO, "索引创建完成")
        
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        if not self._index_ready:
            await self.create_index(vectors, chunks)
            return
            
        await self._log(LogLevel.INFO, f"添加向量，数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 获取当前最大ID
        collection_info = self._client.get_collection(self.config.collection_name)
        start_id = collection_info.points_count
        
        # 准备数据
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors_array)):
            points.append(models.PointStruct(
                id=start_id + i,
                vector=vector.tolist(),
                payload=self._chunk_to_payload(chunk)
            ))
            
        # 批量插入数据
        self._client.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
        
        await self._log(LogLevel.INFO, "向量添加完成")
        
    async def delete_vectors(self, chunk_ids: List[str]) -> None:
        """从索引中删除向量
        
        Args:
            chunk_ids: 要删除的文档块ID列表
        """
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
            top_k: 返回结果数量
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
                    conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=list(value))
                    ))
                else:
                    conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    ))
            if conditions:
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
            chunk = self._payload_to_chunk(hit.payload)
            score = float(hit.score)
            
            # 转换分数
            if self.config.distance == "Cosine":
                score = (1 + score) / 2  # 转换到 [0,1] 区间
            elif self.config.distance == "Euclid":
                score = 1 / (1 + score)  # 转换为相似度
                
            search_results.append(SearchResult(
                chunk=chunk,
                score=score
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
                    conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=list(value))
                    ))
                else:
                    conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    ))
            if conditions:
                filter_selector = models.Filter(must=conditions)
                
        # 执行批量搜索
        k = top_k or self.config.top_k
        batch_results = []
        for query_vector in query_array:
            results = self._client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector.tolist(),
                limit=k,
                query_filter=filter_selector
            )
            
            # 构建结果
            query_results = []
            for hit in results:
                chunk = self._payload_to_chunk(hit.payload)
                score = float(hit.score)
                
                # 转换分数
                if self.config.distance == "Cosine":
                    score = (1 + score) / 2  # 转换到 [0,1] 区间
                elif self.config.distance == "Euclid":
                    score = 1 / (1 + score)  # 转换为相似度
                    
                query_results.append(SearchResult(
                    chunk=chunk,
                    score=score
                ))
                
            batch_results.append(query_results)
            
        return batch_results
        
    async def save(self, path: str) -> None:
        """保存索引到文件
        
        Args:
            path: 保存路径
            
        Note:
            Qdrant 是服务器端的数据库，索引数据由服务器管理，
            这里不需要额外的保存操作。
        """
        pass
        
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 索引文件路径
            
        Note:
            Qdrant 是服务器端的数据库，这里只需要检查集合是否存在。
        """
        collections = self._client.get_collections().collections
        if any(c.name == self.config.collection_name for c in collections):
            self._index_ready = True 