"""
基于 Milvus 的向量数据库实现
"""
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from base.rag.vectordb import VectorDB, VectorDBConfig, SearchResult
from base.rag.chunking import Chunk, ChunkMetadata
from base.core.logging import AsyncLogger, LogLevel


class MilvusVectorDBConfig(VectorDBConfig):
    """Milvus 向量数据库配置"""
    host: str = "localhost"  # Milvus 服务器地址
    port: int = 19530  # Milvus 服务器端口
    user: str = ""  # 用户名
    password: str = ""  # 密码
    collection_name: str = "document_chunks"  # 集合名称
    index_type: str = "IVF_FLAT"  # 索引类型：FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ
    metric_type: str = "L2"  # 距离度量：L2, IP, COSINE
    nlist: int = 1024  # IVF 聚类中心数量
    nprobe: int = 16  # 搜索时探测的聚类中心数量
    index_params: Dict[str, Any] = {}  # 额外的索引参数


class MilvusVectorDB(VectorDB):
    """基于 Milvus 的向量数据库实现"""
    
    def __init__(self, config: MilvusVectorDBConfig, logger: Optional[AsyncLogger] = None):
        """初始化 Milvus 向量数据库
        
        Args:
            config: 数据库配置
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
        self.config: MilvusVectorDBConfig = config
        
        # 连接 Milvus 服务器
        connections.connect(
            alias="default",
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password
        )
        
        # 初始化集合
        self._collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self) -> Collection:
        """获取或创建集合"""
        if utility.has_collection(self.config.collection_name):
            return Collection(self.config.collection_name)
            
        # 定义字段
        fields = [
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                is_primary=True
            ),
            FieldSchema(
                name="doc_id",
                dtype=DataType.VARCHAR,
                max_length=100
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535
            ),
            FieldSchema(
                name="start_char",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="end_char",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="text_len",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="extra",
                dtype=DataType.JSON
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.dimension
            )
        ]
        
        # 创建集合
        schema = CollectionSchema(
            fields=fields,
            description="Document chunks collection"
        )
        collection = Collection(
            name=self.config.collection_name,
            schema=schema
        )
        
        # 创建索引
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {
                "nlist": self.config.nlist,
                **self.config.index_params
            }
        }
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        return collection
        
    def _chunk_to_dict(self, chunk: Chunk, vector: List[float]) -> Dict[str, Any]:
        """将 Chunk 对象转换为字典"""
        return {
            "chunk_id": chunk.metadata.chunk_id,
            "doc_id": chunk.metadata.doc_id,
            "text": chunk.text,
            "start_char": chunk.metadata.start_char,
            "end_char": chunk.metadata.end_char,
            "text_len": chunk.metadata.text_len,
            "extra": chunk.metadata.extra,
            "vector": vector
        }
        
    def _dict_to_chunk(self, data: Dict[str, Any]) -> Chunk:
        """将字典转换为 Chunk 对象"""
        metadata = ChunkMetadata(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            text_len=data["text_len"],
            extra=data["extra"]
        )
        return Chunk(text=data["text"], metadata=metadata)
        
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
        data = []
        for chunk, vector in zip(chunks, vectors_array):
            data.append(self._chunk_to_dict(chunk, vector.tolist()))
            
        # 插入数据
        self._collection.insert(data)
        
        # 加载集合到内存
        self._collection.load()
        
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
        
        # 准备数据
        data = []
        for chunk, vector in zip(chunks, vectors_array):
            data.append(self._chunk_to_dict(chunk, vector.tolist()))
            
        # 插入数据
        self._collection.insert(data)
        
        await self._log(LogLevel.INFO, "向量添加完成")
        
    async def delete_vectors(self, chunk_ids: List[str]) -> None:
        """从索引中删除向量
        
        Args:
            chunk_ids: 要删除的文档块ID列表
        """
        await self._log(LogLevel.INFO, f"删除向量，数量: {len(chunk_ids)}")
        
        # 构建删除表达式
        expr = f'chunk_id in {chunk_ids}'
        
        # 执行删除
        self._collection.delete(expr)
        
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
        
        # 设置搜索参数
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": self.config.nprobe}
        }
        
        # 构建过滤表达式
        expr = None
        if filter_params:
            conditions = []
            for key, value in filter_params.items():
                if isinstance(value, (list, tuple)):
                    conditions.append(f"{key} in {list(value)}")
                else:
                    conditions.append(f"{key} == '{value}'")
            if conditions:
                expr = " && ".join(conditions)
                
        # 执行搜索
        k = top_k or self.config.top_k
        results = self._collection.search(
            data=[query_array[0].tolist()],
            anns_field="vector",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["chunk_id", "doc_id", "text", "start_char", "end_char", "text_len", "extra"]
        )
        
        # 构建结果
        search_results = []
        for hits in results:
            for hit in hits:
                chunk = self._dict_to_chunk(hit)
                score = float(hit.score)
                
                # 转换分数
                if self.config.metric_type == "COSINE":
                    score = 1 - score
                else:  # L2 或 IP
                    score = 1 / (1 + score)
                    
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
        
        # 设置搜索参数
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": self.config.nprobe}
        }
        
        # 构建过滤表达式
        expr = None
        if filter_params:
            conditions = []
            for key, value in filter_params.items():
                if isinstance(value, (list, tuple)):
                    conditions.append(f"{key} in {list(value)}")
                else:
                    conditions.append(f"{key} == '{value}'")
            if conditions:
                expr = " && ".join(conditions)
                
        # 执行批量搜索
        k = top_k or self.config.top_k
        results = self._collection.search(
            data=[v.tolist() for v in query_array],
            anns_field="vector",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["chunk_id", "doc_id", "text", "start_char", "end_char", "text_len", "extra"]
        )
        
        # 构建结果
        batch_results = []
        for query_hits in results:
            query_results = []
            for hit in query_hits:
                chunk = self._dict_to_chunk(hit)
                score = float(hit.score)
                
                # 转换分数
                if self.config.metric_type == "COSINE":
                    score = 1 - score
                else:  # L2 或 IP
                    score = 1 / (1 + score)
                    
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
            Milvus 是服务器端的数据库，索引数据由服务器管理，
            这里不需要额外的保存操作。
        """
        pass
        
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 索引文件路径
            
        Note:
            Milvus 是服务器端的数据库，这里只需要重新加载集合到内存。
        """
        if utility.has_collection(self.config.collection_name):
            self._collection.load()
            self._index_ready = True 