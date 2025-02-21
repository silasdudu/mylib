"""
基于 Weaviate 的向量数据库实现
"""
from typing import List, Optional, Dict, Any
import os
import json
import uuid
import numpy as np
import weaviate
from weaviate.util import generate_uuid5

from base.rag.vectordb import VectorDB, VectorDBConfig, SearchResult
from base.rag.chunking import Chunk, ChunkMetadata
from base.core.logging import LogLevel


class WeaviateVectorDBConfig(VectorDBConfig):
    """Weaviate 向量数据库配置"""
    url: str = "http://localhost:8080"  # Weaviate 服务器地址
    api_key: Optional[str] = None  # API密钥
    class_name: str = "DocumentChunk"  # 类名
    batch_size: int = 100  # 批处理大小
    dynamic_schema: bool = False  # 是否使用动态模式
    vector_index_type: str = "hnsw"  # 向量索引类型
    vector_index_config: Dict[str, Any] = {  # 向量索引配置
        "maxConnections": 64,
        "efConstruction": 128,
        "ef": -1,
        "dynamicEfMin": 100,
        "dynamicEfMax": 500,
        "dynamicEfFactor": 8,
        "vectorCacheMaxObjects": 1000000,
        "flatSearchCutoff": 40000,
        "distance": "cosine"
    }


class WeaviateVectorDB(VectorDB):
    """基于 Weaviate 的向量数据库实现"""
    
    def __init__(self, config: WeaviateVectorDBConfig, logger: Optional[AsyncLogger] = None):
        """初始化 Weaviate 向量数据库
        
        Args:
            config: 数据库配置
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
        self.config: WeaviateVectorDBConfig = config
        
        # 初始化 Weaviate 客户端
        auth_config = weaviate.auth.AuthApiKey(api_key=config.api_key) if config.api_key else None
        self._client = weaviate.Client(
            url=config.url,
            auth_client_secret=auth_config,
            additional_headers={"X-OpenAI-Api-Key": config.api_key} if config.api_key else None
        )
        
        # 创建类
        self._create_class()
        
    def _create_class(self) -> None:
        """创建 Weaviate 类"""
        # 检查类是否存在
        if not self._client.schema.exists(self.config.class_name):
            # 创建类
            class_obj = {
                "class": self.config.class_name,
                "description": "Document chunk for vector search",
                "vectorizer": "none",  # 我们自己提供向量
                "vectorIndexType": self.config.vector_index_type,
                "vectorIndexConfig": self.config.vector_index_config,
                "properties": [
                    {
                        "name": "chunk_id",
                        "dataType": ["string"],
                        "description": "Unique identifier of the chunk"
                    },
                    {
                        "name": "doc_id",
                        "dataType": ["string"],
                        "description": "Document identifier"
                    },
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "Chunk text content"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Chunk metadata"
                    }
                ]
            }
            self._client.schema.create_class(class_obj)
            
    def _object_to_chunk(self, obj: Dict[str, Any]) -> Chunk:
        """从 Weaviate 对象构建 Chunk 对象
        
        Args:
            obj: Weaviate 对象
            
        Returns:
            Chunk 对象
        """
        metadata = obj.get("metadata", {})
        chunk_metadata = ChunkMetadata(
            chunk_id=obj["chunk_id"],
            doc_id=obj.get("doc_id", ""),
            start_char=metadata.get("start_char", 0),
            end_char=metadata.get("end_char", len(obj["text"])),
            text_len=len(obj["text"]),
            extra=metadata
        )
        return Chunk(text=obj["text"], metadata=chunk_metadata)
        
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 准备批量导入数据
        with self._client.batch(
            batch_size=self.config.batch_size,
            dynamic=self.config.dynamic_schema
        ) as batch:
            for vector, chunk in zip(vectors_array, chunks):
                # 准备对象数据
                data_object = {
                    "chunk_id": chunk.metadata.chunk_id,
                    "doc_id": chunk.metadata.doc_id,
                    "text": chunk.text,
                    "metadata": {
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char,
                        "text_len": chunk.metadata.text_len,
                        **(chunk.metadata.extra or {})
                    }
                }
                
                # 添加到批处理
                batch.add_data_object(
                    data_object=data_object,
                    class_name=self.config.class_name,
                    vector=vector.tolist(),
                    uuid=generate_uuid5(chunk.metadata.chunk_id)
                )
                
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
        vectors_array = self._validate_vectors(vectors)
        
        # 准备批量导入数据
        with self._client.batch(
            batch_size=self.config.batch_size,
            dynamic=self.config.dynamic_schema
        ) as batch:
            for vector, chunk in zip(vectors_array, chunks):
                # 准备对象数据
                data_object = {
                    "chunk_id": chunk.metadata.chunk_id,
                    "doc_id": chunk.metadata.doc_id,
                    "text": chunk.text,
                    "metadata": {
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char,
                        "text_len": chunk.metadata.text_len,
                        **(chunk.metadata.extra or {})
                    }
                }
                
                # 添加到批处理
                batch.add_data_object(
                    data_object=data_object,
                    class_name=self.config.class_name,
                    vector=vector.tolist(),
                    uuid=generate_uuid5(chunk.metadata.chunk_id)
                )
                
        await self._log(LogLevel.INFO, "向量添加完成")
        
    async def delete_vectors(self, chunk_ids: List[str]) -> None:
        """从索引中删除向量
        
        Args:
            chunk_ids: 要删除的文档块ID列表
        """
        await self._log(LogLevel.INFO, f"删除向量，数量: {len(chunk_ids)}")
        
        # 批量删除对象
        with self._client.batch() as batch:
            for chunk_id in chunk_ids:
                batch.delete_objects(
                    class_name=self.config.class_name,
                    where={"path": ["chunk_id"], "operator": "Equal", "valueString": chunk_id}
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
        
        # 构建查询
        k = top_k or self.config.top_k
        query = self._client.query.get(
            self.config.class_name,
            ["chunk_id", "doc_id", "text", "metadata", "_additional {certainty}"]
        )
        
        # 添加向量搜索
        query = query.with_near_vector({
            "vector": query_array[0].tolist(),
            "certainty": 0.7  # 最小相似度阈值
        })
        
        # 添加过滤条件
        if filter_params:
            where_filter = {}
            for key, value in filter_params.items():
                if isinstance(value, (list, tuple)):
                    where_filter = {
                        "operator": "Or",
                        "operands": [
                            {
                                "path": [key],
                                "operator": "Equal",
                                "valueString": str(v)
                            }
                            for v in value
                        ]
                    }
                else:
                    where_filter = {
                        "path": [key],
                        "operator": "Equal",
                        "valueString": str(value)
                    }
            query = query.with_where(where_filter)
            
        # 限制返回数量
        query = query.with_limit(k)
        
        # 执行查询
        results = query.do()
        
        # 构建结果
        search_results = []
        if results and "data" in results:
            for item in results["data"]["Get"][self.config.class_name]:
                chunk = self._object_to_chunk(item)
                certainty = item["_additional"]["certainty"]
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=float(certainty)
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
        
        # 执行批量搜索
        batch_results = []
        for vector in query_array:
            results = await self.search(
                query_vector=vector.tolist(),
                top_k=top_k,
                filter_params=filter_params
            )
            batch_results.append(results)
            
        return batch_results
        
    async def save(self, path: str) -> None:
        """保存索引到文件
        
        Args:
            path: 保存路径
        """
        # Weaviate 已经持久化数据，不需要额外保存
        pass
        
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        # 检查类是否存在
        if self._client.schema.exists(self.config.class_name):
            self._index_ready = True 