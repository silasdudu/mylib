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
from base.rag.chunking import Chunk
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
    
    def __init__(self, config: WeaviateVectorDBConfig, **kwargs):
        """初始化 Weaviate 向量数据库
        
        Args:
            config: 数据库配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: WeaviateVectorDBConfig = config
        
        # 创建客户端
        auth_config = weaviate.auth.AuthApiKey(api_key=self.config.api_key) if self.config.api_key else None
        self._client = weaviate.Client(
            url=self.config.url,
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": self.config.api_key
            } if self.config.api_key else None
        )
        
        self._chunks = {}
    
    def _create_class_schema(self) -> Dict[str, Any]:
        """创建类模式
        
        Returns:
            类模式定义
        """
        return {
            "class": self.config.class_name,
            "description": "Document chunk for vector search",
            "vectorizer": "none",  # 我们会手动提供向量
            "vectorIndexType": self.config.vector_index_type,
            "vectorIndexConfig": self.config.vector_index_config,
            "properties": [
                {
                    "name": "chunk_id",
                    "dataType": ["string"],
                    "description": "Unique identifier of the chunk",
                    "indexInverted": True
                },
                {
                    "name": "doc_id",
                    "dataType": ["string"],
                    "description": "Document identifier",
                    "indexInverted": True
                },
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "Chunk text content",
                    "indexInverted": True
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata",
                    "indexInverted": True
                }
            ]
        }
    
    def _prepare_object(
        self,
        chunk: Chunk,
        vector: List[float]
    ) -> Dict[str, Any]:
        """准备要存储的对象
        
        Args:
            chunk: 文档块
            vector: 向量
            
        Returns:
            Weaviate对象
        """
        return {
            "chunk_id": chunk.metadata.chunk_id,
            "doc_id": chunk.metadata.doc_id,
            "text": chunk.text,
            "metadata": chunk.metadata.model_dump(),
            "vector": vector
        }
    
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建 Weaviate 索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 如果类已存在，先删除
        if self._client.schema.exists(self.config.class_name):
            self._client.schema.delete_class(self.config.class_name)
            
        # 创建类模式
        class_schema = self._create_class_schema()
        self._client.schema.create_class(class_schema)
        
        # 准备批量导入的数据
        with self._client.batch(
            batch_size=self.config.batch_size,
            dynamic=self.config.dynamic_schema
        ) as batch:
            for chunk, vector in zip(chunks, vectors_array):
                # 生成UUID
                uuid_str = generate_uuid5(chunk.metadata.chunk_id)
                
                # 准备对象
                data_object = self._prepare_object(chunk, vector.tolist())
                
                # 添加到批处理
                batch.add_data_object(
                    data_object=data_object,
                    class_name=self.config.class_name,
                    uuid=uuid_str
                )
                
        # 存储文档块映射
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        
        self._index_ready = True
        await self._log(LogLevel.INFO, "Weaviate 索引创建完成")
    
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        if not self._index_ready:
            await self.create_index(vectors, chunks)
            return
            
        await self._log(LogLevel.INFO, f"添加向量到 Weaviate 索引，数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 批量添加数据
        with self._client.batch(
            batch_size=self.config.batch_size,
            dynamic=self.config.dynamic_schema
        ) as batch:
            for chunk, vector in zip(chunks, vectors_array):
                # 生成UUID
                uuid_str = generate_uuid5(chunk.metadata.chunk_id)
                
                # 准备对象
                data_object = self._prepare_object(chunk, vector.tolist())
                
                # 添加到批处理
                batch.add_data_object(
                    data_object=data_object,
                    class_name=self.config.class_name,
                    uuid=uuid_str
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
        
        # 批量删除
        with self._client.batch() as batch:
            for chunk_id in chunk_ids:
                uuid_str = generate_uuid5(chunk_id)
                batch.delete_objects(
                    class_name=self.config.class_name,
                    where={
                        "path": ["chunk_id"],
                        "operator": "Equal",
                        "valueString": chunk_id
                    }
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
                chunk_id = item["chunk_id"]
                if chunk_id in self._chunks:
                    chunk = self._chunks[chunk_id]
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
            
        Note:
            Weaviate 是服务器端的数据库，索引数据由服务器管理。
            这里只保存文档块映射等客户端状态。
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return
            
        await self._log(LogLevel.INFO, f"保存 Weaviate 客户端状态到: {path}")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存客户端状态
        state = {
            "class_name": self.config.class_name,
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
            Weaviate 是服务器端的数据库，这里只加载文档块映射等客户端状态。
            索引数据需要在服务器端单独管理。
        """
        await self._log(LogLevel.INFO, f"从文件加载 Weaviate 客户端状态: {path}")
        
        # 加载客户端状态
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # 检查类是否存在
        if self._client.schema.exists(state["class_name"]):
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
                f"类 {state['class_name']} 不存在，请先创建类"
            ) 