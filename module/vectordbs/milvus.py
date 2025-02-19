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
from base.rag.chunking import Chunk
from base.core.logging import LogLevel


class MilvusVectorDBConfig(VectorDBConfig):
    """Milvus 向量数据库配置"""
    host: str = "localhost"  # Milvus 服务器地址
    port: int = 19530  # Milvus 服务器端口
    collection_name: str = "document_chunks"  # 集合名称
    index_type: str = "IVF_FLAT"  # 索引类型
    metric_type: str = "L2"  # 距离度量类型
    nlist: int = 1024  # IVF 聚类中心数量
    nprobe: int = 16  # 搜索时探测的聚类中心数量
    consistency_level: str = "Strong"  # 一致性级别


class MilvusVectorDB(VectorDB):
    """基于 Milvus 的向量数据库实现"""
    
    def __init__(self, config: MilvusVectorDBConfig, **kwargs):
        """初始化 Milvus 向量数据库
        
        Args:
            config: 数据库配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: MilvusVectorDBConfig = config
        self._collection = None
        self._chunks = {}
        
        # 连接 Milvus 服务器
        connections.connect(
            alias="default",
            host=self.config.host,
            port=self.config.port
        )
    
    def _create_collection(self) -> Collection:
        """创建 Milvus 集合
        
        Returns:
            创建的集合对象
        """
        # 定义字段
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=100
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.dimension
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.VARCHAR,
                max_length=65535  # 最大支持64KB的元数据
            )
        ]
        
        # 创建集合
        schema = CollectionSchema(
            fields=fields,
            description="Document chunks collection"
        )
        collection = Collection(
            name=self.config.collection_name,
            schema=schema,
            consistency_level=self.config.consistency_level
        )
        
        # 创建索引
        index_params = {
            "metric_type": self.config.metric_type,
            "index_type": self.config.index_type,
            "params": {
                "nlist": self.config.nlist
            }
        }
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        return collection
    
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建 Milvus 索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 如果集合已存在，先删除
        if utility.has_collection(self.config.collection_name):
            utility.drop_collection(self.config.collection_name)
            
        # 创建新集合
        self._collection = self._create_collection()
        
        # 准备数据
        data = [
            [chunk.metadata.chunk_id for chunk in chunks],  # chunk_ids
            vectors_array.tolist(),  # vectors
            [json.dumps({
                "doc_id": chunk.metadata.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata.dict()
            }) for chunk in chunks]  # metadata
        ]
        
        # 插入数据
        self._collection.insert(data)
        self._collection.flush()
        
        # 加载集合
        self._collection.load()
        
        # 存储文档块映射
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        
        self._index_ready = True
        await self._log(LogLevel.INFO, "Milvus 索引创建完成")
    
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        if not self._index_ready:
            await self.create_index(vectors, chunks)
            return
            
        await self._log(LogLevel.INFO, f"添加向量到 Milvus 索引，数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 准备数据
        data = [
            [chunk.metadata.chunk_id for chunk in chunks],  # chunk_ids
            vectors_array.tolist(),  # vectors
            [json.dumps({
                "doc_id": chunk.metadata.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata.dict()
            }) for chunk in chunks]  # metadata
        ]
        
        # 插入数据
        self._collection.insert(data)
        self._collection.flush()
        
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
        
        # 构建删除表达式
        expr = f'chunk_id in {chunk_ids}'
        
        # 执行删除
        self._collection.delete(expr)
        self._collection.flush()
        
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
        
        # 设置搜索参数
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": filter_params.get("nprobe", self.config.nprobe)}
        }
        
        # 执行搜索
        k = top_k or self.config.top_k
        results = self._collection.search(
            data=query_array.tolist(),
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["chunk_id", "metadata"]
        )
        
        # 构建结果
        search_results = []
        for hits in results:
            for hit in hits:
                chunk_id = hit.entity.get("chunk_id")
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
        
        # 设置搜索参数
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": filter_params.get("nprobe", self.config.nprobe)}
        }
        
        # 执行批量搜索
        k = top_k or self.config.top_k
        results = self._collection.search(
            data=query_array.tolist(),
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["chunk_id", "metadata"]
        )
        
        # 构建结果
        batch_results = []
        for hits in results:
            query_results = []
            for hit in hits:
                chunk_id = hit.entity.get("chunk_id")
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
            Milvus 是服务器端的数据库，索引数据由服务器管理。
            这里只保存文档块映射等客户端状态。
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return
            
        await self._log(LogLevel.INFO, f"保存 Milvus 客户端状态到: {path}")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存客户端状态
        state = {
            "collection_name": self.config.collection_name,
            "chunks": {
                chunk_id: chunk.dict()
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
            Milvus 是服务器端的数据库，这里只加载文档块映射等客户端状态。
            索引数据需要在服务器端单独管理。
        """
        await self._log(LogLevel.INFO, f"从文件加载 Milvus 客户端状态: {path}")
        
        # 加载客户端状态
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # 恢复集合连接
        if utility.has_collection(state["collection_name"]):
            self._collection = Collection(state["collection_name"])
            self._collection.load()
            
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
                f"集合 {state['collection_name']} 不存在，请先在服务器端恢复索引"
            ) 