"""
基于 Elasticsearch 的向量数据库实现
"""
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from base.rag.vectordb import VectorDB, VectorDBConfig, SearchResult
from base.rag.chunking import Chunk
from base.core.logging import LogLevel


class ElasticsearchVectorDBConfig(VectorDBConfig):
    """Elasticsearch 向量数据库配置"""
    hosts: List[str] = ["http://localhost:9200"]  # ES 服务器地址
    index_name: str = "document_chunks"  # 索引名称
    username: Optional[str] = None  # 用户名
    password: Optional[str] = None  # 密码
    api_key: Optional[str] = None  # API密钥
    similarity: str = "cosine"  # 相似度计算方式：cosine, dot_product, l2_norm
    ef_construction: int = 100  # HNSW 构建时的搜索宽度
    m: int = 16  # HNSW 图的每层连接数
    ef_search: int = 50  # HNSW 搜索时的搜索宽度


class ElasticsearchVectorDB(VectorDB):
    """基于 Elasticsearch 的向量数据库实现"""
    
    def __init__(self, config: ElasticsearchVectorDBConfig, **kwargs):
        """初始化 Elasticsearch 向量数据库
        
        Args:
            config: 数据库配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: ElasticsearchVectorDBConfig = config
        
        # 创建客户端
        auth = {}
        if self.config.username and self.config.password:
            auth["basic_auth"] = (self.config.username, self.config.password)
        elif self.config.api_key:
            auth["api_key"] = self.config.api_key
            
        self._client = AsyncElasticsearch(
            hosts=self.config.hosts,
            **auth
        )
        
        self._chunks = {}
    
    async def _create_index(self) -> None:
        """创建 Elasticsearch 索引"""
        # 定义索引映射
        mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.config.dimension,
                        "index": True,
                        "similarity": self.config.similarity,
                        "index_options": {
                            "type": "hnsw",
                            "m": self.config.m,
                            "ef_construction": self.config.ef_construction
                        }
                    },
                    "metadata": {"type": "object"}
                }
            },
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "refresh_interval": "1s"
                }
            }
        }
        
        # 如果索引已存在，先删除
        if await self._client.indices.exists(index=self.config.index_name):
            await self._client.indices.delete(index=self.config.index_name)
            
        # 创建索引
        await self._client.indices.create(
            index=self.config.index_name,
            body=mapping
        )
    
    def _prepare_document(
        self,
        chunk: Chunk,
        vector: List[float]
    ) -> Dict[str, Any]:
        """准备要索引的文档
        
        Args:
            chunk: 文档块
            vector: 向量
            
        Returns:
            ES文档
        """
        return {
            "_index": self.config.index_name,
            "_id": chunk.metadata.chunk_id,
            "_source": {
                "chunk_id": chunk.metadata.chunk_id,
                "doc_id": chunk.metadata.doc_id,
                "text": chunk.text,
                "vector": vector,
                "metadata": chunk.metadata.model_dump()
            }
        }
    
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建 Elasticsearch 索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 创建索引
        await self._create_index()
        
        # 准备批量索引的文档
        actions = [
            self._prepare_document(chunk, vector.tolist())
            for chunk, vector in zip(chunks, vectors_array)
        ]
        
        # 批量索引文档
        await async_bulk(self._client, actions)
        
        # 刷新索引
        await self._client.indices.refresh(index=self.config.index_name)
        
        # 存储文档块映射
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        
        self._index_ready = True
        await self._log(LogLevel.INFO, "Elasticsearch 索引创建完成")
    
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        if not self._index_ready:
            await self.create_index(vectors, chunks)
            return
            
        await self._log(LogLevel.INFO, f"添加向量到 Elasticsearch 索引，数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 准备批量索引的文档
        actions = [
            self._prepare_document(chunk, vector.tolist())
            for chunk, vector in zip(chunks, vectors_array)
        ]
        
        # 批量索引文档
        await async_bulk(self._client, actions)
        
        # 刷新索引
        await self._client.indices.refresh(index=self.config.index_name)
        
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
        
        # 构建删除查询
        body = {
            "query": {
                "terms": {
                    "chunk_id": chunk_ids
                }
            }
        }
        
        # 执行删除
        await self._client.delete_by_query(
            index=self.config.index_name,
            body=body
        )
        
        # 刷新索引
        await self._client.indices.refresh(index=self.config.index_name)
        
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
        body = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "knn_score",
                        "params": {
                            "field": "vector",
                            "query_value": query_array[0].tolist(),
                            "space_type": self.config.similarity
                        }
                    }
                }
            }
        }
        
        # 添加过滤条件
        if filter_params:
            body["query"]["script_score"]["query"] = {
                "bool": {
                    "must": {"match_all": {}},
                    "filter": filter_params
                }
            }
            
        # 执行搜索
        response = await self._client.search(
            index=self.config.index_name,
            body=body
        )
        
        # 构建结果
        search_results = []
        for hit in response["hits"]["hits"]:
            chunk_id = hit["_source"]["chunk_id"]
            if chunk_id in self._chunks:
                chunk = self._chunks[chunk_id]
                score = hit["_score"]
                # ES的得分已经是标准化的相似度分数
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=float(score)
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
        
        # 构建批量查询
        k = top_k or self.config.top_k
        body = []
        for vector in query_array:
            # 添加空行
            body.append({})
            # 添加查询
            query = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "knn_score",
                            "params": {
                                "field": "vector",
                                "query_value": vector.tolist(),
                                "space_type": self.config.similarity
                            }
                        }
                    }
                }
            }
            
            # 添加过滤条件
            if filter_params:
                query["query"]["script_score"]["query"] = {
                    "bool": {
                        "must": {"match_all": {}},
                        "filter": filter_params
                    }
                }
                
            body.append(query)
            
        # 执行批量搜索
        response = await self._client.msearch(
            index=self.config.index_name,
            body=body
        )
        
        # 构建结果
        batch_results = []
        for response_item in response["responses"]:
            query_results = []
            for hit in response_item["hits"]["hits"]:
                chunk_id = hit["_source"]["chunk_id"]
                if chunk_id in self._chunks:
                    chunk = self._chunks[chunk_id]
                    score = hit["_score"]
                    query_results.append(SearchResult(
                        chunk=chunk,
                        score=float(score)
                    ))
            batch_results.append(query_results)
            
        return batch_results
    
    async def save(self, path: str) -> None:
        """保存索引到文件
        
        Args:
            path: 保存路径
            
        Note:
            Elasticsearch 是服务器端的数据库，索引数据由服务器管理。
            这里只保存文档块映射等客户端状态。
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return
            
        await self._log(LogLevel.INFO, f"保存 Elasticsearch 客户端状态到: {path}")
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存客户端状态
        state = {
            "index_name": self.config.index_name,
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
            Elasticsearch 是服务器端的数据库，这里只加载文档块映射等客户端状态。
            索引数据需要在服务器端单独管理。
        """
        await self._log(LogLevel.INFO, f"从文件加载 Elasticsearch 客户端状态: {path}")
        
        # 加载客户端状态
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # 检查索引是否存在
        if await self._client.indices.exists(index=state["index_name"]):
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
                f"索引 {state['index_name']} 不存在，请先在服务器端恢复索引"
            )
            
    async def close(self) -> None:
        """关闭数据库连接"""
        await self._client.close() 