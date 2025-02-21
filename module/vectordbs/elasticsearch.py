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
from base.rag.chunking import Chunk, ChunkMetadata
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
    
    def __init__(self, config: ElasticsearchVectorDBConfig, logger: Optional[AsyncLogger] = None):
        """初始化 Elasticsearch 向量数据库
        
        Args:
            config: 数据库配置
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
        self.config: ElasticsearchVectorDBConfig = config
        
        # 初始化 ES 客户端
        self._client = AsyncElasticsearch(
            hosts=config.hosts,
            api_key=config.api_key,
            basic_auth=(config.username, config.password) if config.username else None
        )
        
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
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    }
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
        
    def _doc_to_chunk(self, doc: Dict[str, Any]) -> Chunk:
        """从 ES 文档构建 Chunk 对象
        
        Args:
            doc: ES 文档
            
        Returns:
            Chunk 对象
        """
        metadata = doc.get("metadata", {})
        chunk_metadata = ChunkMetadata(
            chunk_id=doc["chunk_id"],
            doc_id=doc.get("doc_id", ""),
            start_char=metadata.get("start_char", 0),
            end_char=metadata.get("end_char", len(doc["text"])),
            text_len=len(doc["text"]),
            extra=metadata
        )
        return Chunk(text=doc["text"], metadata=chunk_metadata)
        
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 创建索引
        await self._create_index()
        
        # 准备批量索引数据
        actions = []
        for vector, chunk in zip(vectors_array, chunks):
            action = {
                "_index": self.config.index_name,
                "_id": chunk.metadata.chunk_id,
                "_source": {
                    "chunk_id": chunk.metadata.chunk_id,
                    "doc_id": chunk.metadata.doc_id,
                    "text": chunk.text,
                    "vector": vector.tolist(),
                    "metadata": {
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char,
                        "text_len": chunk.metadata.text_len,
                        **(chunk.metadata.extra or {})
                    }
                }
            }
            actions.append(action)
            
        # 批量索引
        await async_bulk(self._client, actions)
        
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
        
        # 准备批量索引数据
        actions = []
        for vector, chunk in zip(vectors_array, chunks):
            action = {
                "_index": self.config.index_name,
                "_id": chunk.metadata.chunk_id,
                "_source": {
                    "chunk_id": chunk.metadata.chunk_id,
                    "doc_id": chunk.metadata.doc_id,
                    "text": chunk.text,
                    "vector": vector.tolist(),
                    "metadata": {
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char,
                        "text_len": chunk.metadata.text_len,
                        **(chunk.metadata.extra or {})
                    }
                }
            }
            actions.append(action)
            
        # 批量索引
        await async_bulk(self._client, actions)
        await self._log(LogLevel.INFO, "向量添加完成")
        
    async def delete_vectors(self, chunk_ids: List[str]) -> None:
        """从索引中删除向量
        
        Args:
            chunk_ids: 要删除的文档块ID列表
        """
        await self._log(LogLevel.INFO, f"删除向量，数量: {len(chunk_ids)}")
        
        # 构建批量删除请求
        body = []
        for chunk_id in chunk_ids:
            body.extend([
                {"delete": {"_index": self.config.index_name, "_id": chunk_id}}
            ])
            
        # 执行批量删除
        await self._client.bulk(body=body)
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
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {
                            "query_vector": query_array[0].tolist()
                        }
                    }
                }
            },
            "_source": ["chunk_id", "doc_id", "text", "metadata"]
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
            chunk = self._doc_to_chunk(hit["_source"])
            score = hit["_score"] - 1.0  # 恢复原始余弦相似度
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
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {
                                "query_vector": vector.tolist()
                            }
                        }
                    }
                },
                "_source": ["chunk_id", "doc_id", "text", "metadata"]
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
                chunk = self._doc_to_chunk(hit["_source"])
                score = hit["_score"] - 1.0  # 恢复原始余弦相似度
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
        """
        # Elasticsearch 已经持久化数据，不需要额外保存
        pass
        
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        # 检查索引是否存在
        if await self._client.indices.exists(index=self.config.index_name):
            self._index_ready = True 