"""
基于 Chroma 的向量数据库实现
"""
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

from base.rag.vectordb import VectorDB, VectorDBConfig, SearchResult
from base.rag.chunking import Chunk
from base.core.logging import LogLevel


class ChromaVectorDBConfig(VectorDBConfig):
    """Chroma 向量数据库配置"""
    persist_directory: str = "data/chroma"  # 持久化目录
    collection_name: str = "document_chunks"  # 集合名称
    distance_metric: str = "cosine"  # 距离度量方式：cosine, l2, ip
    metadata_fields: List[str] = []  # 要索引的元数据字段


class ChromaVectorDB(VectorDB):
    """基于 Chroma 的向量数据库实现"""
    
    def __init__(self, config: ChromaVectorDBConfig, **kwargs):
        """初始化 Chroma 向量数据库
        
        Args:
            config: 数据库配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: ChromaVectorDBConfig = config
        
        # 创建客户端
        self._client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        self._collection = None
        self._chunks = {}
        
        # 尝试获取已存在的集合
        try:
            self._collection = self._client.get_collection(self.config.collection_name)
            # 检查集合中是否有数据
            collection_data = self._collection.get()
            if collection_data["ids"]:  # 只有在集合中有数据时才设置 _index_ready
                self._index_ready = True
        except ValueError:
            # 集合不存在，保持 _index_ready 为 False
            pass
    
    def _get_or_create_collection(self) -> Collection:
        """获取或创建集合
        
        Returns:
            集合对象
        """
        # 获取现有集合或创建新集合
        collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={
                "dimension": self.config.dimension,
                "distance_metric": self.config.distance_metric
            }
        )
        
        return collection
    
    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """准备要索引的元数据
        
        Args:
            chunk: 文档块
            
        Returns:
            处理后的元数据字典
        """
        metadata = {
            "doc_id": chunk.metadata.doc_id,
            "chunk_id": chunk.metadata.chunk_id
        }
        
        # 添加指定的元数据字段
        chunk_metadata = chunk.metadata.model_dump()
        for field in self.config.metadata_fields:
            if field in chunk_metadata:
                metadata[field] = chunk_metadata[field]
                
        return metadata
    
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建 Chroma 索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 获取或创建集合
        self._collection = self._get_or_create_collection()
        
        # 准备数据
        ids = [chunk.metadata.chunk_id for chunk in chunks]
        embeddings = vectors_array.tolist()
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
        
        # 添加数据
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # 存储文档块映射
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        
        self._index_ready = True
        await self._log(LogLevel.INFO, "Chroma 索引创建完成")
    
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        if not self._index_ready:
            await self.create_index(vectors, chunks)
            return
            
        await self._log(LogLevel.INFO, f"添加向量到 Chroma 索引，数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 准备数据
        ids = [chunk.metadata.chunk_id for chunk in chunks]
        embeddings = vectors_array.tolist()
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
        
        # 添加数据
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
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
        
        # 删除数据
        self._collection.delete(ids=chunk_ids)
        
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
        
        # 执行搜索
        k = top_k or self.config.top_k
        await self._log(LogLevel.INFO, f"执行搜索，top_k={k}")
        
        results = self._collection.query(
            query_embeddings=query_array.tolist(),
            n_results=k,
            where=filter_params
        )
        
        await self._log(LogLevel.INFO, f"搜索结果: {results}")
        
        # 构建结果
        search_results = []
        if results["ids"]:
            for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
                if chunk_id in self._chunks:
                    chunk = self._chunks[chunk_id]
                    # 将距离转换为相似度分数
                    if self.config.distance_metric == "cosine":
                        score = 1 - distance
                    else:  # l2 或 ip
                        score = 1 / (1 + distance)
                    search_results.append(SearchResult(
                        chunk=chunk,
                        score=float(score)
                    ))
                    
        await self._log(LogLevel.INFO, f"构建了 {len(search_results)} 个搜索结果")
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
        k = top_k or self.config.top_k
        results = self._collection.query(
            query_embeddings=query_array.tolist(),
            n_results=k,
            where=filter_params
        )
        
        # 构建结果
        batch_results = []
        if results["ids"]:
            for batch_ids, batch_distances in zip(results["ids"], results["distances"]):
                query_results = []
                for chunk_id, distance in zip(batch_ids, batch_distances):
                    if chunk_id in self._chunks:
                        chunk = self._chunks[chunk_id]
                        # 将距离转换为相似度分数
                        if self.config.distance_metric == "cosine":
                            score = 1 - distance
                        else:  # l2 或 ip
                            score = 1 / (1 + distance)
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
            Chroma 会自动持久化到指定目录，这里只保存额外的状态信息
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return
            
        await self._log(LogLevel.INFO, f"保存 Chroma 客户端状态到: {path}")
        
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
            Chroma 会自动从持久化目录加载数据，这里只加载额外的状态信息
        """
        await self._log(LogLevel.INFO, f"从文件加载 Chroma 客户端状态: {path}")
        
        # 加载客户端状态
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # 恢复集合连接
        try:
            self._collection = self._client.get_collection(state["collection_name"])
            
            # 检查集合中是否有数据
            collection_data = self._collection.get()
            if collection_data["ids"]:  # 只有在集合中有数据时才设置 _index_ready
                # 恢复文档块映射
                self._chunks = {
                    chunk_id: Chunk.parse_obj(chunk_data)
                    for chunk_id, chunk_data in state["chunks"].items()
                }
                
                self._index_ready = True
                await self._log(LogLevel.INFO, "客户端状态加载完成")
            else:
                await self._log(
                    LogLevel.WARNING,
                    "集合存在但没有数据，需要重新构建索引"
                )
        except ValueError:
            await self._log(
                LogLevel.ERROR,
                f"集合 {state['collection_name']} 不存在，请确保数据已正确持久化"
            ) 