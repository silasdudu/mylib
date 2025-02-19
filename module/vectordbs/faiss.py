"""
基于 Faiss 的向量数据库实现
"""
from typing import List, Optional, Dict, Any
import os
import numpy as np
import faiss
import pickle

from base.rag.vectordb import VectorDB, VectorDBConfig, SearchResult
from base.rag.chunking import Chunk
from base.core.logging import LogLevel


class FaissVectorDBConfig(VectorDBConfig):
    """Faiss 向量数据库配置"""
    index_type: str = "IVFFlat"  # 索引类型：Flat, IVFFlat, IVFPQ 等
    nlist: int = 100  # IVF 聚类中心数量
    nprobe: int = 10  # 搜索时探测的聚类中心数量
    use_gpu: bool = False  # 是否使用 GPU


class FaissVectorDB(VectorDB):
    """基于 Faiss 的向量数据库实现"""
    
    def __init__(self, config: FaissVectorDBConfig, **kwargs):
        """初始化 Faiss 向量数据库
        
        Args:
            config: 数据库配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: FaissVectorDBConfig = config
        self._index = None
        self._chunks = {}
        self._chunk_ids = []
    
    def _create_index(self) -> faiss.Index:
        """创建 Faiss 索引
        
        Returns:
            创建的索引对象
        """
        if self.config.index_type == "Flat":
            # 最简单的精确搜索索引
            index = faiss.IndexFlatL2(self.config.dimension)
        elif self.config.index_type == "IVFFlat":
            # IVF 索引，用于大规模数据集
            quantizer = faiss.IndexFlatL2(self.config.dimension)
            index = faiss.IndexIVFFlat(
                quantizer,
                self.config.dimension,
                self.config.nlist,
                faiss.METRIC_L2
            )
        elif self.config.index_type == "IVFPQ":
            # 乘积量化索引，用于超大规模数据集
            quantizer = faiss.IndexFlatL2(self.config.dimension)
            index = faiss.IndexIVFPQ(
                quantizer,
                self.config.dimension,
                self.config.nlist,
                8,  # 每个子向量的比特数
                8   # 每个向量被分成的子向量数
            )
        else:
            raise ValueError(f"不支持的索引类型: {self.config.index_type}")
            
        # 如果使用 GPU
        if self.config.use_gpu:
            if not faiss.get_num_gpus():
                raise RuntimeError("未检测到可用的 GPU")
            index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,  # 使用第一个 GPU
                index
            )
            
        return index
    
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建 Faiss 索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 创建索引
        self._index = self._create_index()
        
        # 如果是 IVF 类型的索引，需要先训练
        if isinstance(self._index, faiss.IndexIVF):
            await self._log(LogLevel.INFO, "训练 IVF 索引...")
            self._index.train(vectors_array)
            
        # 添加向量到索引
        self._index.add(vectors_array)
        
        # 存储文档块
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        self._chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        
        self._index_ready = True
        await self._log(LogLevel.INFO, "Faiss 索引创建完成")
    
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        if not self._index_ready:
            await self.create_index(vectors, chunks)
            return
            
        await self._log(LogLevel.INFO, f"添加向量到 Faiss 索引，数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 添加向量到索引
        self._index.add(vectors_array)
        
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
            
        Note:
            Faiss 不支持直接删除向量，需要重建索引
        """
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return
            
        await self._log(LogLevel.INFO, f"删除向量，数量: {len(chunk_ids)}")
        
        # 找出要保留的向量
        keep_indices = []
        new_chunk_ids = []
        for i, chunk_id in enumerate(self._chunk_ids):
            if chunk_id not in chunk_ids:
                keep_indices.append(i)
                new_chunk_ids.append(chunk_id)
                
        # 提取要保留的向量
        vectors_array = faiss.vector_to_array(self._index).reshape(-1, self.config.dimension)
        keep_vectors = vectors_array[keep_indices]
        
        # 更新文档块
        new_chunks = []
        for chunk_id in new_chunk_ids:
            if chunk_id in self._chunks:
                new_chunks.append(self._chunks[chunk_id])
                
        # 重建索引
        await self.create_index(keep_vectors.tolist(), new_chunks)
        
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
        if isinstance(self._index, faiss.IndexIVF):
            self._index.nprobe = filter_params.get("nprobe", self.config.nprobe)
            
        # 执行搜索
        k = top_k or self.config.top_k
        distances, indices = self._index.search(query_array, k)
        
        # 构建结果
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Faiss 使用 -1 表示无效结果
                chunk_id = self._chunk_ids[idx]
                chunk = self._chunks[chunk_id]
                # 将 L2 距离转换为相似度分数
                score = 1.0 / (1.0 + dist)
                results.append(SearchResult(
                    chunk=chunk,
                    score=float(score)
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
            
        # 验证查询向量
        query_array = self._validate_vectors(query_vectors)
        
        # 设置搜索参数
        if isinstance(self._index, faiss.IndexIVF):
            self._index.nprobe = filter_params.get("nprobe", self.config.nprobe)
            
        # 执行批量搜索
        k = top_k or self.config.top_k
        distances, indices = self._index.search(query_array, k)
        
        # 构建结果
        results = []
        for batch_distances, batch_indices in zip(distances, indices):
            batch_results = []
            for dist, idx in zip(batch_distances, batch_indices):
                if idx != -1:
                    chunk_id = self._chunk_ids[idx]
                    chunk = self._chunks[chunk_id]
                    score = 1.0 / (1.0 + dist)
                    batch_results.append(SearchResult(
                        chunk=chunk,
                        score=float(score)
                    ))
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
            
        await self._log(LogLevel.INFO, f"保存 Faiss 索引到: {path}")
        
        # 如果是 GPU 索引，先转换回 CPU
        if self.config.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self._index)
        else:
            index_cpu = self._index
            
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存索引
        faiss.write_index(index_cpu, f"{path}.index")
        
        # 保存元数据
        metadata = {
            "chunk_ids": self._chunk_ids,
            "chunks": self._chunks
        }
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(metadata, f)
            
        await self._log(LogLevel.INFO, "索引保存完成")
    
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        await self._log(LogLevel.INFO, f"从文件加载 Faiss 索引: {path}")
        
        # 加载索引
        self._index = faiss.read_index(f"{path}.index")
        
        # 如果需要使用 GPU
        if self.config.use_gpu:
            if not faiss.get_num_gpus():
                raise RuntimeError("未检测到可用的 GPU")
            self._index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self._index
            )
            
        # 加载元数据
        with open(f"{path}.meta", "rb") as f:
            metadata = pickle.load(f)
            self._chunk_ids = metadata["chunk_ids"]
            self._chunks = metadata["chunks"]
            
        self._index_ready = True
        await self._log(LogLevel.INFO, "索引加载完成") 