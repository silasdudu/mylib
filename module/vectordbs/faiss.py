"""
基于 Faiss 的向量数据库实现
"""
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
import faiss
import sqlite3
from pathlib import Path

from base.rag.vectordb import VectorDB, VectorDBConfig, SearchResult
from base.rag.chunking import Chunk, ChunkMetadata
from base.core.logging import AsyncLogger, LogLevel


class FaissVectorDBConfig(VectorDBConfig):
    """Faiss 向量数据库配置"""
    index_type: str = "IVFFlat"  # 索引类型：Flat, IVFFlat, IVFPQ, IndexHNSW
    nlist: int = 100  # IVF聚类中心数量
    nprobe: int = 10  # 搜索时探测的聚类中心数量
    storage_path: str = "data/faiss"  # 存储路径
    use_gpu: bool = False  # 是否使用GPU


class FaissVectorDB(VectorDB):
    """基于 Faiss 的向量数据库实现"""
    
    def __init__(self, config: FaissVectorDBConfig, logger: Optional[AsyncLogger] = None):
        """初始化 Faiss 向量数据库
        
        Args:
            config: 数据库配置
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
        self.config: FaissVectorDBConfig = config
        
        # 初始化 Faiss 索引
        self._index = None
        
        # 初始化 SQLite 存储
        os.makedirs(config.storage_path, exist_ok=True)
        self._db_path = os.path.join(config.storage_path, "chunks.db")
        self._init_storage()
        
    def _init_storage(self) -> None:
        """初始化 SQLite 存储"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    text_len INTEGER NOT NULL,
                    extra TEXT
                )
            """)
            conn.commit()
            
    def _store_chunk(self, chunk: Chunk) -> None:
        """将 Chunk 对象存储到 SQLite"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, text, start_char, end_char, text_len, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.metadata.chunk_id,
                    chunk.metadata.doc_id,
                    chunk.text,
                    chunk.metadata.start_char,
                    chunk.metadata.end_char,
                    chunk.metadata.text_len,
                    json.dumps(chunk.metadata.extra)
                )
            )
            conn.commit()
            
    def _load_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """从 SQLite 加载 Chunk 对象"""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?",
                (chunk_id,)
            )
            row = cursor.fetchone()
            
        if row:
            metadata = ChunkMetadata(
                chunk_id=row[0],
                doc_id=row[1],
                start_char=row[3],
                end_char=row[4],
                text_len=row[5],
                extra=json.loads(row[6]) if row[6] else {}
            )
            return Chunk(text=row[2], metadata=metadata)
        return None
        
    def _delete_chunks(self, chunk_ids: List[str]) -> None:
        """从 SQLite 删除 Chunk 对象"""
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "DELETE FROM chunks WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids]
            )
            conn.commit()
            
    def _create_index(self) -> None:
        """创建 Faiss 索引"""
        if self.config.index_type == "Flat":
            self._index = faiss.IndexFlatL2(self.config.dimension)
        elif self.config.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.config.dimension)
            self._index = faiss.IndexIVFFlat(
                quantizer,
                self.config.dimension,
                self.config.nlist,
                faiss.METRIC_L2
            )
        elif self.config.index_type == "IVFPQ":
            quantizer = faiss.IndexFlatL2(self.config.dimension)
            self._index = faiss.IndexIVFPQ(
                quantizer,
                self.config.dimension,
                self.config.nlist,
                8,  # 每个子向量的位数
                8   # 每个维度的比特数
            )
        elif self.config.index_type == "IndexHNSW":
            self._index = faiss.IndexHNSWFlat(
                self.config.dimension,
                32  # HNSW图的每层连接数
            )
        else:
            raise ValueError(f"不支持的索引类型: {self.config.index_type}")
            
        # 如果使用GPU
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            self._index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self._index
            )
            
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
        self._create_index()
        
        # 如果是IVF类型的索引，需要先训练
        if isinstance(self._index, faiss.IndexIVF):
            await self._log(LogLevel.INFO, "训练IVF索引...")
            self._index.train(vectors_array)
            
        # 添加向量到索引
        self._index.add(vectors_array)
        
        # 存储文档块
        for chunk in chunks:
            self._store_chunk(chunk)
            
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
        
        # 添加向量到索引
        self._index.add(vectors_array)
        
        # 存储文档块
        for chunk in chunks:
            self._store_chunk(chunk)
            
        await self._log(LogLevel.INFO, "向量添加完成")
        
    async def delete_vectors(self, chunk_ids: List[str]) -> None:
        """从索引中删除向量
        
        Args:
            chunk_ids: 要删除的文档块ID列表
        """
        await self._log(LogLevel.INFO, f"删除向量，数量: {len(chunk_ids)}")
        
        # Faiss不支持直接删除向量，需要重建索引
        # 这里只从SQLite中删除文档块
        self._delete_chunks(chunk_ids)
        
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
        if isinstance(self._index, faiss.IndexIVF):
            self._index.nprobe = self.config.nprobe
            
        # 执行搜索
        k = top_k or self.config.top_k
        distances, indices = self._index.search(query_array, k)
        
        # 构建结果
        search_results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Faiss返回-1表示无效结果
                continue
                
            # 从SQLite加载chunk
            chunk = self._load_chunk(f"chunk_{idx}")
            if not chunk:
                continue
                
            # 计算相似度分数
            if self.config.distance_metric == "cosine":
                score = 1 - distance
            else:  # l2 或 ip
                score = 1 / (1 + distance)
                
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
        
        # 设置搜索参数
        if isinstance(self._index, faiss.IndexIVF):
            self._index.nprobe = self.config.nprobe
            
        # 执行批量搜索
        k = top_k or self.config.top_k
        distances, indices = self._index.search(query_array, k)
        
        # 构建结果
        batch_results = []
        for query_distances, query_indices in zip(distances, indices):
            query_results = []
            for distance, idx in zip(query_distances, query_indices):
                if idx == -1:  # Faiss返回-1表示无效结果
                    continue
                    
                # 从SQLite加载chunk
                chunk = self._load_chunk(f"chunk_{idx}")
                if not chunk:
                    continue
                    
                # 计算相似度分数
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
        """
        # 创建保存目录
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存Faiss索引
        index_path = save_dir / "index.faiss"
        if isinstance(self._index, faiss.GpuIndex):
            # 如果是GPU索引，先转换回CPU
            cpu_index = faiss.index_gpu_to_cpu(self._index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self._index, str(index_path))
            
        # 复制SQLite数据库
        import shutil
        db_path = save_dir / "chunks.db"
        shutil.copy2(self._db_path, db_path)
        
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        load_dir = Path(path)
        
        # 加载Faiss索引
        index_path = load_dir / "index.faiss"
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
            if self.config.use_gpu and faiss.get_num_gpus() > 0:
                self._index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(),
                    0,
                    self._index
                )
            self._index_ready = True
            
        # 加载SQLite数据库
        db_path = load_dir / "chunks.db"
        if db_path.exists():
            import shutil
            shutil.copy2(db_path, self._db_path) 