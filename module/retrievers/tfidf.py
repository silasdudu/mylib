"""
TF-IDF 检索器实现，基于词频-逆文档频率的文本检索
"""
from typing import List, Optional, Dict, Any, Union
import numpy as np
from collections import Counter
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from base.rag.retriever import Retriever, RetrieverConfig, SearchResult
from base.rag.chunking import Chunk


class TFIDFRetrieverConfig(RetrieverConfig):
    """TF-IDF 检索器配置"""
    use_stopwords: bool = True  # 是否使用停用词
    stopwords_file: Optional[str] = None  # 停用词文件路径
    min_df: int = 1  # 最小文档频率
    max_df: float = 0.95  # 最大文档频率比例
    ngram_range: tuple = (1, 1)  # n-gram范围
    analyzer: str = "word"  # 分析器类型：word, char
    binary: bool = False  # 是否使用二值化TF
    use_idf: bool = True  # 是否使用IDF加权
    smooth_idf: bool = True  # 是否平滑IDF权重
    sublinear_tf: bool = False  # 是否使用次线性TF缩放
    normalize_vectors: bool = True  # 是否对向量进行归一化
    threshold: float = 0.0  # 相似度阈值，低于此值的结果将被过滤


class TFIDFRetriever(Retriever):
    """基于 TF-IDF 的检索器实现"""
    
    def __init__(self, config: TFIDFRetrieverConfig, **kwargs):
        """初始化 TF-IDF 检索器
        
        Args:
            config: 检索器配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: TFIDFRetrieverConfig = config
        
        # 加载停用词
        self._stopwords = set()
        if self.config.use_stopwords:
            if self.config.stopwords_file:
                with open(self.config.stopwords_file, 'r', encoding='utf-8') as f:
                    self._stopwords = set(line.strip() for line in f)
        
        # 初始化 TF-IDF 向量化器
        self._vectorizer = TfidfVectorizer(
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range=config.ngram_range,
            analyzer=config.analyzer,
            binary=config.binary,
            use_idf=config.use_idf,
            smooth_idf=config.smooth_idf,
            sublinear_tf=config.sublinear_tf,
            tokenizer=self._tokenize,
            stop_words=self._stopwords if config.use_stopwords else None
        )
        
        # 初始化存储
        self._vectors = None  # TF-IDF矩阵
        self._chunks = {}  # 文档块字典
        self._chunk_ids = []  # chunk_id列表
    
    def _tokenize(self, text: str) -> List[str]:
        """分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        # 使用jieba分词
        words = jieba.lcut(text)
        
        # 过滤停用词
        if self.config.use_stopwords:
            words = [w for w in words if w not in self._stopwords]
            
        return words
    
    def _compute_similarity(
        self,
        query_vector: np.ndarray,
        index_vectors: np.ndarray
    ) -> np.ndarray:
        """计算查询向量与索引向量的相似度
        
        Args:
            query_vector: 查询向量
            index_vectors: 索引向量数组
            
        Returns:
            相似度分数数组
        """
        # 计算余弦相似度
        if self.config.normalize_vectors:
            # 如果向量已经归一化，直接计算点积
            return index_vectors.dot(query_vector.T).ravel()
        else:
            # 否则需要计算余弦相似度
            norms = np.sqrt(np.sum(index_vectors.power(2), axis=1))
            query_norm = np.sqrt(np.sum(query_vector.power(2)))
            return index_vectors.dot(query_vector.T).ravel() / (norms * query_norm)
    
    async def index(
        self,
        chunks: List[Chunk],
        embeddings: Optional[List[Any]] = None
    ) -> None:
        """索引文档分块
        
        Args:
            chunks: 文档分块列表
            embeddings: 未使用
        """
        # 提取文本内容
        texts = [chunk.text for chunk in chunks]
        
        # 生成 TF-IDF 矩阵
        self._vectors = self._vectorizer.fit_transform(texts)
        
        # 如果需要归一化
        if self.config.normalize_vectors:
            self._vectors = normalize(self._vectors, norm='l2', axis=1)
        
        # 存储文档块
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        self._chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """搜索相关内容
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        if self._vectors is None:
            raise RuntimeError("尚未建立索引")
            
        # 生成查询向量
        query_vector = self._vectorizer.transform([query])
        
        # 如果需要归一化
        if self.config.normalize_vectors:
            query_vector = normalize(query_vector, norm='l2', axis=1)
        
        # 计算相似度
        scores = self._compute_similarity(query_vector, self._vectors)
        
        # 过滤低于阈值的结果
        mask = scores > self.config.threshold
        if not np.any(mask):
            return []
            
        # 获取top_k结果
        k = min(top_k or self.config.top_k, np.sum(mask))
        top_indices = np.argsort(scores[mask])[-k:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks[chunk_id]
            results.append(SearchResult(
                chunk=chunk,
                score=float(scores[idx])
            ))
            
        return results
    
    async def delete(
        self,
        chunk_ids: List[str]
    ) -> None:
        """删除索引
        
        Args:
            chunk_ids: 要删除的chunk_id列表
        """
        if self._vectors is None:
            return
            
        # 找出要删除的索引
        indices_to_delete = []
        for i, chunk_id in enumerate(self._chunk_ids):
            if chunk_id in chunk_ids:
                indices_to_delete.append(i)
                self._chunks.pop(chunk_id, None)
                
        # 更新向量数组和chunk_id列表
        if indices_to_delete:
            mask = np.ones(len(self._chunk_ids), dtype=bool)
            mask[indices_to_delete] = False
            
            # 更新 TF-IDF 矩阵
            self._vectors = self._vectors[mask]
            self._chunk_ids = [
                chunk_id for i, chunk_id in enumerate(self._chunk_ids)
                if i not in indices_to_delete
            ] 