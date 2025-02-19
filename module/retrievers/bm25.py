"""
BM25 检索器实现，基于 BM25 算法的文本检索
"""
from typing import List, Optional, Dict, Any
import numpy as np
from collections import Counter
import jieba
from base.rag.retriever import Retriever, RetrieverConfig, SearchResult
from base.rag.chunking import Chunk


class BM25RetrieverConfig(RetrieverConfig):
    """BM25 检索器配置"""
    k1: float = 1.5  # 词频饱和参数
    b: float = 0.75  # 文档长度归一化参数
    use_stopwords: bool = True  # 是否使用停用词
    stopwords_file: Optional[str] = None  # 停用词文件路径
    min_df: int = 1  # 最小文档频率
    max_df_ratio: float = 0.95  # 最大文档频率比例


class BM25Retriever(Retriever):
    """基于 BM25 算法的检索器实现"""
    
    def __init__(self, config: BM25RetrieverConfig, **kwargs):
        """初始化 BM25 检索器
        
        Args:
            config: 检索器配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: BM25RetrieverConfig = config
        
        # 加载停用词
        self._stopwords = set()
        if self.config.use_stopwords:
            if self.config.stopwords_file:
                with open(self.config.stopwords_file, 'r', encoding='utf-8') as f:
                    self._stopwords = set(line.strip() for line in f)
        
        # 初始化存储
        self._chunks = {}  # 文档块字典
        self._chunk_ids = []  # chunk_id列表
        self._vocab = {}  # 词汇表：word -> id
        self._idf = None  # 逆文档频率
        self._doc_freqs = None  # 文档词频矩阵
        self._doc_lengths = None  # 文档长度
        self._avg_doc_length = 0  # 平均文档长度
    
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
    
    def _build_vocab(self, chunks: List[Chunk]) -> None:
        """构建词汇表
        
        Args:
            chunks: 文档块列表
        """
        # 统计词频
        word_doc_freq = Counter()  # 词在多少个文档中出现
        for chunk in chunks:
            words = set(self._tokenize(chunk.text))  # 使用集合去重
            word_doc_freq.update(words)
            
        # 过滤低频和高频词
        n_docs = len(chunks)
        max_df = int(n_docs * self.config.max_df_ratio)
        
        # 构建词汇表
        self._vocab = {}
        for word, freq in word_doc_freq.items():
            if self.config.min_df <= freq <= max_df:
                self._vocab[word] = len(self._vocab)
    
    def _compute_idf(self, chunks: List[Chunk]) -> None:
        """计算逆文档频率
        
        Args:
            chunks: 文档块列表
        """
        n_docs = len(chunks)
        doc_freq = np.zeros(len(self._vocab))
        
        for chunk in chunks:
            words = set(self._tokenize(chunk.text))  # 使用集合去重
            for word in words:
                if word in self._vocab:
                    doc_freq[self._vocab[word]] += 1
                    
        # 计算 IDF
        self._idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
    
    def _compute_doc_freqs(self, chunks: List[Chunk]) -> None:
        """计算文档词频矩阵
        
        Args:
            chunks: 文档块列表
        """
        n_docs = len(chunks)
        n_vocab = len(self._vocab)
        
        # 初始化文档词频矩阵
        self._doc_freqs = np.zeros((n_docs, n_vocab))
        self._doc_lengths = np.zeros(n_docs)
        
        # 计算词频和文档长度
        for i, chunk in enumerate(chunks):
            words = self._tokenize(chunk.text)
            self._doc_lengths[i] = len(words)
            
            # 统计词频
            word_freq = Counter(words)
            for word, freq in word_freq.items():
                if word in self._vocab:
                    self._doc_freqs[i, self._vocab[word]] = freq
                    
        # 计算平均文档长度
        self._avg_doc_length = np.mean(self._doc_lengths)
    
    def _compute_bm25_scores(self, query_freqs: np.ndarray) -> np.ndarray:
        """计算 BM25 分数
        
        Args:
            query_freqs: 查询词频向量
            
        Returns:
            文档得分数组
        """
        # 计算 BM25 分数
        k1 = self.config.k1
        b = self.config.b
        
        # 文档长度归一化项
        len_norm = (1 - b + b * self._doc_lengths / self._avg_doc_length).reshape(-1, 1)
        
        # 计算分子
        numerator = self._doc_freqs * (k1 + 1)
        
        # 计算分母
        denominator = self._doc_freqs + k1 * len_norm
        
        # 计算最终得分
        scores = np.sum(self._idf * numerator / denominator * query_freqs, axis=1)
        
        return scores
    
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
        # 存储文档块
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        self._chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        
        # 构建词汇表
        self._build_vocab(chunks)
        
        # 计算 IDF
        self._compute_idf(chunks)
        
        # 计算文档词频
        self._compute_doc_freqs(chunks)
    
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
        if not self._vocab:
            raise RuntimeError("尚未建立索引")
            
        # 对查询进行分词
        query_words = self._tokenize(query)
        
        # 计算查询词频向量
        query_freqs = np.zeros(len(self._vocab))
        word_freq = Counter(query_words)
        for word, freq in word_freq.items():
            if word in self._vocab:
                query_freqs[self._vocab[word]] = freq
                
        # 计算 BM25 分数
        scores = self._compute_bm25_scores(query_freqs)
        
        # 获取top_k结果
        k = top_k or self.config.top_k
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有匹配的结果
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
        if not self._vocab:
            return
            
        # 找出要删除的索引
        indices_to_delete = []
        for i, chunk_id in enumerate(self._chunk_ids):
            if chunk_id in chunk_ids:
                indices_to_delete.append(i)
                self._chunks.pop(chunk_id, None)
                
        # 更新数据结构
        if indices_to_delete:
            mask = np.ones(len(self._chunk_ids), dtype=bool)
            mask[indices_to_delete] = False
            
            self._doc_freqs = self._doc_freqs[mask]
            self._doc_lengths = self._doc_lengths[mask]
            self._chunk_ids = [
                chunk_id for i, chunk_id in enumerate(self._chunk_ids)
                if i not in indices_to_delete
            ]
            
            # 重新计算平均文档长度
            self._avg_doc_length = np.mean(self._doc_lengths) 