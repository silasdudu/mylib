"""
基础搜索引擎实现
"""
from typing import List, Dict, Any, Union
from pydantic import Field
from base.rag.search import SearchEngine, SearchConfig, SearchQuery, SearchResult


class BasicSearchConfig(SearchConfig):
    """基础搜索引擎配置"""
    max_results: int = Field(default=10, description="最大返回结果数")
    min_score: float = Field(default=0.0, description="最小相关性分数阈值")
    use_reranking: bool = Field(default=False, description="是否启用重排序")
    dedup_results: bool = Field(default=True, description="是否去重结果")
    normalize_scores: bool = Field(default=True, description="是否对分数进行归一化")


class BasicSearchEngine(SearchEngine):
    """基础搜索引擎"""
    
    def __init__(self, config: BasicSearchConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    async def search(
        self,
        query: Union[str, SearchQuery],
        **kwargs
    ) -> List[SearchResult]:
        """执行基础搜索
        
        Args:
            query: 查询文本或查询对象
            **kwargs: 额外的搜索参数
            
        Returns:
            搜索结果列表
        """
        # 处理查询
        if isinstance(query, str):
            query = SearchQuery(text=query)
            
        # 获取检索结果
        results = await self.retriever.search(
            query.text,
            top_k=self.config.max_results
        )
        
        # 应用过滤器
        if query.filters:
            results = await self.filter_results(results, query.filters)
            
        # 重新排序
        if self.config.use_reranking:
            results = await self.rerank_results(query, results)
            
        # 应用分数阈值
        results = [
            r for r in results
            if r.score >= self.config.min_score
        ]
        
        return results[:self.config.max_results]
    
    async def filter_results(
        self,
        results: List[SearchResult],
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """基于元数据过滤结果
        
        Args:
            results: 搜索结果列表
            filters: 过滤条件字典
            
        Returns:
            过滤后的结果列表
        """
        filtered = []
        
        for result in results:
            match = True
            for key, value in filters.items():
                if key in result.metadata:
                    if isinstance(value, (list, tuple, set)):
                        if result.metadata[key] not in value:
                            match = False
                            break
                    elif result.metadata[key] != value:
                        match = False
                        break
                else:
                    match = False
                    break
                    
            if match:
                filtered.append(result)
                
        return filtered 