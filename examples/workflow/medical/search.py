"""
医疗搜索引擎模块，提供医疗领域特定的搜索功能
"""
from typing import List, Optional, Dict, Any, Set

from base.search.engine import SearchEngine, SearchResult
from module.search import (
    BasicSearchWrapper, 
    BingSearchWrapper,
    GoogleSearchWrapper,
    BraveSearchWrapper,
    TavilySearchWrapper
)
from module.models.llm.custom import CustomLLM
from module.models.rerank.custom import CustomReranker

from .filter import LLMBasedMedicalFilter


class MedicalSearchConfig:
    """医疗搜索引擎配置"""
    
    def __init__(
        self,
        primary_engine: str = "basic",
        medical_keywords: Optional[List[str]] = None,
        min_keyword_match: int = 1,
        boost_medical_results: bool = True,
        use_llm_filter: bool = False,
        use_reranker: bool = False,
        use_hybrid_filter: bool = False,
        llm_threshold: float = 0.7,
        rerank_weight: float = 0.7,
        llm_weight: float = 0.3,
        max_results: int = 10
    ):
        """初始化医疗搜索引擎配置
        
        Args:
            primary_engine: 主要使用的搜索引擎，可选值：basic, bing, google, brave, tavily
            medical_keywords: 医疗相关关键词列表，如果为None则使用默认关键词
            min_keyword_match: 最小关键词匹配数量，默认为1（至少匹配一个关键词）
            boost_medical_results: 是否提升医疗相关结果的排序，默认为True
            use_llm_filter: 是否使用大模型过滤，默认为False
            use_reranker: 是否使用重排序模型过滤，默认为False
            use_hybrid_filter: 是否使用混合过滤（同时使用LLM和重排序），默认为False
            llm_threshold: LLM过滤的医疗相关性阈值，范围0-1
            rerank_weight: 重排序模型分数权重，范围0-1
            llm_weight: LLM分数权重，范围0-1
            max_results: 最大返回结果数
        """
        self.primary_engine = primary_engine
        
        # 默认医疗关键词
        self._default_keywords = [
            "医疗", "健康", "疾病", "症状", "治疗", "药物", "医院", 
            "医生", "患者", "诊断", "手术", "康复", "护理", "预防",
            "病毒", "细菌", "感染", "免疫", "疫苗", "医学", "临床",
            "心脏", "肺", "肝", "肾", "脑", "血液", "骨骼", "肌肉",
            "神经", "内科", "外科", "儿科", "妇产科", "精神科", "皮肤科"
        ]
        
        self.medical_keywords = set(medical_keywords if medical_keywords is not None else self._default_keywords)
        self.min_keyword_match = min_keyword_match
        self.boost_medical_results = boost_medical_results
        
        # 大模型和重排序相关配置
        self.use_llm_filter = use_llm_filter
        self.use_reranker = use_reranker
        self.use_hybrid_filter = use_hybrid_filter
        self.llm_threshold = llm_threshold
        self.rerank_weight = rerank_weight
        self.llm_weight = llm_weight
        self.max_results = max_results


class MedicalSearchEngine(SearchEngine):
    """医疗搜索引擎，组合使用多个搜索引擎并进行医疗领域特定的处理"""
    
    def __init__(
        self, 
        config: Optional[MedicalSearchConfig] = None,
        llm: Optional[CustomLLM] = None,
        reranker: Optional[CustomReranker] = None
    ):
        """初始化医疗搜索引擎
        
        Args:
            config: 医疗搜索引擎配置，如果为None则使用默认配置
            llm: 大语言模型，如果为None则创建默认实例
            reranker: 重排序模型，如果为None则创建默认实例
        """
        self.config = config or MedicalSearchConfig()
        
        self.engines = {
            "basic": BasicSearchWrapper(),
            "bing": BingSearchWrapper(),
            "google": GoogleSearchWrapper(),
            "brave": BraveSearchWrapper(),
            "tavily": TavilySearchWrapper()
        }
        
        if self.config.primary_engine not in self.engines:
            raise ValueError(f"不支持的搜索引擎: {self.config.primary_engine}，可用选项: {list(self.engines.keys())}")
            
        self.primary_engine = self.config.primary_engine
        
        # 初始化医疗内容过滤器
        self.medical_filter = LLMBasedMedicalFilter(llm, reranker)
        
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """搜索医疗相关信息
        
        Args:
            query: 搜索查询
            **kwargs: 额外参数，包括:
                - engine: 指定使用的搜索引擎
                - filter_medical: 是否过滤非医疗相关结果
                - filter_method: 过滤方法，可选值：keyword, llm, rerank, hybrid
                - medical_keywords: 临时指定医疗关键词列表
                - min_keyword_match: 临时指定最小关键词匹配数量
                - llm_threshold: 临时指定LLM过滤阈值
                
        Returns:
            搜索结果列表
        """
        # 确定使用的搜索引擎
        engine_name = kwargs.pop("engine", self.primary_engine)
        if engine_name not in self.engines:
            engine_name = self.primary_engine
            
        engine = self.engines[engine_name]
        
        # 执行搜索
        results = await engine.search(query, **kwargs)
        
        # 是否过滤非医疗相关结果
        filter_medical = kwargs.pop("filter_medical", False)
        if filter_medical:
            # 确定过滤方法
            filter_method = kwargs.pop("filter_method", "keyword")
            
            if filter_method == "keyword":
                # 使用关键词过滤
                medical_keywords = kwargs.pop("medical_keywords", None)
                min_keyword_match = kwargs.pop("min_keyword_match", self.config.min_keyword_match)
                
                results = self._filter_medical_results(
                    results, 
                    medical_keywords=medical_keywords,
                    min_keyword_match=min_keyword_match
                )
            elif filter_method == "llm" or self.config.use_llm_filter:
                # 使用大模型过滤
                llm_threshold = kwargs.pop("llm_threshold", self.config.llm_threshold)
                max_results = kwargs.pop("max_results", self.config.max_results)
                
                results = await self.medical_filter.filter_by_llm(
                    query,
                    results,
                    threshold=llm_threshold,
                    max_results=max_results
                )
            elif filter_method == "rerank" or self.config.use_reranker:
                # 使用重排序模型过滤
                medical_context = kwargs.pop("medical_context", "医疗健康相关信息")
                top_k = kwargs.pop("top_k", self.config.max_results)
                
                results = await self.medical_filter.filter_by_reranker(
                    query,
                    results,
                    medical_context=medical_context,
                    top_k=top_k
                )
            elif filter_method == "hybrid" or self.config.use_hybrid_filter:
                # 使用混合过滤
                llm_threshold = kwargs.pop("llm_threshold", self.config.llm_threshold)
                rerank_weight = kwargs.pop("rerank_weight", self.config.rerank_weight)
                llm_weight = kwargs.pop("llm_weight", self.config.llm_weight)
                max_results = kwargs.pop("max_results", self.config.max_results)
                
                results = await self.medical_filter.hybrid_filter(
                    query,
                    results,
                    llm_threshold=llm_threshold,
                    rerank_weight=rerank_weight,
                    llm_weight=llm_weight,
                    max_results=max_results
                )
            
        return results
    
    def _filter_medical_results(
        self, 
        results: List[SearchResult], 
        medical_keywords: Optional[List[str]] = None,
        min_keyword_match: int = 1
    ) -> List[SearchResult]:
        """过滤出医疗相关的结果（基于关键词）
        
        Args:
            results: 原始搜索结果
            medical_keywords: 临时指定的医疗关键词列表，如果为None则使用配置中的关键词
            min_keyword_match: 最小关键词匹配数量
            
        Returns:
            过滤后的医疗相关结果
        """
        # 使用指定的关键词或配置中的关键词
        keywords = set(medical_keywords) if medical_keywords is not None else self.config.medical_keywords
        
        filtered = []
        for result in results:
            # 检查标题和内容是否包含医疗关键词
            content = result.content.lower()
            title = result.metadata.get("title", "").lower()
            
            # 计算匹配的关键词数量
            matched_keywords = sum(1 for keyword in keywords if keyword in content or keyword in title)
            
            # 如果匹配数量达到阈值，则保留结果
            if matched_keywords >= min_keyword_match:
                # 如果启用了医疗结果提升，则根据匹配关键词数量调整分数
                if self.config.boost_medical_results:
                    # 计算关键词匹配比例作为提升因子
                    boost_factor = matched_keywords / len(keywords) * 0.5  # 最多提升50%
                    result.score = result.score * (1 + boost_factor)
                
                filtered.append(result)
                
        return filtered
    
    def add_medical_keywords(self, keywords: List[str]) -> None:
        """添加医疗关键词到配置
        
        Args:
            keywords: 要添加的关键词列表
        """
        self.config.medical_keywords.update(keywords)
        
    def remove_medical_keywords(self, keywords: List[str]) -> None:
        """从配置中移除医疗关键词
        
        Args:
            keywords: 要移除的关键词列表
        """
        self.config.medical_keywords.difference_update(keywords)
        
    def get_medical_keywords(self) -> List[str]:
        """获取当前配置的医疗关键词列表
        
        Returns:
            医疗关键词列表
        """
        return list(self.config.medical_keywords)
    
    async def get_content(self, url: str) -> str:
        """获取网页内容
        
        Args:
            url: 网页URL
            
        Returns:
            网页内容
        """
        # 使用主搜索引擎获取内容
        return await self.engines[self.primary_engine].get_content(url) 