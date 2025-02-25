"""
Tavily搜索引擎实现
"""
from typing import List, Dict, Any, Union, Optional
import aiohttp
import os
from pydantic import Field

from base.rag.search import SearchEngine, SearchConfig, SearchQuery, SearchResult


class TavilySearchConfig(SearchConfig):
    """Tavily搜索引擎配置"""
    api_key: str = Field(default="", description="Tavily搜索API密钥")
    endpoint: str = Field(default="https://api.tavily.com/search", description="Tavily搜索API端点")
    search_depth: str = Field(default="basic", description="搜索深度，可选值：basic, advanced")
    include_domains: List[str] = Field(default_factory=list, description="包含的域名列表")
    exclude_domains: List[str] = Field(default_factory=list, description="排除的域名列表")
    max_results: int = Field(default=10, description="返回结果数量")
    

class TavilySearch(SearchEngine):
    """Tavily搜索引擎"""
    
    def __init__(self, config: Optional[TavilySearchConfig] = None):
        """初始化Tavily搜索引擎
        
        Args:
            config: 搜索引擎配置，如果为None则使用默认配置或环境变量
        """
        if config is None:
            config = TavilySearchConfig(
                api_key=os.environ.get("TAVILY_API_KEY", ""),
                endpoint=os.environ.get("TAVILY_ENDPOINT", "https://api.tavily.com/search")
            )
        self.config = config
        
    async def search(
        self,
        query: Union[str, SearchQuery],
        **kwargs
    ) -> List[SearchResult]:
        """执行Tavily搜索
        
        Args:
            query: 查询文本或查询对象
            **kwargs: 额外的搜索参数
            
        Returns:
            搜索结果列表
        """
        # 处理查询
        if isinstance(query, str):
            query_text = query
        else:
            query_text = query.text
            
        # 构建请求参数
        payload = {
            "query": query_text,
            "search_depth": self.config.search_depth,
            "max_results": self.config.max_results,
            "api_key": self.config.api_key
        }
        
        # 添加可选域名过滤
        if self.config.include_domains:
            payload["include_domains"] = self.config.include_domains
        if self.config.exclude_domains:
            payload["exclude_domains"] = self.config.exclude_domains
            
        # 添加额外参数
        payload.update(kwargs)
        
        # 发送请求
        headers = {"Content-Type": "application/json"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.endpoint, headers=headers, json=payload) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    
                    # 解析结果
                    results = []
                    if "results" in data:
                        for item in data["results"]:
                            result = SearchResult(
                                content=item.get("content", ""),
                                metadata={
                                    "title": item.get("title", ""),
                                    "url": item.get("url", ""),
                                    "score": item.get("score", 1.0),
                                    "source": "tavily"
                                },
                                score=item.get("score", 1.0)
                            )
                            results.append(result)
                    
                    return results
        except Exception as e:
            print(f"Tavily搜索出错: {str(e)}")
            return []
    
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
        
    async def get_content(self, url: str) -> str:
        """获取网页内容
        
        Args:
            url: 网页URL
            
        Returns:
            网页内容
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    return ""
        except Exception as e:
            print(f"获取网页内容出错: {str(e)}")
            return "" 