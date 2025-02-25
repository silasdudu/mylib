"""
必应搜索引擎实现
"""
from typing import List, Dict, Any, Union, Optional
import aiohttp
import os
from pydantic import Field

from base.rag.search import SearchEngine, SearchConfig, SearchQuery, SearchResult


class BingSearchConfig(SearchConfig):
    """必应搜索引擎配置"""
    api_key: str = Field(default="", description="必应搜索API密钥")
    endpoint: str = Field(default="https://api.bing.microsoft.com/v7.0/search", description="必应搜索API端点")
    market: str = Field(default="zh-CN", description="搜索市场")
    safe_search: str = Field(default="Moderate", description="安全搜索级别")
    count: int = Field(default=10, description="返回结果数量")
    

class BingSearch(SearchEngine):
    """必应搜索引擎"""
    
    def __init__(self, config: Optional[BingSearchConfig] = None):
        """初始化必应搜索引擎
        
        Args:
            config: 搜索引擎配置，如果为None则使用默认配置或环境变量
        """
        if config is None:
            config = BingSearchConfig(
                api_key=os.environ.get("BING_SEARCH_API_KEY", ""),
                endpoint=os.environ.get("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
            )
        self.config = config
        
    async def search(
        self,
        query: Union[str, SearchQuery],
        **kwargs
    ) -> List[SearchResult]:
        """执行必应搜索
        
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
        params = {
            "q": query_text,
            "count": self.config.count,
            "mkt": self.config.market,
            "safeSearch": self.config.safe_search
        }
        
        # 添加额外参数
        params.update(kwargs)
        
        # 发送请求
        headers = {"Ocp-Apim-Subscription-Key": self.config.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.endpoint, headers=headers, params=params) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    
                    # 解析结果
                    results = []
                    if "webPages" in data and "value" in data["webPages"]:
                        for item in data["webPages"]["value"]:
                            result = SearchResult(
                                content=item.get("snippet", ""),
                                metadata={
                                    "title": item.get("name", ""),
                                    "url": item.get("url", ""),
                                    "date": item.get("dateLastCrawled", ""),
                                    "source": "bing"
                                },
                                score=1.0  # 必应API不返回分数，使用默认值
                            )
                            results.append(result)
                    
                    return results
        except Exception as e:
            print(f"必应搜索出错: {str(e)}")
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