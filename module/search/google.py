"""
谷歌搜索引擎实现
"""
from typing import List, Dict, Any, Union, Optional
import aiohttp
import os
from pydantic import Field

from base.rag.search import SearchEngine, SearchConfig, SearchQuery, SearchResult


class GoogleSearchConfig(SearchConfig):
    """谷歌搜索引擎配置"""
    api_key: str = Field(default="", description="谷歌搜索API密钥")
    cx: str = Field(default="", description="谷歌自定义搜索引擎ID")
    endpoint: str = Field(default="https://www.googleapis.com/customsearch/v1", description="谷歌搜索API端点")
    language: str = Field(default="zh-CN", description="搜索语言")
    safe: str = Field(default="medium", description="安全搜索级别")
    num: int = Field(default=10, description="返回结果数量")
    

class GoogleSearch(SearchEngine):
    """谷歌搜索引擎"""
    
    def __init__(self, config: Optional[GoogleSearchConfig] = None):
        """初始化谷歌搜索引擎
        
        Args:
            config: 搜索引擎配置，如果为None则使用默认配置或环境变量
        """
        if config is None:
            config = GoogleSearchConfig(
                api_key=os.environ.get("GOOGLE_SEARCH_API_KEY", ""),
                cx=os.environ.get("GOOGLE_SEARCH_CX", ""),
                endpoint=os.environ.get("GOOGLE_SEARCH_ENDPOINT", "https://www.googleapis.com/customsearch/v1")
            )
        self.config = config
        
    async def search(
        self,
        query: Union[str, SearchQuery],
        **kwargs
    ) -> List[SearchResult]:
        """执行谷歌搜索
        
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
            "key": self.config.api_key,
            "cx": self.config.cx,
            "num": self.config.num,
            "hl": self.config.language,
            "safe": self.config.safe
        }
        
        # 添加额外参数
        params.update(kwargs)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.endpoint, params=params) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    
                    # 解析结果
                    results = []
                    if "items" in data:
                        for item in data["items"]:
                            result = SearchResult(
                                content=item.get("snippet", ""),
                                metadata={
                                    "title": item.get("title", ""),
                                    "url": item.get("link", ""),
                                    "date": item.get("pagemap", {}).get("metatags", [{}])[0].get("date", ""),
                                    "source": "google"
                                },
                                score=1.0  # 谷歌API不返回分数，使用默认值
                            )
                            results.append(result)
                    
                    return results
        except Exception as e:
            print(f"谷歌搜索出错: {str(e)}")
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