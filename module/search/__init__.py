"""
搜索模块，提供各种搜索引擎实现
"""

# 搜索引擎实现
from .basic import BasicSearchEngine, BasicSearchConfig
from .bing import BingSearch, BingSearchConfig
from .google import GoogleSearch, GoogleSearchConfig
from .brave import BraveSearch, BraveSearchConfig
from .tavily import TavilySearch, TavilySearchConfig

# 搜索引擎包装器
from .wrappers import (
    BasicSearchWrapper,
    BingSearchWrapper,
    GoogleSearchWrapper,
    BraveSearchWrapper,
    TavilySearchWrapper
)

__all__ = [
    # 搜索引擎
    "BasicSearchEngine",
    "BasicSearchConfig",
    "BingSearch",
    "BingSearchConfig",
    "GoogleSearch",
    "GoogleSearchConfig",
    "BraveSearch",
    "BraveSearchConfig",
    "TavilySearch",
    "TavilySearchConfig",
    
    # 搜索引擎包装器
    "BasicSearchWrapper",
    "BingSearchWrapper",
    "GoogleSearchWrapper",
    "BraveSearchWrapper",
    "TavilySearchWrapper"
] 