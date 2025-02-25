"""
生成器模块，基于检索结果生成回答
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel

from ..model.interface import LargeModel, ModelConfig, ModelResponse
from .retriever import SearchResult


class GeneratorConfig(BaseModel):
    """生成器配置基类"""
    extra_params: Dict[str, Any] = {}  # 额外的特定实现参数


@dataclass
class GeneratorInput:
    """生成器输入"""
    query: str
    context: List[SearchResult]
    conversation_history: Optional[List[Dict[str, Any]]] = None  # 对话历史
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorOutput:
    """生成器输出"""
    response: str
    context: List[SearchResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Generator(ABC):
    """生成器抽象基类"""
    
    def __init__(
        self,
        config: GeneratorConfig,
        model: LargeModel
    ):
        """初始化生成器
        
        Args:
            config: 生成器配置
            model: 大语言模型
        """
        self.config = config
        self.model = model
    
    @abstractmethod
    async def generate(
        self,
        input_data: GeneratorInput
    ) -> GeneratorOutput:
        """生成回答
        
        Args:
            input_data: 生成器输入
            
        Returns:
            生成器输出
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        input_data: GeneratorInput
    ) -> AsyncIterator[str]:
        """流式生成回答
        
        Args:
            input_data: 生成器输入
            
        Yields:
            生成的文本片段
        """
        pass
    
    @abstractmethod
    def format_prompt(
        self,
        query: str,
        context: List[SearchResult],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """格式化提示词
        
        Args:
            query: 查询文本
            context: 检索结果列表
            conversation_history: 可选的对话历史
            
        Returns:
            格式化后的提示词
        """
        pass


class GeneratorRegistry:
    """生成器注册表"""
    
    def __init__(self):
        self._generators: Dict[str, Generator] = {}
        
    def register(
        self,
        name: str,
        generator: Generator
    ) -> None:
        """注册生成器
        
        Args:
            name: 生成器名称
            generator: 生成器实例
        """
        self._generators[name] = generator
        
    def get_generator(
        self,
        name: str
    ) -> Optional[Generator]:
        """获取生成器
        
        Args:
            name: 生成器名称
            
        Returns:
            生成器实例,如果不存在则返回None
        """
        return self._generators.get(name)
        
    def list_generators(self) -> List[str]:
        """列出所有生成器
        
        Returns:
            生成器名称列表
        """
        return list(self._generators.keys()) 