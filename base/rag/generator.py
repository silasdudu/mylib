"""
生成器模块，基于检索结果生成回答
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel

from ..model.interface import LargeModel, ModelConfig, ModelResponse
from .retriever import SearchResult


class GeneratorConfig(BaseModel):
    """生成器配置"""
    max_input_tokens: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    extra_params: Dict[str, Any] = {}


@dataclass
class GeneratorInput:
    """生成器输入"""
    query: str
    context: List[SearchResult]
    metadata: Dict[str, Any] = None


@dataclass
class GeneratorOutput:
    """生成器输出"""
    response: str
    context: List[SearchResult]
    metadata: Dict[str, Any] = None


class Generator(ABC):
    """生成器抽象基类"""
    
    def __init__(
        self,
        config: GeneratorConfig,
        model: LargeModel
    ):
        self.config = config
        self.model = model
    
    @abstractmethod
    async def generate(
        self,
        input_data: GeneratorInput
    ) -> GeneratorOutput:
        """生成回答"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        input_data: GeneratorInput
    ) -> AsyncIterator[str]:
        """流式生成回答"""
        pass
    
    @abstractmethod
    def format_prompt(
        self,
        query: str,
        context: List[SearchResult]
    ) -> str:
        """格式化提示词"""
        pass


class RAGGenerator(Generator):
    """RAG生成器"""
    
    async def generate(
        self,
        input_data: GeneratorInput
    ) -> GeneratorOutput:
        """生成回答"""
        # 格式化提示词
        prompt = self.format_prompt(
            input_data.query,
            input_data.context
        )
        
        # 调用模型生成
        model_config = ModelConfig(
            model_name=self.model.__class__.__name__,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens
        )
        
        response = await self.model.generate(prompt, model_config)
        
        return GeneratorOutput(
            response=response.text,
            context=input_data.context,
            metadata={
                "tokens_used": response.tokens_used,
                "finish_reason": response.finish_reason
            }
        )
    
    async def generate_stream(
        self,
        input_data: GeneratorInput
    ) -> AsyncIterator[str]:
        """流式生成回答"""
        # 格式化提示词
        prompt = self.format_prompt(
            input_data.query,
            input_data.context
        )
        
        # 调用模型流式生成
        model_config = ModelConfig(
            model_name=self.model.__class__.__name__,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens
        )
        
        async for token in self.model.generate_stream(prompt, model_config):
            yield token
    
    def format_prompt(
        self,
        query: str,
        context: List[SearchResult]
    ) -> str:
        """格式化RAG提示词"""
        # 将上下文组织成文本
        context_text = "\n\n".join([
            f"[{i+1}] {result.chunk.text}"
            for i, result in enumerate(context)
        ])
        
        # 构建提示词模板
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文：
{context_text}

问题：{query}

回答："""
        
        return prompt


class GeneratorRegistry:
    """生成器注册表"""
    
    def __init__(self):
        self._generators: Dict[str, Generator] = {}
        
    def register(
        self,
        name: str,
        generator: Generator
    ) -> None:
        """注册生成器"""
        self._generators[name] = generator
        
    def get_generator(
        self,
        name: str
    ) -> Optional[Generator]:
        """获取生成器"""
        return self._generators.get(name)
        
    def list_generators(self) -> List[str]:
        """列出所有生成器"""
        return list(self._generators.keys()) 