"""
RAG生成器实现
"""
from typing import AsyncIterator, Dict, List, Optional
from pydantic import Field

from base.model.interface import LargeModel, ModelConfig
from base.rag.generator import Generator, GeneratorConfig, GeneratorInput, GeneratorOutput
from base.rag.retriever import SearchResult
from ..prompts.rag import DEFAULT_RAG_TEMPLATE


class RAGGeneratorConfig(GeneratorConfig):
    """RAG生成器配置"""
    max_input_tokens: int = Field(default=4096, description="最大输入token数")
    max_output_tokens: int = Field(default=1024, description="最大输出token数")
    temperature: float = Field(default=0.7, description="生成温度")
    top_p: float = Field(default=0.9, description="核采样概率")
    prompt_template: str = Field(
        default=DEFAULT_RAG_TEMPLATE["prompt_template"],
        description="提示词模板"
    )
    context_format: str = Field(
        default=DEFAULT_RAG_TEMPLATE["context_format"],
        description="上下文格式化模板"
    )
    context_separator: str = Field(
        default=DEFAULT_RAG_TEMPLATE["context_separator"],
        description="上下文分隔符"
    )


class RAGGenerator(Generator):
    """RAG生成器实现"""
    
    def __init__(
        self,
        config: RAGGeneratorConfig,
        model: LargeModel
    ):
        """初始化RAG生成器
        
        Args:
            config: 生成器配置
            model: 大语言模型
        """
        super().__init__(config, model)
        self.config: RAGGeneratorConfig = config
    
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
                "finish_reason": response.finish_reason,
                "prompt": prompt  # 添加提示词到元数据
            }
        )
    
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
        """格式化提示词
        
        Args:
            query: 查询文本
            context: 检索结果列表
            
        Returns:
            格式化后的提示词
        """
        # 格式化上下文
        context_texts = []
        for i, result in enumerate(context):
            text = self.config.context_format.format(
                index=i+1,
                text=result.chunk.text,
                score=result.score,
                metadata=result.metadata
            )
            context_texts.append(text)
            
        # 使用分隔符连接上下文
        context_text = self.config.context_separator.join(context_texts)
        
        # 使用模板格式化最终提示词
        prompt = self.config.prompt_template.format(
            context=context_text,
            query=query
        )
        
        return prompt 