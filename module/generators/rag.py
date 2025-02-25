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
    history_format: str = Field(
        default="{role}: {content}",
        description="对话历史格式化模板"
    )
    history_separator: str = Field(
        default="\n",
        description="对话历史分隔符"
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
            input_data.context,
            input_data.conversation_history
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
            input_data.context,
            input_data.conversation_history
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
        context: List[SearchResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """格式化提示词
        
        Args:
            query: 查询文本
            context: 检索结果列表
            conversation_history: 对话历史
            
        Returns:
            格式化后的提示词
        """
        # 格式化上下文
        context_texts = []
        for i, result in enumerate(context):
            try:
                # 修复：确保只传入格式化字符串需要的参数
                format_args = {
                    "index": i+1,
                    "text": result.chunk.text,
                    "score": result.score
                }
                
                # 不再尝试直接传入metadata
                text = self.config.context_format.format(**format_args)
                context_texts.append(text)
            except Exception as e:
                # 如果格式化失败，使用简单格式
                text = f"[{i+1}] {result.chunk.text}"
                context_texts.append(text)
                print(f"上下文格式化错误: {str(e)}")
            
        # 使用分隔符连接上下文
        context_text = self.config.context_separator.join(context_texts)
        
        # 格式化对话历史（如果有）
        history_text = ""
        if conversation_history:
            history_items = []
            
            # 检查conversation_history是否为字符串
            if isinstance(conversation_history, str):
                # 如果是字符串，直接使用
                history_text = conversation_history
            else:
                # 如果是列表，处理每个消息
                for msg in conversation_history:
                    try:
                        # 确保msg是字典
                        if isinstance(msg, dict):
                            history_items.append(
                                self.config.history_format.format(**msg)
                            )
                        else:
                            # 如果不是字典，转换为字符串
                            history_items.append(str(msg))
                    except Exception as e:
                        # 如果格式化失败，尝试使用简单格式
                        try:
                            if isinstance(msg, dict):
                                role = msg.get("role", "unknown")
                                content = msg.get("content", "")
                                history_items.append(f"{role}: {content}")
                            else:
                                history_items.append(str(msg))
                        except Exception:
                            # 如果还是失败，添加一个占位符
                            history_items.append("[格式化错误的消息]")
                        print(f"历史格式化错误: {str(e)}")
                
                history_text = self.config.history_separator.join(history_items)
            
            # 如果查询中已经包含了历史，就不再添加
            if history_text and "对话历史" not in query:
                query = f"对话历史:\n{history_text}\n\n当前问题: {query}"
        
        # 使用模板格式化最终提示词
        try:
            prompt = self.config.prompt_template.format(
                context=context_text,
                query=query
            )
        except Exception as e:
            # 如果格式化失败，使用简单格式
            prompt = f"参考信息:\n{context_text}\n\n问题: {query}\n\n回答:"
            print(f"提示词格式化错误: {str(e)}")
        
        return prompt 