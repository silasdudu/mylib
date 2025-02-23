"""
自定义大语言模型实现，支持本地模型和API调用两种方式
"""
from typing import List, Optional, Union, AsyncIterator, Dict, Any
import os
import json
import aiohttp
import torch
from pathlib import Path
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from base.model.interface import LargeModel, ModelConfig, ModelResponse


def load_default_config() -> 'CustomLLMConfig':
    """加载默认配置
    
    从项目根目录的 .env 文件中加载配置参数。
    必需的环境变量：
    - CUSTOM_LLM_API_KEY: API密钥
    - CUSTOM_LLM_BASE_URL: API基础URL
    - CUSTOM_LLM_MODEL_NAME: 模型名称
    
    Returns:
        默认的模型配置
        
    Raises:
        ValueError: 当必需的环境变量未设置时抛出
    """
    # 加载 .env 文件
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    # 获取必需的环境变量
    api_key = os.getenv('CUSTOM_LLM_API_KEY')
    base_url = os.getenv('CUSTOM_LLM_BASE_URL')
    model_name = os.getenv('CUSTOM_LLM_MODEL_NAME')
    
    if not all([api_key, base_url, model_name]):
        raise ValueError(
            "必需的环境变量未设置。请在 .env 文件中设置以下变量：\n"
            "- CUSTOM_LLM_API_KEY\n"
            "- CUSTOM_LLM_BASE_URL\n"
            "- CUSTOM_LLM_MODEL_NAME"
        )
    
    return CustomLLMConfig(
        model_name=model_name,
        api_url=base_url,
        api_key=api_key
    )


class CustomLLMConfig(ModelConfig):
    """自定义大语言模型配置"""
    api_url: Optional[str] = None  # API URL，如果不为空则使用 API 模式
    api_key: Optional[str] = None  # API 密钥
    model_path: Optional[str] = None  # 本地模型路径，如果不为空则优先使用本地模型
    max_length: int = 2048  # 最大生成长度
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 运行设备


class CustomLLM(LargeModel):
    """自定义大语言模型实现，支持本地模型和API调用"""
    
    def __init__(self, config: Optional[CustomLLMConfig] = None):
        """初始化模型
        
        Args:
            config: 模型配置，支持本地模型和API两种方式：
                   - 本地模型：需要设置 model_path 或 model_name
                   - API调用：需要设置 api_url、api_key 和 model_name
                   如果不提供，将使用 .env 中的默认配置
        """
        if config is None:
            config = load_default_config()
            
        if not config.model_name:
            raise ValueError("必须提供 model_name，用于指定模型名称")
            
        self.config = config
        
        # 根据配置决定使用哪种模式
        self.use_api = bool(config.api_url and config.api_key)
        
        if self.use_api:
            # API 模式初始化
            self.session = None  # 延迟初始化 aiohttp session
            self.headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }
        else:
            # 本地模型模式初始化
            model_path = config.model_path or config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto" if config.device == "cuda" else None
            )
            
            # 设置为评估模式
            self.model.eval()
    
    async def _ensure_session(self):
        """确保 session 已创建"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def _cleanup(self):
        """清理资源"""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    async def _generate_api(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None
    ) -> ModelResponse:
        """使用 API 生成文本
        
        Args:
            prompt: 输入提示词
            config: 可选的模型配置
            
        Returns:
            生成的文本响应
        """
        try:
            await self._ensure_session()
            
            async with self.session.post(
                self.config.api_url,
                headers=self.headers,
                json={
                    "model": self.config.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": config.temperature if config else self.config.temperature,
                    # "max_tokens": config.max_tokens if config else self.config.max_tokens,
                    # "stop": config.stop_sequences if config else self.config.stop_sequences
                }
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"API 调用失败: {response.status} {await response.text()}")
                    
                result = await response.json()
                return ModelResponse(
                    text=result["choices"][0]["message"]["content"],
                    tokens_used=result["usage"]["total_tokens"],
                    finish_reason=result["choices"][0]["finish_reason"]
                )
                
        except Exception as e:
            raise RuntimeError(f"API 调用出错: {str(e)}")
    
    async def _generate_local(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None
    ) -> ModelResponse:
        """使用本地模型生成文本
        
        Args:
            prompt: 输入提示词
            config: 可选的模型配置
            
        Returns:
            生成的文本响应
        """
        # 对文本进行分词
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        # 移动到正确的设备
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=config.max_tokens if config else self.config.max_tokens,
                temperature=config.temperature if config else self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 计算使用的token数量
        tokens_used = len(outputs[0])
        
        return ModelResponse(
            text=generated_text,
            tokens_used=tokens_used,
            finish_reason="length" if tokens_used >= self.config.max_length else "stop"
        )
    
    async def generate(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None
    ) -> ModelResponse:
        """生成文本
        
        Args:
            prompt: 输入提示词
            config: 可选的模型配置
            
        Returns:
            生成的文本响应
        """
        if self.use_api:
            return await self._generate_api(prompt, config)
        else:
            return await self._generate_local(prompt, config)
    
    async def close(self):
        """关闭资源"""
        await self._cleanup()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None
    ) -> AsyncIterator[str]:
        """流式生成文本"""
        if self.use_api:
            try:
                await self._ensure_session()
                
                async with self.session.post(
                    f"{self.config.api_url}",
                    headers=self.headers,
                    json={
                        "model": self.config.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": config.temperature if config else self.config.temperature,
                        # "max_tokens": config.max_tokens if config else self.config.max_tokens,
                        # "stop": config.stop_sequences if config else self.config.stop_sequences,
                        "stream": True
                    }
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"API 调用失败: {response.status} {await response.text()}")
                        
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8').replace('data: ', '').strip())
                                if data != "[DONE]" and "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                raise RuntimeError(f"API 流式调用出错: {str(e)}")
        else:
            # 本地模型流式生成
            response = await self.generate(prompt, config)
            yield response.text
    
    async def embed(
        self,
        text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """生成文本嵌入
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            文本的向量表示
            
        Raises:
            NotImplementedError: 该方法需要在专门的嵌入模型中实现
        """
        raise NotImplementedError("请使用专门的嵌入模型来生成文本嵌入") 