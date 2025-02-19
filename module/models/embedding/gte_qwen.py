"""
GTE-Qwen 嵌入模型，基于通义千问的文本嵌入模型
支持本地模型加载和 API 调用两种方式
"""
from typing import List, Optional, Union
import json
import os
from pathlib import Path
import aiohttp
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm

from base.model.embedding import DenseEmbeddingModel, EmbeddingConfig, EmbeddingOutput
from base.rag.chunking import Chunk


def load_default_config() -> 'GTEQwenEmbeddingConfig':
    """加载默认配置
    
    从项目根目录的 .env 文件中加载配置参数。
    必需的环境变量：
    - GTE_EMBEDDING_API_KEY: API密钥
    - GTE_EMBEDDING_BASE_URL: API基础URL
    - GTE_EMBEDDING_MODEL_NAME: 模型名称
    
    Returns:
        默认的嵌入模型配置
        
    Raises:
        ValueError: 当必需的环境变量未设置时抛出
    """
    # 加载 .env 文件
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    # 获取必需的环境变量
    api_key = os.getenv('GTE_EMBEDDING_API_KEY')
    base_url = os.getenv('GTE_EMBEDDING_BASE_URL')
    model_name = os.getenv('GTE_EMBEDDING_MODEL_NAME')
    
    if not all([api_key, base_url, model_name]):
        raise ValueError(
            "必需的环境变量未设置。请在 .env 文件中设置以下变量：\n"
            "- GTE_EMBEDDING_API_KEY\n"
            "- GTE_EMBEDDING_BASE_URL\n"
            "- GTE_EMBEDDING_MODEL_NAME"
        )
    
    return GTEQwenEmbeddingConfig(
        model_name=model_name,
        api_url=base_url,
        api_key=api_key,
        dimension=1024,  # GTE-Qwen 默认维度
        normalize=True,  # 默认进行归一化
        batch_size=32    # 默认批处理大小
    )


class GTEQwenEmbeddingConfig(EmbeddingConfig):
    """GTE-Qwen 嵌入模型配置"""
    api_url: Optional[str] = None  # API URL，如果不为空则使用 API 模式
    api_key: Optional[str] = None  # API 密钥
    model_path: Optional[str] = None  # 本地模型路径，如果不为空则优先使用本地模型
    # model_name 从父类继承，必须提供，用于指定模型名称（本地模型或 API 调用时的模型标识）


class GTEQwenEmbedding(DenseEmbeddingModel):
    """GTE-Qwen 嵌入模型实现，支持本地模型和 API 调用"""
    
    def __init__(self, config: Optional[GTEQwenEmbeddingConfig] = None):
        """初始化模型
        
        Args:
            config: 模型配置，支持本地模型和 API 两种方式：
                   - 本地模型：需要设置 model_path 或 model_name
                   - API 调用：需要设置 api_url、api_key 和 model_name
                   如果不提供，将使用 .env 中的默认配置
        """
        if config is None:
            config = load_default_config()
            
        if not config.model_name:
            raise ValueError("必须提供 model_name，用于指定模型名称")
            
        super().__init__(config)
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            
            # 如果有GPU则使用GPU
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            # 设置为评估模式
            self.model.eval()
    
    async def _ensure_session(self):
        """确保 aiohttp session 已初始化"""
        if self.use_api and self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def _cleanup(self):
        """清理资源"""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    def _mean_pooling(self, model_output, attention_mask):
        """平均池化，获取句子表示
        
        Args:
            model_output: 模型输出的隐藏状态
            attention_mask: 注意力掩码
            
        Returns:
            句子的向量表示
        """
        token_embeddings = model_output[0]  # 获取最后一层的隐藏状态
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    async def _embed_api(self, texts: List[str]) -> List[EmbeddingOutput]:
        """使用 API 生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本的向量表示列表
        """
        await self._ensure_session()
        
        try:
            async with self.session.post(
                self.config.api_url,
                headers=self.headers,
                json={
                    "model": self.config.model_name,
                    "input": texts
                }
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"API 调用失败: {response.status} {await response.text()}")
                    
                result = await response.json()
                embeddings = []
                
                if "embeddings" in result:
                    for vector in result["embeddings"]:
                        if self.config.normalize:
                            vector = self.normalize_vector(vector)
                    embeddings.append(EmbeddingOutput(vector=vector))
                elif "data" in result:
                    for _, kv in enumerate(result["data"]):
                        if self.config.normalize:
                            vector = self.normalize_vector(kv['embedding'])
                        embeddings.append(EmbeddingOutput(vector=kv['embedding']))
                elif "embedding" in result:
                    if self.config.normalize:
                        result["embedding"] = self.normalize_vector(result["embedding"])
                    embeddings.append(EmbeddingOutput(vector=result["embedding"]))
                else:
                    raise ValueError(f"API 返回格式不支持: {result}")
                
                return embeddings
                
        except Exception as e:
            raise RuntimeError(f"API 调用出错: {str(e)}")
    
    async def _embed_local(self, texts: List[str]) -> List[EmbeddingOutput]:
        """使用本地模型生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本的向量表示列表
        """
        # 对文本进行分词
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 如果有GPU则使用GPU
        if torch.cuda.is_available():
            encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        
        # 生成嵌入向量
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 如果配置要求归一化，则进行归一化
            if self.config.normalize:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # 转换为列表格式
            return [
                EmbeddingOutput(vector=embedding.cpu().numpy().tolist())
                for embedding in sentence_embeddings
            ]
    
    async def embed(
        self,
        text: Union[str, List[str]]
    ) -> Union[EmbeddingOutput, List[EmbeddingOutput]]:
        """生成文本嵌入向量
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            文本的向量表示
        """
        # 确保输入是列表形式
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False
        
        # 根据模式选择不同的实现
        if self.use_api:
            embeddings = await self._embed_api(text)
        else:
            embeddings = await self._embed_local(text)
        
        # 如果输入是单个文本，则返回单个结果
        return embeddings[0] if single_input else embeddings
    
    async def embed_chunk(self, chunk: Chunk) -> EmbeddingOutput:
        """生成文档块的嵌入向量
        
        Args:
            chunk: 文档块
            
        Returns:
            文档块的向量表示
        """
        return await self.embed(chunk.text)
    
    async def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddingOutput]:
        """批量生成文档块的嵌入向量
        
        Args:
            chunks: 文档块列表
            
        Returns:
            文档块的向量表示列表
        """
        # 提取文本内容
        texts = [chunk.text for chunk in chunks]
        
        # 计算总批次数
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size
        
        # 分批处理，使用tqdm显示进度
        results = []
        with tqdm(total=total_batches, desc="生成文档向量") as pbar:
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_embeddings = await self.embed(batch_texts)
                if not isinstance(batch_embeddings, list):
                    batch_embeddings = [batch_embeddings]
                results.extend(batch_embeddings)
                pbar.update(1)
        
        return results
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self._cleanup() 