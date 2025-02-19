"""
自定义重排序模型实现，支持本地模型和API调用两种方式
"""
from typing import List, Optional, Union, Dict, Any
import os
import json
import aiohttp
import torch
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

from base.model.rerank import Reranker, RerankConfig, RerankOutput
from base.rag.retriever import SearchResult


def load_default_config() -> 'CustomRerankConfig':
    """加载默认配置
    
    从项目根目录的 .env 文件中加载配置参数。
    必需的环境变量：
    - CUSTOM_RERANK_API_KEY: API密钥
    - CUSTOM_RERANK_BASE_URL: API基础URL
    - CUSTOM_RERANK_MODEL_NAME: 模型名称
    
    Returns:
        默认的重排序模型配置
        
    Raises:
        ValueError: 当必需的环境变量未设置时抛出
    """
    # 加载 .env 文件
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    # 获取必需的环境变量
    api_key = os.getenv('CUSTOM_RERANK_API_KEY')
    base_url = os.getenv('CUSTOM_RERANK_BASE_URL')
    model_name = os.getenv('CUSTOM_RERANK_MODEL_NAME')
    
    if not all([api_key, base_url, model_name]):
        raise ValueError(
            "必需的环境变量未设置。请在 .env 文件中设置以下变量：\n"
            "- CUSTOM_RERANK_API_KEY\n"
            "- CUSTOM_RERANK_BASE_URL\n"
            "- CUSTOM_RERANK_MODEL_NAME"
        )
    
    return CustomRerankConfig(
        model_name=model_name,
        api_url=base_url,
        api_key=api_key
    )


class CustomRerankConfig(RerankConfig):
    """自定义重排序模型配置"""
    api_url: Optional[str] = None  # API URL，如果不为空则使用 API 模式
    api_key: Optional[str] = None  # API 密钥
    model_path: Optional[str] = None  # 本地模型路径，如果不为空则优先使用本地模型
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 运行设备


class CustomReranker(Reranker):
    """自定义重排序模型实现，支持本地模型和API调用"""
    
    def __init__(self, config: Optional[CustomRerankConfig] = None):
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
            
        super().__init__(config)
        self.config: CustomRerankConfig = config
        
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
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto" if config.device == "cuda" else None
            )
            
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
    
    async def _compute_similarity_api(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        """使用 API 计算文本相似度
        
        Args:
            text1: 第一个文本或文本列表
            text2: 第二个文本或文本列表
            
        Returns:
            相似度分数或分数列表
        """
        await self._ensure_session()
        
        # 确保输入是列表形式
        if isinstance(text1, str):
            text1 = [text1]
        if isinstance(text2, str):
            text2 = [text2]
            
        try:
            async with self.session.post(
                self.config.api_url,
                headers=self.headers,
                json={
                    "model": self.config.model_name,
                    "text1": text1,
                    "text2": text2
                }
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"API 调用失败: {response.status} {await response.text()}")
                    
                result = await response.json()
                scores = result["scores"]
                
                # 归一化分数
                if self.config.score_type == "normalized":
                    scores = [self.normalize_score(score) for score in scores]
                    
                return scores[0] if len(scores) == 1 else scores
                
        except Exception as e:
            raise RuntimeError(f"API 调用出错: {str(e)}")
    
    async def _compute_similarity_local(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        """使用本地模型计算文本相似度
        
        Args:
            text1: 第一个文本或文本列表
            text2: 第二个文本或文本列表
            
        Returns:
            相似度分数或分数列表
        """
        # 确保输入是列表形式
        if isinstance(text1, str):
            text1 = [text1]
        if isinstance(text2, str):
            text2 = [text2]
            
        # 分批处理
        batch_size = self.config.batch_size
        scores = []
        
        for i in range(0, len(text1), batch_size):
            batch_text1 = text1[i:i + batch_size]
            batch_text2 = text2[i:i + batch_size]
            
            # 对文本对进行分词
            inputs = self.tokenizer(
                batch_text1,
                batch_text2,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # 移动到正确的设备
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            # 计算相似度分数
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                
                # 归一化分数
                if self.config.score_type == "normalized":
                    batch_scores = [self.normalize_score(score) for score in batch_scores]
                    
                scores.extend(batch_scores.tolist())
        
        return scores[0] if len(scores) == 1 else scores
    
    async def compute_similarity(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        """计算文本相似度
        
        Args:
            text1: 第一个文本或文本列表
            text2: 第二个文本或文本列表
            
        Returns:
            相似度分数或分数列表
        """
        if self.use_api:
            return await self._compute_similarity_api(text1, text2)
        else:
            return await self._compute_similarity_local(text1, text2)
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankOutput]:
        """对文档列表进行重新排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的结果数量
            
        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []
            
        # 计算相似度分数
        scores = await self.compute_similarity(
            [query] * len(documents),
            documents
        )
        
        # 创建结果列表
        results = [
            RerankOutput(score=score)
            for score in scores
        ]
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 返回前 top_k 个结果
        if top_k is not None:
            results = results[:top_k]
            
        return results
    
    async def rerank_search_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """对搜索结果进行重新排序
        
        Args:
            query: 查询文本
            results: 搜索结果列表
            top_k: 返回的结果数量
            
        Returns:
            重排序后的搜索结果列表
        """
        if not results:
            return []
            
        # 提取文档文本
        documents = [result.chunk.text for result in results]
        
        # 计算新的相似度分数
        scores = await self.compute_similarity(
            [query] * len(documents),
            documents
        )
        
        # 更新搜索结果的分数
        for result, score in zip(results, scores):
            result.score = score
            
        # 按新分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 返回前 top_k 个结果
        if top_k is not None:
            results = results[:top_k]
            
        return results
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self._cleanup() 