"""
嵌入模型模块，支持将文本转换为向量表示
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel

from .chunking import Chunk


class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""
    model_name: str
    dimension: int
    batch_size: int = 32
    normalize: bool = True
    extra_params: Dict[str, Any] = {}


class EmbeddingOutput(BaseModel):
    """嵌入输出"""
    vector: List[float]
    metadata: Dict[str, Any] = {}


class EmbeddingModel(ABC):
    """嵌入模型抽象基类"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
    @abstractmethod
    async def embed(
        self,
        text: Union[str, List[str]]
    ) -> Union[EmbeddingOutput, List[EmbeddingOutput]]:
        """生成文本嵌入向量"""
        pass
    
    @abstractmethod
    async def embed_chunk(
        self,
        chunk: Chunk
    ) -> EmbeddingOutput:
        """生成分块的嵌入向量"""
        pass
    
    @abstractmethod
    async def embed_chunks(
        self,
        chunks: List[Chunk]
    ) -> List[EmbeddingOutput]:
        """批量生成分块的嵌入向量"""
        pass
    
    def normalize_vector(
        self,
        vector: Union[List[float], np.ndarray]
    ) -> List[float]:
        """归一化向量"""
        if isinstance(vector, list):
            vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


class DenseEmbeddingModel(EmbeddingModel):
    """密集向量嵌入模型"""
    
    async def embed(
        self,
        text: Union[str, List[str]]
    ) -> Union[EmbeddingOutput, List[EmbeddingOutput]]:
        """生成密集向量嵌入"""
        # 实现具体的嵌入逻辑
        raise NotImplementedError()
    
    async def embed_chunk(
        self,
        chunk: Chunk
    ) -> EmbeddingOutput:
        """生成分块的密集向量嵌入"""
        if isinstance(chunk.content, str):
            result = await self.embed(chunk.content)
            if isinstance(result, list):
                result = result[0]
            return result
        raise ValueError(f"不支持的分块内容类型: {type(chunk.content)}")
    
    async def embed_chunks(
        self,
        chunks: List[Chunk]
    ) -> List[EmbeddingOutput]:
        """批量生成分块的密集向量嵌入"""
        results = []
        batch = []
        
        for chunk in chunks:
            if isinstance(chunk.content, str):
                batch.append(chunk.content)
                
                if len(batch) >= self.config.batch_size:
                    embeddings = await self.embed(batch)
                    results.extend(embeddings)
                    batch = []
                    
        if batch:  # 处理剩余的分块
            embeddings = await self.embed(batch)
            results.extend(embeddings)
            
        return results


class SparseEmbeddingModel(EmbeddingModel):
    """稀疏向量嵌入模型"""
    
    async def embed(
        self,
        text: Union[str, List[str]]
    ) -> Union[EmbeddingOutput, List[EmbeddingOutput]]:
        """生成稀疏向量嵌入"""
        # 实现具体的嵌入逻辑
        raise NotImplementedError()
    
    async def embed_chunk(
        self,
        chunk: Chunk
    ) -> EmbeddingOutput:
        """生成分块的稀疏向量嵌入"""
        if isinstance(chunk.content, str):
            result = await self.embed(chunk.content)
            if isinstance(result, list):
                result = result[0]
            return result
        raise ValueError(f"不支持的分块内容类型: {type(chunk.content)}")
    
    async def embed_chunks(
        self,
        chunks: List[Chunk]
    ) -> List[EmbeddingOutput]:
        """批量生成分块的稀疏向量嵌入"""
        results = []
        batch = []
        
        for chunk in chunks:
            if isinstance(chunk.content, str):
                batch.append(chunk.content)
                
                if len(batch) >= self.config.batch_size:
                    embeddings = await self.embed(batch)
                    results.extend(embeddings)
                    batch = []
                    
        if batch:  # 处理剩余的分块
            embeddings = await self.embed(batch)
            results.extend(embeddings)
            
        return results


class EmbeddingModelRegistry:
    """嵌入模型注册表"""
    
    def __init__(self):
        self._models: Dict[str, EmbeddingModel] = {}
        
    def register(
        self,
        name: str,
        model: EmbeddingModel
    ) -> None:
        """注册嵌入模型"""
        self._models[name] = model
        
    def get_model(
        self,
        name: str
    ) -> Optional[EmbeddingModel]:
        """获取嵌入模型"""
        return self._models.get(name)
        
    def list_models(self) -> List[str]:
        """列出所有可用的嵌入模型"""
        return list(self._models.keys()) 