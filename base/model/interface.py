"""
大模型接口模块，定义统一的模型调用接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """模型配置"""
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stop_sequences: List[str] = []
    extra_params: Dict[str, Any] = {}


@dataclass
class ModelResponse:
    """模型响应"""
    text: str
    tokens_used: int
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None


class LargeModel(ABC):
    """大模型抽象基类"""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str,
        config: Optional[ModelConfig] = None
    ) -> ModelResponse:
        """生成文本"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[ModelConfig] = None
    ) -> AsyncIterator[str]:
        """流式生成文本"""
        pass
    
    @abstractmethod
    async def embed(
        self,
        text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """生成文本嵌入"""
        pass


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self._models: Dict[str, LargeModel] = {}
        
    def register_model(self, name: str, model: LargeModel) -> None:
        """注册模型"""
        self._models[name] = model
        
    def get_model(self, name: str) -> Optional[LargeModel]:
        """获取模型"""
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """列出所有已注册的模型"""
        return list(self._models.keys())


class PromptTemplate(BaseModel):
    """提示词模板"""
    template: str
    variables: Dict[str, str] = {}
    
    def format(self, **kwargs) -> str:
        """格式化提示词"""
        return self.template.format(**{**self.variables, **kwargs})


class ModelEvaluator(ABC):
    """模型评测器抽象基类"""
    
    @abstractmethod
    async def evaluate(
        self,
        model: LargeModel,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """评测模型"""
        pass 