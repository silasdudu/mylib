"""
模型微调模块，提供统一的微调接口和工具
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class TuningMethod(str, Enum):
    """微调方法"""
    FULL_FINETUNE = "full_finetune"  # 全量微调
    LORA = "lora"  # LoRA微调
    PTUNING = "ptuning"  # P-Tuning
    PREFIX = "prefix"  # Prefix-tuning
    PROMPT = "prompt"  # Prompt-tuning


@dataclass
class TuningMetrics:
    """微调指标"""
    loss: float
    accuracy: float
    epoch: int
    step: int
    learning_rate: float
    batch_size: int
    metadata: Dict[str, Any] = None


class TuningConfig(BaseModel):
    """微调配置"""
    method: TuningMethod
    model_name: str
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    extra_params: Dict[str, Any] = {}


class DataProcessor(ABC):
    """数据处理器抽象基类"""
    
    @abstractmethod
    async def process(
        self,
        data: Any,
        config: TuningConfig
    ) -> Dict[str, Any]:
        """处理训练数据"""
        pass
    
    @abstractmethod
    async def tokenize(
        self,
        texts: Union[str, List[str]],
        config: TuningConfig
    ) -> Dict[str, Any]:
        """文本标记化"""
        pass
    
    @abstractmethod
    async def create_dataset(
        self,
        data: Any,
        config: TuningConfig
    ) -> Any:
        """创建数据集"""
        pass


class TuningCallback(ABC):
    """微调回调抽象基类"""
    
    @abstractmethod
    async def on_epoch_begin(
        self,
        epoch: int,
        logs: Dict[str, Any]
    ) -> None:
        """epoch开始时的回调"""
        pass
    
    @abstractmethod
    async def on_epoch_end(
        self,
        epoch: int,
        metrics: TuningMetrics
    ) -> None:
        """epoch结束时的回调"""
        pass
    
    @abstractmethod
    async def on_batch_begin(
        self,
        batch: int,
        logs: Dict[str, Any]
    ) -> None:
        """batch开始时的回调"""
        pass
    
    @abstractmethod
    async def on_batch_end(
        self,
        batch: int,
        metrics: TuningMetrics
    ) -> None:
        """batch结束时的回调"""
        pass


class ModelTuner(ABC):
    """模型微调器抽象基类"""
    
    def __init__(
        self,
        config: TuningConfig,
        processor: DataProcessor,
        callbacks: Optional[List[TuningCallback]] = None
    ):
        self.config = config
        self.processor = processor
        self.callbacks = callbacks or []
    
    @abstractmethod
    async def prepare_model(self) -> None:
        """准备模型"""
        pass
    
    @abstractmethod
    async def train(
        self,
        train_data: Any,
        eval_data: Optional[Any] = None
    ) -> List[TuningMetrics]:
        """训练模型"""
        pass
    
    @abstractmethod
    async def evaluate(
        self,
        eval_data: Any
    ) -> TuningMetrics:
        """评估模型"""
        pass
    
    @abstractmethod
    async def save(
        self,
        path: str
    ) -> None:
        """保存模型"""
        pass
    
    @abstractmethod
    async def load(
        self,
        path: str
    ) -> None:
        """加载模型"""
        pass


class HFModelTuner(ModelTuner):
    """基于Hugging Face的模型微调器"""
    
    async def prepare_model(self) -> None:
        """准备HF模型"""
        # 实现具体的HF模型准备逻辑
        pass
    
    async def train(
        self,
        train_data: Any,
        eval_data: Optional[Any] = None
    ) -> List[TuningMetrics]:
        """使用HF训练器训练模型"""
        # 实现具体的HF训练逻辑
        pass
    
    async def evaluate(
        self,
        eval_data: Any
    ) -> TuningMetrics:
        """使用HF训练器评估模型"""
        # 实现具体的HF评估逻辑
        pass
    
    async def save(
        self,
        path: str
    ) -> None:
        """保存HF模型"""
        # 实现具体的HF模型保存逻辑
        pass
    
    async def load(
        self,
        path: str
    ) -> None:
        """加载HF模型"""
        # 实现具体的HF模型加载逻辑
        pass


class TuningRegistry:
    """微调组件注册表"""
    
    def __init__(self):
        self._processors: Dict[str, DataProcessor] = {}
        self._callbacks: Dict[str, TuningCallback] = {}
        self._tuners: Dict[str, ModelTuner] = {}
        
    def register_processor(
        self,
        name: str,
        processor: DataProcessor
    ) -> None:
        """注册数据处理器"""
        self._processors[name] = processor
        
    def register_callback(
        self,
        name: str,
        callback: TuningCallback
    ) -> None:
        """注册回调"""
        self._callbacks[name] = callback
        
    def register_tuner(
        self,
        name: str,
        tuner: ModelTuner
    ) -> None:
        """注册微调器"""
        self._tuners[name] = tuner
        
    def get_processor(
        self,
        name: str
    ) -> Optional[DataProcessor]:
        """获取数据处理器"""
        return self._processors.get(name)
        
    def get_callback(
        self,
        name: str
    ) -> Optional[TuningCallback]:
        """获取回调"""
        return self._callbacks.get(name)
        
    def get_tuner(
        self,
        name: str
    ) -> Optional[ModelTuner]:
        """获取微调器"""
        return self._tuners.get(name) 