"""
配置管理模块，支持多种格式的配置加载和动态更新
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel


class ConfigSource(ABC):
    """配置源抽象基类"""
    
    @abstractmethod
    async def load(self) -> Dict[str, Any]:
        """加载配置"""
        pass
    
    @abstractmethod
    async def save(self, config: Dict[str, Any]) -> None:
        """保存配置"""
        pass
    
    @abstractmethod
    async def watch(self) -> None:
        """监听配置变化"""
        pass


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_source: ConfigSource):
        self._source = config_source
        self._config: Dict[str, Any] = {}
        
    async def load(self) -> Dict[str, Any]:
        """加载配置"""
        self._config = await self._source.load()
        return self._config
    
    async def save(self) -> None:
        """保存当前配置"""
        await self._source.save(self._config)
        
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        self._config[key] = value
        
    async def watch(self) -> None:
        """监听配置变化"""
        await self._source.watch()


class FileConfigSource(ConfigSource):
    """文件配置源"""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        
    async def load(self) -> Dict[str, Any]:
        """从文件加载配置"""
        # 具体实现将根据文件扩展名选择不同的解析器
        raise NotImplementedError()
    
    async def save(self, config: Dict[str, Any]) -> None:
        """保存配置到文件"""
        raise NotImplementedError()
    
    async def watch(self) -> None:
        """监听文件变化"""
        raise NotImplementedError()


class ConfigModel(BaseModel):
    """配置模型基类"""
    
    class Config:
        extra = "allow"  # 允许额外字段 