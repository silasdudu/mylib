"""
数据集模块，提供统一的数据加载和处理接口
"""
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Generic, Iterator, List, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class DatasetConfig(BaseModel):
    """数据集配置"""
    batch_size: int = 32
    shuffle: bool = True
    cache_size: Optional[int] = None
    num_workers: int = 1


class Dataset(Generic[T], ABC):
    """数据集抽象基类"""
    
    @abstractmethod
    def __len__(self) -> int:
        """返回数据集大小"""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> T:
        """获取单个数据项"""
        pass
    
    @abstractmethod
    async def get_batch(self, batch_size: int) -> List[T]:
        """获取数据批次"""
        pass


class DataLoader(Generic[T]):
    """数据加载器"""
    
    def __init__(
        self,
        dataset: Dataset[T],
        config: Optional[DatasetConfig] = None
    ):
        self.dataset = dataset
        self.config = config or DatasetConfig()
        
    def __iter__(self) -> Iterator[List[T]]:
        """同步迭代器"""
        for i in range(0, len(self.dataset), self.config.batch_size):
            yield [
                self.dataset[j]
                for j in range(
                    i,
                    min(i + self.config.batch_size, len(self.dataset))
                )
            ]
            
    async def __aiter__(self) -> AsyncIterator[List[T]]:
        """异步迭代器"""
        for i in range(0, len(self.dataset), self.config.batch_size):
            yield await self.dataset.get_batch(self.config.batch_size)


class DataProcessor(ABC):
    """数据处理器抽象基类"""
    
    @abstractmethod
    async def process(self, data: T) -> T:
        """处理单个数据项"""
        pass
    
    @abstractmethod
    async def process_batch(self, batch: List[T]) -> List[T]:
        """处理数据批次"""
        pass


class DataCache(Generic[T]):
    """数据缓存"""
    
    def __init__(self, capacity: Optional[int] = None):
        self._cache: Dict[str, T] = {}
        self._capacity = capacity
        
    async def get(self, key: str) -> Optional[T]:
        """获取缓存数据"""
        return self._cache.get(key)
        
    async def put(self, key: str, value: T) -> None:
        """存入缓存数据"""
        if self._capacity and len(self._cache) >= self._capacity:
            # 简单的LRU策略：删除第一个键值对
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value
        
    async def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        
    def __len__(self) -> int:
        """返回缓存大小"""
        return len(self._cache) 