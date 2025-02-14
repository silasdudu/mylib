"""
缓存管理模块，提供统一的缓存接口
"""
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel


class CacheConfig(BaseModel):
    """缓存配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    prefix: str = ""
    default_ttl: int = 3600  # 默认过期时间(秒)
    enable_compression: bool = False
    compression_threshold: int = 1024  # 压缩阈值(字节)


class Cache(ABC):
    """缓存抽象基类"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
    
    @abstractmethod
    async def get(self, key: str) -> Any:
        """获取缓存值"""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """清空缓存"""
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取缓存值"""
        pass
    
    @abstractmethod
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """批量设置缓存值"""
        pass
    
    @abstractmethod
    async def delete_many(self, keys: List[str]) -> bool:
        """批量删除缓存值"""
        pass


class RedisCache(Cache):
    """Redis缓存实现"""
    
    async def connect(self) -> None:
        """连接Redis"""
        import aioredis
        
        self.redis = await aioredis.create_redis_pool(
            f"redis://{self.config.host}:{self.config.port}",
            db=self.config.db,
            password=self.config.password,
            encoding="utf-8"
        )
    
    async def disconnect(self) -> None:
        """断开连接"""
        self.redis.close()
        await self.redis.wait_closed()
    
    async def get(self, key: str) -> Any:
        """获取缓存值"""
        import pickle
        
        key = f"{self.config.prefix}{key}"
        value = await self.redis.get(key)
        
        if value is None:
            return None
            
        try:
            return pickle.loads(value)
        except:
            return value.decode()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """设置缓存值"""
        import pickle
        
        key = f"{self.config.prefix}{key}"
        
        if isinstance(value, (str, int, float, bool)):
            data = str(value).encode()
        else:
            data = pickle.dumps(value)
            
        if self.config.enable_compression and len(data) > self.config.compression_threshold:
            import zlib
            data = zlib.compress(data)
            
        if ttl is None:
            ttl = self.config.default_ttl
            
        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())
            
        try:
            await self.redis.set(key, data, expire=ttl)
            return True
        except:
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        key = f"{self.config.prefix}{key}"
        try:
            await self.redis.delete(key)
            return True
        except:
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        key = f"{self.config.prefix}{key}"
        return await self.redis.exists(key)
    
    async def clear(self) -> bool:
        """清空缓存"""
        try:
            await self.redis.flushdb()
            return True
        except:
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取缓存值"""
        import pickle
        
        prefixed_keys = [f"{self.config.prefix}{key}" for key in keys]
        values = await self.redis.mget(*prefixed_keys)
        
        result = {}
        for key, value in zip(keys, values):
            if value is None:
                continue
                
            try:
                result[key] = pickle.loads(value)
            except:
                result[key] = value.decode()
                
        return result
    
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """批量设置缓存值"""
        import pickle
        
        if ttl is None:
            ttl = self.config.default_ttl
            
        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())
            
        pipe = self.redis.pipeline()
        
        for key, value in mapping.items():
            key = f"{self.config.prefix}{key}"
            
            if isinstance(value, (str, int, float, bool)):
                data = str(value).encode()
            else:
                data = pickle.dumps(value)
                
            if self.config.enable_compression and len(data) > self.config.compression_threshold:
                import zlib
                data = zlib.compress(data)
                
            pipe.set(key, data, expire=ttl)
            
        try:
            await pipe.execute()
            return True
        except:
            return False
    
    async def delete_many(self, keys: List[str]) -> bool:
        """批量删除缓存值"""
        prefixed_keys = [f"{self.config.prefix}{key}" for key in keys]
        try:
            await self.redis.delete(*prefixed_keys)
            return True
        except:
            return False


class MemcachedCache(Cache):
    """Memcached缓存实现"""
    
    async def connect(self) -> None:
        """连接Memcached"""
        import aiomcache
        
        self.memcached = aiomcache.Client(
            self.config.host,
            self.config.port
        )
    
    async def disconnect(self) -> None:
        """断开连接"""
        self.memcached.close()
    
    async def get(self, key: str) -> Any:
        """获取缓存值"""
        import pickle
        
        key = f"{self.config.prefix}{key}".encode()
        value = await self.memcached.get(key)
        
        if value is None:
            return None
            
        try:
            return pickle.loads(value)
        except:
            return value.decode()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """设置缓存值"""
        import pickle
        
        key = f"{self.config.prefix}{key}".encode()
        
        if isinstance(value, (str, int, float, bool)):
            data = str(value).encode()
        else:
            data = pickle.dumps(value)
            
        if self.config.enable_compression and len(data) > self.config.compression_threshold:
            import zlib
            data = zlib.compress(data)
            
        if ttl is None:
            ttl = self.config.default_ttl
            
        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())
            
        try:
            await self.memcached.set(key, data, exptime=ttl)
            return True
        except:
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        key = f"{self.config.prefix}{key}".encode()
        try:
            await self.memcached.delete(key)
            return True
        except:
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        key = f"{self.config.prefix}{key}".encode()
        value = await self.memcached.get(key)
        return value is not None
    
    async def clear(self) -> bool:
        """清空缓存"""
        try:
            await self.memcached.flush_all()
            return True
        except:
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取缓存值"""
        import pickle
        
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
                
        return result
    
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """批量设置缓存值"""
        success = True
        for key, value in mapping.items():
            if not await self.set(key, value, ttl):
                success = False
        return success
    
    async def delete_many(self, keys: List[str]) -> bool:
        """批量删除缓存值"""
        success = True
        for key in keys:
            if not await self.delete(key):
                success = False
        return success


class CacheRegistry:
    """缓存注册表"""
    
    def __init__(self):
        self._caches: Dict[str, Cache] = {}
        
    def register(
        self,
        name: str,
        cache: Cache
    ) -> None:
        """注册缓存"""
        self._caches[name] = cache
        
    def get(
        self,
        name: str
    ) -> Optional[Cache]:
        """获取缓存"""
        return self._caches.get(name)
        
    def list_caches(self) -> List[str]:
        """列出所有缓存"""
        return list(self._caches.keys()) 