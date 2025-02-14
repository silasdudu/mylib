"""
弹性模块，提供熔断和重试机制
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel


T = TypeVar("T")


class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态
    OPEN = "open"         # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态


class CircuitBreakerConfig(BaseModel):
    """熔断器配置"""
    failure_threshold: int = 5  # 失败阈值
    success_threshold: int = 3  # 成功阈值
    timeout: int = 60  # 熔断超时时间(秒)
    include_exceptions: List[str] = ["Exception"]  # 计入失败的异常类型
    exclude_exceptions: List[str] = []  # 不计入失败的异常类型


class RetryConfig(BaseModel):
    """重试配置"""
    max_attempts: int = 3  # 最大重试次数
    initial_delay: float = 1.0  # 初始延迟时间(秒)
    max_delay: float = 30.0  # 最大延迟时间(秒)
    exponential_base: float = 2.0  # 指数退避基数
    jitter: bool = True  # 是否添加随机抖动


class CircuitBreaker:
    """熔断器"""
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig
    ):
        self.name = name
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_attempt_time = None
    
    @property
    def state(self) -> CircuitState:
        """获取当前状态"""
        return self._state
    
    def _should_allow_request(self) -> bool:
        """判断是否允许请求"""
        now = datetime.now()
        
        if self._state == CircuitState.CLOSED:
            return True
            
        if self._state == CircuitState.OPEN:
            if (now - self._last_failure_time).total_seconds() >= self.config.timeout:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
            
        if self._state == CircuitState.HALF_OPEN:
            if not self._last_attempt_time or (now - self._last_attempt_time).total_seconds() >= 1:
                self._last_attempt_time = now
                return True
            return False
            
        return True
    
    def _on_success(self) -> None:
        """处理成功请求"""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                
        self._last_attempt_time = None
    
    def _on_failure(self, exception: Exception) -> None:
        """处理失败请求"""
        exception_name = exception.__class__.__name__
        
        if (
            exception_name in self.config.exclude_exceptions or
            (
                self.config.include_exceptions != ["Exception"] and
                exception_name not in self.config.include_exceptions
            )
        ):
            return
            
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._success_count = 0
            
        self._last_attempt_time = None
    
    async def call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """执行被保护的函数调用"""
        if not self._should_allow_request():
            raise Exception(f"Circuit breaker {self.name} is {self._state}")
            
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise


class Retry:
    """重试机制"""
    
    def __init__(
        self,
        config: RetryConfig
    ):
        self.config = config
    
    def _get_delay(
        self,
        attempt: int
    ) -> float:
        """计算重试延迟时间"""
        import random
        
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay
        )
        
        if self.config.jitter:
            delay *= (0.5 + random.random())
            
        return delay
    
    async def call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """执行带重试的函数调用"""
        import asyncio
        
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_attempts:
                    delay = self._get_delay(attempt)
                    await asyncio.sleep(delay)
                    
        raise last_exception


class ResilienceRegistry:
    """弹性组件注册表"""
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._retries: Dict[str, Retry] = {}
    
    def register_circuit_breaker(
        self,
        name: str,
        circuit_breaker: CircuitBreaker
    ) -> None:
        """注册熔断器"""
        self._circuit_breakers[name] = circuit_breaker
    
    def register_retry(
        self,
        name: str,
        retry: Retry
    ) -> None:
        """注册重试机制"""
        self._retries[name] = retry
    
    def get_circuit_breaker(
        self,
        name: str
    ) -> Optional[CircuitBreaker]:
        """获取熔断器"""
        return self._circuit_breakers.get(name)
    
    def get_retry(
        self,
        name: str
    ) -> Optional[Retry]:
        """获取重试机制"""
        return self._retries.get(name)
    
    def list_circuit_breakers(self) -> List[str]:
        """列出所有熔断器"""
        return list(self._circuit_breakers.keys())
    
    def list_retries(self) -> List[str]:
        """列出所有重试机制"""
        return list(self._retries.keys()) 