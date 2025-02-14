"""
异常管理模块，提供统一的异常定义和处理机制
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
import asyncio
T = TypeVar("T")


class BaseError(Exception):
    """基础异常类"""
    
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


class ConfigError(BaseError):
    """配置相关异常"""
    pass


class TaskError(BaseError):
    """任务相关异常"""
    pass


class ModelError(BaseError):
    """模型相关异常"""
    pass


@dataclass
class ErrorContext:
    """错误上下文"""
    error: Exception
    traceback: str
    additional_info: Dict[str, Any]


class ErrorHandler(ABC):
    """错误处理器抽象基类"""
    
    @abstractmethod
    async def handle(self, error: ErrorContext) -> None:
        """处理错误"""
        pass


class ErrorManager:
    """错误管理器"""
    
    def __init__(self):
        self._handlers: List[ErrorHandler] = []
        self._error_history: List[ErrorContext] = []
        
    def add_handler(self, handler: ErrorHandler) -> None:
        """添加错误处理器"""
        self._handlers.append(handler)
        
    async def handle_error(self, error: Exception, **kwargs) -> None:
        """处理错误"""
        import traceback
        
        context = ErrorContext(
            error=error,
            traceback=traceback.format_exc(),
            additional_info=kwargs
        )
        
        self._error_history.append(context)
        
        for handler in self._handlers:
            await handler.handle(context)
            
    def get_error_history(self) -> List[ErrorContext]:
        """获取错误历史"""
        return self._error_history.copy()


def error_boundary(error_manager: ErrorManager):
    """错误边界装饰器"""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                await error_manager.handle_error(e, function=func.__name__)
                raise
                
        return wrapper
        
    return decorator 