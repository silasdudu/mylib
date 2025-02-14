"""
日志模块，提供异步日志记录和多种输出方式
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogHandler(ABC):
    """日志处理器抽象基类"""
    
    @abstractmethod
    async def emit(self, record: Dict[str, Any]) -> None:
        """输出日志记录"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭处理器"""
        pass


class AsyncLogger:
    """异步日志记录器"""
    
    def __init__(self):
        self._handlers: List[LogHandler] = []
        
    def add_handler(self, handler: LogHandler) -> None:
        """添加日志处理器"""
        self._handlers.append(handler)
        
    async def log(self, level: Union[LogLevel, str], message: str, **kwargs) -> None:
        """记录日志"""
        record = {
            "level": level if isinstance(level, str) else level.value,
            "message": message,
            **kwargs
        }
        
        for handler in self._handlers:
            await handler.emit(record)
            
    async def close(self) -> None:
        """关闭所有处理器"""
        for handler in self._handlers:
            await handler.close()


class FileLogHandler(LogHandler):
    """文件日志处理器"""
    
    def __init__(self, file_path: str, rotation: str = "500 MB"):
        logger.add(file_path, rotation=rotation)
        self._file_path = file_path
        
    async def emit(self, record: Dict[str, Any]) -> None:
        """输出日志到文件"""
        logger.log(record["level"], record["message"])
        
    async def close(self) -> None:
        """关闭文件处理器"""
        logger.remove()


class ConsoleLogHandler(LogHandler):
    """控制台日志处理器"""
    
    async def emit(self, record: Dict[str, Any]) -> None:
        """输出日志到控制台"""
        logger.log(record["level"], record["message"])
        
    async def close(self) -> None:
        """关闭控制台处理器"""
        pass 