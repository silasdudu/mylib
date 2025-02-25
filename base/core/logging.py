"""
日志模块，提供异步日志记录和多种输出方式
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import sys
import os

from loguru import logger

# 移除默认处理器
logger.remove()

# 添加彩色控制台处理器
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[module]}:{function}:{line}</cyan> - <level>{message}</level>",
    colorize=True,
    backtrace=False,
    diagnose=False
)


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
        """初始化日志记录器"""
        self._handlers: List[LogHandler] = []
        
    def add_handler(self, handler: LogHandler) -> None:
        """添加日志处理器"""
        self._handlers.append(handler)
        
    async def log(self, level: Union[LogLevel, str], message: str, **kwargs) -> None:
        """记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数
        """
        # 获取调用者信息
        import inspect
        frame = inspect.currentframe()
        if frame:
            caller_frame = frame.f_back
            if caller_frame:
                # 获取完整文件路径
                filename = caller_frame.f_code.co_filename
                
                # 获取模块名（相对于项目根目录的路径）
                try:
                    # 首先尝试通过__file__属性获取模块信息
                    module_path = None
                    module_name = caller_frame.f_globals.get('__name__', '')
                    
                    # 如果不是__main__，直接使用模块名
                    if module_name and module_name != '__main__':
                        module_path = module_name
                    else:
                        # 否则尝试从文件路径推断模块路径
                        root_dir = None
                        for path in sys.path:
                            if path and os.path.exists(path) and filename.startswith(path):
                                if root_dir is None or len(path) > len(root_dir):
                                    root_dir = path
                        
                        if root_dir:
                            # 计算相对路径
                            rel_path = os.path.relpath(filename, root_dir)
                            # 将路径分隔符转换为点号，去掉.py扩展名
                            module_path = rel_path.replace(os.path.sep, '.')
                            if module_path.endswith('.py'):
                                module_path = module_path[:-3]
                    
                    # 保存模块路径供loguru使用
                    if module_path:
                        kwargs['_module_path'] = module_path
                except Exception as e:
                    # 记录异常但继续执行
                    print(f"获取模块路径时出错: {e}")
        
        # 将日志级别转换为字符串
        level_str = level if isinstance(level, str) else level.value
        
        # 创建记录
        record = {
            "level": level_str,
            "message": message,
            **kwargs
        }
        
        # 将日志传递给所有处理器
        for handler in self._handlers:
            await handler.emit(record)
            
    async def close(self) -> None:
        """关闭所有处理器"""
        for handler in self._handlers:
            await handler.close()


class FileLogHandler(LogHandler):
    """文件日志处理器"""
    
    def __init__(self, file_path: str, rotation: str = "500 MB"):
        """初始化文件日志处理器
        
        Args:
            file_path: 日志文件路径
            rotation: 日志轮转设置
        """
        # 添加文件处理器
        self._handler_id = logger.add(
            file_path, 
            rotation=rotation,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[module]}:{function}:{line} - {message}",
            colorize=False,
            backtrace=False,
            diagnose=False
        )
        self._file_path = file_path
        
    async def emit(self, record: Dict[str, Any]) -> None:
        """输出日志到文件
        
        Args:
            record: 日志记录
        """
        # 检查是否有自定义模块路径
        module_path = record.get('_module_path', None)
        
        if module_path:
            # 如果有自定义模块路径，使用它
            logger.bind(module=module_path).opt(depth=2).log(record["level"], record["message"])
        else:
            # 否则使用默认行为
            logger.opt(depth=2).log(record["level"], record["message"])
        
    async def close(self) -> None:
        """关闭文件处理器"""
        if hasattr(self, '_handler_id'):
            logger.remove(self._handler_id)


class ConsoleLogHandler(LogHandler):
    """控制台日志处理器"""
    
    def __init__(self):
        """初始化控制台日志处理器"""
        # 不需要添加处理器，因为我们已经在模块级别添加了
        pass
    
    async def emit(self, record: Dict[str, Any]) -> None:
        """输出日志到控制台
        
        Args:
            record: 日志记录
        """
        # 检查是否有自定义模块路径
        module_path = record.get('_module_path', None)
        
        if module_path:
            # 如果有自定义模块路径，使用它
            logger.bind(module=module_path).opt(depth=2).log(record["level"], record["message"])
        else:
            # 否则使用默认行为
            logger.opt(depth=2).log(record["level"], record["message"])
        
    async def close(self) -> None:
        """关闭控制台处理器"""
        # 不需要关闭，因为我们使用的是模块级别的处理器 