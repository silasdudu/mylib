"""
异步日志模块，提供异步日志记录功能
"""
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import sys
import os
import time
import traceback

from loguru import logger


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AsyncLogHandler(ABC):
    """异步日志处理器抽象基类"""
    
    @abstractmethod
    async def emit(self, record: Dict[str, Any]) -> None:
        """输出日志记录
        
        Args:
            record: 日志记录
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭处理器"""
        pass


class AsyncLogger:
    """异步日志记录器，支持多种处理器和异步记录"""
    
    def __init__(self, module_name: Optional[str] = None):
        """初始化异步日志记录器
        
        Args:
            module_name: 模块名称，如果为None则自动检测
        """
        self._handlers: List[AsyncLogHandler] = []
        self._module_name = module_name
        self._queue = asyncio.Queue()
        self._running = False
        self._worker_task = None
        
    def add_handler(self, handler: AsyncLogHandler) -> None:
        """添加日志处理器
        
        Args:
            handler: 日志处理器
        """
        self._handlers.append(handler)
        
    def remove_handler(self, handler: AsyncLogHandler) -> None:
        """移除日志处理器
        
        Args:
            handler: 日志处理器
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    async def start(self) -> None:
        """启动日志处理工作线程"""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._process_logs())
    
    async def stop(self) -> None:
        """停止日志处理工作线程"""
        if self._running:
            self._running = False
            # 添加一个None记录作为停止信号
            await self._queue.put(None)
            if self._worker_task:
                await self._worker_task
            
            # 关闭所有处理器
            for handler in self._handlers:
                await handler.close()
    
    async def _process_logs(self) -> None:
        """处理日志队列中的记录"""
        while self._running:
            try:
                record = await self._queue.get()
                if record is None:  # 停止信号
                    break
                    
                # 将日志传递给所有处理器
                for handler in self._handlers:
                    try:
                        await handler.emit(record)
                    except Exception as e:
                        # 处理器出错，打印到标准错误
                        print(f"日志处理器出错: {e}", file=sys.stderr)
                        
                self._queue.task_done()
            except Exception as e:
                print(f"日志处理线程出错: {e}", file=sys.stderr)
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """获取调用者信息
        
        Returns:
            包含调用者信息的字典
        """
        # 获取调用者信息
        frame = sys._getframe(2)  # 跳过当前函数和log函数
        
        # 获取完整文件路径
        filename = frame.f_code.co_filename
        function = frame.f_code.co_name
        lineno = frame.f_lineno
        
        # 获取模块名（相对于项目根目录的路径）
        module_path = self._module_name
        
        if not module_path:
            try:
                # 首先尝试通过__name__属性获取模块信息
                module_name = frame.f_globals.get('__name__', '')
                
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
            except Exception:
                # 记录异常但继续执行
                module_path = "unknown"
        
        return {
            "file": filename,
            "function": function,
            "line": lineno,
            "module": module_path or "unknown"
        }
        
    async def log(self, level: Union[LogLevel, str], message: str, **kwargs) -> None:
        """异步记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数
        """
        # 将日志级别转换为字符串
        level_str = level if isinstance(level, str) else level.value
        
        # 获取调用者信息
        caller_info = self._get_caller_info()
        
        # 创建记录
        record = {
            "level": level_str,
            "message": message,
            "timestamp": time.time(),
            **caller_info,
            **kwargs
        }
        
        # 添加到队列
        await self._queue.put(record)
    
    def debug(self, message: str, **kwargs) -> None:
        """记录调试级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外参数
        """
        asyncio.create_task(self.log(LogLevel.DEBUG, message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """记录信息级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外参数
        """
        asyncio.create_task(self.log(LogLevel.INFO, message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """记录警告级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外参数
        """
        asyncio.create_task(self.log(LogLevel.WARNING, message, **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """记录错误级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外参数
        """
        asyncio.create_task(self.log(LogLevel.ERROR, message, **kwargs))
    
    def critical(self, message: str, **kwargs) -> None:
        """记录严重错误级别日志
        
        Args:
            message: 日志消息
            **kwargs: 额外参数
        """
        asyncio.create_task(self.log(LogLevel.CRITICAL, message, **kwargs))
    
    def exception(self, message: str, exc_info: Optional[Exception] = None, **kwargs) -> None:
        """记录异常信息
        
        Args:
            message: 日志消息
            exc_info: 异常信息，如果为None则自动获取当前异常
            **kwargs: 额外参数
        """
        if exc_info is None:
            exc_info = sys.exc_info()[1]
            
        if exc_info:
            tb = traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__)
            kwargs["exception"] = "".join(tb)
        
        asyncio.create_task(self.log(LogLevel.ERROR, message, **kwargs)) 