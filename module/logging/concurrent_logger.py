"""
并发日志模块，提供线程安全的日志记录功能
"""
import os
import time
import logging
import threading
import queue
from typing import Optional, Dict, Any, List, Union, Callable
import traceback

from .async_logger import AsyncLogHandler


class ThreadSafeHandler:
    """线程安全的日志处理器基类"""
    
    def __init__(self):
        """初始化线程安全处理器"""
        self._lock = threading.RLock()
    
    def acquire(self):
        """获取锁"""
        self._lock.acquire()
    
    def release(self):
        """释放锁"""
        self._lock.release()
    
    def emit(self, record):
        """输出日志记录，需要子类实现"""
        raise NotImplementedError("子类必须实现emit方法")


class ThreadSafeFileHandler(ThreadSafeHandler, logging.FileHandler):
    """线程安全的文件日志处理器"""
    
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        """初始化线程安全文件处理器"""
        ThreadSafeHandler.__init__(self)
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)
    
    def emit(self, record):
        """线程安全地输出日志记录"""
        try:
            self.acquire()
            super().emit(record)
        finally:
            self.release()


class ConcurrentLogger:
    """并发日志记录器，支持多线程安全记录"""
    
    def __init__(
        self, 
        name: str, 
        log_dir: Optional[str] = None,
        level: Union[int, str] = logging.INFO,
        format_str: Optional[str] = None,
        console: bool = True,
        file: bool = True,
        encoding: str = 'utf-8'
    ):
        """初始化并发日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录，如果为None则不使用文件日志
            level: 日志级别
            format_str: 日志格式字符串
            console: 是否输出到控制台
            file: 是否输出到文件
            encoding: 文件编码
        """
        self.name = name
        self.log_dir = log_dir
        
        # 设置日志级别
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # 设置格式化器
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 防止日志传播到根记录器
        self.logger.propagate = False
        
        # 清除已有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 添加控制台处理器
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        if file and log_dir:
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            # 构建完整的日志文件路径
            log_path = os.path.join(log_dir, f"{name}.log")
            
            # 创建线程安全的文件处理器
            file_handler = ThreadSafeFileHandler(
                filename=log_path,
                encoding=encoding
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 创建线程锁
        self._lock = threading.RLock()
    
    def debug(self, message: str, **kwargs) -> None:
        """记录调试级别日志"""
        with self._lock:
            extra = kwargs.get('extra', {})
            self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs) -> None:
        """记录信息级别日志"""
        with self._lock:
            extra = kwargs.get('extra', {})
            self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs) -> None:
        """记录警告级别日志"""
        with self._lock:
            extra = kwargs.get('extra', {})
            self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs) -> None:
        """记录错误级别日志"""
        with self._lock:
            extra = kwargs.get('extra', {})
            self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs) -> None:
        """记录严重错误级别日志"""
        with self._lock:
            extra = kwargs.get('extra', {})
            self.logger.critical(message, extra=extra)
    
    def exception(self, message: str, exc_info=True, **kwargs) -> None:
        """记录异常信息"""
        with self._lock:
            extra = kwargs.get('extra', {})
            self.logger.exception(message, exc_info=exc_info, extra=extra)
    
    def close(self) -> None:
        """关闭日志记录器"""
        with self._lock:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)


class QueueHandler(logging.Handler):
    """队列日志处理器，将日志记录放入队列"""
    
    def __init__(self, queue):
        """初始化队列处理器
        
        Args:
            queue: 日志队列
        """
        super().__init__()
        self.queue = queue
    
    def emit(self, record):
        """将日志记录放入队列"""
        try:
            self.queue.put_nowait(record)
        except Exception:
            self.handleError(record)


class QueueListener:
    """队列监听器，从队列中获取日志记录并处理"""
    
    def __init__(self, queue, *handlers):
        """初始化队列监听器
        
        Args:
            queue: 日志队列
            *handlers: 日志处理器列表
        """
        self.queue = queue
        self.handlers = handlers
        self._stop_event = threading.Event()
        self._thread = None
    
    def start(self):
        """启动监听线程"""
        if self._thread is not None and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor)
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self):
        """停止监听线程"""
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self.queue.put_nowait(None)  # 发送停止信号
            self._thread.join()
            self._thread = None
    
    def _monitor(self):
        """监听队列并处理日志记录"""
        while not self._stop_event.is_set():
            try:
                record = self.queue.get()
                if record is None:  # 停止信号
                    break
                    
                for handler in self.handlers:
                    try:
                        if record.levelno >= handler.level:
                            handler.emit(record)
                    except Exception:
                        handler.handleError(record)
            except Exception:
                # 忽略异常，继续处理
                pass


class ThreadSafeAsyncHandler(AsyncLogHandler):
    """线程安全的异步日志处理器"""
    
    def __init__(self, handler: AsyncLogHandler):
        """初始化线程安全异步处理器
        
        Args:
            handler: 底层异步处理器
        """
        self._handler = handler
        self._lock = threading.RLock()
    
    async def emit(self, record: Dict[str, Any]) -> None:
        """线程安全地输出日志记录
        
        Args:
            record: 日志记录
        """
        with self._lock:
            await self._handler.emit(record)
    
    async def close(self) -> None:
        """关闭处理器"""
        with self._lock:
            await self._handler.close() 