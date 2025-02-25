"""
文件日志模块，提供简单的文件日志记录功能
"""
import os
import sys
import logging
import datetime
from typing import Optional, Dict, Any, Union

from .async_logger import AsyncLogHandler


class SimpleFileLogger:
    """简单文件日志记录器"""
    
    def __init__(
        self,
        name: str,
        log_file: str,
        level: Union[int, str] = logging.INFO,
        format_str: Optional[str] = None,
        encoding: str = 'utf-8',
        mode: str = 'a'
    ):
        """初始化简单文件日志记录器
        
        Args:
            name: 日志记录器名称
            log_file: 日志文件路径
            level: 日志级别
            format_str: 日志格式字符串
            encoding: 文件编码
            mode: 文件打开模式
        """
        self.name = name
        self.log_file = log_file
        
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志级别
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # 设置格式化器
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 防止日志传播到根记录器
        self.logger.propagate = False
        
        # 清除已有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(format_str)
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file, mode=mode, encoding=encoding)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """记录调试级别日志"""
        extra = kwargs.get('extra', {})
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs) -> None:
        """记录信息级别日志"""
        extra = kwargs.get('extra', {})
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs) -> None:
        """记录警告级别日志"""
        extra = kwargs.get('extra', {})
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs) -> None:
        """记录错误级别日志"""
        extra = kwargs.get('extra', {})
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs) -> None:
        """记录严重错误级别日志"""
        extra = kwargs.get('extra', {})
        self.logger.critical(message, extra=extra)
    
    def exception(self, message: str, exc_info=True, **kwargs) -> None:
        """记录异常信息"""
        extra = kwargs.get('extra', {})
        self.logger.exception(message, exc_info=exc_info, extra=extra)
    
    def close(self) -> None:
        """关闭日志记录器"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class FileHandler(AsyncLogHandler):
    """文件日志处理器，用于异步日志记录器"""
    
    def __init__(self, log_file: str, mode: str = 'a', encoding: str = 'utf-8'):
        """初始化文件日志处理器
        
        Args:
            log_file: 日志文件路径
            mode: 文件打开模式
            encoding: 文件编码
        """
        self.log_file = log_file
        self.mode = mode
        self.encoding = encoding
        self.file = None
        
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 打开日志文件
        self.file = open(log_file, mode=mode, encoding=encoding)
    
    async def emit(self, record: Dict[str, Any]) -> None:
        """输出日志记录
        
        Args:
            record: 日志记录
        """
        try:
            if self.file is None or self.file.closed:
                self.file = open(self.log_file, mode=self.mode, encoding=self.encoding)
            
            # 格式化时间戳
            timestamp = datetime.datetime.fromtimestamp(record.get('timestamp', datetime.datetime.now().timestamp()))
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # 获取日志级别
            level = record.get('level', 'INFO')
            
            # 获取模块、函数和行号
            module = record.get('module', 'unknown')
            function = record.get('function', 'unknown')
            line = record.get('line', 0)
            
            # 构建日志消息
            message = f"{time_str} - {level:8} - {module}:{function}:{line} - {record['message']}"
            
            # 添加异常信息
            if 'exception' in record:
                message += f"\n{record['exception']}"
            
            # 写入文件
            self.file.write(message + '\n')
            self.file.flush()
        except Exception as e:
            # 输出错误信息到标准错误
            print(f"文件日志处理器出错: {e}", file=sys.stderr)
    
    async def close(self) -> None:
        """关闭处理器"""
        if self.file is not None and not self.file.closed:
            self.file.flush()
            self.file.close()
            self.file = None 