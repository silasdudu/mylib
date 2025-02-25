"""
轮转日志模块，提供按日期和大小轮转的日志功能
"""
import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from typing import Optional, Dict, Any, Union
import datetime

from .async_logger import AsyncLogHandler, LogLevel


class DailyRotatingLogger:
    """每日轮转日志记录器，每天创建新的日志文件"""
    
    def __init__(
        self, 
        log_dir: str, 
        base_filename: str, 
        level: Union[int, str] = logging.INFO,
        backup_count: int = 30,
        encoding: str = 'utf-8',
        format_str: Optional[str] = None
    ):
        """初始化每日轮转日志记录器
        
        Args:
            log_dir: 日志目录
            base_filename: 基础文件名
            level: 日志级别
            backup_count: 保留的备份文件数量
            encoding: 文件编码
            format_str: 日志格式字符串
        """
        self.log_dir = log_dir
        self.base_filename = base_filename
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 构建完整的日志文件路径
        log_path = os.path.join(log_dir, base_filename)
        
        # 创建处理器
        self.handler = TimedRotatingFileHandler(
            filename=log_path,
            when='midnight',  # 每天午夜轮转
            interval=1,       # 间隔为1天
            backupCount=backup_count,
            encoding=encoding
        )
        
        # 设置日志级别
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.handler.setLevel(level)
        
        # 设置格式化器
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)
        self.handler.setFormatter(formatter)
        
        # 创建日志记录器
        self.logger = logging.getLogger(f"daily_rotating_{base_filename}")
        self.logger.setLevel(level)
        self.logger.addHandler(self.handler)
        
        # 防止日志传播到根记录器
        self.logger.propagate = False
    
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
        self.handler.close()
        self.logger.removeHandler(self.handler)


class SizeRotatingLogger:
    """大小轮转日志记录器，当日志文件达到指定大小时创建新文件"""
    
    def __init__(
        self, 
        log_dir: str, 
        base_filename: str, 
        level: Union[int, str] = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 默认10MB
        backup_count: int = 5,
        encoding: str = 'utf-8',
        format_str: Optional[str] = None
    ):
        """初始化大小轮转日志记录器
        
        Args:
            log_dir: 日志目录
            base_filename: 基础文件名
            level: 日志级别
            max_bytes: 单个日志文件的最大字节数
            backup_count: 保留的备份文件数量
            encoding: 文件编码
            format_str: 日志格式字符串
        """
        self.log_dir = log_dir
        self.base_filename = base_filename
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 构建完整的日志文件路径
        log_path = os.path.join(log_dir, base_filename)
        
        # 创建处理器
        self.handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding
        )
        
        # 设置日志级别
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.handler.setLevel(level)
        
        # 设置格式化器
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)
        self.handler.setFormatter(formatter)
        
        # 创建日志记录器
        self.logger = logging.getLogger(f"size_rotating_{base_filename}")
        self.logger.setLevel(level)
        self.logger.addHandler(self.handler)
        
        # 防止日志传播到根记录器
        self.logger.propagate = False
    
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
        self.handler.close()
        self.logger.removeHandler(self.handler)


class DailyRotatingLogHandler(AsyncLogHandler):
    """每日轮转日志处理器，用于异步日志记录器"""
    
    def __init__(
        self, 
        log_dir: str, 
        base_filename: str, 
        backup_count: int = 30,
        encoding: str = 'utf-8'
    ):
        """初始化每日轮转日志处理器
        
        Args:
            log_dir: 日志目录
            base_filename: 基础文件名
            backup_count: 保留的备份文件数量
            encoding: 文件编码
        """
        self.log_dir = log_dir
        self.base_filename = base_filename
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 构建完整的日志文件路径
        self.log_path = os.path.join(log_dir, base_filename)
        
        # 创建处理器
        self.handler = TimedRotatingFileHandler(
            filename=self.log_path,
            when='midnight',  # 每天午夜轮转
            interval=1,       # 间隔为1天
            backupCount=backup_count,
            encoding=encoding
        )
        
        # 设置格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)
        
        # 创建日志记录器
        self.logger = logging.getLogger(f"async_daily_{base_filename}")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)  # 设置为最低级别，由记录决定是否记录
        
        # 防止日志传播到根记录器
        self.logger.propagate = False
    
    async def emit(self, record: Dict[str, Any]) -> None:
        """输出日志记录
        
        Args:
            record: 日志记录
        """
        # 映射日志级别
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        level = level_map.get(record["level"], logging.INFO)
        
        # 构建日志消息
        timestamp = datetime.datetime.fromtimestamp(record.get("timestamp", time.time()))
        module = record.get("module", "unknown")
        function = record.get("function", "unknown")
        line = record.get("line", 0)
        
        message = f"[{module}:{function}:{line}] {record['message']}"
        
        # 添加异常信息
        if "exception" in record:
            message += f"\n{record['exception']}"
        
        # 记录日志
        self.logger.log(level, message)
    
    async def close(self) -> None:
        """关闭处理器"""
        self.handler.close()
        self.logger.removeHandler(self.handler) 