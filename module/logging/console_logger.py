"""
控制台日志模块，提供彩色输出的控制台日志功能
"""
import sys
import logging
from typing import Optional, Dict, Any, Union
import datetime
import os

from .async_logger import AsyncLogHandler


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'BLACK': '\033[0;30m',
        'RED': '\033[0;31m',
        'GREEN': '\033[0;32m',
        'YELLOW': '\033[0;33m',
        'BLUE': '\033[0;34m',
        'MAGENTA': '\033[0;35m',
        'CYAN': '\033[0;36m',
        'WHITE': '\033[0;37m',
        'BOLD_BLACK': '\033[1;30m',
        'BOLD_RED': '\033[1;31m',
        'BOLD_GREEN': '\033[1;32m',
        'BOLD_YELLOW': '\033[1;33m',
        'BOLD_BLUE': '\033[1;34m',
        'BOLD_MAGENTA': '\033[1;35m',
        'BOLD_CYAN': '\033[1;36m',
        'BOLD_WHITE': '\033[1;37m',
        'RESET': '\033[0m',
    }
    
    # 日志级别对应的颜色
    LEVEL_COLORS = {
        logging.DEBUG: COLORS['BLUE'],
        logging.INFO: COLORS['GREEN'],
        logging.WARNING: COLORS['YELLOW'],
        logging.ERROR: COLORS['RED'],
        logging.CRITICAL: COLORS['BOLD_RED'],
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%', use_color=True):
        """初始化彩色格式化器
        
        Args:
            fmt: 格式字符串
            datefmt: 日期格式字符串
            style: 格式化风格
            use_color: 是否使用颜色
        """
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color
    
    def format(self, record):
        """格式化日志记录
        
        Args:
            record: 日志记录
            
        Returns:
            格式化后的日志字符串
        """
        # 获取原始格式化结果
        message = super().format(record)
        
        # 如果不使用颜色或者是Windows系统且不支持ANSI颜色，则直接返回
        if not self.use_color or (os.name == 'nt' and not self._win_supports_ansi()):
            return message
        
        # 添加颜色
        color = self.LEVEL_COLORS.get(record.levelno, self.COLORS['RESET'])
        return f"{color}{message}{self.COLORS['RESET']}"
    
    @staticmethod
    def _win_supports_ansi():
        """检查Windows是否支持ANSI颜色"""
        # Windows 10 build 10586+ 支持ANSI颜色
        if os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                return kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7) != 0
            except:
                return False
        return True


class ColoredConsoleLogger:
    """彩色控制台日志记录器"""
    
    def __init__(
        self, 
        name: str, 
        level: Union[int, str] = logging.INFO,
        format_str: Optional[str] = None,
        use_color: bool = True
    ):
        """初始化彩色控制台日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别
            format_str: 日志格式字符串
            use_color: 是否使用颜色
        """
        self.name = name
        
        # 设置日志级别
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # 设置格式化器
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 创建彩色格式化器
        formatter = ColoredFormatter(format_str, use_color=use_color)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 防止日志传播到根记录器
        self.logger.propagate = False
        
        # 清除已有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
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


class ColoredConsoleHandler(AsyncLogHandler):
    """彩色控制台日志处理器，用于异步日志记录器"""
    
    # 颜色代码
    COLORS = {
        'BLACK': '\033[0;30m',
        'RED': '\033[0;31m',
        'GREEN': '\033[0;32m',
        'YELLOW': '\033[0;33m',
        'BLUE': '\033[0;34m',
        'MAGENTA': '\033[0;35m',
        'CYAN': '\033[0;36m',
        'WHITE': '\033[0;37m',
        'BOLD_BLACK': '\033[1;30m',
        'BOLD_RED': '\033[1;31m',
        'BOLD_GREEN': '\033[1;32m',
        'BOLD_YELLOW': '\033[1;33m',
        'BOLD_BLUE': '\033[1;34m',
        'BOLD_MAGENTA': '\033[1;35m',
        'BOLD_CYAN': '\033[1;36m',
        'BOLD_WHITE': '\033[1;37m',
        'RESET': '\033[0m',
    }
    
    # 日志级别对应的颜色
    LEVEL_COLORS = {
        'DEBUG': COLORS['BLUE'],
        'INFO': COLORS['GREEN'],
        'WARNING': COLORS['YELLOW'],
        'ERROR': COLORS['RED'],
        'CRITICAL': COLORS['BOLD_RED'],
    }
    
    def __init__(self, use_color: bool = True, stream=sys.stdout):
        """初始化彩色控制台处理器
        
        Args:
            use_color: 是否使用颜色
            stream: 输出流
        """
        self.use_color = use_color
        self.stream = stream
    
    async def emit(self, record: Dict[str, Any]) -> None:
        """输出日志记录
        
        Args:
            record: 日志记录
        """
        try:
            # 获取日志级别和颜色
            level = record.get('level', 'INFO')
            color = self.LEVEL_COLORS.get(level, self.COLORS['RESET']) if self.use_color else ''
            reset = self.COLORS['RESET'] if self.use_color else ''
            
            # 格式化时间戳
            timestamp = datetime.datetime.fromtimestamp(record.get('timestamp', datetime.datetime.now().timestamp()))
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # 获取模块、函数和行号
            module = record.get('module', 'unknown')
            function = record.get('function', 'unknown')
            line = record.get('line', 0)
            
            # 构建日志消息
            message = f"{time_str} - {color}{level:8}{reset} - {module}:{function}:{line} - {record['message']}"
            
            # 添加异常信息
            if 'exception' in record:
                message += f"\n{record['exception']}"
            
            # 输出到流
            print(message, file=self.stream)
            self.stream.flush()
        except Exception as e:
            # 输出错误信息到标准错误
            print(f"日志处理器出错: {e}", file=sys.stderr)
    
    async def close(self) -> None:
        """关闭处理器"""
        # 控制台处理器不需要关闭 