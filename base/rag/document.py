"""
文档处理模块，支持多种类型文档的读取、解析和预处理
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from pydantic import BaseModel
from base.core.logging import AsyncLogger, LogLevel

from ..core.exceptions import BaseError


class DocumentType(str, Enum):
    """文档类型"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"
    HTML = "html"
    WORD = "word"
    EXCEL = "excel"
    MARKDOWN = "markdown"


class DocumentMetadata(BaseModel):
    """文档元数据"""
    doc_id: str
    doc_type: DocumentType
    source: str
    created_at: str
    updated_at: Optional[str] = None
    extra: Dict[str, Any] = {}


class Document(ABC):
    """文档抽象基类"""
    
    def __init__(
        self,
        content: Any,
        metadata: DocumentMetadata
    ):
        self.content = content
        self.metadata = metadata
        
    @abstractmethod
    async def preprocess(self) -> None:
        """预处理文档内容"""
        pass
    
    @abstractmethod
    async def to_text(self) -> str:
        """将文档转换为文本形式"""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """获取文档大小"""
        pass


class DocumentParser(ABC):
    """文档解析器抽象基类"""
    
    def __init__(self, logger: Optional[AsyncLogger] = None):
        """初始化解析器
        
        Args:
            logger: 可选的日志记录器
        """
        self.logger = logger
    
    async def _log(self, level: LogLevel, message: str):
        """内部日志记录方法
        
        Args:
            level: 日志级别
            message: 日志消息
        """
        if self.logger:
            await self.logger.log(level, message)
    
    @abstractmethod
    async def parse(
        self,
        file_path: Union[str, Path],
        doc_type: DocumentType
    ) -> Document:
        """解析文档"""
        pass
    
    @abstractmethod
    async def parse_batch(
        self,
        file_paths: List[Union[str, Path]],
        doc_types: List[DocumentType]
    ) -> List[Document]:
        """批量解析文档"""
        pass


class DocumentParserRegistry:
    """文档解析器注册表，管理文档类型、解析器和文件扩展名的映射关系"""
    
    def __init__(self):
        self._parsers: Dict[DocumentType, DocumentParser] = {}
        self._ext_mappings: Dict[str, DocumentType] = {}
        
    def register(
        self,
        doc_type: DocumentType,
        parser: DocumentParser,
        file_extensions: List[str]
    ) -> None:
        """注册解析器和文件扩展名映射
        
        Args:
            doc_type: 文档类型
            parser: 对应的解析器实例
            file_extensions: 支持的文件扩展名列表（如 ['.pdf', '.PDF']）
        """
        # 注册解析器
        self._parsers[doc_type] = parser
        
        # 注册文件扩展名映射
        for ext in file_extensions:
            # 统一转换为小写
            ext = ext.lower()
            if not ext.startswith('.'):
                ext = f'.{ext}'
            self._ext_mappings[ext] = doc_type
    
    def get_parser(
        self,
        doc_type: DocumentType
    ) -> Optional[DocumentParser]:
        """获取指定文档类型的解析器"""
        return self._parsers.get(doc_type)
    
    def get_parser_for_file(
        self,
        file_path: Union[str, Path]
    ) -> Tuple[DocumentParser, DocumentType]:
        """根据文件路径获取对应的解析器和文档类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析器实例和文档类型的元组
            
        Raises:
            ValueError: 当文件类型不支持时抛出
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        ext = file_path.suffix.lower()
        doc_type = self._ext_mappings.get(ext)
        
        if not doc_type:
            raise ValueError(f"不支持的文件类型: {ext}")
            
        parser = self._parsers.get(doc_type)
        if not parser:
            raise ValueError(f"未找到文档类型 {doc_type} 的解析器")
            
        return parser, doc_type
    
    def list_supported_types(self) -> List[DocumentType]:
        """列出所有支持的文档类型"""
        return list(self._parsers.keys())
    
    def list_supported_extensions(self) -> List[str]:
        """列出所有支持的文件扩展名"""
        return list(self._ext_mappings.keys())
    
    def is_supported_extension(self, extension: str) -> bool:
        """检查是否支持指定的文件扩展名
        
        Args:
            extension: 文件扩展名（如 '.pdf' 或 'pdf'）
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        return extension.lower() in self._ext_mappings 