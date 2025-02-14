"""
文档处理模块，支持多种类型文档的读取、解析和预处理
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ..core.exceptions import BaseError


class DocumentType(str, Enum):
    """文档类型"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"
    HTML = "html"


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


class TextDocument(Document):
    """文本文档"""
    
    def __init__(
        self,
        content: str,
        metadata: DocumentMetadata
    ):
        super().__init__(content, metadata)
        
    async def preprocess(self) -> None:
        """预处理文本，如清理、标准化等"""
        # 实现文本预处理逻辑
        pass
        
    async def to_text(self) -> str:
        """直接返回文本内容"""
        return self.content
        
    def get_size(self) -> int:
        """返回文本长度"""
        return len(self.content)


class ImageDocument(Document):
    """图像文档"""
    
    async def preprocess(self) -> None:
        """预处理图像，如调整大小、标准化等"""
        # 实现图像预处理逻辑
        pass
        
    async def to_text(self) -> str:
        """将图像转换为文本描述（如通过图像识别）"""
        # 实现图像到文本的转换
        raise NotImplementedError()
        
    def get_size(self) -> int:
        """返回图像大小（字节）"""
        # 实现图像大小计算
        raise NotImplementedError()


class DocumentParser(ABC):
    """文档解析器抽象基类"""
    
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
    """文档解析器注册表"""
    
    def __init__(self):
        self._parsers: Dict[DocumentType, DocumentParser] = {}
        
    def register(
        self,
        doc_type: DocumentType,
        parser: DocumentParser
    ) -> None:
        """注册解析器"""
        self._parsers[doc_type] = parser
        
    def get_parser(
        self,
        doc_type: DocumentType
    ) -> Optional[DocumentParser]:
        """获取解析器"""
        return self._parsers.get(doc_type)
        
    def list_supported_types(self) -> List[DocumentType]:
        """列出支持的文档类型"""
        return list(self._parsers.keys()) 